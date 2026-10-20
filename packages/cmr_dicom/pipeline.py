"""Cine reconstruction pipeline (Task D, R2): folder -> CineVolume.

Composition of: scan -> series grouping -> per-instance geometry + timing
extraction (classic single-frame AND first-path Enhanced MR multi-frame)
-> tolerance-based physical-slice clustering -> coherent temporal ordering
per slice -> canonical :class:`CineVolume`.

Ambiguity and missing data are surfaced as warnings; nothing is silently
guessed. When reconstruction is impossible (empty series, incompatible
geometry, unprovable frame order, ragged frame counts), the pipeline
returns None plus warnings. Slice ordering is geometric along the slice
normal — never a directional anatomical claim.

S2C-001 cluster A (manifest-backed source identity): every file is read
EXACTLY ONCE through :func:`cmr_dicom.reader.read_dicom_auto_with_digest`,
so the SHA-256 recorded in the source manifest hashes the very bytes that
were parsed (no hash/read TOCTOU gap). Path, study UID, series UID, SOP UID
and file digest are retained through reconstruction; on success the pipeline
builds a :class:`~cmr_core.source_manifest.DicomSourceManifest` from the
files that actually contributed frames to the accepted cine, computes the
decoded-volume digest and attaches BOTH identities to the returned
:class:`CineVolume` (verified at construction) and the returned
:class:`~cmr_core.study.CMRSeries`. The public contract is UNCHANGED: a
``(CineVolume | None, CMRSeries | None)`` two-tuple. When contributor
identity is incomplete (e.g. instances without ``StudyInstanceUID``) the
reconstruction succeeds LEGACY-UNBOUND instead — no manifest, no silent
guessing. An EXPLICIT ``series_uid`` argument stays authoritative: blank or
whitespace-only strings are REJECTED (never silently treated as "auto"),
and one selected series UID appearing under MULTIPLE study UIDs fails
closed. Selection uses ``series_uid is not None``, never truthiness.

Entry point note: ``cmr_dicom/cmr_series`` does not exist as a module.
The real entry point for folder -> cine reconstruction is
:func:`cmr_dicom.pipeline.load_cine_volume` (re-exported as
``cmr_dicom.load_cine_volume``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from cmr_core.cine import CardiacFrame, CineVolume, compute_cine_volume_digest
from cmr_core.errors import (
    AMBIGUOUS_FRAME_ORDER,
    INVALID_ARGUMENT,
    MISSING_PIXEL_DATA,
    MULTIFRAME_UNSUPPORTED,
    SERIES_NOT_A_CINE,
    TIMING_DEGRADED,
    CMRError,
    CMRWarning,
    IssueSet,
)
from cmr_core.geometry import ImageGeometry
from cmr_core.source_manifest import DicomSourceFile, DicomSourceManifest, normalize_relative_path
from cmr_core.study import CMRSeries
from cmr_geometry.ordering import (
    FrameItem,
    SliceItem,
    TimingBasis,
    cluster_slices,
    normals_parallel,
    order_frames,
    order_slices,
    select_timing_basis,
)
from cmr_geometry.stack import spacing_metadata_consistency
from cmr_geometry.transforms import slice_position_mm

from . import tags as T
from .classify import classify_series
from .enhanced import extract_enhanced_frames, is_enhanced_mr
from .metadata import extract_geometry, extract_timing
from .reader import iter_dicom_files, read_dicom_auto_with_digest

if TYPE_CHECKING:
    from pydicom.dataset import Dataset

SLICE_CLUSTER_TOLERANCE_MM = 0.5
ORIENTATION_SPLIT = T.ORIENTATION_SPLIT


class _FileOrigin:
    """Identity of ONE contributing source file (S2C-001).

    Carried verbatim through reconstruction so the manifest can be built from
    exactly the files whose frames survived into the accepted cine.
    """

    __slots__ = ("path", "relative_path", "sop_instance_uid", "file_sha256", "study_uid")

    def __init__(
        self,
        *,
        path: Path,
        relative_path: str,
        sop_instance_uid: str,
        file_sha256: str,
        study_uid: str | None,
    ) -> None:
        self.path = path
        self.relative_path = relative_path
        self.sop_instance_uid = sop_instance_uid
        self.file_sha256 = file_sha256
        self.study_uid = study_uid


class _OrientationGroup:
    """Group of instances sharing a parallel slice-normal vector (R2)."""

    __slots__ = ("representative_normal", "candidates", "distinct_positions")

    def __init__(
        self,
        representative_normal: tuple[float, float, float],
        first_candidate: tuple[CardiacFrame, dict, ImageGeometry, str, int],
    ) -> None:
        self.representative_normal = representative_normal
        self.candidates: list[tuple[CardiacFrame, dict, ImageGeometry, str, int]] = [
            first_candidate
        ]
        self.distinct_positions: list[float] = []



def load_cine_volume(
    directory: Path,
    series_uid: str | None = None,
    issues: IssueSet | None = None,
    slice_tolerance_mm: float = SLICE_CLUSTER_TOLERANCE_MM,
) -> tuple[CineVolume | None, CMRSeries | None]:
    """Scan ``directory`` and reconstruct a cine volume.

    Args:
        directory: folder with DICOM files (files only, non-recursive).
        series_uid: optional selection; when ``None`` the first series with
            usable cine data is chosen. A blank/whitespace-only string is
            REJECTED fail-closed (S2C-001: no default fallback for an
            explicit-but-blank request); a requested UID appearing under more
            than one study UID is rejected as well.
        issues: shared issue collector (created if None).
        slice_tolerance_mm: physical-slice clustering tolerance (mm);
            instances closer than this form one slice cluster (multiple
            cardiac phases of the same slice must share a position).

    Returns:
        ``(CineVolume, CMRSeries)`` on success — with manifest-bound source
        identities attached whenever contributor identity is complete;
        ``(None, None)`` when no usable cine series exists (warnings describe
        why).
    """
    issues = IssueSet() if issues is None else issues
    # S2C-001: distinguish None (auto) from an explicit-but-blank request.
    if series_uid is not None and (
        not isinstance(series_uid, str) or not series_uid.strip()
    ):
        raise CMRError(
            INVALID_ARGUMENT,
            f"explicit series_uid must be a non-blank string, got {series_uid!r}; "
            "pass None to auto-select instead",
            source=str(directory),
        )
    files = iter_dicom_files(directory, issues.warnings)

    series_map: dict[str, list[tuple[Dataset, str, _FileOrigin]]] = {}
    for f in files:
        ds, file_sha256 = read_dicom_auto_with_digest(f, issues.warnings)
        if ds is None or file_sha256 is None:
            continue
        try:
            suid = T.get_uid(ds, T.series_uid_key, issues.warnings)
            sop = T.get_uid(ds, T.instance_uid_key, issues.warnings)
        except Exception as exc:  # noqa: BLE001 - invalid series uid reported as warning
            issues.add(CMRWarning("dicom.invalid_series", str(exc), source=str(f)))
            continue
        try:
            study_uid = T.get_uid(ds, T.study_uid_key, issues.warnings)
        except Exception:  # noqa: BLE001 - missing study UID keeps legacy-unbound path alive
            study_uid = None
            issues.add(
                CMRWarning(
                    T.MISSING_OPTIONAL_TAG,
                    "StudyInstanceUID missing; this instance cannot participate "
                    "in a manifest-bound workflow identity",
                    source=str(f),
                )
            )
        origin = _FileOrigin(
            path=f,
            relative_path=normalize_relative_path(str(f.relative_to(directory))),
            sop_instance_uid=sop,
            file_sha256=file_sha256,
            study_uid=None if study_uid is None else str(study_uid),
        )
        series_map.setdefault(suid, []).append((ds, sop, origin))

    if not series_map:
        issues.add(CMRWarning(SERIES_NOT_A_CINE, "no DICOM series found", source=str(directory)))
        return None, None

    # S2C-001: one selected series UID under MULTIPLE study UIDs is a
    # cross-study splice hazard -> reject instead of guessing.
    if series_uid is not None and series_uid in series_map:
        bound_studies = {
            origin.study_uid
            for _ds, _sop, origin in series_map[series_uid]
            if origin.study_uid is not None
        }
        if len(bound_studies) > 1:
            raise CMRError(
                INVALID_ARGUMENT,
                f"requested series_uid {series_uid!r} appears under multiple "
                f"study UIDs {sorted(bound_studies)!r}; refusing to reconstruct "
                "a series that crosses study boundaries",
                source=str(directory),
            )

    # Classification is annotation-only; preserve the pre-classifier default
    # ordering. An explicit series UID remains authoritative below.
    candidates = (
        [suid for suid in series_map if suid == series_uid]
        if series_uid is not None
        else sorted(series_map)
    )
    for suid in candidates:
        volume, series = _build_one(series_map[suid], issues, slice_tolerance_mm)
        if volume is not None:
            return volume, series
    issues.add(CMRWarning(SERIES_NOT_A_CINE, "no series could be reconstructed as a cine volume"))
    return None, None


def _can_partition_by_trigger_time(
    ordered_reps: list[SliceItem],
    cluster_by_position: dict[float, list[SliceItem]],
    entry_by_source: dict[str, tuple[CardiacFrame, dict, ImageGeometry, str, int]],
    n_frames_per_slice: int,
) -> bool:
    """True when instances can be partitioned into a consistent cine grid by TriggerTime."""
    if n_frames_per_slice <= 1:
        return True

    slice_tts: list[list[float]] = []
    for rep in ordered_reps:
        cluster = cluster_by_position[round(rep.position, 6)]
        tts: list[float] = []
        for c in cluster:
            entry = entry_by_source[c.source]
            tt = entry[1].get("trigger_time_ms")
            if tt is None:
                return False
            tts.append(float(tt))
        if len(tts) != n_frames_per_slice or len(set(tts)) != n_frames_per_slice:
            return False
        slice_tts.append(sorted(tts))

    if len(slice_tts) <= 1:
        return True

    ref_tts = slice_tts[0]
    for other_tts in slice_tts[1:]:
        for t_ref, t_other in zip(ref_tts, other_tts):
            if not math.isclose(t_ref, t_other, abs_tol=1e-3):
                return False

    return True


def _mark_temporal_axis(
    target: object,
    is_derived: bool,
    phase_axis: list[float] | None = None,
) -> None:
    setattr(target, "is_temporal_axis_derived", is_derived)
    setattr(target, "temporal_axis_derived", is_derived)
    setattr(target, "is_derived_phase_axis", is_derived)
    if phase_axis is not None:
        setattr(target, "phase_axis", list(phase_axis))
        setattr(target, "temporal_axis", list(phase_axis))


def _mark_manifest_temporal_axis(
    manifest: DicomSourceManifest | None,
    is_derived: bool,
    phase_axis: list[float] | None = None,
) -> None:
    if manifest is None:
        return
    if phase_axis is not None:
        object.__setattr__(manifest, "phase_axis", list(phase_axis))
        object.__setattr__(manifest, "temporal_axis", list(phase_axis))


def _build_one(
    entries: list[tuple[Dataset, str, _FileOrigin]],
    issues: IssueSet,
    slice_tolerance_mm: float,
) -> tuple[CineVolume | None, CMRSeries | None]:
    """Reconstruct one series. Returns (None, None) + warnings on failure."""
    sops = [sop for _ds, sop, _origin in entries]
    if len(sops) != len(set(sops)):
        issues.add(
            CMRWarning(
                AMBIGUOUS_FRAME_ORDER,
                "duplicate SOPInstanceUID found in series; "
                "refusing to reconstruct ambiguous frame set",
            )
        )
        return None, None

    origins: list[_FileOrigin] = []
    candidates: list[tuple[CardiacFrame, dict, ImageGeometry, str, int]] = []
    source_desc = ""
    source_protocol = ""
    source_manufacturer = ""

    def _origin_index(origin: _FileOrigin) -> int:
        try:
            return origins.index(origin)
        except ValueError:
            origins.append(origin)
            return len(origins) - 1

    for ds, sop, origin in entries:
        source_desc = source_desc or str(getattr(ds, "SeriesDescription", "") or "")
        source_protocol = source_protocol or str(getattr(ds, "ProtocolName", "") or "")
        source_manufacturer = source_manufacturer or str(getattr(ds, "Manufacturer", "") or "")
        origin_idx = _origin_index(origin)
        if is_enhanced_mr(ds):
            for frame in extract_enhanced_frames(ds, issues.warnings):
                timing = {
                    "trigger_time_ms": frame.trigger_time_ms,
                    "temporal_position": frame.temporal_position,
                    "instance_number": None,
                    "number_of_frames": 1,
                    "cardiac_number_of_images": None,
                    "enhanced_temporal_index": frame.enhanced_temporal_index,
                }
                candidates.append(
                    (
                        CardiacFrame(
                            slice_index=-1,
                            frame_index=-1,
                            pixel_data=frame.pixel_data,
                            sop_instance_uid=sop,
                            trigger_time_ms=frame.trigger_time_ms,
                        ),
                        timing,
                        frame.geometry,
                        frame.source,
                        origin_idx,
                    )
                )
            continue
        try:
            geom = extract_geometry(ds, issues.warnings)
        except Exception as exc:  # noqa: BLE001 - reported as warning, instance skipped
            issues.add(CMRWarning("geometry.invalid", str(exc), source=sop))
            continue
        timing = extract_timing(ds, issues.warnings)
        if timing["number_of_frames"] and timing["number_of_frames"] > 1:
            issues.add(
                CMRWarning(
                    MULTIFRAME_UNSUPPORTED,
                    "multi-frame (non-Enhanced) DICOM instances are not supported in this cycle; "
                    "skipping instance",
                    source=sop,
                )
            )
            continue
        try:
            pixels = ds.pixel_array.astype("float32", copy=False)
        except Exception as exc:  # noqa: BLE001 - missing/unsupported pixel data
            issues.add(CMRWarning(MISSING_PIXEL_DATA, f"pixel data unavailable: {exc}", source=sop))
            continue
        candidates.append(
            (
                CardiacFrame(
                    slice_index=-1,
                    frame_index=-1,
                    pixel_data=pixels,
                    sop_instance_uid=sop,
                    trigger_time_ms=timing["trigger_time_ms"],
                ),
                timing,
                geom,
                sop,
                origin_idx,
            )
        )

    if not candidates:
        issues.add(
            CMRWarning(SERIES_NOT_A_CINE, "series has no usable instances", source=source_desc)
        )
        return None, None

    # --- orientation-group splitting: multiple slice normals in same series (R2) ---
    first_normal = candidates[0][2].slice_normal
    if not all(normals_parallel(first_normal, cand[2].slice_normal) for cand in candidates[1:]):
        groups: list[_OrientationGroup] = []
        for cand in candidates:
            c_norm = cand[2].slice_normal
            matched_group: _OrientationGroup | None = None
            for g in groups:
                if normals_parallel(g.representative_normal, c_norm) and all(
                    normals_parallel(existing[2].slice_normal, c_norm) for existing in g.candidates
                ):
                    matched_group = g
                    break
            if matched_group is not None:
                matched_group.candidates.append(cand)
            else:
                groups.append(_OrientationGroup(c_norm, cand))

        for g in groups:
            positions = sorted(slice_position_mm(cand[2]) for cand in g.candidates)
            distinct_positions: list[float] = []
            for pos in positions:
                if not distinct_positions or abs(pos - distinct_positions[-1]) > slice_tolerance_mm:
                    distinct_positions.append(pos)
            g.distinct_positions = distinct_positions

        # Deterministic ranking to select the largest orientation group:
        # 1. Most instances (highest len(candidates))
        # 2. Most distinct slice positions (tie-break)
        # 3. Deterministic key: rounded slice-normal vector, then lowest SOPInstanceUID
        groups.sort(
            key=lambda g: (
                -len(g.candidates),
                -len(g.distinct_positions),
                (
                    round(g.representative_normal[0], 6),
                    round(g.representative_normal[1], 6),
                    round(g.representative_normal[2], 6),
                ),
                min(str(cand[0].sop_instance_uid or cand[3]) for cand in g.candidates),
            )
        )

        chosen_group = groups[0]
        excluded_groups = groups[1:]

        chosen_instance_count = len(chosen_group.candidates)
        chosen_slice_count = len(chosen_group.distinct_positions)
        excluded_instance_count = sum(len(g.candidates) for g in excluded_groups)
        excluded_group_count = len(excluded_groups)

        excluded_details = ", ".join(
            f"group {i + 1} ({len(g.candidates)} inst, "
            f"normal {[round(v, 4) for v in g.representative_normal]})"
            for i, g in enumerate(excluded_groups)
        )
        series_uid_str = str(getattr(entries[0][0], "SeriesInstanceUID", str(entries[0][1])))
        issues.add(
            CMRWarning(
                ORIENTATION_SPLIT,
                f"orientation group split: retained largest slab with "
                f"{chosen_instance_count} instances ({chosen_slice_count} slices); "
                f"excluded {excluded_instance_count} instances across "
                f"{excluded_group_count} groups [{excluded_details}] "
                f"due to incompatible slice normals in same series",
                source=source_desc or series_uid_str,
            )
        )

        candidates = chosen_group.candidates
        chosen_sops = {cand[0].sop_instance_uid for cand in candidates}
        entries = [e for e in entries if e[1] in chosen_sops]

        source_desc = ""
        source_protocol = ""
        source_manufacturer = ""
        for ds, _sop, _orig in entries:
            source_desc = source_desc or str(getattr(ds, "SeriesDescription", "") or "")
            source_protocol = source_protocol or str(getattr(ds, "ProtocolName", "") or "")
            source_manufacturer = source_manufacturer or str(getattr(ds, "Manufacturer", "") or "")
            if source_desc and source_protocol and source_manufacturer:
                break

    # --- physical-slice clustering: tolerance-based, validated geometry ---
    entry_by_source: dict[str, tuple[CardiacFrame, dict, ImageGeometry, str, int]] = {
        entry[3]: entry for entry in candidates
    }
    slice_items = [
        SliceItem(geometry=geom, position=slice_position_mm(geom), source=src)
        for _frame, _timing, geom, src, _idx in candidates
    ]
    try:
        clusters = cluster_slices(slice_items, tolerance_mm=slice_tolerance_mm, issues=issues)
    except Exception as exc:  # noqa: BLE001 - fatal geometry surfaces as warning, series refused
        issues.add(CMRWarning("geometry.invalid", str(exc)))
        return None, None

    # --- slice ordering: geometry-based (monotonic along the normal) ---
    representatives = [cluster[0] for cluster in clusters]
    try:
        ordered_reps = order_slices(representatives, tolerance_mm=slice_tolerance_mm, issues=issues)
    except Exception as exc:  # noqa: BLE001
        issues.add(CMRWarning("geometry.invalid", str(exc)))
        return None, None
    cluster_by_position = {round(c[0].position, 6): c for c in clusters}

    # Verify all slice clusters carry the SAME number of instances (fail-closed if not)
    cluster_sizes = [len(cluster_by_position[round(rep.position, 6)]) for rep in ordered_reps]
    distinct_sizes = set(cluster_sizes)
    if len(distinct_sizes) > 1:
        issues.add(
            CMRWarning(
                "cine.incomplete_frames",
                f"incomplete frame set: slices have {sorted(distinct_sizes)} frame counts; "
                "refusing to guess missing phases",
                source=source_desc,
            )
        )
        return None, None

    n_frames_per_slice = cluster_sizes[0] if cluster_sizes else 0
    if n_frames_per_slice == 0:
        issues.add(
            CMRWarning(SERIES_NOT_A_CINE, "series has no ordered frames", source=source_desc)
        )
        return None, None

    series_uid = str(getattr(entries[0][0], "SeriesInstanceUID", str(entries[0][1])))

    has_enhanced = any(is_enhanced_mr(ds) for ds, _, _ in entries)
    if has_enhanced or _can_partition_by_trigger_time(
        ordered_reps, cluster_by_position, entry_by_source, n_frames_per_slice
    ):
        # --- PRIMARY STRATEGY: measured TriggerTime / coherent timing basis per slice ---
        frames_out: list[list[CardiacFrame]] = []
        used_origin_indexes: set[int] = set()
        geometries_out: list[ImageGeometry] = []
        timing_bases: list[TimingBasis] = []
        refused = False
        for rep in ordered_reps:
            cluster = cluster_by_position[round(rep.position, 6)]
            geometries_out.append(rep.geometry)
            frame_items = [
                FrameItem(
                    source=src,
                    trigger_time_ms=entry[1]["trigger_time_ms"],
                    temporal_position=entry[1]["temporal_position"],
                    instance_number=entry[1]["instance_number"],
                    enhanced_temporal_index=entry[1]["enhanced_temporal_index"],
                    position=rep.position,
                )
                for src, entry in ((c.source, entry_by_source[c.source]) for c in cluster)
            ]
            basis = select_timing_basis(frame_items, issues=issues)
            timing_bases.append(basis)
            try:
                ordered = order_frames(frame_items, issues=issues, basis=basis)
            except Exception as exc:  # noqa: BLE001 - unprovable frame order: refuse reconstruction
                issues.add(CMRWarning(AMBIGUOUS_FRAME_ORDER, f"{exc}", source=rep.source))
                refused = True
                break
            frames_out.append([entry_by_source[fi.source][0] for fi in ordered])
            for fi in ordered:
                used_origin_indexes.add(entry_by_source[fi.source][4])
        if refused:
            return None, None

        if not frames_out or any(len(f) == 0 for f in frames_out):
            issues.add(
                CMRWarning(SERIES_NOT_A_CINE, "series has no ordered frames", source=source_desc)
            )
            return None, None

        counts = {len(f) for f in frames_out}
        if len(counts) > 1:
            issues.add(
                CMRWarning(
                    "cine.incomplete_frames",
                    f"incomplete frame set: slices have {sorted(counts)} frame counts; "
                    "refusing to guess missing phases",
                    source=source_desc,
                )
            )
            return None, None

        for s, slice_frames in enumerate(frames_out):
            for f_index, f in enumerate(slice_frames):
                f.slice_index = s
                f.frame_index = f_index

        spacing_metadata_consistency(geometries_out, issues)
        is_degraded = any(b is TimingBasis.INSTANCE_NUMBER for b in timing_bases)
        primary_basis = timing_bases[0].value if timing_bases else None
        volume = CineVolume(
            geometries=geometries_out,
            frames=frames_out,
            is_timing_degraded=is_degraded,
            timing_basis=primary_basis,
            warnings=list(issues.warnings),
        )
        _mark_temporal_axis(volume, False)

        manifest = _build_source_manifest(
            origins,
            used_origin_indexes,
            volume,
            series_uid,
            issues,
            temporal_axis_provenance="measured",
        )
        if manifest is not None:
            _mark_manifest_temporal_axis(manifest, False)
            volume = CineVolume(
                geometries=geometries_out,
                frames=frames_out,
                is_timing_degraded=is_degraded,
                timing_basis=primary_basis,
                warnings=list(issues.warnings),
                source_manifest=manifest,
            )
            _mark_temporal_axis(volume, False)

        series = CMRSeries(
            series_uid=series_uid,
            series_description=source_desc,
            protocol_name=source_protocol,
            manufacturer=source_manufacturer,
            sop_instance_uids=[sop for _ds, sop, _origin in entries],
            geometry=geometries_out[0],
            warnings=list(issues.warnings),
            study_uid=None if manifest is None else manifest.study_uid,
            workflow_id=None if manifest is None else manifest.workflow_id,
            cine_volume_digest=(
                None if manifest is None else manifest.cine_volume_digest
            ),
        )
        _mark_temporal_axis(series, False)
        series.sequence_class = classify_series(series)
        return volume, series

    # --- FALLBACK STRATEGY: phase-grouping when TriggerTime cannot partition into cine grid ---
    # 1. Slice groups were formed by cluster_slices (ordered_reps).
    # 2. Within each slice cluster, order instances deterministically
    #    by InstanceNumber (fallback: file path sort).
    frames_out = []
    used_origin_indexes = set()
    geometries_out = []
    for rep in ordered_reps:
        cluster = cluster_by_position[round(rep.position, 6)]
        geometries_out.append(rep.geometry)

        def _instance_sort_key(c: SliceItem) -> tuple[int, int, str]:
            entry = entry_by_source[c.source]
            inst_num = entry[1].get("instance_number")
            origin = origins[entry[4]]
            path_key = str(origin.relative_path or origin.path.name or c.source)
            inst_order = 0 if inst_num is not None else 1
            inst_val = inst_num if inst_num is not None else 0
            return (inst_order, inst_val, path_key)

        ordered_cluster = sorted(cluster, key=_instance_sort_key)
        slice_frames = [entry_by_source[c.source][0] for c in ordered_cluster]
        for c in ordered_cluster:
            used_origin_indexes.add(entry_by_source[c.source][4])
        frames_out.append(slice_frames)

    # 4. Synthesize a monotonically increasing phase axis for the manifest when TriggerTime
    # is absent/non-groupable: use equal-count partitioning based on CardiacNumberOfImages
    # when present, else the per-slice instance count.
    cardiac_num: int | None = None
    for cand in candidates:
        c_val = cand[1].get("cardiac_number_of_images")
        if c_val is not None and c_val > 0:
            cardiac_num = c_val
            break
    if cardiac_num is None:
        for ds, _, _ in entries:
            t = extract_timing(ds)
            c_val = t.get("cardiac_number_of_images")
            if c_val is not None and c_val > 0:
                cardiac_num = c_val
                break

    if cardiac_num is not None and cardiac_num != n_frames_per_slice:
        issues.add(
            CMRWarning(
                "cine.incomplete_frames",
                f"incomplete frame set: CardiacNumberOfImages ({cardiac_num}) differs from "
                f"frames per slice ({n_frames_per_slice}); refusing to guess missing phases",
                source=source_desc or series_uid,
            )
        )
        return None, None

    n_phases = cardiac_num if cardiac_num is not None else n_frames_per_slice

    nominal_rr = 1000.0
    for ds, _, _ in entries:
        nom = getattr(ds, "NominalInterval", None)
        if nom is not None:
            try:
                nom_val = float(nom)
                if nom_val > 0:
                    nominal_rr = nom_val
                    break
            except (ValueError, TypeError):
                pass
        hr = getattr(ds, "HeartRate", None)
        if hr is not None:
            try:
                hr_val = float(hr)
                if hr_val > 0:
                    nominal_rr = 60000.0 / hr_val
                    break
            except (ValueError, TypeError):
                pass

    dt = nominal_rr / float(n_phases)
    derived_phase_axis = [float(p * dt) for p in range(n_frames_per_slice)]

    for s, slice_frames in enumerate(frames_out):
        for p, f in enumerate(slice_frames):
            f.slice_index = s
            f.frame_index = p
            f.trigger_time_ms = derived_phase_axis[p]

    # Record clearly (warning/issue) that the temporal axis is DERIVED, not measured
    issues.add(
        CMRWarning(
            TIMING_DEGRADED,
            "cardiac temporal phase axis is DERIVED, not measured "
            f"(TriggerTime absent or non-groupable across slices; "
            f"synthesized from equal-count partitioning across {n_phases} phases)",
            source=source_desc or series_uid,
        )
    )
    issues.add(
        CMRWarning(
            "timing.derived_axis",
            f"cardiac temporal phase axis is DERIVED, not measured "
            f"(equal-count partitioning across {n_phases} phases)",
            source=source_desc or series_uid,
        )
    )

    spacing_metadata_consistency(geometries_out, issues)
    volume = CineVolume(
        geometries=geometries_out,
        frames=frames_out,
        is_timing_degraded=True,
        timing_basis="instance_number",
        warnings=list(issues.warnings),
    )
    _mark_temporal_axis(volume, True, derived_phase_axis)

    manifest = _build_source_manifest(
        origins,
        used_origin_indexes,
        volume,
        series_uid,
        issues,
        temporal_axis_provenance="derived",
    )
    if manifest is not None:
        _mark_manifest_temporal_axis(manifest, True, derived_phase_axis)
        volume = CineVolume(
            geometries=geometries_out,
            frames=frames_out,
            is_timing_degraded=True,
            timing_basis="instance_number",
            warnings=list(issues.warnings),
            source_manifest=manifest,
        )
        _mark_temporal_axis(volume, True, derived_phase_axis)

    series = CMRSeries(
        series_uid=series_uid,
        series_description=source_desc,
        protocol_name=source_protocol,
        manufacturer=source_manufacturer,
        sop_instance_uids=[sop for _ds, sop, _origin in entries],
        geometry=geometries_out[0],
        warnings=list(issues.warnings),
        study_uid=None if manifest is None else manifest.study_uid,
        workflow_id=None if manifest is None else manifest.workflow_id,
        cine_volume_digest=(
            None if manifest is None else manifest.cine_volume_digest
        ),
    )
    _mark_temporal_axis(series, True, derived_phase_axis)
    series.sequence_class = classify_series(series)
    return volume, series


def _build_source_manifest(
    origins: list[_FileOrigin],
    used_origin_indexes: set[int],
    volume: CineVolume,
    series_uid: str,
    issues: IssueSet,
    temporal_axis_provenance: Literal["measured", "derived"] = "measured",
) -> DicomSourceManifest | None:
    """Build the manifest from contributing files, or fall back to unbound.

    Identity requires every contributing file to carry a readable, agreeing
    StudyInstanceUID; otherwise the reconstruction stays legacy-unbound (a
    documented degradation, never a guess).
    """
    contributing = [origins[index] for index in sorted(used_origin_indexes)]
    if not contributing:
        return None
    study_uids = {origin.study_uid for origin in contributing}
    if None in study_uids or len(study_uids) != 1:
        issues.add(
            CMRWarning(
                SERIES_NOT_A_CINE,
                "contributing instances carry missing or conflicting "
                "StudyInstanceUIDs; returning a legacy-unbound cine volume "
                "without a source manifest",
                source=contributing[0].path.name,
            )
        )
        return None
    resolved_study_uid = next(uid for uid in study_uids if uid is not None)
    try:
        files = tuple(
            DicomSourceFile(
                relative_path=origin.relative_path,
                sop_instance_uid=origin.sop_instance_uid,
                file_sha256=origin.file_sha256,
            )
            for origin in sorted(
                contributing, key=lambda item: (item.relative_path, item.sop_instance_uid)
            )
        )
        return DicomSourceManifest(
            study_uid=resolved_study_uid,
            series_uid=series_uid,
            cine_volume_digest=compute_cine_volume_digest(volume),
            files=files,
            temporal_axis_provenance=temporal_axis_provenance,
        )
    except (ValueError, CMRError) as exc:
        issues.add(
            CMRWarning(
                SERIES_NOT_A_CINE,
                f"source manifest could not be built ({exc}); returning a "
                "legacy-unbound cine volume",
                source=contributing[0].path.name,
            )
        )
        return None
