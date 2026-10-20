"""Filesystem staging for one library case (slice 2).

Thin copy layer over the frozen :func:`cmr_dicom.discovery.scan_library`
grouping: given a :class:`~cmr_dicom.discovery.LibraryScan` produced from a
source directory, :func:`stage_case` copies exactly one study's files into a
fresh destination laid out as ``dest_root/dicom/<SeriesUID>/`` flat files so
the EXISTING folder loader (:func:`cmr_dicom.pipeline.load_cine_volume`) and
the EXISTING :meth:`cmr_workflow.controller.WorkflowController.select_series`
verification path keep working unchanged. No new DICOM parse path, no sqlite
writes here, no clinical arithmetic, no active-study switch.

Fail-closed validation order (no partial state on refusal):

1. ``study_uid`` must be a non-blank string and must exist in ``scan``
   (``library.unknown_study``) — checked BEFORE any filesystem mutation, so
   an unknown study never creates ``dest_root``.
2. ``source_root`` (the directory ``scan`` was produced from) is required —
   the scan carries relative paths only, so overlap guards cannot run
   without it (``library.bad_path`` when missing/not-a-directory).
3. ``dest_root`` must not equal ``source_root``
   (``library.dest_equals_source``) and must not lie inside it
   (``library.dest_inside_source``).
4. ``dest_root`` must not exist, or must be an existing EMPTY real
   directory (``library.dest_not_empty`` otherwise — covers existing
   files, non-directories and symlinks).
5. Every staged filename mapping is pre-validated (per-series basename
   collisions across nested source subdirectories refuse with
   ``library.filename_collision``; missing source files refuse with
   ``library.source_missing``) BEFORE anything is created or copied.

Copies use :func:`shutil.copy2` (byte-preserving) in sorted series/file
order, so staging is deterministic. The HTTP layer lives in
``services/workstation/api.py`` (additive routes only); this module never
touches the network, the host/origin policy, or application state.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cmr_core.errors import INVALID_ARGUMENT, CMRError
from cmr_dicom.discovery import LibraryScan

UNKNOWN_STUDY = "library.unknown_study"
DEST_NOT_EMPTY = "library.dest_not_empty"
DEST_INSIDE_SOURCE = "library.dest_inside_source"
DEST_EQUALS_SOURCE = "library.dest_equals_source"
BAD_PATH = "library.bad_path"
SOURCE_MISSING = "library.source_missing"
FILENAME_COLLISION = "library.filename_collision"


@dataclass(frozen=True)
class StagedCase:
    """Identity of one staged study root (JSON-serializable)."""

    study_uid: str
    dest_root: str
    series_uids: tuple[str, ...] = ()
    file_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_uid": self.study_uid,
            "dest_root": self.dest_root,
            "series_uids": list(self.series_uids),
            "file_count": self.file_count,
        }


def scan_to_payload(scan: LibraryScan) -> dict[str, Any]:
    """Render a :class:`LibraryScan` as the frozen ``/api/library/scan`` body.

    Studies and series keep the discovery sort order (by UID); per-file
    paths are NOT exposed (counts + descriptors only). The shape is
    additive over the contract minimum (``protocol_name`` and
    ``file_count`` ride along for browser display).
    """
    studies: list[dict[str, Any]] = []
    for study in scan.studies:
        studies.append({
            "study_uid": study.study_uid,
            "file_count": study.file_count,
            "series": [
                {
                    "series_uid": series.series_uid,
                    "count": series.file_count,
                    "series_description": series.series_description,
                    "protocol_name": series.protocol_name,
                    "modality": series.modality,
                }
                for series in study.series
            ],
        })
    return {
        "studies": studies,
        "errors": [
            {"path": err.relative_path, "reason": err.reason} for err in scan.errors
        ],
    }


def stage_case(
    scan: LibraryScan,
    study_uid: str,
    dest_root: str | Path,
    source_root: str | Path | None = None,
) -> StagedCase:
    """Copy one scanned study into ``dest_root/dicom/<SeriesUID>/`` flat files.

    Args:
        scan: grouping produced from ``source_root`` via ``scan_library``.
        study_uid: exact StudyInstanceUID to stage (no guessing, no default).
        dest_root: fresh destination (must not exist or must be an empty dir;
            must not equal nor lie inside ``source_root``).
        source_root: directory ``scan`` was produced from. Required: the
            scan stores relative paths only, so copying and the
            inside-source guard cannot run without it. ``None`` (or a
            missing/non-directory value) refuses fail-closed with
            ``library.bad_path`` and copies nothing.

    Returns:
        The staged identity (study UID, resolved destination, sorted series
        UIDs, total file count).

    Raises:
        CMRError: typed fail-closed refusals (unknown study, bad/occupied
            destination, source overlap, missing source file, filename
            collision). An unknown study never creates ``dest_root``.
    """
    if not isinstance(study_uid, str) or not study_uid.strip():
        raise CMRError(
            INVALID_ARGUMENT,
            f"study_uid must be a non-blank string, got {study_uid!r}",
        )
    matches = [entry for entry in scan.studies if entry.study_uid == study_uid]
    if not matches:
        raise CMRError(
            UNKNOWN_STUDY,
            "unknown study_uid refused; no destination was created",
            source=study_uid,
        )
    study = matches[0]

    if source_root is None or not isinstance(source_root, (str, Path)):
        raise CMRError(
            BAD_PATH,
            "source_root is required: the scan carries relative paths only",
        )
    source_path = Path(source_root)
    if not source_path.is_dir() or os.path.islink(source_path):
        raise CMRError(
            BAD_PATH,
            f"source root is not a real directory: {source_path}",
        )
    resolved_source = source_path.resolve()

    dest_path = Path(dest_root)
    resolved_dest = dest_path.resolve()
    if resolved_dest == resolved_source:
        raise CMRError(
            DEST_EQUALS_SOURCE,
            "destination must not equal the scan source root",
        )
    if resolved_source in resolved_dest.parents:
        raise CMRError(
            DEST_INSIDE_SOURCE,
            "destination must not lie inside the scan source root",
        )

    if os.path.islink(dest_path) or (
        dest_path.exists() and not dest_path.is_dir()
    ):
        raise CMRError(
            DEST_NOT_EMPTY,
            "destination must be a fresh path or an empty directory",
        )
    if dest_path.is_dir() and any(dest_path.iterdir()):
        raise CMRError(
            DEST_NOT_EMPTY,
            "destination directory is not empty; refusing to overwrite",
        )

    # Pre-validate the full copy plan BEFORE creating anything: every
    # source file must exist, and per-series basenames must not collide
    # (nested source subdirectories flatten into one series directory).
    plan: list[tuple[Path, Path]] = []
    for series in sorted(study.series, key=lambda entry: entry.series_uid):
        seen: set[str] = set()
        for relative in sorted(series.files):
            name = Path(relative).name
            if not name or name in seen:
                raise CMRError(
                    FILENAME_COLLISION,
                    f"series {series.series_uid!r} has colliding staged "
                    f"filename {name!r}; refusing rather than overwriting",
                    source=relative,
                )
            seen.add(name)
            origin = resolved_source / Path(relative)
            if not origin.is_file() or os.path.islink(origin):
                raise CMRError(
                    SOURCE_MISSING,
                    f"scan-listed source file is missing: {relative}",
                    source=relative,
                )
            plan.append(
                (origin, resolved_dest / "dicom" / series.series_uid / name)
            )

    series_uids = sorted({series.series_uid for series in study.series})
    for series_uid in series_uids:
        (resolved_dest / "dicom" / series_uid).mkdir(parents=True, exist_ok=True)
    for origin, target in plan:
        shutil.copy2(origin, target)

    return StagedCase(
        study_uid=study_uid,
        dest_root=str(resolved_dest),
        series_uids=tuple(series_uids),
        file_count=len(plan),
    )


__all__ = [
    "BAD_PATH",
    "DEST_EQUALS_SOURCE",
    "DEST_INSIDE_SOURCE",
    "DEST_NOT_EMPTY",
    "FILENAME_COLLISION",
    "SOURCE_MISSING",
    "UNKNOWN_STUDY",
    "StagedCase",
    "scan_to_payload",
    "stage_case",
]
