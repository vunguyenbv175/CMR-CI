"""Slice and frame ordering with explicit ambiguity detection (Task D, R2).

Slice ordering is geometry-based and robust to vendor float noise:

- slice normals are compared with a cosine tolerance (never exact
  component equality), per the ROUND1_REVIEW tolerance policy;
- instances whose slice positions agree within a tolerance form one
  physical-slice cluster; the cluster is validated for in-plane
  consistency before a representative is used;
- the resulting order is monotonic along the slice normal — it is a
  geometric order, NOT a directional anatomical claim.

Frame ordering selects ONE coherent timing basis per slice:

- priority: TriggerTime (ms) > TemporalPositionIdentifier >
  enhanced-MR temporal index > InstanceNumber (degraded-confidence,
  always warned);
- keys from different bases are never interleaved;
- when >1 frames share a slice and no complete, unique basis exists,
  ordering refuses (fatal AMBIGUOUS_FRAME_ORDER) instead of guessing;
- the first frame is NOT claimed to be end-diastole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from cmr_core.errors import (
    AMBIGUOUS_FRAME_ORDER,
    AMBIGUOUS_SLICE_ORDER,
    INVALID_GEOMETRY,
    TIMING_DEGRADED,
    CMRError,
    CMRWarning,
    IssueSet,
)
from cmr_core.geometry import NORMAL_PARALLEL_COSINE, ImageGeometry, vec3_dot


@dataclass(frozen=True)
class SliceItem:
    """One instance participating in slice ordering."""

    geometry: ImageGeometry
    position: float
    source: str


@dataclass(frozen=True)
class FrameItem:
    """One instance participating in frame (cardiac phase) ordering."""

    source: str
    trigger_time_ms: float | None
    temporal_position: int | None
    instance_number: int | None
    position: float  # slice position, tie-break only
    enhanced_temporal_index: int | None = None


class TimingBasis(Enum):
    """A single coherent timing source chosen for one slice's frames."""

    TRIGGER_TIME = "trigger_time"
    TEMPORAL_POSITION = "temporal_position"
    ENHANCED_TEMPORAL_INDEX = "enhanced_temporal_index"
    INSTANCE_NUMBER = "instance_number"  # degraded-confidence fallback
    NONE = "none"


_PRIORITY_ATTRS: tuple[tuple[str, TimingBasis], ...] = (
    ("trigger_time_ms", TimingBasis.TRIGGER_TIME),
    ("temporal_position", TimingBasis.TEMPORAL_POSITION),
    ("enhanced_temporal_index", TimingBasis.ENHANCED_TEMPORAL_INDEX),
    ("instance_number", TimingBasis.INSTANCE_NUMBER),
)


def normals_parallel(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    tolerance_cos: float = NORMAL_PARALLEL_COSINE,
) -> bool:
    """True when two unit normals point along the same direction within a
    cosine tolerance. Anti-parallel (flipped) normals are NOT compatible
    for a single stack: ordering by signed position would flip sign
    depending on the reference normal."""
    dot = vec3_dot(a, b)
    return dot >= 1.0 - tolerance_cos


def cluster_slices(
    items: list[SliceItem],
    tolerance_mm: float = 0.5,
    issues: IssueSet | None = None,
) -> list[list[SliceItem]]:
    """Group instances into physical-slice clusters by position.

    Instances whose positions agree within ``tolerance_mm`` form one
    cluster (multiple cardiac phases of the same slice). Each cluster is
    validated: parallel slice normals (cosine tolerance) and consistent
    in-plane geometry (rows/cols, pixel spacing). Inconsistent clusters
    emit AMBIGUOUS_SLICE_ORDER warnings.
    """
    if not items:
        raise CMRError("input.empty", "cluster_slices called with no items")
    ordered = sorted(items, key=lambda it: it.position)
    ref_normal = ordered[0].geometry.slice_normal
    for it in ordered[1:]:
        dot = vec3_dot(ref_normal, it.geometry.slice_normal)
        if dot < 1.0 - NORMAL_PARALLEL_COSINE:
            raise CMRError(
                INVALID_GEOMETRY,
                f"slices have incompatible slice normals "
                f"{it.geometry.slice_normal} vs {ref_normal} "
                f"(cosine {dot:.6f} below {1.0 - NORMAL_PARALLEL_COSINE:.6f}); "
                f"tilted or flipped acquisition directions are not stackable",
                source=it.source,
            )
    clusters: list[list[SliceItem]] = []
    for it in ordered:
        if clusters and abs(it.position - clusters[-1][0].position) <= tolerance_mm:
            clusters[-1].append(it)
        else:
            clusters.append([it])
    for cluster in clusters:
        _validate_cluster_geometry(cluster, issues)
    return clusters


def _validate_cluster_geometry(cluster: list[SliceItem], issues: IssueSet | None = None) -> None:
    ref = cluster[0].geometry
    for it in cluster[1:]:
        g = it.geometry
        sp_match = math.isclose(
            g.pixel_spacing[0], ref.pixel_spacing[0], rel_tol=1e-3, abs_tol=1e-3
        ) and math.isclose(g.pixel_spacing[1], ref.pixel_spacing[1], rel_tol=1e-3, abs_tol=1e-3)
        if g.rows != ref.rows or g.cols != ref.cols or not sp_match:
            raise CMRError(
                INVALID_GEOMETRY,
                f"slice cluster at {cluster[0].position:.2f} mm has incompatible "
                f"in-plane geometry: {ref.rows}x{ref.cols} px {ref.pixel_spacing} mm "
                f"vs {g.rows}x{g.cols} px {g.pixel_spacing} mm; "
                "frames within the same physical slice must have compatible geometry",
                source=it.source,
            )


def order_slices(
    items: list[SliceItem],
    tolerance_mm: float = 0.5,
    issues: IssueSet | None = None,
) -> list[SliceItem]:
    """Order slice instances along their common slice normal.

    Instances whose positions agree within ``tolerance_mm`` are treated as
    the same physical slice: the first instance wins, extras are reported
    as AMBIGUOUS_SLICE_ORDER warnings. Raises a fatal :class:`CMRError`
    when orientations are incompatible (slices not parallel).
    """
    clusters = cluster_slices(items, tolerance_mm=tolerance_mm, issues=issues)
    result: list[SliceItem] = []
    for cluster in clusters:
        if len(cluster) > 1:
            for extra in cluster[1:]:
                if issues is not None:
                    issues.add(
                        CMRWarning(
                            AMBIGUOUS_SLICE_ORDER,
                            f"multiple instances at slice position {cluster[0].position:.2f} mm; "
                            f"keeping first ({cluster[0].source}), ignoring {extra.source}",
                            source=extra.source,
                        )
                    )
        result.append(cluster[0])
    return result


def select_timing_basis(
    items: list[FrameItem],
    issues: IssueSet | None = None,
) -> TimingBasis:
    """Choose one coherent timing basis for one slice's frames.

    Rules (ROUND2_INSTRUCTIONS 8.2):

    - a basis is usable when EVERY frame carries a value and the values
      are all distinct for that basis;
    - the highest-priority usable basis wins;
    - a fallback to InstanceNumber always emits a degraded-confidence
      warning; a fallback from a partially-present primary metadata also
      explains why (missing or duplicated);
    - when no usable basis exists, returns ``NONE`` (the caller must
      refuse automatic ordering).
    """
    n = len(items)
    if n <= 1:
        return TimingBasis.NONE

    present_partial: list[str] = []
    duplicate_reasons: list[str] = []
    picked: TimingBasis = TimingBasis.NONE
    for attr, basis in _PRIORITY_ATTRS:
        values = [getattr(it, attr) for it in items]
        present = [v for v in values if v is not None]
        if len(present) == 0:
            continue
        if len(present) < n and attr != "instance_number":
            present_partial.append(attr)
            continue
        if len(present) < n:
            continue
        if len(set(values)) == n:
            picked = basis
            break
        duplicate_reasons.append(attr)

    if picked is TimingBasis.TRIGGER_TIME or picked is TimingBasis.NONE:
        return picked

    if picked is TimingBasis.INSTANCE_NUMBER and issues is not None:
        reasons = present_partial + duplicate_reasons
        issues.add(
            CMRWarning(
                TIMING_DEGRADED,
                "no complete cardiac timing basis; falling back to InstanceNumber "
                f"({'mixed/duplicated: ' + ', '.join(reasons) if reasons else 'no timing'}) "
                "— degraded-confidence frame order",
                source=items[0].source,
            )
        )
    elif (
        picked is not TimingBasis.NONE
        and (present_partial or duplicate_reasons)
        and issues is not None
    ):
        issues.add(
            CMRWarning(
                AMBIGUOUS_FRAME_ORDER,
                f"cardiac timing metadata incomplete or duplicated "
                f"({'mixed: ' + ', '.join(present_partial) if present_partial else ''}"
                f"{'duplicated: ' + ', '.join(duplicate_reasons) if duplicate_reasons else ''}); "
                f"ordering by {picked.value}",
                source=items[0].source,
            )
        )
    return picked


def _timing_value(it: FrameItem, basis: TimingBasis) -> float | None:
    if basis is TimingBasis.TRIGGER_TIME:
        return it.trigger_time_ms
    if basis is TimingBasis.TEMPORAL_POSITION:
        return None if it.temporal_position is None else float(it.temporal_position)
    if basis is TimingBasis.ENHANCED_TEMPORAL_INDEX:
        return None if it.enhanced_temporal_index is None else float(it.enhanced_temporal_index)
    return None if it.instance_number is None else float(it.instance_number)


def order_frames(
    items: list[FrameItem],
    issues: IssueSet | None = None,
    basis: TimingBasis | None = None,
) -> list[FrameItem]:
    """Order cardiac phases of one slice using a single coherent basis.

    ``basis`` may be forced by the caller (e.g. validated enhanced-MR
    temporal index). Otherwise the basis is selected once for the whole
    slice; keys from different bases are never interleaved.

    Raises fatal AMBIGUOUS_FRAME_ORDER when >1 frames exist and no
    coherent basis can be selected.
    """
    if not items:
        raise CMRError("input.empty", "order_frames called with no items")
    if basis is None:
        basis = select_timing_basis(items, issues)
    if basis is TimingBasis.NONE:
        if len(items) <= 1:
            return list(items)
        raise CMRError(
            AMBIGUOUS_FRAME_ORDER,
            "no coherent cardiac timing basis (TriggerTime/TemporalPosition/"
            "temporal index/InstanceNumber absent or non-unique); "
            "refusing automatic quantitative reconstruction",
            source=items[0].source,
        )

    def key(it: FrameItem) -> tuple[int, float]:
        v = _timing_value(it, basis)
        if v is None:
            return (0, 0.0)  # frames lacking the (forced) key sort first, as before
        return (1, v)

    ordered = sorted(items, key=key)
    result: list[FrameItem] = []
    seen: set[tuple[int, float]] = set()
    for it in ordered:
        k = key(it)
        if k in seen:
            if issues is not None:
                v = _timing_value(it, basis)
                issues.add(
                    CMRWarning(
                        AMBIGUOUS_FRAME_ORDER,
                        f"duplicate frame timing key {v!r} for basis {basis.value}; "
                        f"keeping first instance, ignoring {it.source}",
                        source=it.source,
                    )
                )
            continue
        seen.add(k)
        result.append(it)
    return result
