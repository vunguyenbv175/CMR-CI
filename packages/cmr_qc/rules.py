"""Deterministic AI-QC rules (Task I, Phase 11 of the master plan).

Every rule is a pure function returning a list of QC-severity
:class:`CMRWarning` flags. Rules operate on contours (polygons) and on
volume curves — never on AI internals.
"""

from __future__ import annotations

from cmr_contours.model import Contour, ContourSet
from cmr_core.errors import (
    QC_CONTAINMENT,
    QC_MISSING_STRUCTURE,
    QC_SELF_INTERSECTION,
    QC_VOLUME_CURVE,
    CMRWarning,
)
from cmr_core.segmentation import Structure
from cmr_geometry.polygon import polygon_contains, polygon_self_intersects


def check_missing_structures(
    contour_set: ContourSet, expected: list[Structure]
) -> list[CMRWarning]:
    """Flag structures that should be present but are missing."""
    flags: list[CMRWarning] = []
    for s in expected:
        if s not in contour_set.contours:
            flags.append(
                CMRWarning.qc(
                    QC_MISSING_STRUCTURE,
                    f"structure {s.value} missing on slice {contour_set.slice_index}, "
                    f"frame {contour_set.frame_index}",
                    source=f"s{contour_set.slice_index}f{contour_set.frame_index}",
                )
            )
    return flags


def check_self_intersection(contour: Contour) -> list[CMRWarning]:
    """Flag contours whose non-adjacent edges cross (degenerate polygon)."""
    if polygon_self_intersects(contour.points):
        return [
            CMRWarning.qc(
                QC_SELF_INTERSECTION,
                f"{contour.structure.value} contour self-intersects on "
                f"slice {contour.slice_index}, frame {contour.frame_index}",
                source=f"s{contour.slice_index}f{contour.frame_index}",
            )
        ]
    return []


def check_containment(inner: Contour, outer: Contour) -> list[CMRWarning]:
    """Flag when ``inner`` is not fully inside ``outer``
    (e.g. endocardium not inside epicardium)."""
    if not polygon_contains(outer.points, inner.points):
        return [
            CMRWarning.qc(
                QC_CONTAINMENT,
                f"{inner.structure.value} contour is not fully inside {outer.structure.value} "
                f"on slice {inner.slice_index}, frame {inner.frame_index}",
                source=f"s{inner.slice_index}f{inner.frame_index}",
            )
        ]
    return []


def check_volume_curve(volumes: list[float], min_volume_ml: float = 0.0) -> list[CMRWarning]:
    """Flag implausible volume curves: negative volumes, flat curves,
    or ESV not < EDV."""
    flags: list[CMRWarning] = []
    if not volumes:
        return [CMRWarning.qc(QC_VOLUME_CURVE, "empty volume curve")]
    if any(v < min_volume_ml for v in volumes):
        flags.append(
            CMRWarning.qc(QC_VOLUME_CURVE, f"volume curve contains values below {min_volume_ml} ml")
        )
    if len(volumes) > 1 and max(volumes) == min(volumes):
        flags.append(CMRWarning.qc(QC_VOLUME_CURVE, "volume curve is flat (max == min)"))
    if len(volumes) > 1 and min(volumes) > max(volumes):
        flags.append(CMRWarning.qc(QC_VOLUME_CURVE, "ESV exceeds EDV in volume curve"))
    return flags


def run_qc_on_contour_set(
    contour_set: ContourSet,
    expected: list[Structure] | None = None,
    inner_outer: list[tuple[Structure, Structure]] | None = None,
) -> list[CMRWarning]:
    """Run the standard rule bundle on one (slice, frame) contour set."""
    expected = expected or list(Structure)
    inner_outer = inner_outer or [(Structure.LV_CAVITY, Structure.LV_MYOCARDIUM)]
    flags: list[CMRWarning] = check_missing_structures(contour_set, expected)
    for c in contour_set.contours.values():
        flags.extend(check_self_intersection(c))
    for inner_s, outer_s in inner_outer:
        inner = contour_set.get(inner_s)
        outer = contour_set.get(outer_s)
        if inner is not None and outer is not None:
            flags.extend(check_containment(inner, outer))
    return flags
