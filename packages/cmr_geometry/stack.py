"""Stack-level geometry: IPP-derived spacing and consistency checks (R2).

Slice positions come from projecting each ImagePositionPatient onto the
validated slice normal (``slice_position_mm`` in ``transforms``). All
downstream spacing values are derived from actual positions, NOT from
SpacingBetweenSlices or SliceThickness:

- ``neighbor_spacings_mm``: distance between consecutive sorted positions;
- ``stack_gaps_mm``: per-slice slab length for volumetry; interior slices
  use the mid-distance between their two neighbours, end slices use the
  single neighbour distance;
- ``spacing_metadata_consistency``: warns when SpacingBetweenSlices
  disagrees with the IPP-derived neighbour spacing.

Per-instance ``SliceThickness`` describes the excited slab, not the
centre-to-centre distance; it is used only as a last-resort fallback when
IPP gives no usable stack (single slice).
"""

from __future__ import annotations

from cmr_core.errors import INVALID_GEOMETRY, CMRError, CMRWarning, IssueSet
from cmr_core.geometry import ImageGeometry

from .transforms import slice_position_mm

STACK_POSITION_TOLERANCE_MM = 0.5
SPACING_METADATA_TOLERANCE_MM = 0.5


def stack_positions_mm(geometries: list[ImageGeometry]) -> list[float]:
    """Sorted slice positions (mm) along the common slice normal."""
    if not geometries:
        raise CMRError("input.empty", "stack_positions_mm called with no geometries")
    return sorted(slice_position_mm(g) for g in geometries)


def neighbor_spacings_mm(
    geometries: list[ImageGeometry],
    position_tolerance_mm: float = STACK_POSITION_TOLERANCE_MM,
) -> list[float]:
    """Consecutive centre-to-centre distances (mm) for a sorted stack.

    Raises when fewer than 2 slices are given, or when two slices share a
    position within ``position_tolerance_mm`` (duplicate positions make
    the stacking ambiguous).
    """
    positions = stack_positions_mm(geometries)
    if len(positions) < 2:
        raise CMRError(
            INVALID_GEOMETRY,
            "at least 2 slices are required to derive IPP stack spacing",
        )
    _validate_unique_positions(positions, position_tolerance_mm)
    return [b - a for a, b in zip(positions, positions[1:])]


def _validate_unique_positions(positions: list[float], tolerance_mm: float) -> None:
    for a, b in zip(positions, positions[1:]):
        if abs(b - a) <= tolerance_mm:
            raise CMRError(
                INVALID_GEOMETRY,
                f"duplicate slice positions {a:.2f} mm and {b:.2f} mm within "
                f"{tolerance_mm} mm tolerance; clean slice stack required for IPP spacing",
            )


def is_uniform_spacing_mm(spacings: list[float], tolerance_mm: float = 0.05) -> bool:
    """True when all neighbour spacings agree within ``tolerance_mm``."""
    if not spacings:
        return True
    reference = spacings[0]
    return all(abs(s - reference) <= tolerance_mm for s in spacings[1:])


def average_neighbor_spacing_mm(geometries: list[ImageGeometry]) -> float:
    spacings = neighbor_spacings_mm(geometries)
    return sum(spacings) / len(spacings)


def stack_gaps_mm(
    geometries: list[ImageGeometry],
    position_tolerance_mm: float = STACK_POSITION_TOLERANCE_MM,
) -> list[float]:
    """Per-slice slab length (mm) in sorted-position order.

    Interior slice i: (pos[i+1] - pos[i-1]) / 2 (mid-distance to its two
    neighbours). End slices: the single neighbour distance. Raises when
    fewer than 2 slices exist — a single-slice stack has no IPP-derived
    spacing; the caller must fall back to per-instance metadata.
    """
    positions = stack_positions_mm(geometries)
    if len(positions) < 2:
        raise CMRError(
            INVALID_GEOMETRY,
            "cannot derive IPP gaps for a single-slice stack; "
            "fall back to ImageGeometry.slice_gap_mm",
        )
    _validate_unique_positions(positions, position_tolerance_mm)
    n = len(positions)
    gaps: list[float] = []
    for i in range(n):
        if i == 0:
            gaps.append(positions[1] - positions[0])
        elif i == n - 1:
            gaps.append(positions[-1] - positions[-2])
        else:
            gaps.append((positions[i + 1] - positions[i - 1]) / 2.0)
    return gaps


def spacing_metadata_consistency(
    geometries: list[ImageGeometry],
    issues: IssueSet | None = None,
    tolerance_mm: float = SPACING_METADATA_TOLERANCE_MM,
) -> None:
    """Warn when SpacingBetweenSlices disagrees with IPP-derived spacing.

    IPP-derived positions are the source of truth for stack spacing;
    SpacingBetweenSlices is a nominal value that may be absent or stale.
    A mismatch beyond ``tolerance_mm`` is reported, never silently relied
    on. Silently returns when the stack is too small or SBS is absent.
    """
    if issues is None or len(geometries) < 2:
        return
    sbs = [g.spacing_between_slices for g in geometries]
    if any(v is None for v in sbs):
        return
    try:
        neighbor = average_neighbor_spacing_mm(geometries)
    except CMRError:
        return
    first = sbs[0]
    assert first is not None  # any-None guard above already returned
    if abs(first - neighbor) > tolerance_mm:
        issues.add(
            CMRWarning(
                "geometry.spacing_mismatch",
                f"SpacingBetweenSlices ({sbs[0]} mm) disagrees with IPP-derived "
                f"neighbour spacing ({neighbor:.3f} mm, tolerance {tolerance_mm} mm); "
                "volumetry uses IPP-derived spacing",
            )
        )
