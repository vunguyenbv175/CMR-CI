"""ImageGeometry: physical image geometry with explicit validation.

Round 2 decision (ROUND2_INSTRUCTIONS 12, Option A): the ``ImageGeometry``
value object lives in ``cmr_core`` — it is core imaging-domain metadata and
is referenced by domain objects without importing an algorithm package.
Algorithms (transforms, ordering, clustering) stay in ``cmr_geometry`` and
import this type.

Validation policy (ROUND1_REVIEW 4.1): vendor floating-point variation is
handled by cosine/angular tolerances, never exact component comparison:

- row and column direction vectors must be finite, unit length within
  ``ORIENT_UNIT_TOLERANCE``, near-orthogonal within
  ``ORIENT_ORTHO_TOLERANCE`` (cosine), and their cross product must be
  non-degenerate (norm > ``ORIENT_CROSS_MIN``).

Normalized direction vectors are exposed via ``row_direction`` /
``col_direction`` / ``slice_normal`` so downstream math uses unit vectors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import INVALID_GEOMETRY, MISSING_ESSENTIAL_GEOMETRY, CMRError

# ImageOrientationPatient: row direction (x,y,z) cosine + column direction (x,y,z) cosine
OrientationTuple = tuple[float, float, float, float, float, float]

DEFAULT_ORIENTATION: OrientationTuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

# Cosine tolerance: |abs(dot) - 1| <= tol counts as unit length / parallel.
ORIENT_UNIT_TOLERANCE = 1e-3
# Cosine tolerance for near-orthogonality of row vs column directions.
ORIENT_ORTHO_TOLERANCE = 1e-3
# Minimum allowed norm of cross(row, col) (degenerate-plane guard).
ORIENT_CROSS_MIN = 1e-6
# Cosine tolerance for "same slice normal direction" in stack validation.
NORMAL_PARALLEL_COSINE = 1e-3


def _validate_vec3(label: str, v: tuple[float, float, float]) -> None:
    if not all(math.isfinite(x) for x in v):
        raise CMRError(INVALID_GEOMETRY, f"{label} direction must be finite, got {v}")


def vec3_norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = vec3_norm(v)
    if n <= 0:
        raise CMRError(INVALID_GEOMETRY, f"cannot normalize zero-length vector {v}")
    return (v[0] / n, v[1] / n, v[2] / n)


def vec3_dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize_orientation(
    orientation: OrientationTuple,
    unit_tolerance: float = ORIENT_UNIT_TOLERANCE,
    ortho_tolerance: float = ORIENT_ORTHO_TOLERANCE,
    cross_min: float = ORIENT_CROSS_MIN,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Validate an IOP tuple approximately and return normalized directions.

    Checks (ROUND1_REVIEW 4.1): finite components, unit-length row and
    column directions, near-orthogonality, non-degenerate cross product.
    Raises :class:`CMRError` on violation.
    """
    if len(orientation) != 6:
        raise CMRError(INVALID_GEOMETRY, "orientation must have 6 components")
    row = (orientation[0], orientation[1], orientation[2])
    col = (orientation[3], orientation[4], orientation[5])
    _validate_vec3("row", row)
    _validate_vec3("column", col)
    row_norm = vec3_norm(row)
    col_norm = vec3_norm(col)
    if abs(row_norm - 1.0) > unit_tolerance:
        raise CMRError(
            INVALID_GEOMETRY,
            f"row direction is not unit length (norm {row_norm:.6f}, tolerance {unit_tolerance})",
        )
    if abs(col_norm - 1.0) > unit_tolerance:
        raise CMRError(
            INVALID_GEOMETRY,
            f"column direction is not unit length "
            f"(norm {col_norm:.6f}, tolerance {unit_tolerance})",
        )
    row_u = vec3_normalize(row)
    col_u = vec3_normalize(col)
    cos_angle = abs(vec3_dot(row_u, col_u))
    if cos_angle > ortho_tolerance:
        raise CMRError(
            INVALID_GEOMETRY,
            f"row and column directions are not near-orthogonal "
            f"(|dot| = {cos_angle:.6f} > {ortho_tolerance})",
        )
    cross = vec3_cross(row_u, col_u)
    if vec3_norm(cross) <= cross_min:
        raise CMRError(
            INVALID_GEOMETRY, "row and column directions are degenerate (zero cross product)"
        )
    return row_u, col_u


def _positive(label: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0.0):
        raise CMRError(INVALID_GEOMETRY, f"{label} must be finite and > 0, got {value}")


@dataclass(frozen=True)
class ImageGeometry:
    """Physical interpretation of one 2D image grid.

    Attributes:
        rows, cols: pixel grid dimensions.
        pixel_spacing: (row spacing mm, col spacing mm) — DICOM convention
            (first value is between rows, i.e. along the column axis of the
            image plane; we keep the DICOM order and expose helpers).
        slice_thickness: slice thickness in mm (may be None). This is the
            excited slab thickness, NOT the distance between neighbouring
            slice centres; use stack-level IPP-derived spacing for volumetry.
        spacing_between_slices: nominal gap in mm (may be None); a fallback
            for volumetry, not more authoritative than actual IPP positions.
        image_position_patient: (x, y, z) mm of the centre of the first
            pixel (top-left of the image plane).
        image_orientation_patient: 6 cosines (row dir, col dir), validated
            and normalized internally before use.
    """

    rows: int
    cols: int
    pixel_spacing: tuple[float, float]
    slice_thickness: float | None = None
    spacing_between_slices: float | None = None
    image_position_patient: tuple[float, float, float] = (0.0, 0.0, 0.0)
    image_orientation_patient: OrientationTuple = DEFAULT_ORIENTATION

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise CMRError(INVALID_GEOMETRY, f"rows/cols must be > 0, got {self.rows}x{self.cols}")
        _positive("pixel_spacing[0]", self.pixel_spacing[0])
        _positive("pixel_spacing[1]", self.pixel_spacing[1])
        if self.slice_thickness is not None:
            _positive("slice_thickness", self.slice_thickness)
        if self.spacing_between_slices is not None:
            _positive("spacing_between_slices", self.spacing_between_slices)
        if self.image_position_patient is None or len(self.image_position_patient) != 3:
            raise CMRError(INVALID_GEOMETRY, "image position must have 3 components")
        if not all(math.isfinite(v) for v in self.image_position_patient):
            raise CMRError(INVALID_GEOMETRY, "image position must be finite")
        if len(self.image_orientation_patient) != 6:
            raise CMRError(INVALID_GEOMETRY, "orientation must have 6 components")
        normalize_orientation(self.image_orientation_patient)

    # --- derived quantities -------------------------------------------------

    @property
    def row_spacing_mm(self) -> float:
        return self.pixel_spacing[0]

    @property
    def col_spacing_mm(self) -> float:
        return self.pixel_spacing[1]

    @property
    def pixel_area_mm2(self) -> float:
        return self.row_spacing_mm * self.col_spacing_mm

    @property
    def slice_gap_mm(self) -> float | None:
        """Nominal per-instance gap: spacing-between-slices preferred, else
        slice thickness. Use only as a fallback when actual stack geometry
        (IPP-derived neighbouring slice distances) is unavailable."""
        if self.spacing_between_slices is not None and self.spacing_between_slices > 0:
            return self.spacing_between_slices
        return self.slice_thickness

    @property
    def row_direction(self) -> tuple[float, float, float]:
        """Normalized row direction cosine."""
        return vec3_normalize(self.image_orientation_patient[0:3])

    @property
    def col_direction(self) -> tuple[float, float, float]:
        """Normalized column direction cosine."""
        return vec3_normalize(self.image_orientation_patient[3:6])

    @property
    def slice_normal(self) -> tuple[float, float, float]:
        """Unit normal of the image plane = cross(row, col), normalized."""
        return vec3_normalize(vec3_cross(self.row_direction, self.col_direction))


def validate_geometry_present(
    geometry: ImageGeometry | None, source: str = "series"
) -> ImageGeometry:
    """Raise a fatal CMRError when essential geometry is absent."""
    if geometry is None:
        raise CMRError(
            MISSING_ESSENTIAL_GEOMETRY,
            f"essential geometry missing for {source}",
            source=source,
        )
    return geometry


def geometry_with_gap(geometry: ImageGeometry) -> ImageGeometry:
    """Return geometry with a mandatory slab gap (falls back to thickness)."""
    if geometry.slice_gap_mm is None:
        raise CMRError(
            MISSING_ESSENTIAL_GEOMETRY,
            "neither SpacingBetweenSlices nor SliceThickness is available; "
            "volume integration impossible",
            source=str(geometry),
        )
    return geometry


# --- S3-001 hole-aware polygon predicates (design spec §2.1 + §4 conventions) --
#
# Binding conventions (CMR_S3_AI_001_DESIGN_SPEC §4):
# - Area = shoelace (Green's theorem) on PIXEL coordinates; exterior CCW yields
#   a positive signed area, holes are CW. Net region area =
#   ``|signed_area(exterior)| - sum(|signed_area(hole_i)|)``.
# - Coordinates are pixel-space ``(x, y)`` matching ``cmr_contours.Contour``;
#   ring closure is implicit (the first vertex is never repeated at the end).
# - Exact predicates: segment-intersection tests use orientation cross-products
#   with ZERO epsilon; degeneracy is handled by an explicit collinear-overlap
#   branch, never by tolerance fudging.
#
# This module must stay importable from ``cmr_core.domain_types`` without an
# import cycle, so these helpers speak plain point tuples — no domain imports.

#: Signed-area magnitude below which a ring counts as degenerate/collinear.
_AREA_EPSILON = 1e-9

RingPoint = tuple[float, float]
RingPoints = tuple[RingPoint, ...]


def _signed_area(points: Sequence[RingPoint]) -> float:
    """Shoelace signed area on pixel coordinates.

    Counter-clockwise vertex order yields a POSITIVE area; clockwise yields
    negative. The ring is implicitly closed (last -> first edge included).
    """
    total = 0.0
    count = len(points)
    for index in range(count):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % count]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def _orientation(
    a: RingPoint,
    b: RingPoint,
    c: RingPoint,
) -> float:
    """Exact orientation cross-product of (b - a) x (c - a), zero epsilon."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _collinear_point_on_segment(a: RingPoint, b: RingPoint, p: RingPoint) -> bool:
    """Whether collinear point ``p`` lies within the bounding box of a->b.

    Callers MUST establish collinearity first (``_orientation(a, b, p) == 0``);
    this predicate only bounds the parameter range, it never tolerates
    non-zero cross products.
    """
    return (
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def _segments_intersect(p1: RingPoint, p2: RingPoint, p3: RingPoint, p4: RingPoint) -> bool:
    """Exact closed-segment intersection test (CLRS style).

    Proper crossings are detected by opposite-sign orientation pairs. The
    explicit collinear-overlap branch covers endpoint-on-segment contact and
    collinear overlap with zero epsilon instead of tolerance fudging.
    """
    d1 = _orientation(p3, p4, p1)
    d2 = _orientation(p3, p4, p2)
    d3 = _orientation(p1, p2, p3)
    d4 = _orientation(p1, p2, p4)
    if ((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)) and (
        (d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0)
    ):
        return True
    if d1 == 0.0 and _collinear_point_on_segment(p3, p4, p1):
        return True
    if d2 == 0.0 and _collinear_point_on_segment(p3, p4, p2):
        return True
    if d3 == 0.0 and _collinear_point_on_segment(p1, p2, p3):
        return True
    if d4 == 0.0 and _collinear_point_on_segment(p1, p2, p4):
        return True
    return False


def _validate_no_self_intersection(points: Sequence[RingPoint]) -> None:
    """Reject any self-intersection of the implicitly closed ring.

    O(n^2) segment-pair sweep with exact predicates. Adjacent segments share a
    vertex by construction and are skipped for the shared-endpoint contact;
    every other contact — including a doubled-back edge whose endpoint lies on
    a non-incident segment — fails closed.
    """
    count = len(points)
    segments: list[tuple[RingPoint, RingPoint]] = [
        (points[i], points[(i + 1) % count]) for i in range(count)
    ]
    for i in range(count):
        for j in range(i + 1, count):
            adjacent = j == i + 1 or (i == 0 and j == count - 1)
            if adjacent:
                continue
            p1, p2 = segments[i]
            p3, p4 = segments[j]
            if _segments_intersect(p1, p2, p3, p4):
                raise ValueError(
                    "polygon ring self-intersection between segments "
                    f"{i} and {j}"
                )


def _point_strictly_inside(point: RingPoint, ring: Sequence[RingPoint]) -> bool:
    """Exact strict-containment test against one implicitly closed ring.

    Boundary contact (vertex on an edge) returns ``False`` by construction:
    callers treat touching as a topology violation before consulting this
    predicate. Interior detection uses the half-open even-odd ray cast, which
    is deterministic for vertex-on-ray degeneracies.
    """
    px, py = point
    count = len(ring)
    for index in range(count):
        a = ring[index]
        b = ring[(index + 1) % count]
        if _orientation(a, b, point) == 0.0 and _collinear_point_on_segment(a, b, point):
            return False  # boundary/touching contact is never "strictly inside"
    inside = False
    previous = count - 1
    for index in range(count):
        xi, yi = ring[index]
        xj, yj = ring[previous]
        if (yi > py) != (yj > py):
            x_cross = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_cross:
                inside = not inside
        previous = index
    return inside


def _validate_hole_topology(
    exterior: Sequence[RingPoint],
    holes: Sequence[Sequence[RingPoint]],
) -> None:
    """Fail closed on any hole-topology violation (spec §2.1 steps 1-4).

    Checks, in order:
    1. every hole lies STRICTLY inside the exterior: no edge crossings with
       the exterior ring, no vertex touching the exterior boundary;
    2. holes are pairwise disjoint: no crossings, no touching, no nesting;
    3. the summed absolute hole area is STRICTLY less than the exterior area;
    4. the net region area after hole subtraction stays strictly positive.

    Winding normalization is NOT this function's job: callers normalize
    exterior-CCW / holes-CW first via :func:`_ring_winding_normalized`.
    """
    exterior_area = abs(_signed_area(exterior))
    hole_areas = [abs(_signed_area(hole)) for hole in holes]
    total_hole_area = sum(hole_areas)
    if total_hole_area >= exterior_area:
        raise ValueError(
            "summed hole area must be strictly less than the exterior area "
            f"(holes {total_hole_area!r} >= exterior {exterior_area!r})"
        )
    net_area = exterior_area - total_hole_area
    if net_area <= 0.0:
        raise ValueError(
            f"net region area after hole subtraction must be positive, got {net_area!r}"
        )
    for hole_index, hole in enumerate(holes):
        for h_index in range(len(hole)):
            hp1 = hole[h_index]
            hp2 = hole[(h_index + 1) % len(hole)]
            for e_index in range(len(exterior)):
                ep1 = exterior[e_index]
                ep2 = exterior[(e_index + 1) % len(exterior)]
                if _segments_intersect(hp1, hp2, ep1, ep2):
                    raise ValueError(
                        f"hole {hole_index} crosses or touches the exterior boundary"
                    )
        if not _point_strictly_inside(hole[0], exterior):
            raise ValueError(f"hole {hole_index} is not strictly inside the exterior")
        for other_index, other in enumerate(holes):
            if other_index <= hole_index:
                continue
            for h_index in range(len(hole)):
                hp1 = hole[h_index]
                hp2 = hole[(h_index + 1) % len(hole)]
                for o_index in range(len(other)):
                    op1 = other[o_index]
                    op2 = other[(o_index + 1) % len(other)]
                    if _segments_intersect(hp1, hp2, op1, op2):
                        raise ValueError(
                            f"hole {hole_index} crosses or touches hole {other_index}"
                        )
            if not _point_strictly_inside(other[0], hole) and not _point_strictly_inside(
                hole[0], other
            ):
                # No crossings and neither contains the other => disjoint.
                continue
            raise ValueError(
                f"hole {hole_index} overlaps or nests with hole {other_index}"
            )


def _ring_winding_normalized(points: Sequence[RingPoint], *, ccw: bool) -> RingPoints:
    """Return the ring with counter-clockwise (or clockwise) orientation.

    Reversal preserves simplicity and area magnitude, so callers may normalize
    winding either before or after topology validation without changing the
    verdict.
    """
    area = _signed_area(points)
    if (ccw and area < 0.0) or (not ccw and area > 0.0):
        return tuple(reversed(tuple(points)))
    return tuple(points)
