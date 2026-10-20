"""mask <-> contour conversions (Task F).

Conversions are explicit about coordinate frames: masks live in the pixel
grid, contours are point lists in pixel coordinates (see
:mod:`cmr_contours.model`); mm conversion happens through
``ImageGeometry`` at the call site, never implicitly.

Contour extraction uses deterministic marching squares on the binary
grid. Grid convention: pixel (x, y) covers the unit square
[x, x+1) x [y, y+1); outside the image counts as background. Boundary
segments lie on grid lines between foreground and background cells and
are assembled into closed loops. The polygon of a loop is the exact
boundary of that component of the foreground pixel union, so its
shoelace area equals the component pixel count. Loops are normalized to
counter-clockwise winding (positive shoelace in math convention).

For masks with holes (e.g. the myocardium annulus) all loops are
returned; :func:`mask_to_contour` keeps the loop with the largest area
(the outer, epicardial boundary), which is the clinically relevant
contour for LV/RV structures in this cycle.

S2C-001 cluster F disclosure (documentation only, no behavior change):
``mask_to_contour`` DROPS every internal loop — a mask whose foreground
carries holes (papillary islands, a true cavity annulus) converts to ONE
outer polygon, so cavity holes and papillary topology CANNOT be preserved
through this function and no downstream mass formula can reconstruct them.
Hole-aware representation is owned by milestone CMR-01 before any clinical
freeze of papillary/trabeculation semantics
(see ``cmr_quantification.measurements.PAPILLARY_TRABECULATION_CONVENTION_V1``).
"""

from __future__ import annotations

import numpy as np
from cmr_core.geometry import ImageGeometry
from cmr_core.segmentation import Structure

from .model import Contour


def mask_to_contour(
    mask: np.ndarray,
    slice_index: int,
    frame_index: int,
    structure: Structure,
    geometry: ImageGeometry | None = None,
    model=None,
    source: str = "ai",
) -> Contour:
    """Trace the outer boundary of a binary mask into a contour polygon.

    Deterministic marching squares; no randomness, no GPU. The returned
    points are pixel (grid) coordinates, CCW-wound.

    S2C-001 cluster F disclosure: ALL INTERNAL LOOPS ARE DROPPED. A mask
    with holes converts to a single outer polygon — cavity holes and
    papillary islands are NOT representable in the returned Contour and are
    not preserved (see the module docstring; CMR-01 owns hole-aware
    representation).
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if not mask.any():
        raise ValueError("cannot trace an empty mask")
    loops = marching_squares_loops(mask)
    if not loops:
        raise ValueError("mask produced no contour loops")
    outer = max(loops, key=lambda loop: _abs_area(loop))
    return Contour(
        slice_index=slice_index,
        frame_index=frame_index,
        structure=structure,
        points=list(outer),
        source=source,
        model=model,
    )


def contour_to_mask(contour: Contour, geometry: ImageGeometry) -> np.ndarray:
    """Rasterize a contour polygon into a boolean mask via even-odd fill.

    A pixel is filled when its centre (x+0.5, y+0.5) lies strictly inside
    the polygon, matching the grid convention of the contour extraction
    (fill = ceil(x0 - 0.5) .. floor(x1 - 0.5) per scanline crossing xi).
    """
    rows, cols = geometry.rows, geometry.cols
    mask = np.zeros((rows, cols), dtype=bool)
    pts = contour.points
    if len(pts) < 3:
        return mask
    for row in range(rows):
        crossings: list[float] = []
        y_top = float(row) + 0.5  # scanline through pixel centres
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            if (y1 <= y_top) != (y2 <= y_top):
                x_cross = x1 + (y_top - y1) * (x2 - x1) / (y2 - y1)
                crossings.append(x_cross)
        crossings.sort()
        for k in range(0, len(crossings) - 1, 2):
            lo = int(np.ceil(crossings[k] - 0.5))
            hi = int(np.floor(crossings[k + 1] - 0.5))
            lo = max(0, lo)
            hi = min(cols - 1, hi)
            if hi >= lo:
                mask[row, lo : hi + 1] = True
    return mask


def contour_area_px2(contour: Contour) -> float:
    """Shoelace area (abs) of the contour polygon in pixel^2."""
    return _abs_area(contour.points)


def mask_area_px2(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask))


Point = tuple[float, float]


def marching_squares_loops(mask: np.ndarray) -> list[list[Point]]:
    """Deterministic extraction of all boundary loops of a binary mask.

    Returns loops with counter-clockwise winding (positive shoelace).
    """
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask

    segments: list[tuple[Point, Point]] = []
    height, width = padded.shape

    # horizontal adjacency (y, x) vs (y, x+1): vertical edge at x+1
    for y in range(height):
        for x in range(width - 1):
            if padded[y, x] != padded[y, x + 1]:
                a = (float(x + 1.0), float(y))
                b = (float(x + 1.0), float(y + 1.0))
                if padded[y, x]:
                    a, b = b, a  # keep foreground on the left
                segments.append((a, b))
    # vertical adjacency (y, x) vs (y+1, x): horizontal edge at y+1
    for y in range(height - 1):
        for x in range(width):
            if padded[y, x] != padded[y + 1, x]:
                a = (float(x), float(y + 1.0))
                b = (float(x + 1.0), float(y + 1.0))
                if padded[y, x]:
                    a, b = b, a  # keep foreground on the left
                segments.append((a, b))

    if not segments:
        return []

    unused = set(range(len(segments)))
    loops: list[list[Point]] = []
    while unused:
        first = unused.pop()
        a0, b0 = segments[first]
        loop = [a0, b0]
        current = b0
        start = a0
        while current != start:
            nxt = _next_segment_at_junction(
                current, loop[-2] if len(loop) >= 2 else a0, segments, unused
            )
            if nxt is None:
                break  # open chain: cannot happen for closed pixel unions
            if _close(nxt, start):
                break  # closed; do not duplicate the start point
            loop.append(nxt)
            current = nxt
        loops.append(loop)

    # shift back out of the padding border and normalise winding to CCW
    shifted = []
    for loop in loops:
        moved = [(x - 1.0, y - 1.0) for x, y in loop]
        if _signed_area(moved) < 0:
            moved = list(reversed(moved))
        shifted.append(moved)
    return shifted


def _next_segment_at_junction(
    at: Point,
    came_from: Point,
    segments: list[tuple[Point, Point]],
    unused: set[int],
) -> Point | None:
    """Choose the next segment endpoint at a junction.

    At ordinary boundary vertices exactly one unused segment continues the
    loop. At 4-way saddle junctions (2x2 checkerboard of the binary grid)
    several segments meet: we prefer a 90-degree TURN to the incoming
    direction, which resolves the ambiguity deterministically without
    producing crossing loops (straight-through pairing would cross).
    """
    incoming = (at[0] - came_from[0], at[1] - came_from[1])
    candidates: list[tuple[int, Point]] = []
    for idx in list(unused):
        pa, pb = segments[idx]
        if _close(pa, at):
            candidates.append((idx, pb))
        elif _close(pb, at):
            candidates.append((idx, pa))
    if not candidates:
        return None
    turns = []
    straights = []
    for idx, other in candidates:
        dx, dy = other[0] - at[0], other[1] - at[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            continue
        dot = (dx * incoming[0] + dy * incoming[1]) / length
        if abs(dot) < 0.5:
            turns.append((idx, other))
        else:
            straights.append((idx, other))
    pool = turns if turns else straights
    idx, other = pool[0]
    unused.discard(idx)
    return other


def _close(a: Point, b: Point, eps: float = 1e-6) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def _signed_area(points: list[Point]) -> float:
    from cmr_geometry.polygon import polygon_area

    return polygon_area(points)


def _abs_area(points: list[Point]) -> float:
    return abs(_signed_area(points))
