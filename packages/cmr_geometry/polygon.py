"""Deterministic polygon operations used by contours and QC.

Pure float math, no numpy. Points are (x, y) pairs in any consistent
coordinate frame (pixel or mm) — callers choose and must not mix frames.
"""

from __future__ import annotations

import math

Point = tuple[float, float]


def polygon_area(points: list[Point]) -> float:
    """Shoelace area (signed). Positive when vertices are CCW."""
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        s += p[0] * q[1] - q[0] * p[1]
    return 0.5 * s


def polygon_perimeter(points: list[Point]) -> float:
    return sum(math.dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray-casting even-odd rule test."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def point_on_segment(p: Point, a: Point, b: Point, eps: float = 1e-9) -> bool:
    """True when p lies on segment ab (within eps tolerance)."""
    cross = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    if abs(cross) > eps:
        return False
    dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
    if dot < -eps:
        return False
    len2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    return dot <= len2 + eps


def _same(a: Point, b: Point, eps: float = 1e-9) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point, eps: float = 1e-9) -> bool:
    """True when segments [a1,a2] and [b1,b2] properly cross.

    Touching at a shared endpoint is NOT an intersection (valid closed
    polygon vertices touch exactly once). A zero-length segment never
    intersects anything.
    """
    if _same(a1, a2, eps) or _same(b1, b2, eps):
        return False
    if _same(a1, b1, eps) or _same(a1, b2, eps) or _same(a2, b1, eps) or _same(a2, b2, eps):
        return False

    def cross(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    # T-junction: endpoint of one segment strictly inside the other
    if _strictly_on(a1, b1, b2, eps) or _strictly_on(a2, b1, b2, eps):
        return True
    if _strictly_on(b1, a1, a2, eps) or _strictly_on(b2, a1, a2, eps):
        return True

    d1 = cross(b1, b2, a1)
    d2 = cross(b1, b2, a2)
    d3 = cross(a1, a2, b1)
    d4 = cross(a1, a2, b2)
    return ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
        (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
    )


def _strictly_on(p: Point, a: Point, b: Point, eps: float = 1e-9) -> bool:
    """p lies strictly in the interior of segment ab."""
    if _same(p, a, eps) or _same(p, b, eps):
        return False
    cross = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    if abs(cross) > eps:
        return False
    dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
    len2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    return -eps <= dot <= len2 + eps


def polygon_self_intersects(points: list[Point], eps: float = 1e-9) -> bool:
    """True when a polygon's non-adjacent edges cross or touch."""
    n = len(points)
    if n < 4:
        return False
    for i in range(n):
        a1, a2 = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b1, b2 = points[j], points[(j + 1) % n]
            if segments_intersect(a1, a2, b1, b2, eps):
                return True
    return False


def polygon_contains(polygon_outer: list[Point], polygon_inner: list[Point]) -> bool:
    """True when every vertex of ``polygon_inner`` lies strictly inside
    (or on) ``polygon_outer``."""
    for p in polygon_inner:
        if not point_in_polygon(p, polygon_outer):
            return False
    return True
