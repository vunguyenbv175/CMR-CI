"""Pixel <-> world coordinate transforms and slice-position algebra.

Everything is deterministic float math with explicit units (mm).
"""

from __future__ import annotations

import math

from cmr_core.geometry import ImageGeometry


def pixel_to_world(geometry: ImageGeometry, col: float, row: float) -> tuple[float, float, float]:
    """Map image-pixel (col, row) to patient coordinates in mm.

    ``row``/``col`` are continuous pixel indices with 0 = center of the
    first pixel, matching DICOM ImagePositionPatient semantics.
    """
    rx, ry, rz = geometry.row_direction
    cx, cy, cz = geometry.col_direction
    ox, oy, oz = geometry.image_position_patient
    rs, cs = geometry.row_spacing_mm, geometry.col_spacing_mm
    return (
        ox + row * rs * rx + col * cs * cx,
        oy + row * rs * ry + col * cs * cy,
        oz + row * rs * rz + col * cs * cz,
    )


def world_to_pixel(
    geometry: ImageGeometry, world: tuple[float, float, float]
) -> tuple[float, float]:
    """Inverse of :func:`pixel_to_world` (projection onto the plane)."""
    dx, dy, dz = (
        world[0] - geometry.image_position_patient[0],
        world[1] - geometry.image_position_patient[1],
        world[2] - geometry.image_position_patient[2],
    )
    rx, ry, rz = geometry.row_direction
    cx, cy, cz = geometry.col_direction
    row = (dx * rx + dy * ry + dz * rz) / geometry.row_spacing_mm
    col = (dx * cx + dy * cy + dz * cz) / geometry.col_spacing_mm
    return (row, col)


def slice_position_mm(geometry: ImageGeometry) -> float:
    """Signed distance of the slice origin along its own normal (mm).

    The geometric ordering key for spatial sorting of a slice stack
    (monotonic along the normal, no directional anatomical claim).
    Direction convention: normal = cross(row_direction, col_direction)
    per DICOM (LPS).
    """
    nx, ny, nz = geometry.slice_normal
    ox, oy, oz = geometry.image_position_patient
    return ox * nx + oy * ny + oz * nz


def in_plane_distance_mm(
    geometry: ImageGeometry, a_px: tuple[float, float], b_px: tuple[float, float]
) -> float:
    """Euclidean distance between two pixel positions, in mm."""
    a = pixel_to_world(geometry, *a_px)
    b = pixel_to_world(geometry, *b_px)
    return math.dist(a, b)


def mask_centroid_px(mask) -> tuple[float, float]:
    """Mask centroid in pixel coordinates; deterministic and pure."""
    ys, xs = mask.nonzero()
    if ys.size == 0:
        raise ValueError("cannot compute centroid of an empty mask")
    return (float(xs.mean()), float(ys.mean()))
