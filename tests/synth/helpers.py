"""Synthetic (non-patient) mask/image builders with known ground truth.

All helpers are deterministic and produce simple analytic shapes
(circles, rings, ellipses) whose areas/volumes are known in closed form,
so tests can detect silent regressions in the mathematics.
"""

from __future__ import annotations

import math

import numpy as np
from cmr_core.cine import CardiacFrame, CineVolume
from cmr_core.geometry import ImageGeometry


def circle_mask(rows: int, cols: int, cx: float, cy: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:cols]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def ring_mask(
    rows: int, cols: int, cx: float, cy: float, r_inner: float, r_outer: float
) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:cols]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (d2 <= r_outer**2) & (d2 > r_inner**2)


def circle_area_mm2(radius: float) -> float:
    return math.pi * radius**2


def ring_area_mm2(r_inner: float, r_outer: float) -> float:
    return math.pi * (r_outer**2 - r_inner**2)


def make_geometry(
    rows: int = 64,
    cols: int = 64,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    gap_mm: float = 2.0,
    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ImageGeometry:
    return ImageGeometry(
        rows=rows,
        cols=cols,
        pixel_spacing=pixel_spacing,
        slice_thickness=gap_mm,
        spacing_between_slices=gap_mm,
        image_position_patient=position_mm,
    )


def make_cine_volume(
    n_slices: int,
    n_frames: int,
    rows: int = 64,
    cols: int = 64,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    gap_mm: float = 2.0,
    radius_fn=None,
) -> CineVolume:
    """Cine volume whose LV cavity is a circle with per-(slice, frame) radius.

    ``radius_fn(slice_index, frame_index)`` returns the radius in px
    (default: constant 20 px).
    """
    radius_fn = radius_fn or (lambda s, f: 20.0)
    geometries: list[ImageGeometry] = []
    frames: list[list[CardiacFrame]] = []
    for s in range(n_slices):
        geom = make_geometry(rows, cols, pixel_spacing, gap_mm, (0.0, 0.0, float(s * gap_mm)))
        geometries.append(geom)
        slice_frames: list[CardiacFrame] = []
        for f in range(n_frames):
            mask = circle_mask(rows, cols, cols / 2, rows / 2, radius_fn(s, f))
            img = np.where(mask, 150, 5).astype("float32")
            slice_frames.append(
                CardiacFrame(
                    slice_index=s,
                    frame_index=f,
                    pixel_data=img,
                    sop_instance_uid=f"SYNTH.{s}.{f}",
                    trigger_time_ms=float(f * 100),
                )
            )
        frames.append(slice_frames)
    return CineVolume(geometries=geometries, frames=frames)


def volume_of_circle_cine(
    n_slices: int, gap_mm: float, pixel_spacing: tuple[float, float], radius_fn
) -> float:
    """Expected total cavity volume (ml) of the synthetic cine above."""
    total = 0.0
    for s in range(n_slices):
        area = circle_area_mm2(radius_fn(s, 0))
        total += area * pixel_spacing[0] * pixel_spacing[1] * gap_mm / 1000.0
    return total
