"""Deterministic volume mathematics (Task G, R2 corrections).

Assumptions (documented, never hidden):

- Area of a binary mask = number of foreground pixels x pixel area (mm^2).
  No partial-volume correction — this is a foundation primitive.
- Slice volume = area x slab length (mm^3 / 1000 => ml). For a slice
  STACK, the slab length comes from IPP-derived neighbour distances
  (``cmr_geometry.stack.stack_gaps_mm``) — actual centre-to-centre
  spacing is the truth, `SpacingBetweenSlices` is only validated against
  it, and `SliceThickness` (excited slab) is never used as the gap when
  the stack provides positions. Per-instance metadata gaps
  (:attr:`ImageGeometry.slice_gap_mm`) are the documented last-resort
  fallback for a single-slice stack only.
- Non-uniform (variable) slice spacing is supported: each slice gets its
  own IPP-derived gap.
- This is the disc/cylinder method; Simpson's rule is intentionally not
  implemented in this cycle.

ED/ES detection: ED = frame with maximum cavity volume, ES = frame with
minimum cavity volume. Ties resolve to the first frame; flat curves are
reported as a warning instead of silently picking a frame.
"""

from __future__ import annotations

import numpy as np
from cmr_core.errors import QC_VOLUME_CURVE, CMRError, CMRWarning, IssueSet
from cmr_core.geometry import ImageGeometry
from cmr_geometry.stack import stack_gaps_mm
from cmr_geometry.transforms import slice_position_mm


def mask_area_mm2(mask: np.ndarray, geometry: ImageGeometry) -> float:
    """Area of a binary mask in mm^2."""
    return float(np.count_nonzero(mask)) * geometry.pixel_area_mm2


def slice_volume_ml(mask: np.ndarray, geometry: ImageGeometry, gap_mm: float) -> float:
    """Volume of one slice slab in ml: area(mm^2) * gap(mm) / 1000."""
    return mask_area_mm2(mask, geometry) * gap_mm / 1000.0


def volume_time_curve_ml(
    masks: list[np.ndarray],
    geometry: ImageGeometry,
    gap_mm: float | None = None,
) -> list[float]:
    """Volumes (ml) per cardiac phase for one structure across one slice stack.

    ``masks``: one mask per frame for the same slice (same geometry).
    Uses the per-instance metadata gap unless ``gap_mm`` is given
    explicitly; prefer the stack-level API where possible.
    """
    gap = geometry.slice_gap_mm if gap_mm is None else gap_mm
    if gap is None:
        raise ValueError("no slice gap available; pass gap_mm explicitly")
    return [slice_volume_ml(m, geometry, gap) for m in masks]


def stack_volumes_ml(
    masks: list[np.ndarray],
    geometries: list[ImageGeometry],
) -> list[float]:
    """Per-slice slab volumes (ml) using IPP-derived per-slice gaps.

    ``masks`` and ``geometries`` are aligned positionally; the stack is
    sorted by slice position internally, and results are returned in the
    same sorted-position order (monotonic along the slice normal).

    - For a real multi-slice stack: each slice gets its IPP-derived gap
      (``cmr_geometry.stack.stack_gaps_mm``), supporting variable spacing.
    - For a single-slice stack: the per-instance metadata gap
      (:attr:`ImageGeometry.slice_gap_mm`) is the documented fallback;
      raises when even that is absent.
    """
    if len(masks) != len(geometries):
        raise ValueError(f"masks ({len(masks)}) and geometries ({len(geometries)}) must align")
    if not geometries:
        return []
    if len(geometries) == 1:
        gap = geometries[0].slice_gap_mm
        if gap is None:
            raise CMRError(
                QC_VOLUME_CURVE,
                "single-slice stack without SpacingBetweenSlices or SliceThickness; "
                "volume integration impossible",
            )
        return [slice_volume_ml(masks[0], geometries[0], gap)]
    ordered = sorted(zip(masks, geometries), key=lambda mg: slice_position_mm(mg[1]))
    gaps = stack_gaps_mm([g for _, g in ordered])
    return [slice_volume_ml(m, g, gap) for (m, g), gap in zip(ordered, gaps)]


def volume_curve_stack_ml(
    masks_by_slice: list[list[np.ndarray]],
    geometries: list[ImageGeometry],
) -> list[list[float]]:
    """Total stack volume (ml) per cardiac phase.

    ``masks_by_slice``: per slice (index s), per frame (index f) masks.
    Returns per-frame volumes, each summing all slices via IPP-derived
    gaps. Every slice must have the same frame count.
    """
    if not masks_by_slice:
        raise ValueError("empty input")
    n_frames = len(masks_by_slice[0])
    if any(len(s) != n_frames for s in masks_by_slice):
        raise ValueError("ragged volume curve input")
    if len(masks_by_slice) != len(geometries):
        raise ValueError("masks_by_slice and geometries slice counts disagree")
    out: list[list[float]] = []
    for f in range(n_frames):
        per_slice = stack_volumes_ml([s[f] for s in masks_by_slice], geometries)
        out.append(per_slice)
    return out


def cavity_volume_curve_ml(
    volumes_by_slice: list[list[float]],
) -> list[float]:
    """Total cavity volume per frame across all slices.

    ``volumes_by_slice``: per slice (index s), per frame (index f) volumes.
    Returns per-frame summed volumes. All slices must have the same frame
    count.
    """
    if not volumes_by_slice:
        return []
    n_frames = len(volumes_by_slice[0])
    if any(len(v) != n_frames for v in volumes_by_slice):
        raise ValueError("ragged volume curve input")
    return [sum(s[f] for s in volumes_by_slice) for f in range(n_frames)]


def detect_ed_es(volumes: list[float], issues: IssueSet | None = None) -> tuple[int, int]:
    """ED = argmax volume, ES = argmin volume (first occurrence on ties).

    A flat curve (max == min) emits a QC warning.
    """
    if not volumes:
        raise ValueError("cannot detect ED/ES on an empty volume curve")
    ed = max(range(len(volumes)), key=lambda i: volumes[i])
    es = min(range(len(volumes)), key=lambda i: volumes[i])
    if volumes[ed] == volumes[es]:
        if issues is not None:
            issues.add(
                CMRWarning.qc(
                    QC_VOLUME_CURVE,
                    "volume-time curve is flat; ED/ES selection is arbitrary",
                )
            )
    return ed, es
