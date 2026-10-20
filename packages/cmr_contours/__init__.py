"""cmr_contours — contour model, mask<->contour conversion, QC helpers."""

from .conversion import (
    contour_area_px2,
    contour_to_mask,
    mask_area_px2,
    mask_to_contour,
)
from .model import Contour, ContourSet

__all__ = [
    "Contour",
    "ContourSet",
    "contour_area_px2",
    "contour_to_mask",
    "mask_area_px2",
    "mask_to_contour",
]
