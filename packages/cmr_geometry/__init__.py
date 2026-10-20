"""cmr_geometry — physical image geometry, transforms, ordering, polygons.

``ImageGeometry`` itself lives in ``cmr_core.geometry`` (the value object is
core domain metadata); this package provides the algorithms over it.
"""

from .ordering import FrameItem, SliceItem, order_frames, order_slices
from .polygon import (
    point_in_polygon,
    polygon_area,
    polygon_contains,
    polygon_perimeter,
    polygon_self_intersects,
    segments_intersect,
)
from .transforms import (
    mask_centroid_px,
    pixel_to_world,
    slice_position_mm,
    world_to_pixel,
)

__all__ = [
    "FrameItem",
    "SliceItem",
    "mask_centroid_px",
    "order_frames",
    "order_slices",
    "pixel_to_world",
    "point_in_polygon",
    "polygon_area",
    "polygon_contains",
    "polygon_perimeter",
    "polygon_self_intersects",
    "segments_intersect",
    "slice_position_mm",
    "world_to_pixel",
]
