"""Extract CMR-relevant metadata and ImageGeometry from a DICOM dataset."""

from __future__ import annotations

from cmr_core.errors import MISSING_ESSENTIAL_GEOMETRY, CMRError, CMRWarning
from cmr_core.geometry import ImageGeometry

from . import tags as T


def extract_geometry(ds, warnings: list[CMRWarning] | None = None) -> ImageGeometry:
    """Build :class:`ImageGeometry` from DICOM tags.

    PixelSpacing, ImagePositionPatient and ImageOrientationPatient are
    essential: a missing one raises a fatal :class:`CMRError`.
    SliceThickness/SpacingBetweenSlices are optional (warnings only).
    """
    warnings = [] if warnings is None else warnings
    pixel_spacing = T.get_optional_float2(ds, T.pixel_spacing_key, warnings)
    position = T.get_optional_float3(ds, T.image_position_key, warnings)
    orientation = T.get_optional_float6(ds, T.image_orientation_key, warnings)
    if pixel_spacing is None or position is None or orientation is None:
        raise CMRError(
            MISSING_ESSENTIAL_GEOMETRY,
            "essential geometry tags missing or unreadable",
            source=T.source_uid(ds),
        )
    rows = int(getattr(ds, "Rows", 0) or 0)
    cols = int(getattr(ds, "Columns", 0) or 0)
    return ImageGeometry(
        rows=rows,
        cols=cols,
        pixel_spacing=(pixel_spacing[0], pixel_spacing[1]),
        slice_thickness=T.get_optional_float(ds, T.slice_thickness_key, warnings),
        spacing_between_slices=T.get_optional_float(ds, T.spacing_between_slices_key, warnings),
        image_position_patient=(position[0], position[1], position[2]),
        image_orientation_patient=(
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
            orientation[4],
            orientation[5],
        ),
    )


def extract_series_metadata(ds, warnings: list[CMRWarning] | None = None) -> dict:
    """Collect the optional metadata fields of a series into a plain dict.

    Missing optional fields become CMRWarning entries — never silent.
    """
    warnings = [] if warnings is None else warnings
    return {
        "series_description": T.get_optional_str(ds, T.series_description_key, warnings),
        "protocol_name": T.get_optional_str(ds, T.protocol_name_key, warnings),
        "modality": T.get_optional_str(ds, T.modality_key, warnings),
        "manufacturer": T.get_optional_str(ds, T.manufacturer_key, warnings),
        "manufacturer_model": T.get_optional_str(ds, T.manufacturer_model_key, warnings),
        "magnetic_field_strength_t": T.get_optional_float(ds, T.field_strength_key, warnings),
    }


def extract_timing(ds, warnings: list[CMRWarning] | None = None) -> dict:
    """Cardiac timing metadata (all optional)."""
    warnings = [] if warnings is None else warnings
    c_num = getattr(ds, T.cardiac_number_of_images_key, None)
    cardiac_num: int | None = None
    if c_num is not None:
        try:
            val = int(float(c_num))
            if val > 0:
                cardiac_num = val
        except (ValueError, TypeError):
            pass
    return {
        "trigger_time_ms": T.get_optional_float(ds, T.trigger_time_key, warnings),
        "temporal_position": _as_int(T.get_optional_float(ds, T.temporal_position_key, warnings)),
        "instance_number": _as_int(T.get_optional_float(ds, T.instance_number_key, warnings)),
        "number_of_frames": _as_int(T.get_optional_float(ds, T.number_of_frames_key, warnings)),
        "cardiac_number_of_images": cardiac_num,
        "enhanced_temporal_index": None,
    }


def _as_int(value: float | None) -> int | None:
    if value is None:
        return None
    return int(value)
