"""DICOM tag keyword constants and safe accessors.

Every accessor treats a missing tag as a recoverable warning (never a
silent default that could mislead downstream code).
"""

from __future__ import annotations

from typing import Any

from cmr_core.errors import INVALID_ARGUMENT, MISSING_OPTIONAL_TAG, CMRError, CMRWarning

study_uid_key = "StudyInstanceUID"
series_uid_key = "SeriesInstanceUID"
instance_uid_key = "SOPInstanceUID"
patient_id_key = "PatientID"
patient_name_key = "PatientName"
study_date_key = "StudyDate"
series_description_key = "SeriesDescription"
protocol_name_key = "ProtocolName"
modality_key = "Modality"
manufacturer_key = "Manufacturer"
manufacturer_model_key = "ManufacturerModelName"
field_strength_key = "MagneticFieldStrength"
image_position_key = "ImagePositionPatient"
image_orientation_key = "ImageOrientationPatient"
pixel_spacing_key = "PixelSpacing"
slice_thickness_key = "SliceThickness"
spacing_between_slices_key = "SpacingBetweenSlices"
trigger_time_key = "TriggerTime"
temporal_position_key = "TemporalPositionIdentifier"
instance_number_key = "InstanceNumber"
number_of_frames_key = "NumberOfFrames"
cardiac_number_of_images_key = "CardiacNumberOfImages"
nominal_interval_key = "NominalInterval"
heart_rate_key = "HeartRate"
ORIENTATION_SPLIT = "geometry.orientation_split"



def get_uid(ds, keyword: str, warnings: list[CMRWarning] | None = None) -> str:
    """Read a UID as str; missing UID is fatal (grouping is impossible)."""
    value = None
    if warnings is not None:
        value, _ = _get(ds, keyword, warnings)
    else:
        try:
            value = getattr(ds, keyword, None)
        except Exception:  # noqa: BLE001 - malformed tags surface as warnings
            value = None
    if value is None:
        raise CMRError(
            INVALID_ARGUMENT,
            f"essential UID tag {keyword} missing",
            source=source_uid(ds),
        )
    return str(value).strip()


def get_optional_str(ds, keyword: str, warnings: list[CMRWarning] | None = None) -> str | None:
    """Optional string tag; None when missing (a warning is recorded)."""
    value, _ = _get(ds, keyword, [] if warnings is None else warnings)
    if value is None:
        return None
    return str(value).strip()


def get_optional_float(ds, keyword: str, warnings: list[CMRWarning] | None = None) -> float | None:
    """Optional numeric tag; None when missing or unparsable."""
    value, collected = _get(ds, keyword, [] if warnings is None else warnings)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        collected.append(
            CMRWarning(
                MISSING_OPTIONAL_TAG,
                f"DICOM tag {keyword} is not numeric: {value!r}",
                source=source_uid(ds),
            )
        )
        return None


def get_optional_float2(
    ds, keyword: str, warnings: list[CMRWarning] | None = None
) -> tuple[float, float] | None:
    """Optional two-valued numeric tag (e.g. PixelSpacing)."""
    value, collected = _get(ds, keyword, [] if warnings is None else warnings)
    if value is None:
        return None
    try:
        a, b = float(value[0]), float(value[1])
        return (a, b)
    except (TypeError, ValueError, IndexError):
        collected.append(
            CMRWarning(
                MISSING_OPTIONAL_TAG,
                f"DICOM tag {keyword} does not have two numeric values: {value!r}",
                source=source_uid(ds),
            )
        )
        return None


def get_optional_float6(
    ds, keyword: str, warnings: list[CMRWarning] | None = None
) -> tuple[float, float, float, float, float, float] | None:
    """Optional six-valued numeric tag (e.g. ImageOrientationPatient)."""
    value, collected = _get(ds, keyword, [] if warnings is None else warnings)
    if value is None:
        return None
    try:
        return (
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
            float(value[4]),
            float(value[5]),
        )
    except (TypeError, ValueError):
        collected.append(
            CMRWarning(
                MISSING_OPTIONAL_TAG,
                f"DICOM tag {keyword} does not have six numeric values: {value!r}",
                source=source_uid(ds),
            )
        )
        return None


def get_optional_float3(
    ds, keyword: str, warnings: list[CMRWarning] | None = None
) -> tuple[float, float, float] | None:
    """Optional three-valued numeric tag (e.g. ImagePositionPatient)."""
    value, collected = _get(ds, keyword, [] if warnings is None else warnings)
    if value is None:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        collected.append(
            CMRWarning(
                MISSING_OPTIONAL_TAG,
                f"DICOM tag {keyword} does not have three numeric values: {value!r}",
                source=source_uid(ds),
            )
        )
        return None


def source_uid(ds) -> str | None:
    try:
        return str(getattr(ds, "SOPInstanceUID", None))
    except Exception:  # noqa: BLE001
        return None


def _get(ds, keyword: str, warnings: list[CMRWarning]) -> tuple[Any, list[CMRWarning]]:
    try:
        value = getattr(ds, keyword, None)
    except Exception:  # noqa: BLE001 - malformed tag values surface as warnings
        value = None
    if value is None:
        warnings.append(
            CMRWarning(
                MISSING_OPTIONAL_TAG,
                f"DICOM tag {keyword} is missing or unreadable",
                source=source_uid(ds),
            )
        )
    return value, warnings
