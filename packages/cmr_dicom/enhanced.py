"""First Enhanced MR (multi-frame) support (ROUND2_INSTRUCTIONS 11).

Scope of the standard path: ``Enhanced MR Image Storage`` cine instances
whose per-frame functional groups map every frame to a (physical slice
position, cardiac phase):

- shared functional groups provide pixel measures (pixel spacing,
  slice thickness) for the whole instance;
- per-frame functional groups provide Plane Position / Plane Orientation
  per frame, and Frame Content / Cardiac Synchronization for timing
  (``TemporalPositionIndex`` / ``NominalCardiacTriggerDelayTime``);
- every frame becomes an :class:`EnhancedFrame` that runs through the
  identical slice-clustering and timing-basis machinery as classic
  single-frame cine.

Variants that are explicitly characterized as unsupported this cycle
emit ``UNSUPPORTED_ENHANCED_VARIANT`` warnings instead of failing or
silently guessing:

- missing NumberOfFrames / PerFrameFunctionalGroupsSequence or a count
  mismatch;
- a frame lacking plane position/orientation (frame skipped);
- a frame lacking pixel spacing even via shared functional groups;
- non-cine content (no temporal index, no nominal trigger delay) —
  surfaces downstream as an AMBIGUOUS_FRAME_ORDER refusal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cmr_core.errors import MISSING_PIXEL_DATA, UNSUPPORTED_ENHANCED_VARIANT, CMRWarning
from cmr_core.geometry import ImageGeometry

ENHANCED_MR_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.4.1"


@dataclass(frozen=True)
class EnhancedFrame:
    """One frame of an Enhanced MR instance: pixels + geometry + timing."""

    pixel_data: np.ndarray
    geometry: ImageGeometry
    trigger_time_ms: float | None
    temporal_position: int | None
    enhanced_temporal_index: int | None
    source: str


def is_enhanced_mr(ds) -> bool:
    try:
        return str(getattr(ds, "SOPClassUID", "")) == ENHANCED_MR_STORAGE_UID
    except Exception:  # noqa: BLE001 - malformed SOP class surfaces nowhere
        return False


def _first(fg, keyword: str):
    """First item of the sequence element ``keyword`` inside ``fg``, or None."""
    seq = getattr(fg, keyword, None)
    if seq is None or len(seq) == 0:
        return None
    try:
        return seq[0]
    except Exception:  # noqa: BLE001
        return None


def _pixel_measures(fg):
    """(pixel_spacing, slice_thickness) from a functional-group item, or None."""
    pm = _first(fg, "PixelMeasuresSequence")
    if pm is None:
        return None
    try:
        spacing = tuple(float(v) for v in pm.PixelSpacing)
        thickness = (
            float(pm.SliceThickness) if getattr(pm, "SliceThickness", None) is not None else None
        )
        if len(spacing) != 2:
            return None
        return (spacing, thickness)
    except Exception:  # noqa: BLE001 - malformed pixel measures
        return None


def _plane_position(fg) -> tuple[float, float, float] | None:
    pos = _first(fg, "PlanePositionSequence")
    if pos is None or getattr(pos, "ImagePositionPatient", None) is None:
        return None
    try:
        v = tuple(float(x) for x in pos.ImagePositionPatient)
        return v if len(v) == 3 else None
    except Exception:  # noqa: BLE001
        return None


def _plane_orientation(fg) -> tuple[float, float, float, float, float, float] | None:
    orient = _first(fg, "PlaneOrientationSequence")
    if orient is None or getattr(orient, "ImageOrientationPatient", None) is None:
        return None
    try:
        v = tuple(float(x) for x in orient.ImageOrientationPatient)
        return v if len(v) == 6 else None
    except Exception:  # noqa: BLE001
        return None


def extract_enhanced_frames(
    ds,
    warnings: list[CMRWarning] | None = None,
) -> list[EnhancedFrame]:
    """Expand one Enhanced MR dataset into per-frame cine candidates.

    Returns an empty list with explanatory warnings for unsupported
    variants (see module docstring).
    """
    warnings = [] if warnings is None else warnings
    sop = str(getattr(ds, "SOPInstanceUID", "?"))
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    try:
        n_frames = int(getattr(ds, "NumberOfFrames", 0) or 0)
    except Exception:  # noqa: BLE001
        n_frames = 0
    if per_frame is None or len(per_frame) == 0:
        warnings.append(
            CMRWarning(
                UNSUPPORTED_ENHANCED_VARIANT,
                "Enhanced MR instance without PerFrameFunctionalGroupsSequence",
                source=sop,
            )
        )
        return []
    if n_frames <= 0 or len(per_frame) != n_frames:
        warnings.append(
            CMRWarning(
                UNSUPPORTED_ENHANCED_VARIANT,
                f"NumberOfFrames ({n_frames}) disagrees with "
                f"PerFrameFunctionalGroupsSequence ({len(per_frame)})",
                source=sop,
            )
        )
        return []
    try:
        rows = int(getattr(ds, "Rows", 0) or 0)
        cols = int(getattr(ds, "Columns", 0) or 0)
    except Exception:  # noqa: BLE001
        rows = cols = 0
    if rows <= 0 or cols <= 0:
        warnings.append(
            CMRWarning(
                UNSUPPORTED_ENHANCED_VARIANT,
                "Enhanced MR instance without Rows/Columns",
                source=sop,
            )
        )
        return []
    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    shared_measures = _pixel_measures(shared[0]) if shared is not None and len(shared) else None

    try:
        pixels = ds.pixel_array
        if isinstance(pixels, list):
            pixels = np.stack(pixels)
        arr = np.asarray(pixels)
    except Exception as exc:  # noqa: BLE001 - pixel decoding failed
        warnings.append(
            CMRWarning(MISSING_PIXEL_DATA, f"pixel data unavailable: {exc}", source=sop)
        )
        return []
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3 or arr.shape[0] != n_frames:
        warnings.append(
            CMRWarning(
                UNSUPPORTED_ENHANCED_VARIANT,
                f"pixel array shape {arr.shape} does not match NumberOfFrames {n_frames}",
                source=sop,
            )
        )
        return []

    out: list[EnhancedFrame] = []
    for i in range(n_frames):
        fg = per_frame[i]
        position = _plane_position(fg)
        orientation = _plane_orientation(fg)
        if position is None or orientation is None:
            warnings.append(
                CMRWarning(
                    UNSUPPORTED_ENHANCED_VARIANT,
                    f"frame {i} lacks Plane Position / Plane Orientation; frame skipped",
                    source=sop,
                )
            )
            continue
        measures = _pixel_measures(fg)
        if measures is None and shared_measures is not None:
            measures = shared_measures
        if measures is None:
            warnings.append(
                CMRWarning(
                    UNSUPPORTED_ENHANCED_VARIANT,
                    f"frame {i} lacks PixelSpacing even via shared functional groups; "
                    f"frame skipped",
                    source=sop,
                )
            )
            continue
        (pixel_spacing, thickness) = measures
        try:
            geometry = ImageGeometry(
                rows=rows,
                cols=cols,
                pixel_spacing=(pixel_spacing[0], pixel_spacing[1]),
                slice_thickness=thickness,
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
        except Exception as exc:  # noqa: BLE001 - invalid geometry surfaces as warning
            warnings.append(
                CMRWarning(
                    UNSUPPORTED_ENHANCED_VARIANT, f"frame {i} geometry invalid: {exc}", source=sop
                )
            )
            continue
        frame_content = _first(fg, "FrameContentSequence")
        cardiac = _first(fg, "CardiacSynchronizationSequence")
        temporal_index = None
        if frame_content is not None:
            try:
                v = getattr(frame_content, "TemporalPositionIndex", None)
                temporal_index = int(v) if v is not None else None
            except Exception:  # noqa: BLE001
                temporal_index = None
        trigger_ms = None
        if cardiac is not None:
            try:
                v = getattr(cardiac, "NominalCardiacTriggerDelayTime", None)
                trigger_ms = float(v) if v is not None else None
            except Exception:  # noqa: BLE001
                trigger_ms = None
        out.append(
            EnhancedFrame(
                pixel_data=arr[i].astype("float32", copy=False),
                geometry=geometry,
                trigger_time_ms=trigger_ms,
                temporal_position=None,
                enhanced_temporal_index=temporal_index,
                source=f"{sop}#f{i}",
            )
        )
    return out
