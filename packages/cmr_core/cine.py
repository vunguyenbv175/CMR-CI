"""Cine representation: slices x cardiac phases with preserved geometry.

``CineVolume`` is the canonical internal representation a cine study is
reconstructed into before segmentation/quantification. Indices:

- slice index: monotonic spatial order along the canonical slice normal
  (see ``cmr_geometry.ordering``). The order is geometric, not
  anatomical: the index conveys only spatial monotonicity along the
  plane normal — never a claimed direction toward a cardiac landmark;
- frame index: temporal order reconstructed from a single coherent timing
  basis per slice (trigger time, temporal position, or instance number
  fallback — see ``cmr_geometry.ordering``). The first frame is NOT
  end-diastole and carries no automatic phase claim: ED/ES are assigned
  ONLY by the persisted physician :class:`~cmr_core.domain_types
  .EDESSelection`, which is the sole authority. Automated phase detection
  may ASSIST later, but never becomes authoritative by itself
  (S2C-001 cluster F disclosure).

Pixel arrays are numpy (shape ``[rows, cols]``); geometry is preserved
per slice so physical coordinates can always be recovered.

S2C-001 cluster A: a reconstructed volume may carry its
:class:`~cmr_core.source_manifest.DicomSourceManifest` via the trailing
optional ``source_manifest`` argument; when present, the manifest's cine
volume digest is verified against THIS decoded volume at construction and
the read-only identity properties (:attr:`study_uid`, :attr:`series_uid`,
:attr:`workflow_id`, :attr:`cine_volume_digest`) expose the bound workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from .errors import INVALID_GEOMETRY, MISSING_PIXEL_DATA, CMRError, CMRWarning
from .geometry import ImageGeometry

#: Version tag of the decoded-volume digest payload (S2C-001 §2.2). The raw
#: file hashes remain the source-of-truth identity; this digest identifies
#: the exact DECODED/reconstructed volume used clinically.
CINE_DIGEST_VERSION = "cine-volume-sha256-v1"

#: Encoding identifier stamped onto every per-frame pixel digest entry.
_CINE_PIXEL_ENCODING = "<f4"

# Exact json.dumps parameter set of ``cmr_core.domain_types.canonical_json``.
# This module must NOT import :mod:`cmr_core.domain_types`: doing so would
# reintroduce the documented eager-import cycle (see ``cmr_core/__init__``).
# Every value entering the cine digest payload is already a JSON primitive,
# so serializing with the identical parameter set IS the canonical form;
# tests/unit/test_cine_semantics.py pins byte equality against
# ``canonical_digest`` for representative volumes.
def _canonical_payload_text(payload: dict[str, Any]) -> str:
    """Serialize an all-primitive payload with the exact canonical kwargs."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _geometry_payload(geometry: ImageGeometry) -> dict[str, Any]:
    """Every :class:`ImageGeometry` field as canonical JSON primitives."""
    return {
        "rows": geometry.rows,
        "cols": geometry.cols,
        "pixel_spacing": [float(v) for v in geometry.pixel_spacing],
        "slice_thickness": (
            None if geometry.slice_thickness is None else float(geometry.slice_thickness)
        ),
        "spacing_between_slices": (
            None
            if geometry.spacing_between_slices is None
            else float(geometry.spacing_between_slices)
        ),
        "image_position_patient": [
            float(v) for v in geometry.image_position_patient
        ],
        "image_orientation_patient": [
            float(v) for v in geometry.image_orientation_patient
        ],
    }


def _frame_pixel_digest(frame: CardiacFrame) -> str:
    """Normalized pixel-array digest (S2C-001 §2.2 normalization rule)."""
    pixels = np.ascontiguousarray(frame.pixel_data, dtype=np.dtype(_CINE_PIXEL_ENCODING))
    return hashlib.sha256(pixels.tobytes(order="C")).hexdigest()


def compute_cine_volume_digest(volume: CineVolume) -> str:
    """Content digest of the DECODED/reconstructed cine volume (S2C-001 §2.2).

    The payload covers the digest version; slice/frame/row/column dimensions;
    every :class:`ImageGeometry` field in canonical slice order; the timing
    basis and degraded-timing flag; and, for every ordered frame, its slice
    index, frame index, SOP UID, trigger time, shape, encoding identifier and
    normalized pixel digest. Identity therefore changes when any contributing
    byte, UID, relative path (via the manifest), decoded geometry, ordered
    frame or pixel value changes.
    """
    frames_payload: list[dict[str, Any]] = []
    for slice_index, slice_frames in enumerate(volume.frames):
        for frame in slice_frames:
            rows, cols = frame.pixel_data.shape
            trigger_time_ms = frame.trigger_time_ms
            if trigger_time_ms is not None and not math.isfinite(float(trigger_time_ms)):
                raise CMRError(
                    INVALID_GEOMETRY,
                    f"frame ({slice_index},{frame.frame_index}) carries a "
                    "non-finite trigger time; cine digest refuses to guess",
                )
            frames_payload.append(
                {
                    "slice_index": slice_index,
                    "frame_index": frame.frame_index,
                    "sop_instance_uid": frame.sop_instance_uid,
                    "trigger_time_ms": (
                        None if trigger_time_ms is None else float(trigger_time_ms)
                    ),
                    "shape": [int(rows), int(cols)],
                    "encoding": _CINE_PIXEL_ENCODING,
                    "pixel_sha256": _frame_pixel_digest(frame),
                }
            )
    payload = {
        "digest_version": CINE_DIGEST_VERSION,
        "n_slices": volume.n_slices,
        "n_frames": volume.n_frames,
        "geometries": [_geometry_payload(geometry) for geometry in volume.geometries],
        "timing_basis": volume.timing_basis,
        "is_timing_degraded": bool(volume.is_timing_degraded),
        "frames": frames_payload,
    }
    return hashlib.sha256(
        _canonical_payload_text(payload).encode("utf-8")
    ).hexdigest()


class CardiacFrame:
    """One image at (slice_index, frame_index) tied to its source instance."""

    __slots__ = ("slice_index", "frame_index", "pixel_data", "sop_instance_uid", "trigger_time_ms")

    def __init__(
        self,
        slice_index: int,
        frame_index: int,
        pixel_data: np.ndarray,
        sop_instance_uid: str | None = None,
        trigger_time_ms: float | None = None,
    ) -> None:
        if pixel_data.ndim != 2:
            raise CMRError(
                MISSING_PIXEL_DATA, f"frame pixel data must be 2D, got ndim={pixel_data.ndim}"
            )
        self.slice_index = slice_index
        self.frame_index = frame_index
        self.pixel_data = pixel_data
        self.sop_instance_uid = sop_instance_uid
        self.trigger_time_ms = trigger_time_ms


class CineVolume:
    """A 3D+T cine volume: slices in monotonic spatial order along the
    canonical normal, `frames` per slice in coherent temporal order."""

    def __init__(
        self,
        geometries: list[ImageGeometry],
        frames: list[list[CardiacFrame]],
        is_timing_degraded: bool = False,
        timing_basis: str | None = None,
        warnings: list[CMRWarning] | None = None,
        source_manifest: Any | None = None,
    ) -> None:
        if len(geometries) == 0:
            raise CMRError("input.empty", "cannot build a CineVolume with no slices")
        if len(frames) != len(geometries):
            raise CMRError(INVALID_GEOMETRY, "frames and geometries slice counts disagree")
        n_frames = len(frames[0])
        if n_frames == 0:
            raise CMRError("input.empty", "cannot build a CineVolume with no frames")
        for s, slice_frames in enumerate(frames):
            if len(slice_frames) != n_frames:
                raise CMRError(
                    INVALID_GEOMETRY,
                    f"slice {s} has {len(slice_frames)} frames, expected {n_frames}",
                )
            expected_shape = (geometries[s].rows, geometries[s].cols)
            for f in slice_frames:
                if f.pixel_data.shape != expected_shape:
                    raise CMRError(
                        INVALID_GEOMETRY,
                        f"frame ({s},{f.frame_index}) shape {f.pixel_data.shape} "
                        f"!= geometry {expected_shape}",
                    )
        self.geometries = list(geometries)
        self.frames = frames
        self.is_timing_degraded = is_timing_degraded
        self.timing_basis = timing_basis
        self.warnings = list(warnings or [])
        # S2C-001 cluster A (additive trailing optional): when a manifest is
        # attached it must describe EXACTLY this decoded volume.
        self._source_manifest = source_manifest
        if source_manifest is not None:
            computed = compute_cine_volume_digest(self)
            if computed != source_manifest.cine_volume_digest:
                raise CMRError(
                    INVALID_GEOMETRY,
                    "source manifest cine_volume_digest "
                    f"{source_manifest.cine_volume_digest!r} does not match this "
                    f"decoded volume ({computed!r}); the manifest must be built "
                    "from the same reconstruction",
                )

    @property
    def source_manifest(self) -> Any | None:
        """The attached :class:`~cmr_core.source_manifest.DicomSourceManifest`."""
        return self._source_manifest

    @property
    def workflow_id(self) -> str | None:
        """Manifest-bound workflow identity, else ``None`` (legacy volumes)."""
        return None if self._source_manifest is None else self._source_manifest.workflow_id

    @property
    def study_uid(self) -> str | None:
        return None if self._source_manifest is None else self._source_manifest.study_uid

    @property
    def series_uid(self) -> str | None:
        return None if self._source_manifest is None else self._source_manifest.series_uid

    @property
    def cine_volume_digest(self) -> str | None:
        return (
            None
            if self._source_manifest is None
            else self._source_manifest.cine_volume_digest
        )

    @property
    def n_slices(self) -> int:
        return len(self.geometries)

    @property
    def n_frames(self) -> int:
        return len(self.frames[0])

    def frame(self, slice_index: int, frame_index: int) -> CardiacFrame:
        return self.frames[slice_index][frame_index]

    def stacked_pixels(self) -> np.ndarray:
        """Return pixels as float32 array of shape [slices, frames, rows, cols].

        This is the canonical array view for segmentation providers.
        """
        return np.stack(
            [
                np.stack([f.pixel_data for f in slice_frames]).astype(np.float32)
                for slice_frames in self.frames
            ]
        )

    def validate(self) -> None:
        """Re-run structural validation (geometry sizes vs pixel shapes)."""
        for s, geom in enumerate(self.geometries):
            for f in self.frames[s]:
                if f.pixel_data.shape != (geom.rows, geom.cols):
                    raise CMRError(
                        INVALID_GEOMETRY,
                        f"frame ({s},{f.frame_index}) shape {f.pixel_data.shape} "
                        f"!= geometry {geom}",
                    )
