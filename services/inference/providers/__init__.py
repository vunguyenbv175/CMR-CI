"""Segmentation provider implementations and registry (Task E).

The workstation only sees :class:`SegmentationProvider` from
``cmr_core.segmentation``; providers here are swappable by configuration
(provider name string). No PyTorch, no BAAI, no GPU, no network.

This ``inference.providers`` package intentionally registers ONLY the
deterministic in-process providers (``dummy``, ``manual``). Experimental
deep-learning providers (``inference.providers.baai``) are NOT registered
by default: they require runtime capabilities (checkpoints, PyTorch/GPU)
that must not affect default workstation behaviour. They are importable
directly and can be added to the registry by an explicit deployment
configuration later.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
from cmr_core.errors import UNKNOWN_PROVIDER, CMRError
from cmr_core.provenance import ModelMetadata, Provenance
from cmr_core.segmentation import (
    SegmentationProvider,
    SegmentationRequest,
    SegmentationResult,
    Structure,
)

DUMMY_LABEL_MAP = {s.value: i for i, s in enumerate(Structure, start=1)}


class DummySegmentationProvider:
    """Deterministic synthetic segmentation for tests and development.

    Produces fixed-shape synthetic masks (ellipse / ring / offset ellipse)
    with per-slice, per-frame, per-structure perturbations derived from an
    MD5 hash of the request identity — the same request always yields the
    same result on every platform (no numpy RNG, no system entropy).

    Never meant for clinical use; exists to prove the provider contract.
    """

    name = "dummy"

    def __init__(
        self, radius_fraction: float = 0.30, ring_thickness_fraction: float = 0.08
    ) -> None:
        self.radius_fraction = radius_fraction
        self.ring_thickness_fraction = ring_thickness_fraction

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        volume = request.volume
        frames = request.requested_frames or list(range(volume.n_frames))
        model = ModelMetadata(
            name=self.name,
            version="0.1.0",
            label_map=dict(DUMMY_LABEL_MAP),
            preprocessing_version="none",
        )
        result = SegmentationResult(
            study_uid=request.study_uid,
            series_uid=request.series_uid,
            sequence_type=request.sequence_type,
            model=model,
            provenance=Provenance(
                study_uid=request.study_uid,
                series_uid=request.series_uid,
                model=model,
            ),
        )
        for s in range(volume.n_slices):
            geometry = volume.geometries[s]
            rows, cols = geometry.rows, geometry.cols
            for f in frames:
                rng = self._rng(request, s, f)
                cx = cols / 2 + rng.uniform(-cols * 0.02, cols * 0.02)
                cy = rows / 2 + rng.uniform(-rows * 0.02, rows * 0.02)
                rx = cols * self.radius_fraction
                ry = rows * self.radius_fraction
                lv_cavity = _ellipse_mask(rows, cols, cx, cy, rx, ry)
                if Structure.LV_CAVITY in request.requested_structures:
                    result.masks[(s, f, Structure.LV_CAVITY)] = lv_cavity
                if Structure.LV_MYOCARDIUM in request.requested_structures:
                    ring_rx = rx * (1 + self.ring_thickness_fraction)
                    ring_ry = ry * (1 + self.ring_thickness_fraction)
                    outer = _ellipse_mask(rows, cols, cx, cy, ring_rx, ring_ry)
                    result.masks[(s, f, Structure.LV_MYOCARDIUM)] = outer & ~lv_cavity
                if Structure.RV_CAVITY in request.requested_structures:
                    rv_center = (cx + cols * 0.28, cy + rows * 0.05)
                    result.masks[(s, f, Structure.RV_CAVITY)] = _ellipse_mask(
                        rows, cols, rv_center[0], rv_center[1], rx * 0.55, ry * 0.5
                    )
        return result

    def _rng(self, request: SegmentationRequest, s: int, f: int) -> random.Random:
        seed_input = f"{request.study_uid}|{request.series_uid}|{s}|{f}"
        digest = hashlib.md5(seed_input.encode("utf-8")).hexdigest()
        return random.Random(int(digest[:12], 16))

    def describe(self) -> str:
        return "dummy deterministic provider (synthetic ellipses, for tests only)"


class ManualSegmentationProvider:
    """Provider wrapping physician-supplied masks into a SegmentationResult.

    Masks are supplied as ``provider_settings["masks"]``: dict keyed by
    ``(slice_index, frame_index, Structure)`` with boolean arrays. This is
    how the workstation hands manual tracings back through the same
    contract (source=manual, model=None).
    """

    name = "manual"

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        masks = request.provider_settings.get("masks")
        if not isinstance(masks, dict) or not masks:
            raise CMRError(
                "provider.manual_no_masks", "manual provider requires provider_settings['masks']"
            )
        result = SegmentationResult(
            study_uid=request.study_uid,
            series_uid=request.series_uid,
            sequence_type=request.sequence_type,
            model=None,
            provenance=Provenance(
                study_uid=request.study_uid,
                series_uid=request.series_uid,
                model=None,
            ),
        )
        for key, mask in masks.items():
            if not isinstance(mask, np.ndarray) or mask.ndim != 2:
                raise CMRError(
                    "provider.manual_invalid_mask", f"mask for {key} must be a 2D numpy array"
                )
            result.masks[key] = mask.astype(bool)
        return result

    def describe(self) -> str:
        return "manual provider (physician masks -> SegmentationResult)"


def _ellipse_mask(rows: int, cols: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:cols]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    return mask


PROVIDERS: dict[str, type] = {
    DummySegmentationProvider.name: DummySegmentationProvider,
    ManualSegmentationProvider.name: ManualSegmentationProvider,
}


def get_provider(name: str, **kwargs) -> SegmentationProvider:
    """Provider registry lookup — switching models is configuration only."""
    cls = PROVIDERS.get(name)
    if cls is None:
        raise CMRError(
            UNKNOWN_PROVIDER, f"unknown segmentation provider {name!r}; known: {sorted(PROVIDERS)}"
        )
    return cls(**kwargs)


def list_providers() -> list[str]:
    return sorted(PROVIDERS)
