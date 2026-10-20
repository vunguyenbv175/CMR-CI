"""Model-agnostic segmentation contracts (Task E).

The workstation only ever talks to :class:`SegmentationProvider`. Concrete
implementations (dummy, BAAI, nnU-Net, remote, ...) live outside
``packages/cmr_core`` — swapping a model must require configuration only.

No PyTorch tensors and no BAAI types appear here by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from .cine import CineVolume
from .errors import CMRWarning
from .provenance import ModelMetadata, Provenance


class Structure(Enum):
    """Structures that contouring can produce in this cycle."""

    LV_CAVITY = "lv_cavity"
    LV_MYOCARDIUM = "lv_myocardium"
    RV_CAVITY = "rv_cavity"


@dataclass
class SegmentationRequest:
    """Everything a provider needs to segment one cine volume."""

    study_uid: str
    series_uid: str
    sequence_type: str
    volume: CineVolume
    requested_structures: list[Structure] = field(
        default_factory=lambda: [Structure.LV_CAVITY, Structure.LV_MYOCARDIUM, Structure.RV_CAVITY]
    )
    requested_frames: list[int] | None = None  # None = all frames
    provider_settings: dict[str, object] = field(default_factory=dict)


@dataclass
class SegmentationResult:
    """Standardized provider output independent from any model internals.

    ``masks`` maps ``(slice_index, frame_index, Structure)`` to a boolean
    numpy mask in the same pixel grid as the source image.
    """

    study_uid: str
    series_uid: str
    sequence_type: str
    masks: dict[tuple[int, int, Structure], np.ndarray] = field(default_factory=dict)
    model: ModelMetadata | None = None
    processing_time_s: float | None = None
    provenance: Provenance | None = None
    qc_flags: list[CMRWarning] = field(default_factory=list)
    warnings: list[CMRWarning] = field(default_factory=list)
    # Free-form, machine-readable provider details (packing geometry, ordering,
    # model-space spacing, ...). Diagnostics and provenance only: this dict must
    # NEVER feed deterministic clinical math (EDV/ESV/EF/mass/volume), which
    # reads canonical ImageGeometry and IPP-derived spacing exclusively.
    provider_metadata: dict[str, object] = field(default_factory=dict)

    def has(self, slice_index: int, frame_index: int, structure: Structure) -> bool:
        return (slice_index, frame_index, structure) in self.masks

    def mask(self, slice_index: int, frame_index: int, structure: Structure) -> np.ndarray:
        return self.masks[(slice_index, frame_index, structure)]


class SegmentationProvider(Protocol):
    """Contract every segmentation backend implements.

    Implementations MUST be deterministic-or-declared and MUST fill
    ``SegmentationResult.model`` and ``.provenance``.
    """

    name: str

    def segment(self, request: SegmentationRequest) -> SegmentationResult: ...

    def describe(self) -> str:  # pragma: no cover - protocol default
        """Human-readable provider description (for logs/settings UI)."""
        return self.name
