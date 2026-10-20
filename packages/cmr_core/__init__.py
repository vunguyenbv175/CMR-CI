"""cmr_core — domain contracts, error model, provenance (dependency-light).

No PyTorch, no BAAI, no pydicom imports allowed in this package.
"""

from typing import Any

from .cine import CardiacFrame, CineVolume
from .errors import (
    QC_FAILED,
    TOPOLOGY_BLOCKED,
    CMRError,
    CMRWarning,
    IssueSet,
    Severity,
)
from .geometry import ImageGeometry
from .provenance import VERSION, ModelMetadata, Provenance
from .segmentation import (
    SegmentationProvider,
    SegmentationRequest,
    SegmentationResult,
    Structure,
)
from .study import SEQUENCE_CLASSES, CMRSeries, CMRStudy

__version__ = VERSION

# Node 1 domain records are exposed lazily (PEP 562): eager re-export here would
# make ``import cmr_qc`` re-enter this package via
# cmr_core.domain_types -> cmr_qc.report -> cmr_core.errors while cmr_core is
# still partially initialized, breaking a previously valid import order.
_DOMAIN_RECORD_NAMES = (
    "ApprovalSnapshot",
    "ContourRevision",
    "EDESSelection",
    "HoleAwareContour",
    "Measurement",
    "ModelRunRecord",
    "Point2D",
    "PolygonRing",
    "Prediction",
    "SelectedPhasePrediction",
    "TopologyBlock",
    "WholeCinePrediction",
    "WholeCineRevision",
)

# FNC-G2 derivation-only approved-result reference (lazy for the same
# import-order reason as the domain records above).
_RESULT_REFERENCE_NAMES = (
    "ApprovalLookup",
    "ApprovedResultReference",
)


def __getattr__(name: str) -> Any:
    if name in _DOMAIN_RECORD_NAMES:
        from cmr_core import domain_types

        return getattr(domain_types, name)
    if name in _RESULT_REFERENCE_NAMES:
        from cmr_core import result_reference

        return getattr(result_reference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals()) | set(_DOMAIN_RECORD_NAMES) | set(_RESULT_REFERENCE_NAMES)
    )


__all__ = [
    "CMRError",
    "CMRWarning",
    "CMRSeries",
    "CMRStudy",
    "CardiacFrame",
    "CineVolume",
    "ImageGeometry",
    "IssueSet",
    "ModelMetadata",
    "Provenance",
    "QC_FAILED",
    "SEQUENCE_CLASSES",
    "SegmentationProvider",
    "SegmentationRequest",
    "SegmentationResult",
    "Severity",
    "Structure",
    "TOPOLOGY_BLOCKED",
    "VERSION",
]

__all__ += list(_DOMAIN_RECORD_NAMES)
__all__ += list(_RESULT_REFERENCE_NAMES)
