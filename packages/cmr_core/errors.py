"""Levels and containers for the CMR Studio error/warning model.

Three severities are distinguished:

- FATAL (``CMRError``): the operation cannot proceed (impossible geometry,
  unreadable pixel data, incompatible segmentation result, ...).
- WARNING (``CMRWarning``): recoverable problem; processing continues with
  degraded information (missing optional DICOM tag, ambiguous series
  classification, missing TriggerTime, ...).
- QC (``QCWarning``): the *result* is suspect (discontinuous contour,
  implausible volume change, missing structure, ...). QC warnings originate
  from :mod:`cmr_qc`.

Exceptions are never swallowed silently: a warning must be collected
explicitly (see :class:`WarningCollector` / :class:`IssueSet`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

# --- error/warning codes (stable identifiers used by tests and UI) -----------

INVALID_ARGUMENT = "app.invalid_argument"
INVALID_GEOMETRY = "geometry.invalid"
MISSING_ESSENTIAL_GEOMETRY = "geometry.essential_missing"
MISSING_OPTIONAL_TAG = "dicom.missing_optional_tag"
UNREADABLE_FILE = "dicom.unreadable"
NOT_A_DICOM_FILE = "dicom.not_a_file"
MISSING_PIXEL_DATA = "dicom.missing_pixel_data"
MULTIFRAME_UNSUPPORTED = "dicom.multiframe_unsupported"
UNSUPPORTED_ENHANCED_VARIANT = "dicom.enhanced_variant_unsupported"
DUPLICATE_INSTANCE = "dicom.duplicate_instance"
EMPTY_INPUT = "input.empty"
AMBIGUOUS_SLICE_ORDER = "ordering.ambiguous_slice"
AMBIGUOUS_FRAME_ORDER = "ordering.ambiguous_frame"
TIMING_DEGRADED = "timing.degraded_confidence"
SERIES_NOT_A_CINE = "cine.not_a_cine"
CONTOUR_INVALID = "contour.invalid"
TOPOLOGY_BLOCKED = "topology.blocked"
SEGMENTATION_INCOMPATIBLE = "segmentation.incompatible"
UNKNOWN_PROVIDER = "provider.unknown"
QUANTIFICATION_INVALID_INPUT = "quantification.invalid_input"
QC_FAILED = "qc.failed"
QC_MISSING_STRUCTURE = "qc.missing_structure"
QC_SELF_INTERSECTION = "qc.self_intersection"
QC_CONTAINMENT = "qc.containment"
QC_VOLUME_CURVE = "qc.volume_curve"
# FNC-G2 approved-result references: resolving a derivation-only reference
# against sealed approval rows fails closed with these codes.
RESULT_REFERENCE_UNSEALED = "resultref.unsealed"
RESULT_REFERENCE_BINDING_MISMATCH = "resultref.binding_mismatch"


class Severity(Enum):
    """Severity of an issue in the CMR Studio error model."""

    FATAL = "fatal"
    WARNING = "warning"
    QC = "qc"


@dataclass(frozen=True)
class CMRWarning:
    """A recoverable problem. Processing continues but information is degraded.

    Attributes:
        code: stable identifier (use constants above).
        message: human-readable description.
        source: optional identifier of the affected object
            (file path, SOPInstanceUID, series UID, slice/frame index, ...).
        severity: WARNING or QC. QC warnings mark the result as suspect.
    """

    code: str
    message: str
    source: str | None = None
    severity: Severity = Severity.WARNING

    def __str__(self) -> str:  # pragma: no cover - trivial
        origin = f" [{self.source}]" if self.source else ""
        return f"{self.severity.value}: {self.code}{origin}: {self.message}"

    @classmethod
    def qc(cls, code: str, message: str, source: str | None = None) -> CMRWarning:
        """Build a QC-severity warning."""
        return cls(code=code, message=message, source=source, severity=Severity.QC)


class CMRError(Exception):
    """A fatal error: the operation cannot proceed (severity FATAL).

    Use :class:`CMRWarning` for recoverable problems instead.
    """

    def __init__(self, code: str, message: str, source: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.source = source

    def __str__(self) -> str:  # pragma: no cover - trivial
        origin = f" [{self.source}]" if self.source else ""
        return f"CMRError {self.code}{origin}: {self.message}"


@dataclass
class IssueSet:
    """Collects warnings/QC flags produced by an operation.

    Fatal errors are raised as :class:`CMRError`; this container is for
    recoverable and QC-severity issues. It is mutable and accumulates
    ``CMRWarning`` objects.
    """

    warnings: list[CMRWarning] = field(default_factory=list)

    def add(self, warning: CMRWarning) -> None:
        self.warnings.append(warning)

    def extend(self, warnings: Iterable[CMRWarning]) -> None:
        self.warnings.extend(warnings)

    @property
    def qc_flags(self) -> list[CMRWarning]:
        return [w for w in self.warnings if w.severity is Severity.QC]

    @property
    def has_qc_issues(self) -> bool:
        return len(self.qc_flags) > 0

    def __bool__(self) -> bool:
        return bool(self.warnings)

    def __len__(self) -> int:
        return len(self.warnings)

    def __iter__(self):
        return iter(self.warnings)
