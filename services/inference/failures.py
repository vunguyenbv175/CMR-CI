"""The closed ADR-0003 inference failure taxonomy.

The registry in this module is deliberately small and explicit.  Code which
adds a new failure must first add a new reviewed taxonomy version; callers are
therefore not allowed to manufacture an unclassified string at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

TAXONOMY_VERSION = "adr-0003-failure-taxonomy-v1"


class FailureClass(str, Enum):
    INPUT_UNSUPPORTED = "INPUT_UNSUPPORTED"
    SOURCE_IDENTITY = "SOURCE_IDENTITY"
    GEOMETRY = "GEOMETRY"
    PREPROCESSING = "PREPROCESSING"
    ARTIFACT = "ARTIFACT"
    RUNTIME = "RUNTIME"
    RESOURCE_BUDGET = "RESOURCE_BUDGET"
    MODEL_OUTPUT = "MODEL_OUTPUT"
    INVERSE_MAPPING = "INVERSE_MAPPING"
    ANATOMICAL_QC = "ANATOMICAL_QC"
    UNCERTAINTY_OOD = "UNCERTAINTY_OOD"
    PERSISTENCE_PROVENANCE = "PERSISTENCE_PROVENANCE"
    USER_CANCELLED = "USER_CANCELLED"


class FailureSeverity(str, Enum):
    FATAL = "FATAL"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    ABORT_LOCAL = "ABORT_LOCAL"


# The public name ``Severity`` is useful to callers and avoids making the
# classification API depend on the older cmr_core error severity enum.
Severity = FailureSeverity


@dataclass(frozen=True)
class FailureClassification:
    code: str
    failure_class: FailureClass
    severity: FailureSeverity

    @property
    def class_name(self) -> str:
        return self.failure_class.value


def _classification(
    code: str, failure_class: FailureClass, severity: FailureSeverity
) -> FailureClassification:
    return FailureClassification(code, failure_class, severity)


_REGISTRY = (
    _classification("SOURCE_DIGEST_MISMATCH", FailureClass.SOURCE_IDENTITY, Severity.FATAL),
    _classification("PREPROCESSING_DEGENERATE", FailureClass.PREPROCESSING, Severity.FATAL),
    _classification("PREPROCESSING_NONFINITE", FailureClass.PREPROCESSING, Severity.FATAL),
    _classification("LABEL_MISSING", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("LABEL_DUPLICATE", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("LABEL_UNKNOWN", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("LABEL_MANIFEST_UNSUPPORTED", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("LABEL_EXCLUSIVITY_UNSUPPORTED", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("LABEL_PROBABILITY_UNSUPPORTED", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("TRANSFORM_OP_UNKNOWN", FailureClass.PREPROCESSING, Severity.FATAL),
    _classification("TRANSFORM_PARAM_INVALID", FailureClass.PREPROCESSING, Severity.FATAL),
    _classification("TRANSFORM_SHAPE_MISMATCH", FailureClass.MODEL_OUTPUT, Severity.FATAL),
    _classification("ARTIFACT_LAYOUT_INVALID", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("ARTIFACT_MODEL_JSON_INVALID", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("ARTIFACT_CHECKSUM_MISMATCH", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("ARTIFACT_WEIGHTS_INVALID", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("ARTIFACT_KNOWN_FAILURES_INVALID", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("ARTIFACT_PICKLE_REJECTED", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("ARTIFACT_PREPROCESSING_UNSUPPORTED", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("ARTIFACT_POSTPROCESSING_UNSUPPORTED", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("ARTIFACT_CALIBRATION_INVALID", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("ARTIFACT_METRICS_INVALID", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("ARTIFACT_SBOM_INVALID", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("ARTIFACT_SIGNATURE_INVALID", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("REGISTRY_BUNDLE_UNKNOWN", FailureClass.PERSISTENCE_PROVENANCE, Severity.FATAL),
    _classification("REGISTRY_CONFLICT", FailureClass.PERSISTENCE_PROVENANCE, Severity.FATAL),
    _classification("REGISTRY_STATE_CONFLICT", FailureClass.PERSISTENCE_PROVENANCE, Severity.FATAL),
    _classification(
        "REGISTRY_ROLLBACK_UNAVAILABLE", FailureClass.PERSISTENCE_PROVENANCE, Severity.FATAL
    ),
    _classification("REGISTRY_ROOT_INVALID", FailureClass.PERSISTENCE_PROVENANCE, Severity.FATAL),
    _classification("REGISTRY_BUNDLE_FILES_INVALID", FailureClass.ARTIFACT, Severity.FATAL),
    _classification("PROFILE_UNKNOWN", FailureClass.RUNTIME, Severity.FATAL),
    _classification("PROFILE_NOT_INSTALLED", FailureClass.RUNTIME, Severity.FATAL),
    _classification("CONVENTION_VERSION_MISMATCH", FailureClass.INPUT_UNSUPPORTED, Severity.FATAL),
    _classification("NATIVE_OUTPUT_INVALID", FailureClass.MODEL_OUTPUT, Severity.FATAL),
    _classification("INFERENCE_CANCELLED", FailureClass.USER_CANCELLED, Severity.ABORT_LOCAL),
    _classification("RESOURCE_BUDGET_EXCEEDED", FailureClass.RESOURCE_BUDGET, Severity.ABORT_LOCAL),
)

FAILURE_REGISTRY: tuple[FailureClassification, ...] = _REGISTRY
FAILURE_CODES: tuple[str, ...] = tuple(item.code for item in _REGISTRY)
_BY_CODE: Mapping[str, FailureClassification] = MappingProxyType(
    {item.code: item for item in _REGISTRY}
)

# Stable constants are intentionally aliases of the exact registry strings.
(
    SOURCE_DIGEST_MISMATCH,
    PREPROCESSING_DEGENERATE,
    PREPROCESSING_NONFINITE,
    LABEL_MISSING,
    LABEL_DUPLICATE,
    LABEL_UNKNOWN,
    LABEL_MANIFEST_UNSUPPORTED,
    LABEL_EXCLUSIVITY_UNSUPPORTED,
    LABEL_PROBABILITY_UNSUPPORTED,
    TRANSFORM_OP_UNKNOWN,
    TRANSFORM_PARAM_INVALID,
    TRANSFORM_SHAPE_MISMATCH,
    ARTIFACT_LAYOUT_INVALID,
    ARTIFACT_MODEL_JSON_INVALID,
    ARTIFACT_CHECKSUM_MISMATCH,
    ARTIFACT_WEIGHTS_INVALID,
    ARTIFACT_KNOWN_FAILURES_INVALID,
    ARTIFACT_PICKLE_REJECTED,
    ARTIFACT_PREPROCESSING_UNSUPPORTED,
    ARTIFACT_POSTPROCESSING_UNSUPPORTED,
    ARTIFACT_CALIBRATION_INVALID,
    ARTIFACT_METRICS_INVALID,
    ARTIFACT_SBOM_INVALID,
    ARTIFACT_SIGNATURE_INVALID,
    REGISTRY_BUNDLE_UNKNOWN,
    REGISTRY_CONFLICT,
    REGISTRY_STATE_CONFLICT,
    REGISTRY_ROLLBACK_UNAVAILABLE,
    REGISTRY_ROOT_INVALID,
    REGISTRY_BUNDLE_FILES_INVALID,
    PROFILE_UNKNOWN,
    PROFILE_NOT_INSTALLED,
    CONVENTION_VERSION_MISMATCH,
    NATIVE_OUTPUT_INVALID,
    INFERENCE_CANCELLED,
    RESOURCE_BUDGET_EXCEEDED,
) = FAILURE_CODES


def classify(code: str) -> FailureClassification:
    """Return the reviewed classification for ``code`` or fail closed."""

    if not isinstance(code, str) or code not in _BY_CODE:
        raise ValueError(f"unknown ADR-0003 failure code: {code!r}")
    return _BY_CODE[code]


__all__ = [
    *FAILURE_CODES,
    "FAILURE_CODES",
    "FAILURE_REGISTRY",
    "FailureClass",
    "FailureClassification",
    "FailureSeverity",
    "Severity",
    "TAXONOMY_VERSION",
    "classify",
]
