"""Immutable canonical model for CMR Studio structured reports.

This module is deliberately a projection boundary.  It carries the values
already present in a frozen :class:`Measurement`, together with caller-owned
identity and a stored QC snapshot.  It does not calculate or interpret any
clinical quantity.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from cmr_core.domain_types import Measurement, canonical_json
from cmr_qc.report import QCReport
from cmr_quantification.reference_ranges import ReferenceSex

from .errors import ReportError

REPORT_FORMAT_VERSION: Literal["cmr-report-v1"] = "cmr-report-v1"
RESEARCH_USE_ONLY_DISCLAIMER = (
    "RESEARCH USE ONLY — CMR Studio has not completed clinical validation; "
    "this report is not for diagnostic use."
)


@dataclass(frozen=True)
class ReportInput:
    """The stored state and explicit caller context used to build a report."""

    measurement: Measurement
    qc_summary: QCReport | None
    series_uid: str
    study_uid: str
    workflow_id: str
    patient_sex: ReferenceSex | None
    report_created_at: str


@dataclass(frozen=True)
class MetricValue:
    """One stored numeric value and its stored unit spelling."""

    value: float
    unit: str

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ReportError(
                "REPORT_MODEL: MetricValue.value must be a real stored number"
            )
        if not math.isfinite(self.value):
            raise ReportError("REPORT_MODEL: MetricValue.value must be finite")
        if not isinstance(self.unit, str) or not self.unit:
            raise ReportError("REPORT_MODEL: MetricValue.unit must be a non-empty string")

    def to_canonical(self) -> dict[str, str]:
        """Return the frozen six-decimal canonical value projection.

        The dataclass retains the stored number for renderer formatting.  The
        digest payload intentionally carries only this fixed presentation
        projection, matching the repository's canonical artifact doctrine.
        """

        return {"value": format(self.value, ".6f"), "unit": self.unit}


@dataclass(frozen=True)
class LvMetrics:
    edv_ml: MetricValue
    esv_ml: MetricValue
    sv_ml: MetricValue
    ef_pct: MetricValue
    lv_mass_g: MetricValue

    def to_canonical(self) -> dict[str, dict[str, str]]:
        return {
            "edv_ml": self.edv_ml.to_canonical(),
            "esv_ml": self.esv_ml.to_canonical(),
            "sv_ml": self.sv_ml.to_canonical(),
            "ef_pct": self.ef_pct.to_canonical(),
            "lv_mass_g": self.lv_mass_g.to_canonical(),
        }


@dataclass(frozen=True)
class RvMetrics:
    rv_edv_ml: MetricValue
    rv_esv_ml: MetricValue
    rv_sv_ml: MetricValue
    rv_ef_pct: MetricValue

    def to_canonical(self) -> dict[str, dict[str, str]]:
        return {
            "rv_edv_ml": self.rv_edv_ml.to_canonical(),
            "rv_esv_ml": self.rv_esv_ml.to_canonical(),
            "rv_sv_ml": self.rv_sv_ml.to_canonical(),
            "rv_ef_pct": self.rv_ef_pct.to_canonical(),
        }


@dataclass(frozen=True)
class RangeComparison:
    """A stored value beside stored frozen-row bounds.

    No field records an in-range decision.  The two bound strings are kept as
    presentation values from the selected frozen row.
    """

    metric: str
    value: MetricValue
    lower_display: str
    upper_display: str

    def to_canonical(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value.to_canonical(),
            "lower_display": self.lower_display,
            "upper_display": self.upper_display,
        }


@dataclass(frozen=True)
class ReportIdentity:
    series_uid: str
    study_uid: str
    workflow_id: str
    patient_sex: str | None
    report_created_at: str

    def __post_init__(self) -> None:
        for name in ("series_uid", "study_uid", "workflow_id", "report_created_at"):
            if not isinstance(getattr(self, name), str):
                raise ReportError(f"REPORT_MODEL: identity {name} must be a string")
        if self.patient_sex is not None and not isinstance(self.patient_sex, str):
            raise ReportError("REPORT_MODEL: identity patient_sex must be a string or None")

    def to_canonical(self) -> dict[str, str | None]:
        return {
            "series_uid": self.series_uid,
            "study_uid": self.study_uid,
            "workflow_id": self.workflow_id,
            "patient_sex": self.patient_sex,
            "report_created_at": self.report_created_at,
        }


@dataclass(frozen=True)
class QcFlagRow:
    code: str
    message: str
    source: str | None
    severity: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not isinstance(self.message, str):
            raise ReportError("REPORT_MODEL: QC flag code and message must be strings")
        if self.source is not None and not isinstance(self.source, str):
            raise ReportError("REPORT_MODEL: QC flag source must be a string or None")
        if not isinstance(self.severity, str):
            raise ReportError("REPORT_MODEL: QC flag severity must be a string")

    def to_canonical(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "severity": self.severity,
        }


def _digest_payload(payload: Mapping[str, Any]) -> str:
    try:
        canonical = canonical_json(payload)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except ReportError:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise ReportError("REPORT_DIGEST: canonical payload could not be serialized") from exc


@dataclass(frozen=True)
class CmrReportV1:
    report_format_version: Literal["cmr-report-v1"]
    report_id: str = field(init=False)
    identity: ReportIdentity
    lv: LvMetrics
    rv: RvMetrics
    ed_frame: int
    es_frame: int
    algorithm_version: str
    software_version: str
    measurement_id: str
    range_comparisons: Mapping[str, RangeComparison] | None
    qc_verdict: str | None
    qc_flag_count: int | None
    qc_flags: tuple[QcFlagRow, ...]
    disclaimers: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.report_format_version != REPORT_FORMAT_VERSION:
            raise ReportError(
                "REPORT_FORMAT: unsupported report format "
                f"{self.report_format_version!r}"
            )
        if not isinstance(self.identity, ReportIdentity):
            raise ReportError("REPORT_MODEL: identity must be ReportIdentity")
        if not isinstance(self.lv, LvMetrics) or not isinstance(self.rv, RvMetrics):
            raise ReportError("REPORT_MODEL: LV/RV metric groups are incomplete")
        if isinstance(self.ed_frame, bool) or not isinstance(self.ed_frame, int):
            raise ReportError("REPORT_MODEL: ed_frame must be an integer")
        if isinstance(self.es_frame, bool) or not isinstance(self.es_frame, int):
            raise ReportError("REPORT_MODEL: es_frame must be an integer")
        for name in ("algorithm_version", "software_version", "measurement_id"):
            if not isinstance(getattr(self, name), str):
                raise ReportError(f"REPORT_MODEL: {name} must be a string")
        if self.qc_verdict is not None and not isinstance(self.qc_verdict, str):
            raise ReportError("REPORT_MODEL: qc_verdict must be a string or None")
        if self.qc_flag_count is not None and (
            isinstance(self.qc_flag_count, bool)
            or not isinstance(self.qc_flag_count, int)
            or self.qc_flag_count < 0
        ):
            raise ReportError("REPORT_MODEL: qc_flag_count must be a non-negative integer")

        flags = tuple(self.qc_flags)
        if any(not isinstance(flag, QcFlagRow) for flag in flags):
            raise ReportError("REPORT_MODEL: qc_flags must contain QcFlagRow values")
        disclaimers = tuple(self.disclaimers)
        if any(not isinstance(disclaimer, str) for disclaimer in disclaimers):
            raise ReportError("REPORT_MODEL: disclaimers must contain strings")
        object.__setattr__(self, "qc_flags", flags)
        object.__setattr__(self, "disclaimers", disclaimers)

        if self.range_comparisons is not None:
            if not isinstance(self.range_comparisons, Mapping):
                raise ReportError("REPORT_MODEL: range_comparisons must be a mapping or None")
            comparisons = dict(self.range_comparisons)
            if any(
                not isinstance(key, str) or not isinstance(value, RangeComparison)
                for key, value in comparisons.items()
            ):
                raise ReportError("REPORT_MODEL: range_comparisons has invalid entries")
            object.__setattr__(self, "range_comparisons", MappingProxyType(comparisons))

        object.__setattr__(self, "report_id", _digest_payload(self.to_canonical()))

    def to_canonical(self) -> dict[str, Any]:
        """Return the exact digest payload, excluding ``report_id`` itself."""

        payload: dict[str, Any] = {
            "report_format_version": self.report_format_version,
            "identity": self.identity.to_canonical(),
            "lv": self.lv.to_canonical(),
            "rv": self.rv.to_canonical(),
            "ed_frame": self.ed_frame,
            "es_frame": self.es_frame,
            "algorithm_version": self.algorithm_version,
            "software_version": self.software_version,
            "measurement_id": self.measurement_id,
            "range_comparisons": (
                None
                if self.range_comparisons is None
                else {
                    key: value.to_canonical()
                    for key, value in self.range_comparisons.items()
                }
            ),
            "qc_flags": [flag.to_canonical() for flag in self.qc_flags],
            "disclaimers": list(self.disclaimers),
        }
        # An unavailable QC snapshot is intentionally absent from the
        # canonical payload.  Renderers carry the fixed human-facing sentinel.
        if self.qc_verdict is not None:
            payload["qc_verdict"] = self.qc_verdict
        if self.qc_flag_count is not None:
            payload["qc_flag_count"] = self.qc_flag_count
        return payload


def report_payload_digest(report: CmrReportV1) -> str:
    """Return the digest that must equal a report's content identity."""

    if not isinstance(report, CmrReportV1):
        raise ReportError("REPORT_DIGEST: expected a CmrReportV1")
    return _digest_payload(report.to_canonical())


def verify_report_digest(report: CmrReportV1) -> None:
    """Fail closed if a report object has been changed after construction."""

    expected = report_payload_digest(report)
    if report.report_id != expected:
        raise ReportError(
            "REPORT_DIGEST: report_id does not match canonical payload "
            f"({report.report_id!r} != {expected!r})"
        )


__all__ = [
    "CmrReportV1",
    "LvMetrics",
    "MetricValue",
    "QcFlagRow",
    "RangeComparison",
    "ReportIdentity",
    "ReportInput",
    "REPORT_FORMAT_VERSION",
    "RESEARCH_USE_ONLY_DISCLAIMER",
    "report_payload_digest",
    "verify_report_digest",
]
