"""CMR Studio deterministic structured reporting package."""

from .assembler import build_report
from .errors import ReportError
from .model import (
    REPORT_FORMAT_VERSION,
    RESEARCH_USE_ONLY_DISCLAIMER,
    CmrReportV1,
    LvMetrics,
    MetricValue,
    QcFlagRow,
    RangeComparison,
    ReportIdentity,
    ReportInput,
    RvMetrics,
    report_payload_digest,
    verify_report_digest,
)
from .render_csv import render_report_csv
from .render_json import render_report_json
from .render_pdf import render_report_pdf
from .reproduce import reproduce_report, verify_reproducible

__all__ = [
    "CmrReportV1",
    "LvMetrics",
    "MetricValue",
    "QcFlagRow",
    "RangeComparison",
    "ReportError",
    "ReportIdentity",
    "ReportInput",
    "RvMetrics",
    "REPORT_FORMAT_VERSION",
    "RESEARCH_USE_ONLY_DISCLAIMER",
    "build_report",
    "render_report_csv",
    "render_report_json",
    "render_report_pdf",
    "reproduce_report",
    "report_payload_digest",
    "verify_report_digest",
    "verify_reproducible",
]
