"""Canonical JSON renderer for structured reports."""

from __future__ import annotations

from cmr_core.domain_types import canonical_json

from .errors import ReportError
from .model import CmrReportV1, verify_report_digest


def render_report_json(report: CmrReportV1) -> bytes:
    """Return canonical UTF-8 JSON bytes without a trailing newline."""

    try:
        verify_report_digest(report)
        return canonical_json(report.to_canonical()).encode("utf-8")
    except ReportError:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise ReportError("REPORT_RENDER: JSON serialization failed") from exc


__all__ = ["render_report_json"]
