"""Public reproduction and byte-identity verification API."""

from __future__ import annotations

from .assembler import build_report
from .errors import ReportError
from .model import ReportInput
from .render_csv import render_report_csv
from .render_json import render_report_json
from .render_pdf import render_report_pdf


def reproduce_report(input_value: ReportInput, fmt: str) -> bytes:
    """Assemble and render one report in the requested frozen format."""

    if fmt not in ("json", "csv", "pdf"):
        raise ReportError(f"REPORT_FORMAT: unsupported format {fmt!r}")
    try:
        report = build_report(input_value)
        if fmt == "json":
            return render_report_json(report)
        if fmt == "csv":
            return render_report_csv(report)
        return render_report_pdf(report)
    except ReportError:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise ReportError("REPORT_REPRODUCE: report reproduction failed") from exc


def verify_reproducible(input_value: ReportInput) -> None:
    """Render each format twice and fail closed if any bytes differ."""

    for fmt in ("json", "csv", "pdf"):
        first = reproduce_report(input_value, fmt)
        second = reproduce_report(input_value, fmt)
        if first != second:
            raise ReportError(f"REPORT_REPRODUCE: {fmt} bytes are not reproducible")


__all__ = ["reproduce_report", "verify_reproducible"]
