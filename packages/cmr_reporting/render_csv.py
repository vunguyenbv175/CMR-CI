"""The single canonical two-column CSV renderer for structured reports."""

from __future__ import annotations

from .errors import ReportError
from .model import CmrReportV1, verify_report_digest

_METRIC_FIELDS = (
    ("edv_ml", "lv", "edv_ml"),
    ("esv_ml", "lv", "esv_ml"),
    ("sv_ml", "lv", "sv_ml"),
    ("ef_pct", "lv", "ef_pct"),
    ("lv_mass_g", "lv", "lv_mass_g"),
    ("rv_edv_ml", "rv", "rv_edv_ml"),
    ("rv_esv_ml", "rv", "rv_esv_ml"),
    ("rv_sv_ml", "rv", "rv_sv_ml"),
    ("rv_ef_pct", "rv", "rv_ef_pct"),
)


def _sanitize(value: str) -> str:
    """Apply the frozen transliteration and delimiter-free CSV dialect."""

    replacements = {
        "±": "+/-",
        "–": "-",
        "—": "-",
        "²": "^2",
        "\u00a0": " ",
        "≤": "<=",
        "≥": ">=",
    }
    output: list[str] = []
    for character in value:
        if character in replacements:
            output.append(replacements[character])
        elif ord(character) > 0x7F:
            output.append("?")
        elif character in ',"\n\r':
            output.append(" ")
        else:
            output.append(character)
    return "".join(output)


def _metric_value(report: CmrReportV1, group_name: str, field_name: str) -> str:
    group = getattr(report, group_name)
    metric = getattr(group, field_name)
    return format(metric.value, ".1f")


def render_report_csv(report: CmrReportV1) -> bytes:
    """Return exactly the frozen header plus 19 key/value rows."""

    try:
        verify_report_digest(report)
        rows: list[tuple[str, str]] = [
            ("measurement_id", report.measurement_id),
            ("series_uid", report.identity.series_uid),
            ("study_uid", report.identity.study_uid),
            ("workflow_id", report.identity.workflow_id),
            ("ed_frame", str(report.ed_frame)),
            ("es_frame", str(report.es_frame)),
            ("algorithm_version", report.algorithm_version),
            ("software_version", report.software_version),
            ("report_format_version", report.report_format_version),
        ]
        rows.extend(
            (field_name, _metric_value(report, group_name, metric_name))
            for field_name, group_name, metric_name in _METRIC_FIELDS
        )
        rows.append(
            (
                "qc_verdict",
                report.qc_verdict if report.qc_verdict is not None else "NOT AVAILABLE",
            )
        )
        lines = ["field,value"]
        lines.extend(f"{_sanitize(field)},{_sanitize(value)}" for field, value in rows)
        return "\n".join(lines).encode("utf-8")
    except ReportError:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise ReportError("REPORT_RENDER: CSV serialization failed") from exc


__all__ = ["render_report_csv"]
