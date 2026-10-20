"""First-party deterministic PDF-1.4 renderer for structured reports.

Only the Python standard library and the local report model are involved.
The writer emits five fixed objects, Standard-14 Helvetica, one text stream,
and a conventional fixed-width cross-reference table.  No metadata dates or
document identifier are emitted.
"""

from __future__ import annotations

from .errors import ReportError
from .model import CmrReportV1, RangeComparison, verify_report_digest

_HEADER = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
_OBJECT_1 = b"<< /Type /Catalog /Pages 2 0 R >>"
_OBJECT_2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
_OBJECT_3 = (
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
)
_OBJECT_5 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
_TRANSLITERATION = {
    "±": "+/-",
    "–": "-",
    "—": "-",
    "²": "^2",
    "\u00a0": " ",
    "≤": "<=",
    "≥": ">=",
}


def _transliterate(value: str) -> str:
    output: list[str] = []
    for character in value:
        if character in _TRANSLITERATION:
            output.append(_TRANSLITERATION[character])
        elif ord(character) > 0x7F:
            output.append("?")
        elif character in "\r\n":
            output.append(" ")
        else:
            output.append(character)
    return "".join(output)


def _literal(value: str) -> bytes:
    text = _transliterate(value)
    # Backslash must be escaped before the parentheses are escaped.
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return text.encode("ascii")


def _value_text(value: object) -> str:
    return "None" if value is None else str(value)


def _metric_line(label: str, metric: object) -> str:
    return f"{label}: {format(metric.value, '.1f')} {metric.unit}"  # type: ignore[attr-defined]


def _range_line(key: str, comparison: RangeComparison) -> str:
    metric = comparison.value
    return (
        f"range_{key}: {format(metric.value, '.1f')} {metric.unit} | "
        f"lower: {comparison.lower_display} | upper: {comparison.upper_display}"
    )


def _content_lines(report: CmrReportV1) -> list[str]:
    lines = [
        "CMR Studio structured report",
        f"report_format_version: {report.report_format_version}",
        f"report_id: {report.report_id}",
        f"measurement_id: {report.measurement_id}",
        f"series_uid: {report.identity.series_uid}",
        f"study_uid: {report.identity.study_uid}",
        f"workflow_id: {report.identity.workflow_id}",
        f"patient_sex: {_value_text(report.identity.patient_sex)}",
        f"report_created_at: {report.identity.report_created_at}",
        f"ed_frame: {report.ed_frame}",
        f"es_frame: {report.es_frame}",
        f"algorithm_version: {report.algorithm_version}",
        f"software_version: {report.software_version}",
        _metric_line("edv_ml", report.lv.edv_ml),
        _metric_line("esv_ml", report.lv.esv_ml),
        _metric_line("sv_ml", report.lv.sv_ml),
        _metric_line("ef_pct", report.lv.ef_pct),
        _metric_line("lv_mass_g", report.lv.lv_mass_g),
        _metric_line("rv_edv_ml", report.rv.rv_edv_ml),
        _metric_line("rv_esv_ml", report.rv.rv_esv_ml),
        _metric_line("rv_sv_ml", report.rv.rv_sv_ml),
        _metric_line("rv_ef_pct", report.rv.rv_ef_pct),
    ]
    if report.range_comparisons is not None:
        lines.extend(
            _range_line(key, report.range_comparisons[key])
            for key in sorted(report.range_comparisons)
        )
    if report.qc_verdict is None:
        lines.append("QC: NOT AVAILABLE")
    else:
        lines.append(f"qc_verdict: {report.qc_verdict}")
        lines.append(f"qc_flag_count: {report.qc_flag_count}")
        lines.extend(
            f"qc_flag: {flag.code} | {flag.message} | {_value_text(flag.source)} | "
            f"{flag.severity}"
            for flag in report.qc_flags[:10]
        )
        if len(report.qc_flags) > 10:
            lines.append(
                f"... +{len(report.qc_flags) - 10} more flags (see QC report)"
            )
    lines.extend(f"disclaimer: {disclaimer}" for disclaimer in report.disclaimers)
    return lines


def _content_stream(report: CmrReportV1) -> bytes:
    operations = ["BT /F1 10 Tf"]
    for index, line in enumerate(_content_lines(report)):
        y = 752 - (14 * index)
        operations.append(f"56 {y} Td ({_literal(line).decode('ascii')}) Tj")
    operations.append("ET")
    return "\n".join(operations).encode("ascii")


def _object(number: int, body: bytes) -> bytes:
    return str(number).encode("ascii") + b" 0 obj\n" + body + b"\nendobj\n"


def render_report_pdf(report: CmrReportV1) -> bytes:
    """Return a deterministic one-page PDF-1.4 byte stream."""

    try:
        verify_report_digest(report)
        stream = _content_stream(report)
        object_4 = (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        objects = (
            _object(1, _OBJECT_1),
            _object(2, _OBJECT_2),
            _object(3, _OBJECT_3),
            _object(4, object_4),
            _object(5, _OBJECT_5),
        )
        output = bytearray(_HEADER)
        offsets: list[int] = []
        for object_bytes in objects:
            offsets.append(len(output))
            output.extend(object_bytes)

        xref_offset = len(output)
        output.extend(b"xref\n0 6\n")
        # PDF cross-reference rows are exactly 20 bytes: ten offset digits,
        # one space, five generation digits, one space, type, space, LF.
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets:
            if offset > 9999999999:
                raise ReportError("REPORT_RENDER: PDF object offset is too large")
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(b"trailer\n<< /Size 6 /Root 1 0 R >>\n")
        output.extend(b"startxref\n")
        output.extend(str(xref_offset).encode("ascii"))
        output.extend(b"\n%%EOF\n")
        return bytes(output)
    except ReportError:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise ReportError("REPORT_RENDER: PDF serialization failed") from exc


__all__ = ["render_report_pdf"]
