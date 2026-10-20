"""Assembly of a canonical report from stored domain records only."""

from __future__ import annotations

from cmr_core.domain_types import Measurement
from cmr_qc.report import QCReport
from cmr_quantification.reference_ranges import (
    NON_DIAGNOSTIC_PAPILLARY_DISCLAIMER,
    NON_DIAGNOSTIC_REFERENCE_DISCLAIMER,
    REFERENCE_TABLE_VERSION,
    ReferenceQuantity,
    ReferenceSex,
    lookup_reference_range,
)
from cmr_quantification.units import Quantity

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
)


def _stored_metric(measurement: Measurement, name: str) -> MetricValue:
    try:
        quantity = getattr(measurement, name)
    except AttributeError as exc:
        raise ReportError(f"REPORT_INPUT: stored measurement is missing {name}") from exc
    if not isinstance(quantity, Quantity):
        raise ReportError(f"REPORT_INPUT: stored measurement field {name} is not a Quantity")
    try:
        unit = quantity.unit.value
    except AttributeError as exc:
        raise ReportError(f"REPORT_INPUT: stored measurement field {name} has no Unit") from exc
    try:
        return MetricValue(value=quantity.value, unit=unit)
    except ReportError:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise ReportError(f"REPORT_INPUT: stored measurement field {name} is invalid") from exc


def _stored_qc_rows(qc_summary: QCReport) -> tuple[QcFlagRow, ...]:
    try:
        flags = tuple(qc_summary.flags)
    except (AttributeError, TypeError) as exc:
        raise ReportError("REPORT_INPUT: QC snapshot flags are unavailable") from exc
    rows: list[QcFlagRow] = []
    for flag in flags:
        try:
            severity = flag.severity.value
            row = QcFlagRow(
                code=flag.code,
                message=flag.message,
                source=flag.source,
                severity=severity,
            )
        except ReportError:
            raise
        except Exception as exc:
            raise ReportError("REPORT_INPUT: QC snapshot contains an invalid flag") from exc
        rows.append(row)
    return tuple(rows)


def _range_comparisons(measurement: Measurement, patient_sex: ReferenceSex | None):
    try:
        table_version = measurement.range_table_version
    except AttributeError as exc:
        raise ReportError("REPORT_INPUT: stored measurement has no range_table_version") from exc

    if patient_sex is not None and not isinstance(patient_sex, ReferenceSex):
        raise ReportError("REPORT_RANGE: patient_sex must be a ReferenceSex or None")
    if table_version is not None and table_version != REFERENCE_TABLE_VERSION:
        raise ReportError(
            "REPORT_RANGE: unsupported range_table_version " f"{table_version!r}"
        )

    # The single frozen condition is intentionally shared: no sex or no table
    # means no range lookup and no invented comparison object.
    if table_version is None or patient_sex is None:
        return None

    try:
        reference = lookup_reference_range(
            patient_sex,
            ReferenceQuantity.LVEF,
            table_version=table_version,
        )
    except Exception as exc:
        raise ReportError("REPORT_RANGE: frozen reference lookup failed") from exc
    if reference.bsa_indexed:
        raise ReportError("REPORT_RANGE: indexed reference row is not eligible")

    # LVEF is the only non-indexed frozen row matching a raw Measurement
    # field.  The bounds are copied as display strings; no value is examined.
    ef_value = _stored_metric(measurement, "ef_pct")
    return {
        "ef_pct": RangeComparison(
            metric="ef_pct",
            value=ef_value,
            lower_display=str(reference.lower),
            upper_display=str(reference.upper),
        )
    }


def build_report(input_value: ReportInput) -> CmrReportV1:
    """Build a digest-bearing report by projecting stored values verbatim."""

    if not isinstance(input_value, ReportInput):
        raise ReportError("REPORT_INPUT: expected a ReportInput")
    measurement = input_value.measurement
    if not isinstance(measurement, Measurement):
        raise ReportError("REPORT_INPUT: measurement is required and must be a Measurement")

    try:
        lv = LvMetrics(
            edv_ml=_stored_metric(measurement, "edv_ml"),
            esv_ml=_stored_metric(measurement, "esv_ml"),
            sv_ml=_stored_metric(measurement, "sv_ml"),
            ef_pct=_stored_metric(measurement, "ef_pct"),
            lv_mass_g=_stored_metric(measurement, "lv_mass_g"),
        )
        rv = (
            _stored_metric(measurement, "rv_edv_ml"),
            _stored_metric(measurement, "rv_esv_ml"),
            _stored_metric(measurement, "rv_sv_ml"),
            _stored_metric(measurement, "rv_ef_pct"),
        )
        rv_group = RvMetrics(
            rv_edv_ml=rv[0],
            rv_esv_ml=rv[1],
            rv_sv_ml=rv[2],
            rv_ef_pct=rv[3],
        )
        range_comparisons = _range_comparisons(measurement, input_value.patient_sex)

        if input_value.qc_summary is None:
            qc_verdict = None
            qc_flag_count = None
            qc_flags: tuple[QcFlagRow, ...] = ()
        else:
            qc_flags = _stored_qc_rows(input_value.qc_summary)
            try:
                qc_verdict = input_value.qc_summary.verdict
            except Exception as exc:
                raise ReportError("REPORT_INPUT: QC snapshot verdict is unavailable") from exc
            if not isinstance(qc_verdict, str):
                raise ReportError("REPORT_INPUT: QC snapshot verdict must be a string")
            qc_flag_count = len(qc_flags)

        patient_sex = (
            None if input_value.patient_sex is None else input_value.patient_sex.value
        )
        report = CmrReportV1(
            report_format_version=REPORT_FORMAT_VERSION,
            identity=ReportIdentity(
                series_uid=input_value.series_uid,
                study_uid=input_value.study_uid,
                workflow_id=input_value.workflow_id,
                patient_sex=patient_sex,
                report_created_at=input_value.report_created_at,
            ),
            lv=lv,
            rv=rv_group,
            ed_frame=measurement.ed_frame,
            es_frame=measurement.es_frame,
            algorithm_version=measurement.algorithm_version,
            software_version=measurement.software_version,
            measurement_id=measurement.measurement_id,
            range_comparisons=range_comparisons,
            qc_verdict=qc_verdict,
            qc_flag_count=qc_flag_count,
            qc_flags=qc_flags,
            disclaimers=(
                NON_DIAGNOSTIC_REFERENCE_DISCLAIMER,
                NON_DIAGNOSTIC_PAPILLARY_DISCLAIMER,
                RESEARCH_USE_ONLY_DISCLAIMER,
            ),
        )
    except ReportError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ReportError("REPORT_INPUT: stored report input is incomplete or invalid") from exc

    expected = report_payload_digest(report)
    if report.report_id != expected:
        raise ReportError("REPORT_DIGEST: report identity assertion failed")
    return report


__all__ = ["build_report"]
