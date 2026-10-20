"""Group DICOM files into CMRStudy with per-series metadata and warnings."""

from __future__ import annotations

from pathlib import Path

from cmr_core.errors import DUPLICATE_INSTANCE, CMRWarning, IssueSet
from cmr_core.study import CMRSeries, CMRStudy

from . import tags as T
from .classify import classify_series
from .metadata import extract_geometry, extract_series_metadata
from .reader import iter_dicom_files, read_dicom


def scan_directory(path: Path, issues: IssueSet | None = None) -> CMRStudy | None:
    """Scan a folder: group files by study -> series, extract metadata.

    Returns None when the folder contains no readable DICOM (warnings
    recorded). Duplicate SOPInstanceUIDs within one series are reported
    and the first instance wins.
    """
    issues = IssueSet() if issues is None else issues
    files = iter_dicom_files(path, issues.warnings)
    study: CMRStudy | None = None
    series_by_uid: dict[str, CMRSeries] = {}
    seen_sops: set[str] = set()

    for f in files:
        ds = read_dicom(f, issues.warnings)
        if ds is None:
            continue
        study_uid = T.get_uid(ds, T.study_uid_key, issues.warnings)
        series_uid = T.get_uid(ds, T.series_uid_key, issues.warnings)
        sop = T.get_uid(ds, T.instance_uid_key, issues.warnings)
        if sop in seen_sops:
            issues.add(
                CMRWarning(
                    DUPLICATE_INSTANCE,
                    f"duplicate SOPInstanceUID {sop}; ignoring repeated file {f}",
                    source=sop,
                )
            )
            continue
        seen_sops.add(sop)

        if study is None:
            study = CMRStudy(
                study_uid=study_uid,
                patient_id=T.get_optional_str(ds, T.patient_id_key, issues.warnings) or "",
                study_date=T.get_optional_str(ds, T.study_date_key, issues.warnings) or "",
            )
        elif study.study_uid != study_uid:
            issues.add(
                CMRWarning(
                    "dicom.multiple_studies",
                    f"folder contains multiple studies; only {study.study_uid} is kept",
                    source=study_uid,
                )
            )
            continue

        series = series_by_uid.get(series_uid)
        if series is None:
            meta = extract_series_metadata(ds, issues.warnings)
            series = CMRSeries(
                series_uid=series_uid,
                series_description=meta["series_description"] or "",
                protocol_name=meta["protocol_name"] or "",
                modality=meta["modality"] or "MR",
                manufacturer=meta["manufacturer"] or "",
                manufacturer_model=meta["manufacturer_model"] or "",
                magnetic_field_strength_t=meta["magnetic_field_strength_t"],
            )
            try:
                series.geometry = extract_geometry(ds, issues.warnings)
            except Exception as exc:  # noqa: BLE001 - fatal geometry error surfaces as warning here;
                # series is still kept so diagnosis is possible
                series.warnings.append(
                    CMRWarning(
                        "geometry.invalid",
                        f"could not extract geometry: {exc}",
                        source=sop,
                    )
                )
            series_by_uid[series_uid] = series
            study.add_series(series)
        else:
            try:
                geom = extract_geometry(ds, issues.warnings)
                if series.geometry is not None and (geom.rows, geom.cols) != (
                    series.geometry.rows,
                    series.geometry.cols,
                ):
                    series.warnings.append(
                        CMRWarning(
                            "geometry.inconsistent",
                            f"instance {sop} has geometry inconsistent with the series",
                            source=sop,
                        )
                    )
            except Exception:  # noqa: BLE001 - already warned via issues
                pass
        series.sop_instance_uids.append(sop)

    if study is not None:
        for series in study.series:
            series.sequence_class = classify_series(series)

    return study
