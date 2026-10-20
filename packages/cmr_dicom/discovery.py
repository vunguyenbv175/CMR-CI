"""Recursive DICOM library discovery grouped Study -> Series.

Slice 1 of the study/case workspace epic: a pure, read-only scan of a
directory tree that groups DICOM files by StudyInstanceUID then
SeriesInstanceUID. No writes, no sqlite, no env vars, no server/API/
desktop/frontend change.

Reuse-first: header reads go through
:func:`cmr_dicom.reader.read_dicom_auto_with_digest`, identity through
:func:`cmr_dicom.tags.get_uid`, and per-series descriptive metadata through
:func:`cmr_dicom.metadata.extract_series_metadata` (read-only). No new
DICOM parse path is introduced here.

Failure semantics are fail-soft per file: a single bad file can never
raise. Non-DICOM files are skipped silently (but counted); files that
look like DICOM yet cannot be parsed yield a ``ScanError`` with
``reason="unreadable"``; files missing any of Study/Series/SOP UID yield
a ``ScanError`` with ``reason="missing_uid"``.

Ordering is fully deterministic: directory traversal sorts dir and file
names, studies sort by StudyInstanceUID, series sort by
SeriesInstanceUID, files sort by relative POSIX path, and errors sort by
path. Two scans of the same tree produce identical ``to_dict()`` output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from cmr_core.errors import CMRError

from .metadata import extract_series_metadata
from .reader import NOT_DICOM, classify_file, read_dicom_auto_with_digest
from .tags import get_uid, instance_uid_key, series_uid_key, study_uid_key

MISSING_UID = "missing_uid"
UNREADABLE = "unreadable"

#: Cap on the representative SOP list per series; the full file list is
#: always preserved in :attr:`SeriesEntry.files`.
MAX_REPRESENTATIVE_SOPS = 32


@dataclass(frozen=True)
class ScanError:
    """One per-file failure; the scan continues past it."""

    relative_path: str
    reason: str  # MISSING_UID | UNREADABLE
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "relative_path": self.relative_path,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SeriesEntry:
    """One DICOM series: sorted file list plus best-effort descriptors."""

    series_uid: str
    file_count: int
    files: tuple[str, ...] = field(default_factory=tuple)
    sop_instance_uids: tuple[str, ...] = field(default_factory=tuple)
    sop_list_truncated: bool = False
    series_description: str | None = None
    protocol_name: str | None = None
    modality: str | None = None

    def to_dict(self) -> dict:
        return {
            "series_uid": self.series_uid,
            "file_count": self.file_count,
            "files": list(self.files),
            "sop_instance_uids": list(self.sop_instance_uids),
            "sop_list_truncated": self.sop_list_truncated,
            "series_description": self.series_description,
            "protocol_name": self.protocol_name,
            "modality": self.modality,
        }


@dataclass(frozen=True)
class StudyEntry:
    """One DICOM study: its series sorted by SeriesInstanceUID."""

    study_uid: str
    series: tuple[SeriesEntry, ...] = field(default_factory=tuple)
    file_count: int = 0

    def to_dict(self) -> dict:
        return {
            "study_uid": self.study_uid,
            "file_count": self.file_count,
            "series": [s.to_dict() for s in self.series],
        }


@dataclass(frozen=True)
class LibraryScan:
    """Result of :func:`scan_library`: studies plus per-file errors."""

    studies: tuple[StudyEntry, ...] = field(default_factory=tuple)
    errors: tuple[ScanError, ...] = field(default_factory=tuple)
    files_scanned: int = 0
    files_skipped_non_dicom: int = 0

    def to_dict(self) -> dict:
        return {
            "studies": [s.to_dict() for s in self.studies],
            "errors": [e.to_dict() for e in self.errors],
            "files_scanned": self.files_scanned,
            "files_skipped_non_dicom": self.files_skipped_non_dicom,
        }


def scan_library(root: str | Path) -> LibraryScan:
    """Recursively scan ``root`` and group DICOM files Study -> Series.

    Pure function of the directory bytes: deterministic, read-only, and
    fail-soft per file. Raises :class:`FileNotFoundError` only when
    ``root`` itself is not a directory; a single bad file never raises.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"not a directory: {root_path}")

    # study_uid -> series_uid -> {"files": [rel], "sops": set, "meta": dict|None}
    grouped: dict[str, dict[str, dict]] = {}
    errors: list[ScanError] = []
    files_scanned = 0
    files_skipped = 0

    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        dirnames.sort()
        # Never descend into symlinked directories (no symlink follow).
        dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
        for name in sorted(filenames):
            full = Path(dirpath) / name
            # No symlink follow for files either; count and skip silently.
            if os.path.islink(full):
                files_skipped += 1
                continue
            if not full.is_file():
                continue
            files_scanned += 1
            rel = full.relative_to(root_path).as_posix()

            # Fast-path classification; non-DICOM skipped silently (counted).
            if classify_file(full) == NOT_DICOM:
                files_skipped += 1
                continue

            warnings: list = []
            ds, _digest = read_dicom_auto_with_digest(full, warnings)
            if ds is None:
                errors.append(
                    ScanError(
                        relative_path=rel,
                        reason=UNREADABLE,
                        detail="file classified as DICOM but header is unreadable",
                    )
                )
                continue
            try:
                study_uid = get_uid(ds, study_uid_key)
                series_uid = get_uid(ds, series_uid_key)
                sop_uid = get_uid(ds, instance_uid_key)
            except CMRError as exc:
                errors.append(ScanError(relative_path=rel, reason=MISSING_UID, detail=str(exc)))
                continue
            if not study_uid.strip() or not series_uid.strip() or not sop_uid.strip():
                errors.append(
                    ScanError(
                        relative_path=rel,
                        reason=MISSING_UID,
                        detail="blank Study/Series/SOP UID",
                    )
                )
                continue

            study = grouped.setdefault(study_uid, {})
            series = study.setdefault(series_uid, {"files": [], "sops": set(), "meta": None})
            series["files"].append(rel)
            series["sops"].add(sop_uid)
            if series["meta"] is None:
                # Best-effort descriptors from the first readable header
                # (traversal is sorted, so this is the smallest path);
                # None-tolerant, read-only.
                meta = extract_series_metadata(ds, [])
                series["meta"] = {
                    "series_description": meta.get("series_description"),
                    "protocol_name": meta.get("protocol_name"),
                    "modality": meta.get("modality"),
                }

    studies: list[StudyEntry] = []
    for study_uid in sorted(grouped):
        series_entries: list[SeriesEntry] = []
        study_files = 0
        for series_uid in sorted(grouped[study_uid]):
            bucket = grouped[study_uid][series_uid]
            files = sorted(bucket["files"])
            sops = sorted(bucket["sops"])
            rep = sops[:MAX_REPRESENTATIVE_SOPS]
            meta = bucket["meta"] or {}
            series_entries.append(
                SeriesEntry(
                    series_uid=series_uid,
                    file_count=len(files),
                    files=tuple(files),
                    sop_instance_uids=tuple(rep),
                    sop_list_truncated=len(sops) > len(rep),
                    series_description=meta.get("series_description"),
                    protocol_name=meta.get("protocol_name"),
                    modality=meta.get("modality"),
                )
            )
            study_files += len(files)
        studies.append(
            StudyEntry(
                study_uid=study_uid,
                series=tuple(series_entries),
                file_count=study_files,
            )
        )

    errors.sort(key=lambda e: e.relative_path)
    return LibraryScan(
        studies=tuple(studies),
        errors=tuple(errors),
        files_scanned=files_scanned,
        files_skipped_non_dicom=files_skipped,
    )
