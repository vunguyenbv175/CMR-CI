"""cmr_dicom — DICOM ingestion and cine reconstruction (pydicom-based)."""

from .classify import classify_series
from .discovery import (
    LibraryScan,
    ScanError,
    SeriesEntry,
    StudyEntry,
    scan_library,
)
from .metadata import extract_geometry, extract_series_metadata, extract_timing
from .pipeline import ORIENTATION_SPLIT, load_cine_volume
from .reader import is_dicom_file, iter_dicom_files, read_dicom
from .study import scan_directory

__all__ = [
    "LibraryScan",
    "ORIENTATION_SPLIT",
    "ScanError",
    "SeriesEntry",
    "StudyEntry",
    "classify_series",
    "extract_geometry",
    "extract_series_metadata",
    "extract_timing",
    "is_dicom_file",
    "iter_dicom_files",
    "load_cine_volume",
    "read_dicom",
    "scan_directory",
    "scan_library",
]
