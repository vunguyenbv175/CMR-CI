"""Low-level DICOM reading: scan a folder, tolerate non-DICOM files.

Uses pydicom. Never crashes on unrelated files — they are reported as
NOT_A_DICOM_FILE warnings (or ignored silently when clearly not DICOM).

Round 2 (ROUND2_INSTRUCTIONS 9): a DICOM preamble (128 zero bytes +
``DICM``) is the fast path, but it is NOT mandatory. Valid DICOM
datasets without a preamble (raw implicit/explicit VR files) are
detected by attempting a tolerant parse and verifying required DICOM
identity tags (SOPInstanceUID / StudyInstanceUID / SeriesInstanceUID /
PixelData). Files that cannot prove DICOM identity are classified as
NOT_A_DICOM_FILE — never guessed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from cmr_core.errors import NOT_A_DICOM_FILE, UNREADABLE_FILE, CMRWarning

from .tags import get_uid, instance_uid_key

if TYPE_CHECKING:
    from pydicom.dataset import Dataset

DICOM_PREAMBLE = "dicom_preamble"
DICOM_RAW = "dicom_raw"
NOT_DICOM = "not_dicom"


def is_dicom_file(path: Path) -> bool:
    """Cheap heuristic: pydicom preamble (128 zero bytes + 'DICM')."""
    try:
        with path.open("rb") as fh:
            header = fh.read(132)
        return len(header) == 132 and header[128:132] == b"DICM"
    except OSError:
        return False


def classify_file(path: Path) -> str:
    """Classify one file: 'dicom_preamble' | 'dicom_raw' | 'not_dicom'.

    ``dicom_raw`` means the file parses tolerantly as a DICOM dataset
    and carries at least one DICOM identity element, but has no Part-10
    preamble/DICM marker.
    """
    if is_dicom_file(path):
        return DICOM_PREAMBLE
    try:
        import pydicom

        ds = pydicom.dcmread(str(path), force=True)
    except Exception:  # noqa: BLE001 - anything else is not a usable DICOM file
        return NOT_DICOM
    return DICOM_RAW if _has_identity(ds) else NOT_DICOM


def iter_dicom_files(
    directory: Path,
    warnings: list[CMRWarning] | None = None,
) -> list[Path]:
    """Return sorted DICOM file paths under ``directory`` (non-recursive).

    Files are classified with :func:`classify_file`: preamble-less but
    valid DICOM files ARE included. Non-DICOM files are appended to
    ``warnings`` as NOT_A_DICOM_FILE.
    """
    out: list[Path] = []
    warnings = [] if warnings is None else warnings
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        kind = classify_file(entry)
        if kind in (DICOM_PREAMBLE, DICOM_RAW):
            out.append(entry)
        else:
            warnings.append(
                CMRWarning(
                    NOT_A_DICOM_FILE,
                    "ignoring file without recognizable DICOM content",
                    source=str(entry),
                )
            )
    return out


def read_dicom(
    path: Path,
    warnings: list[CMRWarning] | None = None,
    *,
    force: bool = False,
) -> Dataset | None:
    """Read one DICOM file, returning the pydicom Dataset or None on failure.

    Set ``force=True`` for files without a Part-10 preamble (tolerant
    parse — see :func:`classify_file`). A dataset that yields none of
    the essential identifiers is treated as unreadable. Failures are
    always reported as warnings — never swallowed silently.
    """
    import pydicom

    warnings = [] if warnings is None else warnings
    try:
        ds = pydicom.dcmread(str(path), force=force)
    except Exception as exc:  # noqa: BLE001 - pydicom raises many exception types
        warnings.append(
            CMRWarning(
                UNREADABLE_FILE,
                f"failed to read DICOM: {exc}",
                source=str(path),
            )
        )
        return None
    if not _has_identity(ds):
        warnings.append(
            CMRWarning(
                UNREADABLE_FILE,
                "file has a DICOM preamble but no usable DICOM content",
                source=str(path),
            )
        )
        return None
    return ds


def read_dicom_auto(
    path: Path,
    warnings: list[CMRWarning] | None = None,
) -> Dataset | None:
    """Read a DICOM file, selecting the tolerant parse when needed.

    Preamble files are read strictly; preamble-less files go through the
    tolerant (force) path used by :func:`classify_file`.
    """
    return read_dicom(path, warnings, force=not is_dicom_file(path))


def read_dicom_auto_with_digest(
    path: Path,
    warnings: list[CMRWarning] | None = None,
) -> tuple[Dataset | None, str | None]:
    """Read a DICOM file ONCE and hash the exact bytes that were parsed.

    S2C-001 cluster A (TOCTOU-free source identity): the raw bytes are read
    in a single ``read_bytes`` call; the returned digest is
    ``sha256(those exact bytes)`` and the dataset is parsed from an in-memory
    buffer over the same bytes — so a file replaced between hashing and
    parsing can never split identity across two reads. Failure semantics
    mirror :func:`read_dicom_auto` (warnings + ``(None, None)``).
    """
    import io

    import pydicom

    warnings = [] if warnings is None else warnings
    try:
        data = path.read_bytes()
    except OSError as exc:
        warnings.append(
            CMRWarning(
                UNREADABLE_FILE,
                f"failed to read file bytes: {exc}",
                source=str(path),
            )
        )
        return None, None
    file_sha256 = hashlib.sha256(data).hexdigest()
    has_preamble = len(data) >= 132 and data[128:132] == b"DICM"
    try:
        ds = pydicom.dcmread(io.BytesIO(data), force=not has_preamble)
    except Exception as exc:  # noqa: BLE001 - pydicom raises many exception types
        warnings.append(
            CMRWarning(
                UNREADABLE_FILE,
                f"failed to read DICOM: {exc}",
                source=str(path),
            )
        )
        return None, None
    if not _has_identity(ds):
        warnings.append(
            CMRWarning(
                UNREADABLE_FILE,
                "file has a DICOM preamble but no usable DICOM content",
                source=str(path),
            )
        )
        return None, None
    return ds, file_sha256


def _has_identity(ds) -> bool:
    for keyword in (
        "SOPInstanceUID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "PixelData",
    ):
        try:
            if getattr(ds, keyword, None) is not None:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def instance_key(ds) -> str:
    """Stable identity of one instance: SOPInstanceUID with fallback."""
    return get_uid(ds, instance_uid_key)
