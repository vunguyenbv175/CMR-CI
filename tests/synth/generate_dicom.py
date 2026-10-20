"""Synthetic DICOM generator for tests (no patient data anywhere).

Creates minimal-but-valid single-frame MR DICOM files with synthetic
circle images, real geometry and cardiac timing tags. Vendors and UIDs
are obviously synthetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian, MRImageStorage

from tests.synth.helpers import circle_mask


def make_instance(
    *,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    position_mm: tuple[float, float, float],
    radius_px: float,
    trigger_time_ms: float | None,
    temporal_position: int | None = None,
    instance_number: int | None = None,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    slice_thickness: float = 2.0,
    spacing_between_slices: float = 2.0,
    rows: int = 64,
    cols: int = 64,
    manufacturer: str = "SYNTH-MR",
    series_description: str = "cine SAX",
    protocol_name: str = "SYNTH-CINE",
    patient_id: str = "SYNTH-PATIENT",
    number_of_frames: int = 1,
    missing: tuple[str, ...] = (),
) -> FileDataset:
    """Build one synthetic DICOM instance (uncompressed, Explicit VR LE)."""

    def drop(ds: FileDataset, keyword: str) -> None:
        if hasattr(ds, keyword):
            delattr(ds, keyword)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = UID(sop_uid)
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("synthetic.dcm", {}, file_meta=meta, preamble=b"\0" * 128)

    ds.PatientName = f"SYNTH^{patient_id}"
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = sop_uid
    ds.SOPClassUID = MRImageStorage
    ds.Modality = "MR"
    ds.SeriesDescription = series_description
    ds.ProtocolName = protocol_name
    ds.Manufacturer = manufacturer
    ds.MagneticFieldStrength = 1.5
    ds.StudyDate = "20240101"

    ds.Rows = rows
    ds.Columns = cols
    ds.PixelSpacing = list(pixel_spacing)
    ds.SliceThickness = slice_thickness
    ds.SpacingBetweenSlices = spacing_between_slices
    ds.ImagePositionPatient = list(position_mm)
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.NumberOfFrames = number_of_frames
    if trigger_time_ms is not None:
        ds.TriggerTime = trigger_time_ms
    if temporal_position is not None:
        ds.TemporalPositionIdentifier = temporal_position
    if instance_number is not None:
        ds.InstanceNumber = instance_number

    if number_of_frames > 1:
        stack = np.stack(
            [
                circle_mask(rows, cols, cols / 2, rows / 2, radius_px)
                for _ in range(number_of_frames)
            ]
        )
        img = np.where(stack, 150, 5).astype("uint16")
    else:
        img = np.where(circle_mask(rows, cols, cols / 2, rows / 2, radius_px), 150, 5).astype(
            "uint16"
        )
    ds.PixelData = img.tobytes()
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    for keyword in missing:
        drop(ds, keyword)
    return ds


def write_synthetic_cine(
    directory: Path,
    *,
    n_slices: int = 4,
    n_frames: int = 6,
    radius_fn=None,
    shuffle: bool = True,
    series_uid: str = "1.2.826.0.1.3680043.10.999.1",
    study_uid: str = "1.2.826.0.1.3680043.10.999.100",
    trigger_times: list[float] | None = None,
    omit_trigger_time: bool = False,
    omit_temporal: bool = False,
    omit_instance_number: bool = False,
    gap_mm: float = 2.0,
) -> list[Path]:
    """Write a synthetic single-frame cine series to disk.

    Returns the list of written file paths (shuffled if ``shuffle``).
    Filenames are deliberately NOT ordered (e.g. ``img_03_02.dcm``
    encodes slice 3 / frame 2 out of order) to prove ordering is driven
    by geometry/timing metadata, not filenames.
    """
    radius_fn = radius_fn or (lambda s, f: 20.0)
    if trigger_times is None:
        trigger_times = [float(i * 120) for i in range(n_frames)]
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    counter = 0
    for s in range(n_slices):
        for f in range(n_frames):
            sop = f"1.2.826.0.1.3680043.10.999.50.{s * 100 + f}"
            ds = make_instance(
                study_uid=study_uid,
                series_uid=series_uid,
                sop_uid=sop,
                position_mm=(0.0, 0.0, float(s * gap_mm)),
                radius_px=radius_fn(s, f),
                trigger_time_ms=None if omit_trigger_time else trigger_times[f],
                temporal_position=None if omit_temporal else f + 1,
                instance_number=None if omit_instance_number else counter + 1,
                spacing_between_slices=gap_mm,
                slice_thickness=gap_mm,
            )
            # deliberately shuffled filename
            path = directory / f"img_{n_frames - f:02d}_{n_slices - s:02d}.dcm"
            ds.save_as(str(path), enforce_file_format=True)
            written.append(path)
            counter += 1
    if shuffle:
        import random

        rng = random.Random(42)
        rng.shuffle(written)
    return written
