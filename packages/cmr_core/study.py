"""Canonical study/series domain objects.

Independent from PyTorch, BAAI, pydicom and any AI library. ``CMRSeries``
carries DICOM UIDs, vendor metadata, geometry, ordered image frames and any
warnings collected during ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import CMRWarning
from .geometry import ImageGeometry

# Sequence classes recognized in Phase 3 of the master plan. Rule-based
# classification lives later; the enum is already the stable vocabulary.
#
# LGE Phase 1 (additive): "lge" is appended AFTER "unknown" so that the
# index/order of every pre-existing member is untouched (nothing may depend on
# position); membership tests are order-insensitive. The LGE Phase-1 classifier
# emits ONLY the canonical "lge" string — it never invents "lge_sax" /
# "lge_long_axis" (view is deliberately NOT part of the Phase-1 class).
SEQUENCE_CLASSES = (
    "cine_sax",
    "cine_2ch",
    "cine_3ch",
    "cine_4ch",
    "lge_sax",
    "lge_long_axis",
    "t1_map",
    "t2_map",
    "t2star",
    "flow_magnitude",
    "flow_phase",
    "perfusion",
    "localizer",
    "unknown",
    "lge",
)


@dataclass
class CMRSeries:
    """One DICOM series with enough metadata to reconstruct a cine volume.

    S2C-001 (additive trailing optionals): when the series was ingested from
    a manifest-backed workflow, ``study_uid``/``workflow_id``/
    ``cine_volume_digest`` carry the bound source identity. They stay ``None``
    for legacy unbound ingestion paths.
    """

    series_uid: str
    series_description: str = ""
    protocol_name: str = ""
    modality: str = "MR"
    manufacturer: str = ""
    manufacturer_model: str = ""
    magnetic_field_strength_t: float | None = None
    sequence_class: str = "unknown"
    sop_instance_uids: list[str] = field(default_factory=list)
    geometry: ImageGeometry | None = None
    warnings: list[CMRWarning] = field(default_factory=list)
    study_uid: str | None = None
    workflow_id: str | None = None
    cine_volume_digest: str | None = None

    def add_warning(self, warning: CMRWarning) -> None:
        self.warnings.append(warning)

    @property
    def is_timing_degraded(self) -> bool:
        return any(
            w.code == "timing.degraded_confidence"
            or (w.code == "ordering.ambiguous_frame" and "degraded-confidence" in w.message)
            for w in self.warnings
        )


@dataclass
class CMRStudy:
    """All series of one patient study. Minimal patient metadata by design."""

    study_uid: str
    patient_id: str = ""
    study_date: str = ""
    series: list[CMRSeries] = field(default_factory=list)

    def add_series(self, series: CMRSeries) -> None:
        self.series.append(series)

    def series_by_uid(self, series_uid: str) -> CMRSeries | None:
        for s in self.series:
            if s.series_uid == series_uid:
                return s
        return None

    @property
    def all_warnings(self) -> list[CMRWarning]:
        out: list[CMRWarning] = []
        for s in self.series:
            out.extend(s.warnings)
        return out
