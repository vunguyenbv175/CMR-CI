"""Provenance and versioning metadata (reproducibility contract).

Every AI result, contour and quantification attaches a
:class:`Provenance` so that any downstream value can be traced back to:

- input study/series references,
- software version,
- model name/version (when an AI result is involved),
- preprocessing version,
- creation time,
- review/approval state and revision history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

VERSION = "0.1.0"


@dataclass(frozen=True)
class ModelMetadata:
    """Identity of the model that produced an AI result."""

    name: str
    version: str
    label_map: dict[str, int] = field(default_factory=dict)
    preprocessing_version: str = "none"
    checksum: str | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.name}@{self.version}"


@dataclass
class Provenance:
    """Traceability record attached to results.

    ``model`` is None for manual results (physician tracing) and the dummy
    provider still records its own identity so provenance tests are
    meaningful without real models.

    ``workflow_id`` (S2C-001, additive trailing optional): the
    :class:`~cmr_core.source_manifest.DicomSourceManifest` content digest
    binding this result to one exact raw-DICOM source workflow. It stays
    ``None`` for legacy unbound records; canonical serialization includes it
    ONLY when non-``None`` so legacy payload bytes never change.
    """

    study_uid: str | None = None
    series_uid: str | None = None
    software_version: str = VERSION
    model: ModelMetadata | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    revision: int = 0
    workflow_id: str | None = None

    def bumped(self, revision: int | None = None) -> Provenance:
        """Return a copy with an incremented (or explicitly set) revision."""
        return Provenance(
            study_uid=self.study_uid,
            series_uid=self.series_uid,
            software_version=self.software_version,
            model=self.model,
            revision=self.revision + 1 if revision is None else revision,
            workflow_id=self.workflow_id,
        )
