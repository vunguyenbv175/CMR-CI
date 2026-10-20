"""FNC-G2 derivation-only references to sealed approved results.

An :class:`ApprovedResultReference` is a frozen, read-only view of one sealed
:class:`~cmr_core.domain_types.ApprovalSnapshot`. Deriving a reference creates
NO new seal and NO persistence: it restates the approval's own content identity
(``approval_id``, the canonical payload digest the store re-verifies on every
load) together with the binding pointers that identity covers — series, head
revision, selection, measurement, QC verdict/digest, inclusion set and
topology-disposition digest.

Resolution is fail-closed and always reads through
``store.load_approvals(series_uid)`` (whose loader re-verifies each stored
payload digest before it reconstructs a snapshot):

* an ``approval_id`` with no exact sealed row refuses ``resultref.unsealed``
  — fabricated or never-sealed references never resolve;
* the full-length content digest must match exactly, never a prefix or a
  substring of a sealed digest;
* every binding pointer must equal the sealed snapshot's value exactly: a
  stale or tampered head / selection / measurement / QC / inclusion /
  topology pointer refuses ``resultref.binding_mismatch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from cmr_core.domain_types import (
    ApprovalSnapshot,
    _validate_optional_hex64,
    canonical_digest,
)
from cmr_core.errors import (
    RESULT_REFERENCE_BINDING_MISMATCH,
    RESULT_REFERENCE_UNSEALED,
    CMRError,
)


class ApprovalLookup(Protocol):
    """The minimal read surface approved-result resolution needs.

    :class:`~cmr_persistence.store.SqliteCmrStore` satisfies it structurally;
    ``cmr_core`` imports no persistence module.
    """

    def load_approvals(self, series_uid: str) -> list[ApprovalSnapshot]: ...


def _require_hex64(name: str, value: Any) -> None:
    """Fail closed unless ``value`` is 64-character lowercase hex."""
    _validate_optional_hex64(name, value)
    if value is None:
        raise ValueError(f"{name} must be 64-character lowercase hex, got None")


@dataclass(frozen=True)
class ApprovedResultReference:
    """Derivation-only reference to one sealed approved result.

    Carries the approval's exact content identity (``approval_id``) plus the
    binding pointers that identity covers. Construction performs no store
    access; only :meth:`resolve` / :meth:`verify` consult sealed rows.
    """

    series_uid: str
    approval_id: str
    head_revision_id: str
    measurement_id: str
    qc_verdict: str
    selection_id: str | None = None
    qc_report_digest: str | None = None
    inclusion_set_id: str | None = None
    topology_disposition_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.series_uid, str) or not self.series_uid:
            raise ValueError("series_uid must be a non-empty string")
        _require_hex64("approval_id", self.approval_id)
        _require_hex64("head_revision_id", self.head_revision_id)
        _require_hex64("measurement_id", self.measurement_id)
        _validate_optional_hex64("selection_id", self.selection_id)
        _validate_optional_hex64("qc_report_digest", self.qc_report_digest)
        _validate_optional_hex64("inclusion_set_id", self.inclusion_set_id)
        _validate_optional_hex64(
            "topology_disposition_digest", self.topology_disposition_digest
        )
        if not isinstance(self.qc_verdict, str) or not self.qc_verdict:
            raise ValueError("qc_verdict must be a non-empty string")

    @classmethod
    def from_snapshot(cls, snapshot: ApprovalSnapshot) -> ApprovedResultReference:
        """Derive the reference view of an approval snapshot (no store read).

        The snapshot's content identity is re-checked against a fresh
        canonical digest before any pointer is copied, so an internally
        inconsistent snapshot can never seed a reference. Whether the
        snapshot was actually sealed is decided by :meth:`resolve` against
        ``store.load_approvals`` — this derivation seals nothing.
        """
        if (
            snapshot.approval_id != snapshot.digest
            or snapshot.approval_id != canonical_digest(snapshot.to_canonical())
        ):
            raise CMRError(
                RESULT_REFERENCE_UNSEALED,
                "approval snapshot content identity is inconsistent",
                source=snapshot.approval_id,
            )
        return cls(
            series_uid=snapshot.series_uid,
            approval_id=snapshot.approval_id,
            head_revision_id=snapshot.source_revision_id,
            measurement_id=snapshot.measurement_id,
            qc_verdict=snapshot.qc_verdict,
            selection_id=snapshot.selection_id,
            qc_report_digest=snapshot.qc_report_digest,
            inclusion_set_id=snapshot.inclusion_set_id,
            topology_disposition_digest=snapshot.topology_disposition_digest,
        )

    def matches(self, snapshot: ApprovalSnapshot) -> bool:
        """True iff every pointer equals ``snapshot``'s sealed binding."""
        return (
            self.approval_id == snapshot.approval_id
            and self.series_uid == snapshot.series_uid
            and self.head_revision_id == snapshot.source_revision_id
            and self.selection_id == snapshot.selection_id
            and self.measurement_id == snapshot.measurement_id
            and self.qc_verdict == snapshot.qc_verdict
            and self.qc_report_digest == snapshot.qc_report_digest
            and self.inclusion_set_id == snapshot.inclusion_set_id
            and self.topology_disposition_digest
            == snapshot.topology_disposition_digest
        )

    def resolve(self, store: ApprovalLookup) -> ApprovalSnapshot:
        """Resolve to the exact sealed snapshot this reference binds to.

        Reads ONLY through ``store.load_approvals`` (every stored payload is
        digest re-verified by that loader) and fails closed:

        * no sealed row carries the exact full ``approval_id`` →
          ``resultref.unsealed``;
        * a sealed row carries the id but any binding pointer disagrees
          (stale head or selection, tampered measurement / QC / inclusion /
          topology) → ``resultref.binding_mismatch``.
        """
        sealed = store.load_approvals(self.series_uid)
        exact = [
            snapshot for snapshot in sealed if snapshot.approval_id == self.approval_id
        ]
        if not exact:
            raise CMRError(
                RESULT_REFERENCE_UNSEALED,
                "approved result reference does not match any sealed approval",
                source=self.approval_id,
            )
        for snapshot in exact:
            if self.matches(snapshot):
                return snapshot
        raise CMRError(
            RESULT_REFERENCE_BINDING_MISMATCH,
            "approved result reference binding disagrees with the sealed approval",
            source=self.approval_id,
        )

    def verify(self, store: ApprovalLookup) -> ApprovedResultReference:
        """Fail-closed store check: returns ``self`` only when it resolves exactly."""
        self.resolve(store)
        return self
