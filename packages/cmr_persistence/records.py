"""Persistence-local decision record type (milestone N7.1).

``QCDecisionRecord`` is the durable artifact for B-LITE sentinel approval
decisions (physician overrides of ``REVIEW REQUIRED`` states where Node 5
legitimately produced no :class:`~cmr_core.domain_types.Measurement`). It
lives HERE, in ``cmr_persistence``, deliberately NOT in
``packages/cmr_core``: the Node 1 domain types are frozen public surface and
must stay byte-identical; this persistence-local record keeps them untouched
while giving the store a first-class, content-addressed record to persist
into the separate append-only ``qc_decision_records`` table (schema v2).

The immutable clinical approval path is unchanged: sealed
:class:`~cmr_core.domain_types.ApprovalSnapshot` objects keep flowing into
``approval_snapshots`` with their foreign key to ``measurements`` intact.
A sentinel decision carries NO numeric clinical metric, so it never touches
that table; it anchors to persisted contour-revision history instead
(``source_revision_id`` -> ``contour_revisions.revision_id``).

Content addressing follows the store convention exactly:
``decision_id == digest == sha256(canonical_json(record))``, where
``canonical_json`` is :func:`cmr_core.domain_types.canonical_json`. Writers
assert the identity before every insert and loaders re-verify it
byte-for-byte after reconstruction; any drift fails closed.

Local duplications (deliberate, documented): the UTC-timestamp check, the
provenance freeze proxy and the provenance rebuilder below mirror private
``cmr_core`` helpers that this package must not import. The QC verdict
vocabulary duplicates the single source tuple in
``cmr_qc.report.QCReport.verdict`` output ("PASS" / "REVIEW REQUIRED") and
the table-level CHECK constraint in ``store.py`` enforces the same set at
the schema level.

Deep immutability (amendment B1): the frozen provenance proxy is a STABLE
module-level class (dataclass ``__eq__`` compares ``other.__class__ is
self.__class__``, so a fresh local subclass per construction would break
value equality of canonically identical records), and a supplied
``Provenance.model`` is DEEP-frozen — its mutable ``label_map`` dict is
re-wrapped as ``MappingProxyType`` over a copy, so mutating the caller's
model after construction can neither change ``canonical_json(record)`` nor
invalidate ``decision_id``.

Fail-closed sentinel semantics (amendment B2): a blank/whitespace-only
``override_reason`` and any ``algorithm_version`` other than
:data:`QC_DECISION_ALGORITHM_VERSION` raise ``ValueError`` at construction —
arbitrary algorithm identities can never be persisted as sentinel decisions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, cast

from cmr_core.domain_types import canonical_digest, canonical_json, head_tombstoned_keys
from cmr_core.provenance import ModelMetadata, Provenance
from cmr_core.segmentation import Structure

__all__ = [
    "CINE_WORKFLOW_REGISTRATION_RECORD_SCHEMA",
    "CineWorkflowRegistrationV1",
    "QC_DECISION_ALGORITHM_VERSION",
    "QC_VERDICTS",
    "QCDecisionRecord",
    "QC_TOPOLOGY_ACQUISITION_OMISSION",
    "QC_TOPOLOGY_MISSING_DATA",
    "QC_TOPOLOGY_PHYSICIAN_EXCLUDED",
    "QC_TOPOLOGY_UNRESOLVED",
    "SelectedPhaseAnalysisRecord",
    "SliceInclusionEntry",
    "SliceInclusionFrameRole",
    "SliceInclusionSet",
    "SliceInclusionStructure",
    "TopologyDisposition",
    "TopologyDispositionRecord",
    "WholeCinePredictionRow",
    "WholeCineRevisionRow",
    "ModelRunRow",
    "ModelBundleRecord",
    "ModelActivationRecord",
    "RegistryState",
    "SelectedPhasePredictionRow",
    "analysis_record_from_payload",
    "chain_key_authoring_times",
    "qc_decision_from_payload",
    "slice_inclusion_set_from_payload",
    "topology_disposition_from_payload",
    "topology_disposition_set_digest",
    "applicable_topology_dispositions",
]

#: Forced algorithm identity on B-LITE decision records (never a real Node 5
#: algorithm version). Single source of truth for the record type;
#: ``cmr_workflow.controller.SENTINEL_ALGORITHM_VERSION`` aliases this value.
QC_DECISION_ALGORITHM_VERSION = "metrics-unavailable-due-to-qc"

#: Repository QC verdict vocabulary, exactly as produced by
#: ``cmr_qc.report.QCReport.verdict``.
QC_VERDICTS = ("PASS", "REVIEW REQUIRED")


# --- S3-004E schema v7 cine-workflow registration authority --------------------

#: Exact frozen ``schema`` literal of the canonical registration-record bytes
#: (design spec §4.1 step (9); the frozen sample is committed at
#: ``tests/ai/fixtures/cine_registration_record_schema_v1.json``).
CINE_WORKFLOW_REGISTRATION_RECORD_SCHEMA = "cine-workflow-registration-record-v1"

#: The store-issued membership MARKER (correction6 / R3 finding 1). It is set
#: by the guarded ``CineWorkflowRegistrationV1.__init__`` ONLY — never by a
#: caller, never by ``object.__setattr__`` outside that path — so the kernel
#: (which holds no store handle) can prove the authority in its hand was
#: issued by ``read_registration``. The value is a module-private sentinel
#: object: unguessable by value, so a from-scratch forgery can never carry it.
_STORE_ISSUED_MARKER = object()

class _RegistrationIssuanceToken:
    """Module-private issuance token (correction6 / R3 finding 1).

    The frozen spec makes ``SqliteCmrStore.read_registration`` the EXACTLY ONE
    construction path of :class:`CineWorkflowRegistrationV1` (spec:598-601).
    The dataclass' own ``__post_init__`` cannot prove that on its own: a
    caller supplying coherent arguments — including the fully coherent
    UNPERSISTED forgery, where ``store_record_bytes`` ==
    ``persisted_record_bytes`` and every digest recomputes — satisfies every
    field-level check while never having been persisted. Provenance cannot be
    inferred from the seven values, so it is carried by this token: the public
    constructor requires the single module-private singleton, which only
    ``read_registration`` holds.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "<_RegistrationIssuanceToken>"

_REGISTRATION_ISSUANCE_TOKEN = _RegistrationIssuanceToken()

def registration_issuance_token() -> _RegistrationIssuanceToken:
    """Return the module-private issuance token for the D8 store read API.

    ``cmr_persistence.store.SqliteCmrStore.read_registration`` is the ONLY
    caller: it constructs the authority FROM the persisted row. Nothing else
    in the repository may obtain the token, so nothing else can construct a
    usable authority (see the guarded ``CineWorkflowRegistrationV1.__init__``).
    """
    return _REGISTRATION_ISSUANCE_TOKEN

@dataclass(frozen=True)
class CineWorkflowRegistrationV1:
    """The store-issued cine-workflow registration AUTHORITY (S3-004E D8).

    EXACTLY seven primitives-only fields (R26-01 — no others):

    ``workflow_id``
        hex64 content digest of the bound ``DicomSourceManifest`` (the
        store's enforced domain, ``store.py`` CHECK(length=64 AND hex)).
    ``manifest_digest``
        hex64 canonical digest of that manifest (``source_manifest.py:179``:
        ``workflow_id == canonical_digest(to_canonical())``).
    ``cine_volume_digest``
        hex64 decoded-volume digest carried by the manifest.
    ``sequence``
        the durable store's monotonic registration revision for that
        ``workflow_id`` (allocated inside ``BEGIN IMMEDIATE``).
    ``store_record_bytes``
        the canonical registration-record bytes AS PARSED from the persisted
        row (frozen schema, canonical JSON + LF).
    ``store_record_sha256``
        hex64 sha256 over ``store_record_bytes``.
    ``persisted_record_bytes``
        the EXACT persisted row bytes read by the D8 store API — the
        store-issued membership evidence that defeats a fully coherent
        UNPERSISTED forgery.

    There is EXACTLY ONE construction path: :meth:`SqliteCmrStore.
    read_registration` builds this object FROM the persisted row. No other
    code path may construct a ``CineWorkflowRegistrationV1`` — that path is
    now ENFORCED by :meth:`__init__` (correction6 / R3 finding 1): the frozen
    seven fields, their names, types and order are unchanged, but the
    constructor additionally requires the module-private issuance token that
    only ``read_registration`` can obtain. A direct public construction
    therefore raises, and the object it returns carries no issuance marker, so
    it is rejected at the kernel seam even when a caller bypasses the guard
    with ``object.__new__`` + ``object.__setattr__``.
    """

    workflow_id: str
    manifest_digest: str
    cine_volume_digest: str
    sequence: int
    store_record_bytes: bytes
    store_record_sha256: str
    persisted_record_bytes: bytes

    def __init__(
        self,
        workflow_id: str,
        manifest_digest: str,
        cine_volume_digest: str,
        sequence: int,
        store_record_bytes: bytes,
        store_record_sha256: str,
        persisted_record_bytes: bytes,
        *,
        _issuance_token: _RegistrationIssuanceToken | None = None,
    ) -> None:
        """Guarded constructor: the frozen seven fields plus the issuance proof.

        ``_issuance_token`` is the SINGLE construction path's proof. It is
        keyword-only and defaults to ``None``, so every legacy/direct
        positional or keyword construction attempt that does not carry the
        token is rejected here. The seven data fields are unchanged: their
        names, types, order and the frozen-dataclass semantics (value
        equality, hashing, ``dataclasses.fields()``) are EXACTLY as before.
        """
        if _issuance_token is not _REGISTRATION_ISSUANCE_TOKEN:
            raise PermissionError(
                "CineWorkflowRegistrationV1 has exactly one construction path: "
                "SqliteCmrStore.read_registration (frozen spec §4.1 / R26-01); "
                "a directly constructed authority is never store-issued"
            )
        object.__setattr__(self, "workflow_id", workflow_id)
        object.__setattr__(self, "manifest_digest", manifest_digest)
        object.__setattr__(self, "cine_volume_digest", cine_volume_digest)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "store_record_bytes", store_record_bytes)
        object.__setattr__(self, "store_record_sha256", store_record_sha256)
        object.__setattr__(self, "persisted_record_bytes", persisted_record_bytes)
        # The store-issued membership marker is set ONLY here: an object built
        # from scratch with ``object.__new__`` + ``object.__setattr__`` (the
        # corrupted-host / pickle-replay mutation vector) lacks it and is
        # rejected at the kernel seam (sax_kernel step (9)).
        object.__setattr__(self, "_store_issued", _STORE_ISSUED_MARKER)
        self.__post_init__()

    def __post_init__(self) -> None:
        _require_digest("workflow_id", self.workflow_id)
        _require_digest("manifest_digest", self.manifest_digest)
        _require_digest("cine_volume_digest", self.cine_volume_digest)
        _require_digest("store_record_sha256", self.store_record_sha256)
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise ValueError("sequence must be a positive integer")
        for name in ("store_record_bytes", "persisted_record_bytes"):
            value = getattr(self, name)
            if not isinstance(value, bytes) or not value:
                raise ValueError(f"{name} must be non-empty bytes")
        # The authority is self-consistent BY CONSTRUCTION: the persisted row
        # bytes ARE the membership proof, so a caller passing diverging byte
        # strings is rejected here rather than at kernel step (9).
        if self.store_record_bytes != self.persisted_record_bytes:
            raise ValueError(
                "store_record_bytes must equal persisted_record_bytes; the "
                "persisted row is the single registration membership proof"
            )
        if self.store_record_sha256 != hashlib.sha256(self.store_record_bytes).hexdigest():
            raise ValueError("store_record_sha256 does not match store_record_bytes")

def _store_issued_proof(authority: object) -> bool:
    """True IFF ``authority`` was built by the single store issuance path.

    The kernel holds NO store handle, so the issuance proof must travel with
    the object: the guarded ``__init__`` is the only writer of the marker.
    A direct construction is blocked outright; a from-scratch
    ``object.__new__`` + ``object.__setattr__`` forgery (or a pickle replay)
    simply never carries the marker, so it is detected here. Reading via
    ``object.__getattribute__`` keeps the check out of ``__dict__`` and off
    the seven-field surface.
    """
    if not isinstance(authority, CineWorkflowRegistrationV1):
        return False
    try:
        return object.__getattribute__(authority, "_store_issued") is _STORE_ISSUED_MARKER
    except AttributeError:
        return False


def _require_utc_timestamp(value: str) -> None:
    """Mirror of the private Node 1 UTC check (see module docstring)."""
    if not isinstance(value, str):
        raise ValueError(f"timestamp must be ISO-8601 UTC: {value!r}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"timestamp must be ISO-8601 UTC: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"timestamp must be ISO-8601 UTC: {value!r}")


class _LockedProvenance(Provenance):
    """STABLE module-level locked provenance proxy (amendment B1).

    ``Provenance`` is a legacy MUTABLE dataclass; without a lock a caller
    could mutate ``record.provenance.software_version`` in place and corrupt
    the content-addressed identity. The class MUST live at module level:
    dataclass-generated ``__eq__`` requires ``other.__class__ is
    self.__class__``, so a fresh local subclass per construction would make
    two canonically identical records compare UNEQUAL. One shared subclass
    keeps value equality intact while every attribute write after the
    constructor finishes fails closed.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise TypeError("immutable domain record")
        super().__setattr__(name, value)


def _freeze_model(model: ModelMetadata | None) -> ModelMetadata | None:
    """Deep-freeze a model snapshot: copy-on-wrap its mutable ``label_map``.

    ``ModelMetadata`` itself is frozen, but its ``label_map`` dict is a
    mutable container shared with the caller. Re-wrapping a COPY as
    ``MappingProxyType`` guarantees that post-construction mutation of the
    caller's mapping can neither change ``canonical_json(record)`` nor
    invalidate ``decision_id``.

    The ``cast`` documents that ``ModelMetadata.label_map`` is DECLARED as
    ``dict`` while holding a read-only mapping here: every consumer in this
    package and in ``cmr_core._canonical`` uses only the ``Mapping``
    protocol (iteration/lookup/equality), never dict-only mutation, so the
    wrapper satisfies the actual runtime contract.
    """
    if model is None:
        return None
    return ModelMetadata(
        name=model.name,
        version=model.version,
        # Runtime-safe: MappingProxyType implements the full Mapping protocol
        # that every reader of this field uses; only writers would care.
        label_map=cast("dict[str, int]", MappingProxyType(dict(model.label_map))),
        preprocessing_version=model.preprocessing_version,
        checksum=model.checksum,
    )


def _freeze_provenance(provenance: Provenance | None) -> Provenance | None:
    """Persistence-local clone of the private Node 1 provenance freeze.

    Clones the supplied (possibly caller-mutable) provenance into the locked
    :class:`_LockedProvenance` proxy and deep-freezes any attached model via
    :func:`_freeze_model`, so the record's canonical bytes are immune to all
    post-construction mutations of the caller's objects. S2C-001: the
    additive ``workflow_id`` binding rides along.
    """
    if provenance is None:
        return None
    clone = _LockedProvenance(
        provenance.study_uid,
        provenance.series_uid,
        provenance.software_version,
        _freeze_model(provenance.model),
        provenance.created_at,
        provenance.revision,
        provenance.workflow_id,
    )
    object.__setattr__(clone, "_locked", True)
    return clone


@dataclass(frozen=True)
class QCDecisionRecord:
    """One immutable, content-addressed B-LITE sentinel approval decision.

    Carries no numeric clinical metric by design: this record exists precisely
    for decisions where Node 5 legitimately produced no Measurement. Fields
    mirror the ``qc_decision_records`` columns (schema v2); ``digest`` and
    ``decision_id`` are derived in ``__post_init__`` per the store's
    content-addressing convention and are not constructor arguments.
    """

    series_uid: str
    source_revision_id: str
    ed_frame: int
    es_frame: int
    qc_verdict: str
    approved_by: str
    approved_at: str
    override_reason: str
    provenance: Provenance
    software_version: str
    algorithm_version: str = QC_DECISION_ALGORITHM_VERSION
    selection_id: str | None = None
    decision_id: str = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.ed_frame < 0 or self.es_frame < 0:
            raise ValueError("ED/ES frame indices must be non-negative")
        if self.qc_verdict not in QC_VERDICTS:
            raise ValueError(f"qc_verdict must be one of {', '.join(QC_VERDICTS)}")
        # S2C-001 additive binding (cluster B): a sentinel decision may carry
        # the explicit EDESSelection content id; populated values must be
        # 64-character lowercase hex, absent values stay omitted from the
        # canonical payload so legacy decision bytes never change.
        if self.selection_id is not None and (
            not isinstance(self.selection_id, str)
            or len(self.selection_id) != 64
            or any(c not in "0123456789abcdef" for c in self.selection_id)
        ):
            raise ValueError(
                "selection_id must be 64-character lowercase hex when populated, "
                f"got {self.selection_id!r}"
            )
        _require_utc_timestamp(self.approved_at)
        # Amendment B2: sentinel semantics fail closed. A decision record
        # without a documented physician rationale is not sealable, and no
        # algorithm identity other than the forced sentinel value may pass
        # through (the constructor default is the ONLY sanctioned value).
        if not isinstance(self.override_reason, str) or not self.override_reason.strip():
            raise ValueError(
                "override_reason must be a non-empty string documenting the "
                f"physician's decision, got {self.override_reason!r}"
            )
        if self.algorithm_version != QC_DECISION_ALGORITHM_VERSION:
            raise ValueError(
                "algorithm_version on a QCDecisionRecord must be the sentinel "
                f"identity {QC_DECISION_ALGORITHM_VERSION!r}; got "
                f"{self.algorithm_version!r} — arbitrary algorithm identities "
                "cannot be persisted as sentinel decisions"
            )
        if self.provenance.series_uid not in (None, self.series_uid):
            raise ValueError("provenance series_uid does not match record")
        _require_utc_timestamp(self.provenance.created_at)
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        payload = self.to_canonical()
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "decision_id", digest)

    def to_canonical(self) -> dict[str, Any]:
        payload = {
            "domain": "QCDecisionRecord",
            "series_uid": self.series_uid,
            "source_revision_id": self.source_revision_id,
            "ed_frame": self.ed_frame,
            "es_frame": self.es_frame,
            "qc_verdict": self.qc_verdict,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "override_reason": self.override_reason,
            "provenance": self.provenance,
            "software_version": self.software_version,
            "algorithm_version": self.algorithm_version,
        }
        if self.selection_id is not None:
            payload["selection_id"] = self.selection_id
        return payload


def qc_decision_from_payload(data: dict[str, Any]) -> QCDecisionRecord:
    """Rebuild a :class:`QCDecisionRecord` from its parsed canonical payload.

    Parsed values pass through VERBATIM into the constructor — no
    ``float()``/``int()`` coercion anywhere — so the store's byte-for-byte
    re-verification after reconstruction can only pass for payloads this
    record type actually produced.
    """

    def model_from_payload(value: dict[str, Any] | None) -> ModelMetadata | None:
        if value is None:
            return None
        return ModelMetadata(
            name=value["name"],
            version=value["version"],
            label_map=dict(value["label_map"]),
            preprocessing_version=value["preprocessing_version"],
            checksum=value["checksum"],
        )

    raw_provenance = data["provenance"]
    provenance = Provenance(
        study_uid=raw_provenance["study_uid"],
        series_uid=raw_provenance["series_uid"],
        software_version=raw_provenance["software_version"],
        model=model_from_payload(raw_provenance["model"]),
        created_at=raw_provenance["created_at"],
        revision=raw_provenance["revision"],
        workflow_id=raw_provenance.get("workflow_id"),
    )
    return QCDecisionRecord(
        series_uid=data["series_uid"],
        source_revision_id=data["source_revision_id"],
        ed_frame=data["ed_frame"],
        es_frame=data["es_frame"],
        qc_verdict=data["qc_verdict"],
        approved_by=data["approved_by"],
        approved_at=data["approved_at"],
        override_reason=data["override_reason"],
        provenance=provenance,
        software_version=data["software_version"],
        algorithm_version=data["algorithm_version"],
        selection_id=data.get("selection_id"),
    )


# --- S2C-001 cluster G: schema-ready basal-slice inclusion sets -----------------

#: Canonical RANKS for entry ordering (CFM-001 correction F8): role_order is
#: PRIMARY with ED < ES, then structure_order lv_cavity < rv_cavity <
#: lv_myocardium. Lexical string order does NOT match this frozen order.
_ROLE_RANK: dict[SliceInclusionFrameRole, int] = {}
_STRUCTURE_RANK: dict[SliceInclusionStructure, int] = {}


def _entry_rank(entry: SliceInclusionEntry) -> tuple[int, int, int]:
    """Canonical sort key ``(slice_index, role_rank, structure_rank)``."""
    return (
        entry.slice_index,
        _ROLE_RANK[entry.frame_role],
        _STRUCTURE_RANK[entry.structure],
    )


#: Frame roles of one inclusion entry (ED / ES phase authorship is independent).
class SliceInclusionFrameRole(Enum):
    """Cardiac phase a basal-inclusion decision applies to."""

    ED = "ED"
    ES = "ES"


#: Structures whose slice inclusion can be authored independently.
class SliceInclusionStructure(Enum):
    """Anatomical structure an inclusion entry speaks about."""

    LV_CAVITY = "lv_cavity"
    RV_CAVITY = "rv_cavity"
    LV_MYOCARDIUM = "lv_myocardium"


_ROLE_RANK[SliceInclusionFrameRole.ED] = 0
_ROLE_RANK[SliceInclusionFrameRole.ES] = 1
_STRUCTURE_RANK[SliceInclusionStructure.LV_CAVITY] = 0
_STRUCTURE_RANK[SliceInclusionStructure.RV_CAVITY] = 1
_STRUCTURE_RANK[SliceInclusionStructure.LV_MYOCARDIUM] = 2


@dataclass(frozen=True)
class SliceInclusionEntry:
    """One immutable ``(slice, role, structure) -> included`` decision.

    Part of the append-only :class:`SliceInclusionSet` (S2C-001 cluster G):
    SCHEMA-READY but deliberately UNWIRED — quantification continues the
    disclosed legacy all-slices behavior until CMR-01 defines completeness
    rules, defaults and metric effects.

    Correction round 1 P2: ``frame_index`` is an ADDITIVE trailing optional
    (omitted from the canonical payload when None so every spec-shaped set
    keeps its identity bytes). When populated it names the exact ED/ES frame
    the decision speaks about and is verified against the referenced
    selection at save time (wrong frames fail closed).
    """

    slice_index: int
    frame_role: SliceInclusionFrameRole
    structure: SliceInclusionStructure
    included: bool
    frame_index: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.slice_index, bool) or not isinstance(self.slice_index, int):
            raise ValueError(f"slice_index must be an integer, got {self.slice_index!r}")
        if self.slice_index < 0:
            raise ValueError(f"slice_index must be non-negative, got {self.slice_index}")
        if not isinstance(self.frame_role, SliceInclusionFrameRole):
            raise ValueError(
                f"frame_role must be a SliceInclusionFrameRole member, got {self.frame_role!r}"
            )
        if not isinstance(self.structure, SliceInclusionStructure):
            raise ValueError(
                f"structure must be a SliceInclusionStructure member, got {self.structure!r}"
            )
        if not isinstance(self.included, bool):
            raise ValueError(f"included must be a bool, got {self.included!r}")
        if self.frame_index is not None and (
            isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int)
        ):
            raise ValueError(f"frame_index must be an integer or None, got {self.frame_index!r}")
        if self.frame_index is not None and self.frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {self.frame_index}")

    def to_canonical(self) -> dict[str, Any]:
        payload = {
            "slice_index": self.slice_index,
            "frame_role": self.frame_role.value,
            "structure": self.structure.value,
            "included": self.included,
        }
        if self.frame_index is not None:
            payload["frame_index"] = self.frame_index
        return payload

    @property
    def identity_key(self) -> tuple[int, str, str]:
        """Duplicate-detection key ``(slice_index, role, structure)``."""
        return (
            self.slice_index,
            self.frame_role.value,
            self.structure.value,
        )


def _require_hex64(name: str, value: str | None) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64-character lowercase hex when populated, got {value!r}")


@dataclass(frozen=True)
class SliceInclusionSet:
    """One atomically content-addressed basal-slice inclusion set (cluster G).

    Deliberately a SEPARATE append-only record instead of a mutable boolean on
    :class:`~cmr_core.domain_types.EDESSelection`: phase selection and basal
    inclusion have independent authorship/revision lifecycles, inclusion may
    differ per ED/ES role and per structure, and only a complete set is
    honestly content-addressable. ``inclusion_set_id`` is derived as
    ``canonical_digest(to_canonical())``. Entries are stored as a sorted,
    duplicate-free immutable tuple keyed by
    ``(slice_index, frame_role, structure)``.
    """

    series_uid: str
    source_revision_id: str
    selection_id: str
    entries: tuple[SliceInclusionEntry, ...]
    selected_by: str
    selected_at: str
    provenance: Provenance
    workflow_id: str | None = None
    inclusion_set_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_hex64("workflow_id", self.workflow_id)
        _require_hex64("source_revision_id", self.source_revision_id)
        _require_hex64("selection_id", self.selection_id)
        if not isinstance(self.series_uid, str) or not self.series_uid.strip():
            raise ValueError("series_uid must be a non-empty string")
        if not isinstance(self.selected_by, str) or not self.selected_by.strip():
            raise ValueError("selected_by must be a non-empty string")
        _require_utc_timestamp(self.selected_at)
        if self.provenance.series_uid not in (None, self.series_uid):
            raise ValueError("provenance series_uid does not match record")
        _require_utc_timestamp(self.provenance.created_at)
        if isinstance(self.entries, list):
            object.__setattr__(self, "entries", tuple(self.entries))
        seen: dict[tuple[int, str, str], SliceInclusionEntry] = {}
        for entry in self.entries:
            if not isinstance(entry, SliceInclusionEntry):
                raise ValueError(
                    f"entries must be SliceInclusionEntry instances, got {type(entry).__name__}"
                )
            key = entry.identity_key
            if key in seen:
                raise ValueError(
                    "duplicate inclusion key "
                    f"(slice_index={key[0]}, frame_role={key[1]}, "
                    f"structure={key[2]}); one set carries exactly one decision "
                    "per key"
                )
            seen[key] = entry
        # CFM-001 correction F8: canonical rank order (role primary), NOT
        # lexical string order — this participates in inclusion_set_id.
        ordered = tuple(sorted(self.entries, key=_entry_rank))
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(self, "inclusion_set_id", canonical_digest(self.to_canonical()))

    def to_canonical(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "domain": "SliceInclusionSet",
            "series_uid": self.series_uid,
            "source_revision_id": self.source_revision_id,
            "selection_id": self.selection_id,
            "entries": [entry.to_canonical() for entry in self.entries],
            "selected_by": self.selected_by,
            "selected_at": self.selected_at,
            "provenance": self.provenance,
        }
        if self.workflow_id is not None:
            payload["workflow_id"] = self.workflow_id
        return payload


def slice_inclusion_set_from_payload(data: dict[str, Any]) -> SliceInclusionSet:
    """Rebuild a :class:`SliceInclusionSet` from its parsed canonical payload."""

    raw_provenance = data["provenance"]
    provenance = Provenance(
        study_uid=raw_provenance["study_uid"],
        series_uid=raw_provenance["series_uid"],
        software_version=raw_provenance["software_version"],
        model=None
        if raw_provenance["model"] is None
        else ModelMetadata(
            name=raw_provenance["model"]["name"],
            version=raw_provenance["model"]["version"],
            label_map=dict(raw_provenance["model"]["label_map"]),
            preprocessing_version=raw_provenance["model"]["preprocessing_version"],
            checksum=raw_provenance["model"]["checksum"],
        ),
        created_at=raw_provenance["created_at"],
        revision=raw_provenance["revision"],
        workflow_id=raw_provenance.get("workflow_id"),
    )
    return SliceInclusionSet(
        series_uid=data["series_uid"],
        source_revision_id=data["source_revision_id"],
        selection_id=data["selection_id"],
        entries=tuple(
            SliceInclusionEntry(
                slice_index=entry["slice_index"],
                frame_role=SliceInclusionFrameRole(entry["frame_role"]),
                structure=SliceInclusionStructure(entry["structure"]),
                included=entry["included"],
                frame_index=entry.get("frame_index"),
            )
            for entry in data["entries"]
        ),
        selected_by=data["selected_by"],
        selected_at=data["selected_at"],
        provenance=provenance,
        workflow_id=data.get("workflow_id"),
    )


# --- Topology disposition records (Astra F1-F5, 2026-09-09) ---------------------
#
# One durable, content-addressed physician decision about ONE non-projectable
# topology block key ``(series_uid, slice_index, frame_index, structure)``.
# It answers the audit question the durable block alone cannot: WHO resolved
# this unsupported-topology key, HOW (``REPLACED`` by physician-authoritative
# geometry in the chain head, or ``EXCLUDED`` from quantification), WHY, and
# AGAINST WHICH EXACT clinical state (the current contour-chain head plus the
# EDESSelection identity). The design language mirrors
# :class:`SliceInclusionSet`: frozen dataclass, canonical payload,
# ``canonical_digest(to_canonical())`` identity, deep-frozen provenance,
# append-only storage, and NO quantitative clinical content.
#
# A disposition never supplies geometry and never zero-fills: ``EXCLUDED``
# means the key is intentionally absent from quantification, ``REPLACED``
# means the exact current chain head carries physician geometry for it. A
# disposition whose bound head or selection no longer matches the CURRENT
# head/selection is STALE and is never a resolution (F5).


class TopologyDisposition(Enum):
    """How one blocked topology key was resolved for the selected metrics.

    ``REPLACED``
        The CURRENT revision-chain head carries physician-authoritative
        geometry for the exact key (an edit superseded the blocked state).
    ``EXCLUDED``
        The physician explicitly excluded the key from quantification; the
        key stays absent from the clinical index and metrics.
    """

    REPLACED = "REPLACED"
    EXCLUDED = "EXCLUDED"


#: QC distinction vocabulary for the approval-blocking path (contract item 7).
QC_TOPOLOGY_UNRESOLVED = "topology.unresolved"
QC_TOPOLOGY_PHYSICIAN_EXCLUDED = "topology.physician_excluded"
QC_TOPOLOGY_MISSING_DATA = "topology.missing_data"
QC_TOPOLOGY_ACQUISITION_OMISSION = "topology.acquisition_omission"


@dataclass(frozen=True)
class TopologyDispositionRecord:
    """One immutable, content-addressed physician topology disposition.

    Field set (Astra MANAGER IMPLEMENTATION CONTRACT item 4): the exact block
    key (``series_uid``, ``slice_index``, ``frame_index``, ``structure``),
    the source ``block_id`` it resolves, the CURRENT chain head and selection
    the disposition is bound to, the frame role the key speaks about, the
    :class:`TopologyDisposition` verdict, the authoring physician, the
    creation stamp, a NON-EMPTY reason, provenance, and the derived
    content-addressed ``disposition_id``.
    """

    series_uid: str
    slice_index: int
    frame_index: int
    structure: str
    source_block_id: str
    source_revision_id: str
    selection_id: str
    frame_role: SliceInclusionFrameRole
    disposition: TopologyDisposition
    physician: str
    created_at: str
    reason: str
    provenance: Provenance
    workflow_id: str | None = None
    disposition_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.series_uid, str) or not self.series_uid.strip():
            raise ValueError("series_uid must be a non-empty string")
        for name in ("slice_index", "frame_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value!r}")
        if not isinstance(self.structure, str) or not self.structure.strip():
            raise ValueError("structure must be a non-empty string")
        _require_hex64("source_block_id", self.source_block_id)
        _require_hex64("source_revision_id", self.source_revision_id)
        _require_hex64("selection_id", self.selection_id)
        _require_hex64("workflow_id", self.workflow_id)
        if not isinstance(self.frame_role, SliceInclusionFrameRole):
            raise ValueError(
                f"frame_role must be a SliceInclusionFrameRole member, got {self.frame_role!r}"
            )
        if not isinstance(self.disposition, TopologyDisposition):
            raise ValueError(
                f"disposition must be a TopologyDisposition member, got {self.disposition!r}"
            )
        if not isinstance(self.physician, str) or not self.physician.strip():
            raise ValueError("physician must be a non-empty string")
        _require_utc_timestamp(self.created_at)
        # A disposition without a documented reason is exactly the audit gap
        # Astra F2 names: blank/whitespace-only reasons fail closed here.
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if self.provenance.series_uid not in (None, self.series_uid):
            raise ValueError("provenance series_uid does not match record")
        _require_utc_timestamp(self.provenance.created_at)
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(
            self, "disposition_id", canonical_digest(self.to_canonical())
        )

    @property
    def key(self) -> tuple[str, int, int, str]:
        """The exact durable block key this disposition resolves."""
        return (self.series_uid, self.slice_index, self.frame_index, self.structure)

    @property
    def identity_key(self) -> tuple[int, int, str]:
        """Per-series duplicate-detection key ``(slice, frame, structure)``."""
        return (self.slice_index, self.frame_index, self.structure)

    def to_canonical(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "domain": "TopologyDispositionRecord",
            "series_uid": self.series_uid,
            "slice_index": self.slice_index,
            "frame_index": self.frame_index,
            "structure": self.structure,
            "source_block_id": self.source_block_id,
            "source_revision_id": self.source_revision_id,
            "selection_id": self.selection_id,
            "frame_role": self.frame_role.value,
            "disposition": self.disposition.value,
            "physician": self.physician,
            "created_at": self.created_at,
            "reason": self.reason,
            "provenance": self.provenance,
        }
        if self.workflow_id is not None:
            payload["workflow_id"] = self.workflow_id
        return payload


def topology_disposition_from_payload(
    data: dict[str, Any],
) -> TopologyDispositionRecord:
    """Rebuild a :class:`TopologyDispositionRecord` from its canonical payload."""

    raw_provenance = data["provenance"]
    provenance = Provenance(
        study_uid=raw_provenance["study_uid"],
        series_uid=raw_provenance["series_uid"],
        software_version=raw_provenance["software_version"],
        model=None
        if raw_provenance["model"] is None
        else ModelMetadata(
            name=raw_provenance["model"]["name"],
            version=raw_provenance["model"]["version"],
            label_map=dict(raw_provenance["model"]["label_map"]),
            preprocessing_version=raw_provenance["model"]["preprocessing_version"],
            checksum=raw_provenance["model"]["checksum"],
        ),
        created_at=raw_provenance["created_at"],
        revision=raw_provenance["revision"],
        workflow_id=raw_provenance.get("workflow_id"),
    )
    return TopologyDispositionRecord(
        series_uid=data["series_uid"],
        slice_index=int(data["slice_index"]),
        frame_index=int(data["frame_index"]),
        structure=str(data["structure"]),
        source_block_id=data["source_block_id"],
        source_revision_id=data["source_revision_id"],
        selection_id=data["selection_id"],
        frame_role=SliceInclusionFrameRole(data["frame_role"]),
        disposition=TopologyDisposition(data["disposition"]),
        physician=data["physician"],
        created_at=data["created_at"],
        reason=data["reason"],
        provenance=provenance,
        workflow_id=data.get("workflow_id"),
    )


def topology_disposition_set_digest(
    entries: Sequence[TopologyDispositionRecord],
) -> str:
    """Content-addressed digest of a disposition SET (approval binding).

    The set is canonicalized by ``(slice_index, frame_index, structure)`` so
    the digest is independent of insertion order, then hashed. Mirrors the
    ``inclusion_set_id`` discipline: the approval binds this digest, so any
    added, removed or altered disposition changes the sealed identity (Astra
    required test 5).
    """
    ordered = sorted(entries, key=lambda record: record.identity_key)
    payload = [record.to_canonical() for record in ordered]
    return canonical_digest(payload)


def applicable_topology_dispositions(
    entries: Sequence[TopologyDispositionRecord],
    *,
    head_revision_id: str,
    selection_id: str,
) -> list[TopologyDispositionRecord]:
    """The APPLICABLE subset of a series' durable dispositions (C1 / F2).

    ``topology_dispositions`` is append-only: after a head change (or a
    selection change) the physician records a NEW disposition for the same
    ``(slice, frame, structure)`` key, bound to the new head+selection. The
    historical rows are preserved for provenance, so a store can carry
    SEVERAL records for one key — but exactly ONE of them may speak about
    the clinical state being resolved now: the one bound to the CURRENT
    ``head_revision_id`` and ``selection_id``. This function is that single
    deterministic projection, and it is the ONE place it is computed so the
    controller, the store's digest verification and the wire projection
    cannot drift.

    Determinism: at most one record per (key, head, selection) binding is
    admitted at write time (``save_topology_disposition_set_checked``), so
    the applicable set is unique; the result keeps the caller's (store)
    order, which is already canonical.
    """
    return [
        record
        for record in entries
        if record.source_revision_id == head_revision_id
        and record.selection_id == selection_id
    ]


def chain_key_authoring_times(
    chain: Sequence[Any],
) -> dict[tuple[int, int, Structure], str]:
    """Authoring stamp of every key the revision chain actually AUTHORED.

    R1 review F1 (P1), extracted HERE so the controller's relevance resolver
    and the store's seal last line cannot drift: a revision that merely
    re-carries identical contours (a ``snapshot_qc`` refresh copies its
    parent's contour map verbatim) is NOT a geometry-authoring event, so the
    map records the FIRST revision whose content for a key differs from what
    the chain carried before it. Keys only ever carried verbatim keep their
    earliest authoring stamp.

    ``chain`` is the ordered revision chain (any object exposing ``contours``
    mapping ``Structure -> contour`` and ``created_at``), oldest first.

    Tombstone rule (CMR_CREATE_A_001, R2 C1): keys whose newest chain event
    is a ``remove`` tombstone are ABSENT from both maps. A removal is never
    a silent ``resolved_by_replacement`` event (that would clear a durable
    block without a physician disposition), and ordinary ancestor merging
    must not resurrect removed geometry into disposition or seal checks.
    """
    suppressed = head_tombstoned_keys(chain)
    carried: dict[tuple[int, int, Structure], str] = {}
    authored: dict[tuple[int, int, Structure], str] = {}
    for revision in chain:
        for structure, contour in revision.contours.items():
            key = (contour.slice_index, contour.frame_index, structure)
            if key in suppressed:
                continue
            digest = canonical_digest(contour)
            if carried.get(key) != digest:
                carried[key] = digest
                authored[key] = revision.created_at
    return authored


# --- S3-001 schema v5 row projections -------------------------------------------
#
# Frozen row dataclasses for the three additive v5 tables. They are pure
# storage projections (column TEXT/INTEGER values exactly as stored), used by
# the store loaders to verify denormalized columns against the canonical
# payload fields before any record is returned. They carry no identity of
# their own: the content identities live in the reconstructed domain records.


@dataclass(frozen=True)
class ModelRunRow:
    """Stored ``model_runs`` row projection (schema v5)."""

    model_run_id: str
    model_name: str
    model_version: str
    model_sha256: str
    framework: str
    study_uid: str
    series_uid: str
    input_content_hash: str
    mask_artifact_sha256: str
    uq_method: str | None
    payload: str
    record_digest: str
    created_at: str


@dataclass(frozen=True)
class WholeCinePredictionRow:
    """Stored ``whole_cine_predictions`` row projection (schema v5)."""

    prediction_id: str
    study_uid: str
    series_uid: str
    source: str
    model_run_id: str | None
    slice_count: int
    frame_count: int
    payload: str
    created_at: str


@dataclass(frozen=True)
class SelectedPhasePredictionRow:
    """Stored ``selected_phase_predictions`` row projection (schema v8, FNC-C)."""

    prediction_id: str
    study_uid: str
    series_uid: str
    source: str
    model_run_id: str | None
    slice_count: int
    frame_count: int
    source_frame_indices: str
    payload: str
    created_at: str


@dataclass(frozen=True)
class WholeCineRevisionRow:
    """Stored ``whole_cine_revisions`` row projection (schema v5)."""

    revision_id: str
    study_uid: str
    series_uid: str
    version: int
    parent_prediction_id: str
    parent_revision_id: str | None
    editor: str
    edited_keys_json: str
    payload: str
    created_at: str


# --- S3-003 schema v6 model-registry records -----------------------------------


def _require_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64-character lowercase hex")


@dataclass(frozen=True)
class ModelBundleRecord:
    """Immutable registry projection for one signed model attestation."""

    bundle_id: str
    model_id: str
    semantic_version: str
    weights_format: str
    payload_sha256: str
    manifest_json: str
    install_root_relpath: str
    installed_at: str
    software_version: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("bundle_id", self.bundle_id)
        _require_digest("payload_sha256", self.payload_sha256)
        if self.weights_format not in ("safetensors-v1", "onnx-v1"):
            raise ValueError("unsupported weights_format")
        if not isinstance(self.manifest_json, str) or not self.manifest_json:
            raise ValueError("manifest_json must be non-empty canonical JSON text")
        if (
            not isinstance(self.install_root_relpath, str)
            or self.install_root_relpath != self.bundle_id
            or "/" in self.install_root_relpath
            or "\\" in self.install_root_relpath
        ):
            raise ValueError("install_root_relpath must equal the relative bundle id")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be non-blank")
        if not isinstance(self.semantic_version, str) or not self.semantic_version.strip():
            raise ValueError("semantic_version must be non-blank")
        if not isinstance(self.software_version, str) or not self.software_version.strip():
            raise ValueError("software_version must be non-blank")
        _require_utc_timestamp(self.installed_at)
        object.__setattr__(self, "digest", canonical_digest(self.to_canonical()))

    def to_canonical(self) -> dict[str, Any]:
        return {
            "domain": "ModelBundleRecord",
            "bundle_id": self.bundle_id,
            "model_id": self.model_id,
            "semantic_version": self.semantic_version,
            "weights_format": self.weights_format,
            "payload_sha256": self.payload_sha256,
            "manifest_json": self.manifest_json,
            "install_root_relpath": self.install_root_relpath,
            "installed_at": self.installed_at,
            "software_version": self.software_version,
        }


@dataclass(frozen=True)
class ModelActivationRecord:
    """One append-only pointer transition in the model activation history."""

    bundle_id: str
    previous_activation_id: str | None
    reason: str
    activated_at: str
    activation_id: str = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest("bundle_id", self.bundle_id)
        if self.previous_activation_id is not None:
            _require_digest("previous_activation_id", self.previous_activation_id)
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be non-blank")
        _require_utc_timestamp(self.activated_at)
        identity = canonical_digest(self.to_canonical())
        object.__setattr__(self, "activation_id", identity)
        object.__setattr__(self, "digest", identity)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "domain": "ModelActivationRecord",
            "bundle_id": self.bundle_id,
            "previous_activation_id": self.previous_activation_id,
            "reason": self.reason,
            "activated_at": self.activated_at,
        }


@dataclass(frozen=True)
class RegistryState:
    """The single mutable registry pointer (the history remains append-only)."""

    active_activation_id: str
    updated_at: str

    def __post_init__(self) -> None:
        _require_digest("active_activation_id", self.active_activation_id)
        _require_utc_timestamp(self.updated_at)


# --- F14 selected-phase analysis result authority --------------------------------
#
# CMR_FUNCTION_V1_ANALYSIS_001: ONE persisted binding that makes the 9 Cine
# Function metrics, the QC report, staleness and approval/seal resolve to the
# SAME exact inputs. The record is deliberately small and reuses every existing
# identity:
#
# * ``head_revision_id``       the exact QC-bearing contour-chain head used
#                              (contour/revision authority);
# * ``selection_id`` + frames  the exact physician-confirmed ED/ES selection;
# * ``inclusion_set_id``       the exact applicable basal-inclusion grid (None
#                              only for disclosed legacy null-set states);
# * ``measurement_id``         the content-addressed 9-metric Measurement
#                              (algorithm/software/range versions ride inside);
# * ``qc_report_digest``       the exact QC payload the metrics were paired
#                              with (stored on the same head revision);
# * ``workflow_id``/``study``  the raw-source workflow identity.
#
# It seals nothing: an approval still runs its own atomic head/selection/
# measurement/QC/inclusion verification. The binding exists so no reader can
# pair a metric record with a QC report from a different state, and so a change
# to ANY bound input makes the previous analysis visibly non-current instead of
# silently mixable.

#: ``analysis_id == sha256(canonical_json(payload))``, the store convention.
@dataclass(frozen=True)
class SelectedPhaseAnalysisRecord:
    """One immutable, content-addressed selected-phase analysis binding."""

    series_uid: str
    study_uid: str
    head_revision_id: str
    selection_id: str
    ed_frame: int
    es_frame: int
    measurement_id: str
    qc_report_digest: str
    qc_verdict: str
    algorithm_version: str
    software_version: str
    provenance: Provenance
    workflow_id: str | None = None
    inclusion_set_id: str | None = None
    range_table_version: str | None = None
    analysis_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.series_uid, str) or not self.series_uid.strip():
            raise ValueError("series_uid must be a non-empty string")
        if not isinstance(self.study_uid, str) or not self.study_uid.strip():
            raise ValueError("study_uid must be a non-empty string")
        for name in ("head_revision_id", "selection_id", "measurement_id", "qc_report_digest"):
            _require_hex64(name, getattr(self, name))
        _require_hex64("workflow_id", self.workflow_id)
        _require_hex64("inclusion_set_id", self.inclusion_set_id)
        for name in ("ed_frame", "es_frame"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value!r}")
        if self.qc_verdict not in QC_VERDICTS:
            raise ValueError(
                f"qc_verdict must be one of {', '.join(QC_VERDICTS)}, "
                f"got {self.qc_verdict!r}"
            )
        for name in ("algorithm_version", "software_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.range_table_version is not None and (
            not isinstance(self.range_table_version, str)
            or not self.range_table_version.strip()
        ):
            raise ValueError(
                "range_table_version must be a non-empty string when populated, "
                f"got {self.range_table_version!r}"
            )
        if self.provenance.series_uid not in (None, self.series_uid):
            raise ValueError("provenance series_uid does not match record")
        _require_utc_timestamp(self.provenance.created_at)
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(self, "analysis_id", canonical_digest(self.to_canonical()))

    @property
    def binding_key(self) -> tuple[str, str, str | None, str, str]:
        """The UNIQUE clinical binding this analysis speaks about.

        ``(head_revision_id, selection_id, inclusion_set_id, measurement_id,
        qc_report_digest)`` — at most one persisted analysis row may carry one
        binding (enforced at schema level), so an approval can always resolve
        exactly ONE analysis record from its own sealed pointers.
        """
        return (
            self.head_revision_id,
            self.selection_id,
            self.inclusion_set_id,
            self.measurement_id,
            self.qc_report_digest,
        )

    def binds(
        self,
        *,
        head_revision_id: str,
        selection_id: str | None,
        inclusion_set_id: str | None,
        measurement_id: str,
        qc_report_digest: str | None,
    ) -> bool:
        """True IFF every supplied identity equals this analysis' binding."""
        return self.binding_key == (
            head_revision_id,
            selection_id,
            inclusion_set_id,
            measurement_id,
            qc_report_digest,
        )

    def to_canonical(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "domain": "SelectedPhaseAnalysisRecord",
            "series_uid": self.series_uid,
            "study_uid": self.study_uid,
            "head_revision_id": self.head_revision_id,
            "selection_id": self.selection_id,
            "ed_frame": self.ed_frame,
            "es_frame": self.es_frame,
            "measurement_id": self.measurement_id,
            "qc_report_digest": self.qc_report_digest,
            "qc_verdict": self.qc_verdict,
            "algorithm_version": self.algorithm_version,
            "software_version": self.software_version,
            "provenance": self.provenance,
        }
        if self.workflow_id is not None:
            payload["workflow_id"] = self.workflow_id
        if self.inclusion_set_id is not None:
            payload["inclusion_set_id"] = self.inclusion_set_id
        if self.range_table_version is not None:
            payload["range_table_version"] = self.range_table_version
        return payload


def analysis_record_from_payload(data: dict[str, Any]) -> SelectedPhaseAnalysisRecord:
    """Rebuild a :class:`SelectedPhaseAnalysisRecord` from its canonical payload."""

    raw_provenance = data["provenance"]
    provenance = Provenance(
        study_uid=raw_provenance["study_uid"],
        series_uid=raw_provenance["series_uid"],
        software_version=raw_provenance["software_version"],
        model=None
        if raw_provenance["model"] is None
        else ModelMetadata(
            name=raw_provenance["model"]["name"],
            version=raw_provenance["model"]["version"],
            label_map=dict(raw_provenance["model"]["label_map"]),
            preprocessing_version=raw_provenance["model"]["preprocessing_version"],
            checksum=raw_provenance["model"]["checksum"],
        ),
        created_at=raw_provenance["created_at"],
        revision=raw_provenance["revision"],
        workflow_id=raw_provenance.get("workflow_id"),
    )
    return SelectedPhaseAnalysisRecord(
        series_uid=data["series_uid"],
        study_uid=data["study_uid"],
        head_revision_id=data["head_revision_id"],
        selection_id=data["selection_id"],
        ed_frame=int(data["ed_frame"]),
        es_frame=int(data["es_frame"]),
        measurement_id=data["measurement_id"],
        qc_report_digest=data["qc_report_digest"],
        qc_verdict=data["qc_verdict"],
        algorithm_version=data["algorithm_version"],
        software_version=data["software_version"],
        provenance=provenance,
        workflow_id=data.get("workflow_id"),
        inclusion_set_id=data.get("inclusion_set_id"),
        range_table_version=data.get("range_table_version"),
    )
