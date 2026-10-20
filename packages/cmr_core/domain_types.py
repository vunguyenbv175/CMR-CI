"""Immutable, content-addressed Node 1 clinical domain records.

This module deliberately contains records only. It does not persist data, select ED/ES,
or calculate measurements. Canonical payloads are JSON objects with sorted keys and
SHA-256 digests; nested legacy domain objects are snapshotted at construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, cast

from cmr_contours.model import Contour
from cmr_qc.report import QCReport
from cmr_quantification.units import Quantity, Unit

from cmr_core.errors import CMRWarning
from cmr_core.geometry import (
    _AREA_EPSILON,
    _ring_winding_normalized,
    _signed_area,
    _validate_hole_topology,
    _validate_no_self_intersection,
)
from cmr_core.provenance import ModelMetadata, Provenance
from cmr_core.segmentation import Structure


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        ordered = sorted(value.items(), key=lambda item: str(item[0]))
        return {str(_canonical(k)): _canonical(v) for k, v in ordered}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical payload cannot contain non-finite numbers")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, ModelMetadata):
        return {
            "name": value.name,
            "version": value.version,
            "label_map": _canonical(value.label_map),
            "preprocessing_version": value.preprocessing_version,
            "checksum": value.checksum,
        }
    if isinstance(value, Provenance):
        # S2C-001: workflow_id is included ONLY when populated, so every
        # legacy (unbound) canonical payload stays byte-identical.
        payload = {
            "study_uid": value.study_uid,
            "series_uid": value.series_uid,
            "software_version": value.software_version,
            "model": _canonical(value.model),
            "created_at": value.created_at,
            "revision": value.revision,
        }
        if value.workflow_id is not None:
            payload["workflow_id"] = value.workflow_id
        return payload
    if isinstance(value, Quantity):
        return {"value": value.value, "unit": value.unit.value}
    if isinstance(value, Contour):
        return {
            "slice_index": value.slice_index,
            "frame_index": value.frame_index,
            "structure": value.structure.value,
            "points": _canonical(value.points),
            "source": value.source,
            "edited": value.edited,
            "approved": value.approved,
            "revision": value.revision,
            "model": _canonical(value.model),
            "provenance": _canonical(value.provenance),
        }
    if isinstance(value, CMRWarning):
        return {
            "code": value.code,
            "message": value.message,
            "source": value.source,
            "severity": value.severity.value,
        }
    if isinstance(value, QCReport):
        return {"flags": [_canonical(f) for f in value.flags], "verdict": value.verdict}
    if isinstance(value, PolygonRing):
        return {"points": _canonical(value.points)}
    if isinstance(value, HoleAwareContour):
        return {
            "structure": value.structure.value,
            "exterior": _canonical(value.exterior),
            "holes": [_canonical(hole) for hole in value.holes],
        }
    if isinstance(value, WholeCinePrediction):
        return _canonical(value.to_canonical())
    if isinstance(value, WholeCineRevision):
        return _canonical(value.to_canonical())
    if isinstance(value, ModelRunRecord):
        return _canonical(value.to_canonical())
    if hasattr(value, "to_canonical"):
        return _canonical(value.to_canonical())
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the one stable serialization used by all Node 1 digests."""
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    """Public content identity: ``sha256(canonical_json(value).encode("utf-8"))``.

    S2C-001 promotes the previously private digest helper to the public
    identity primitive used by every content address in the coherence chain
    (manifest workflow ids, selection ids, QC payload digests). The private
    alias below keeps every existing call site and derived id unchanged.
    """
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# Internal alias (S2C-001): existing derived identities keep their exact
# computation path; ``canonical_digest`` is the same function by definition.
_digest = canonical_digest


def _validate_optional_hex64(name: str, value: str | None) -> None:
    """Fail closed unless ``value`` is None or 64-character lowercase hex."""
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{name} must be 64-character lowercase hex when populated, got {value!r}"
        )


def _utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"timestamp must be ISO-8601 UTC: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"timestamp must be ISO-8601 UTC: {value!r}")
    return value


class _FrozenContour(Contour):
    """A Contour-compatible immutable snapshot used inside immutable records."""

    def __init__(self, source: Contour) -> None:
        object.__setattr__(self, "_locked", False)
        super().__init__(
            source.slice_index,
            source.frame_index,
            source.structure,
            cast(Any, tuple(tuple(p) for p in source.points)),
            source.source,
            source.edited,
            source.approved,
            source.revision,
            _freeze_model(source.model),
            _freeze_provenance(source.provenance),
        )
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise TypeError("immutable domain record")
        object.__setattr__(self, name, value)


def _freeze_model(model: ModelMetadata | None) -> ModelMetadata | None:
    if model is None:
        return None
    clone = ModelMetadata(
        model.name,
        model.version,
        cast(Any, MappingProxyType(dict(model.label_map))),
        model.preprocessing_version,
        model.checksum,
    )
    return clone


def _freeze_provenance(provenance: Provenance | None) -> Provenance | None:
    if provenance is None:
        return None
    clone = Provenance(
        provenance.study_uid,
        provenance.series_uid,
        provenance.software_version,
        _freeze_model(provenance.model),
        provenance.created_at,
        provenance.revision,
        provenance.workflow_id,
    )
    object.__setattr__(clone, "_locked", True)
    # Provenance is a legacy mutable dataclass; use a dedicated immutable proxy.
    return _FrozenProvenance.from_value(clone)


class _FrozenProvenance(Provenance):
    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise TypeError("immutable domain record")
        super().__setattr__(name, value)

    @classmethod
    def from_value(cls, value: Provenance) -> _FrozenProvenance:
        obj = cls(
            value.study_uid,
            value.series_uid,
            value.software_version,
            value.model,
            value.created_at,
            value.revision,
            value.workflow_id,
        )
        object.__setattr__(obj, "_locked", True)
        return obj


def _validate_contours_mapping(contours: Mapping[Structure, Contour]) -> None:
    """Mapping-key coherence (review e5614a6 P1-3): every mapping key must
    equal the contour's own ``structure`` BEFORE freezing/digesting; a contour
    stored under a foreign key corrupts canonical state and fails closed.
    :func:`save_to_revision` performs the same check for edit composition."""
    for key, value in contours.items():
        if key != value.structure:
            raise ValueError(
                f"contours key {key.value!r} does not match contour structure "
                f"{value.structure.value!r}; mapping keys must equal "
                "Contour.structure"
            )


def _freeze_contours(contours: Mapping[Structure, Contour]) -> Mapping[Structure, Contour]:
    return MappingProxyType({key: _FrozenContour(value) for key, value in contours.items()})


def _validate_provenance(provenance: Provenance, series_uid: str) -> None:
    if provenance.series_uid not in (None, series_uid):
        raise ValueError("provenance series_uid does not match record")
    _utc(provenance.created_at)


@dataclass(frozen=True)
class Prediction:
    study_uid: str
    series_uid: str
    source: str
    contours: Mapping[Structure, Contour]
    model: ModelMetadata | None
    provenance: Provenance
    created_at: str
    prediction_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.source not in ("ai", "manual-seed"):
            raise ValueError("prediction source must be 'ai' or 'manual-seed'")
        if self.provenance.revision != 0:
            raise ValueError("prediction provenance revision must be 0")
        if self.source == "ai":
            if self.provenance.model is None:
                raise ValueError("ai predictions require model identity in provenance.model")
            if self.model is not None and self.model != self.provenance.model:
                raise ValueError("prediction model must agree with provenance.model")
        _utc(self.created_at)
        _validate_provenance(self.provenance, self.series_uid)
        _validate_contours_mapping(self.contours)
        object.__setattr__(self, "contours", _freeze_contours(self.contours))
        object.__setattr__(self, "model", _freeze_model(self.model))
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        payload = self.to_canonical()
        object.__setattr__(self, "prediction_id", _digest(payload))

    def to_canonical(self) -> dict[str, Any]:
        return {
            "domain": "Prediction",
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "source": self.source,
            "contours": self.contours,
            "model": self.model,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ContourRevision:
    """One immutable append-only physician revision of legacy contour state.

    Create/remove operation identity (CMR_CREATE_A_001, R2 C1): a revision
    composed by the bounded create/remove path carries ``edit_op`` plus the
    exact single ``edit_key`` it authored. A ``remove`` revision additionally
    carries ``inverse_of`` (the ``revision_id`` of the single-key ``create``
    revision it inverts) and acts as a parent-linked TOMBSTONE: its own
    contour map omits the removed key while every effective-state reader
    suppresses the key chain-wide (see :func:`head_tombstoned_keys`). All
    three fields are additive payload-level data (omitted from the canonical
    form when unset, so every legacy payload stays byte-identical); no DDL
    or CAS-writer change is involved.
    """

    series_uid: str
    parent_prediction_id: str | None
    parent_revision_id: str | None
    version: int
    contours: Mapping[Structure, Contour]
    editor: str
    created_at: str
    provenance: Provenance
    qc_report: QCReport | None = None
    edit_op: str | None = None
    edit_key: tuple[int, int, Structure] | None = None
    inverse_of: str | None = None
    revision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("revision version must be >= 1")
        if self.version == 1 and self.parent_revision_id is not None:
            raise ValueError("version 1 cannot have a parent revision")
        if self.version > 1 and not self.parent_revision_id:
            raise ValueError("versions after 1 require a parent revision")
        _utc(self.created_at)
        _validate_provenance(self.provenance, self.series_uid)
        _validate_contours_mapping(self.contours)
        _validate_edit_operation(self.edit_op, self.edit_key, self.inverse_of)
        if self.edit_key is not None:
            # Normalized once so every reader sees one (slice, frame,
            # Structure) spelling regardless of the caller's tuple shape.
            object.__setattr__(
                self,
                "edit_key",
                (self.edit_key[0], self.edit_key[1], self.edit_key[2]),
            )
        object.__setattr__(self, "contours", _freeze_contours(self.contours))
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(self, "qc_report", _freeze_qc(self.qc_report))
        object.__setattr__(self, "revision_id", _digest(self.to_canonical()))

    @classmethod
    def from_parent(
        cls,
        parent: ContourRevision | Prediction,
        *,
        series_uid: str,
        contours: Mapping[Structure, Contour],
        editor: str,
        created_at: str,
        provenance: Provenance,
        qc_report: QCReport | None = None,
        edit_op: str | None = None,
        edit_key: tuple[int, int, Structure] | None = None,
        inverse_of: str | None = None,
    ) -> ContourRevision:
        if series_uid != parent.series_uid:
            raise ValueError("child revision series_uid must match parent series_uid")
        expected_revision = parent.provenance.revision + 1
        if provenance.revision != expected_revision:
            raise ValueError(
                f"provenance revision must be {expected_revision} "
                f"(parent revision {parent.provenance.revision} + 1)"
            )
        if isinstance(parent, Prediction):
            return cls(
                series_uid,
                parent.prediction_id,
                None,
                1,
                contours,
                editor,
                created_at,
                provenance,
                qc_report,
                edit_op,
                edit_key,
                inverse_of,
            )
        return cls(
            series_uid,
            parent.parent_prediction_id,
            parent.revision_id,
            parent.version + 1,
            contours,
            editor,
            created_at,
            provenance,
            qc_report,
            edit_op,
            edit_key,
            inverse_of,
        )

    def to_canonical(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "domain": "ContourRevision",
            "series_uid": self.series_uid,
            "parent_prediction_id": self.parent_prediction_id,
            "parent_revision_id": self.parent_revision_id,
            "version": self.version,
            "contours": self.contours,
            "editor": self.editor,
            "created_at": self.created_at,
            "provenance": self.provenance,
            "qc_report": self.qc_report,
        }
        # Additive tombstone/operation identity (R2 C1): omitted when unset
        # so every legacy canonical payload — and therefore every legacy
        # revision_id — stays byte-identical.
        if self.edit_op is not None:
            payload["edit_op"] = self.edit_op
        if self.edit_key is not None:
            slice_index, frame_index, structure = self.edit_key
            payload["edit_key"] = {
                "slice_index": slice_index,
                "frame_index": frame_index,
                "structure": structure.value,
            }
        if self.inverse_of is not None:
            payload["inverse_of"] = self.inverse_of
        return payload


#: Operation kinds for parent-linked create/remove revisions (R2 C1).
EDIT_OP_CREATE = "create"
EDIT_OP_REMOVE = "remove"


def _validate_edit_operation(
    edit_op: str | None,
    edit_key: tuple[int, int, Structure] | None,
    inverse_of: str | None,
) -> None:
    """Fail closed on any incoherent create/remove operation identity."""
    if edit_op is None:
        if edit_key is not None:
            raise ValueError("edit_key requires edit_op 'create' or 'remove'")
        if inverse_of is not None:
            raise ValueError("inverse_of requires edit_op 'remove'")
        return
    if edit_op not in (EDIT_OP_CREATE, EDIT_OP_REMOVE):
        raise ValueError(f"edit_op must be 'create' or 'remove', got {edit_op!r}")
    if edit_key is None:
        raise ValueError(f"{edit_op} revision requires edit_key")
    try:
        slice_index, frame_index, structure = edit_key
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{edit_op} edit_key must be a (slice_index, frame_index, Structure) "
            f"triple, got {edit_key!r}"
        ) from exc
    for name, value in (("slice_index", slice_index), ("frame_index", frame_index)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{edit_op} {name} must be an integer, got {value!r}")
        if value < 0:
            raise ValueError(f"{edit_op} {name} must be non-negative, got {value!r}")
    if not isinstance(structure, Structure):
        raise ValueError(
            f"{edit_op} edit_key structure must be a Structure member, "
            f"got {structure!r}"
        )
    if edit_op == EDIT_OP_CREATE:
        if inverse_of is not None:
            raise ValueError("create revision must not carry inverse_of")
    else:
        _validate_optional_hex64("inverse_of", inverse_of)
        if inverse_of is None:
            raise ValueError("remove revision requires inverse_of")


def revision_operation(
    revision: Any,
) -> tuple[str | None, tuple[int, int, Structure] | None, str | None]:
    """The durable ``(edit_op, edit_key, inverse_of)`` of one chain record.

    Duck-typed over any object exposing the three attributes (legacy
    :class:`Prediction` records and hand-built fakes simply yield
    ``(None, None, None)``); malformed operation spellings likewise read as
    "no operation" so readers never resurrect geometry from a half-written
    marker.
    """
    edit_op = getattr(revision, "edit_op", None)
    if edit_op not in (EDIT_OP_CREATE, EDIT_OP_REMOVE):
        return (None, None, None)
    edit_key = getattr(revision, "edit_key", None)
    try:
        slice_index, frame_index, structure = edit_key
    except (TypeError, ValueError):
        return (None, None, None)
    if (
        isinstance(slice_index, bool)
        or not isinstance(slice_index, int)
        or isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
    ):
        return (None, None, None)
    if not isinstance(structure, Structure):
        try:
            structure = Structure(structure)
        except (ValueError, TypeError):
            return (None, None, None)
    return (edit_op, (slice_index, frame_index, structure), getattr(revision, "inverse_of", None))


def _carried_legacy_contours(
    revision: Any,
) -> tuple[tuple[tuple[int, int, Structure], Contour], ...]:
    """``((slice, frame, structure), contour)`` pairs one record asserts.

    Only legacy :class:`Contour` values count: whole-cine mappings (whose
    values are component tuples) never participate in legacy tombstone
    resolution.
    """
    contours = getattr(revision, "contours", None)
    if not isinstance(contours, Mapping):
        return ()
    carried: list[tuple[tuple[int, int, Structure], Contour]] = []
    for map_key, contour in contours.items():
        if not isinstance(contour, Contour):
            continue
        if map_key != contour.structure:
            continue
        carried.append(
            ((contour.slice_index, contour.frame_index, contour.structure), contour)
        )
    return tuple(carried)


def head_tombstoned_keys(
    chain: Any,
) -> set[tuple[int, int, Structure]]:
    """Keys whose NEWEST chain event is a ``remove`` tombstone.

    Newest-event-wins, mirroring the overwrite merge of
    :meth:`ViewerState.merged_contours_at <cmr_viewer.state.ViewerState
    .merged_contours_at>`: a key re-created after removal (redo) resolves
    live; a key whose latest event is removal reads absent in every
    effective-state reader. Chains without operations yield the empty set,
    so legacy behavior is byte-for-byte unchanged.
    """
    tombstoned: set[tuple[int, int, Structure]] = set()
    resolved: set[tuple[int, int, Structure]] = set()
    for revision in reversed(list(chain or ())):
        edit_op, edit_key, _ = revision_operation(revision)
        if edit_op is not None and edit_key is not None and edit_key not in resolved:
            resolved.add(edit_key)
            if edit_op == EDIT_OP_REMOVE:
                tombstoned.add(edit_key)
        for carried_key, _ in _carried_legacy_contours(revision):
            if carried_key not in resolved:
                resolved.add(carried_key)
    return tombstoned


def effective_chain_contour(
    chain: Any,
    key: tuple[int, int, Structure],
) -> Contour | None:
    """Newest effective legacy contour for ``key``, honoring tombstones.

    A ``remove`` tombstone stops the backwards search (the key reads absent
    even though ancestors still carry it); a ``create`` assertion likewise
    terminates the search. Returns ``None`` when the key is tombstoned or
    was never authored.
    """
    slice_index, frame_index, structure = key
    for revision in reversed(list(chain or ())):
        edit_op, edit_key, _ = revision_operation(revision)
        if edit_op == EDIT_OP_REMOVE and edit_key == key:
            return None
        contours = getattr(revision, "contours", None)
        if isinstance(contours, Mapping):
            candidate = contours.get(structure)
            if (
                isinstance(candidate, Contour)
                and candidate.slice_index == slice_index
                and candidate.frame_index == frame_index
            ):
                return candidate
        if edit_op == EDIT_OP_CREATE and edit_key == key:
            return None
    return None


def chain_effective_contours(
    chain: Any,
) -> dict[tuple[int, int, Structure], Contour]:
    """Oldest-first effective ``(slice, frame, structure) -> Contour`` map.

    Later revisions overwrite earlier ones; a ``remove`` tombstone deletes
    its key so ordinary ancestor merging can never resurrect removed
    geometry. Used by chain-wide presence readers (adoption preservation,
    topology presence) and their tests.
    """
    live: dict[tuple[int, int, Structure], Contour] = {}
    for revision in list(chain or ()):
        edit_op, edit_key, _ = revision_operation(revision)
        if edit_op == EDIT_OP_REMOVE and edit_key is not None:
            live.pop(edit_key, None)
        for carried_key, contour in _carried_legacy_contours(revision):
            live[carried_key] = contour
    return live


class _FrozenQCReport(QCReport):
    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise TypeError("immutable domain record")
        super().__setattr__(name, value)


def _freeze_qc(report: QCReport | None) -> QCReport | None:
    if report is None:
        return None
    clone = _FrozenQCReport(list(report.flags))
    object.__setattr__(clone, "flags", tuple(report.flags))
    object.__setattr__(clone, "_locked", True)
    return clone


@dataclass(frozen=True)
class EDESSelection:
    """One physician-authored ED/ES phase selection (manual authority).

    S2C-001: ``selection_id`` is the content identity
    ``canonical_digest(to_canonical())`` over EXACTLY the persistence
    projection field set that storage already used in schema v2 — so every
    previously stored selection id stays byte-stable while new measurements,
    approvals and QC decisions can bind to the explicit id.
    """

    series_uid: str
    ed_frame: int
    es_frame: int
    selected_by: str
    selected_at: str
    provenance: Provenance
    n_frames: int | None = None
    selection_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.ed_frame < 0 or self.es_frame < 0:
            raise ValueError("ED/ES frame indices must be non-negative")
        if self.n_frames is not None and (
            self.n_frames <= 0 or self.ed_frame >= self.n_frames or self.es_frame >= self.n_frames
        ):
            raise ValueError("ED/ES frame index is outside the series")
        _utc(self.selected_at)
        _validate_provenance(self.provenance, self.series_uid)
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(self, "selection_id", canonical_digest(self.to_canonical()))

    def to_canonical(self) -> dict[str, Any]:
        """Canonical projection: EXACTLY the frozen v2 persistence fields.

        Field set is fixed by the storage contract (see
        ``cmr_persistence.reconstruct.edes_selection_payload``): adding or
        removing any key would churn every stored selection id.
        """
        return {
            "domain": "EDESSelection",
            "series_uid": self.series_uid,
            "ed_frame": self.ed_frame,
            "es_frame": self.es_frame,
            "selected_by": self.selected_by,
            "selected_at": self.selected_at,
            "n_frames": self.n_frames,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class Measurement:
    series_uid: str
    ed_frame: int
    es_frame: int
    source_revision_id: str
    edv_ml: Quantity
    esv_ml: Quantity
    sv_ml: Quantity
    ef_pct: Quantity
    lv_mass_g: Quantity
    rv_edv_ml: Quantity
    rv_esv_ml: Quantity
    rv_sv_ml: Quantity
    rv_ef_pct: Quantity
    stack_gaps_mm: tuple[float, ...]
    software_version: str
    provenance: Provenance
    created_at: str
    algorithm_version: str = "node1-record-only"
    selection_id: str | None = None
    inclusion_set_id: str | None = None
    # CFM-001 additive trailing optional (spec §2.6): reference-range table
    # version bound into the measurement identity. Omitted from the canonical
    # payload when None so every legacy payload stays byte-identical.
    range_table_version: str | None = None
    measurement_id: str = field(init=False)

    def __post_init__(self) -> None:
        expected = (
            ("edv_ml", Unit.ML),
            ("esv_ml", Unit.ML),
            ("sv_ml", Unit.ML),
            ("ef_pct", Unit.PERCENT),
            ("lv_mass_g", Unit.G),
            ("rv_edv_ml", Unit.ML),
            ("rv_esv_ml", Unit.ML),
            ("rv_sv_ml", Unit.ML),
            ("rv_ef_pct", Unit.PERCENT),
        )
        for name, unit in expected:
            if getattr(self, name).unit is not unit:
                raise ValueError(f"{name} must use {unit.value}")
        if self.ed_frame < 0 or self.es_frame < 0:
            raise ValueError("ED/ES frame indices must be non-negative")
        if any(not math.isfinite(g) or g <= 0 for g in self.stack_gaps_mm):
            raise ValueError("stack_gaps_mm must contain finite positive values")
        _validate_optional_hex64("selection_id", self.selection_id)
        _validate_optional_hex64("inclusion_set_id", self.inclusion_set_id)
        # CFM-001 §2.6: populated range_table_version must be a non-empty
        # string (no hex constraint — it is a semantic version label).
        if self.range_table_version is not None and (
            not isinstance(self.range_table_version, str)
            or not self.range_table_version.strip()
        ):
            raise ValueError(
                "range_table_version must be a non-empty string when populated, "
                f"got {self.range_table_version!r}"
            )
        _utc(self.created_at)
        _validate_provenance(self.provenance, self.series_uid)
        object.__setattr__(self, "stack_gaps_mm", tuple(self.stack_gaps_mm))
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(self, "measurement_id", _digest(self.to_canonical()))

    def to_canonical(self) -> dict[str, Any]:
        # S2C-001 additive binding fields are omitted when None so every
        # legacy measurement payload stays byte-identical (no serialized
        # ``"field": null`` is ever introduced).
        payload = {
            "domain": "Measurement",
            "series_uid": self.series_uid,
            "ed_frame": self.ed_frame,
            "es_frame": self.es_frame,
            "source_revision_id": self.source_revision_id,
            "edv_ml": self.edv_ml,
            "esv_ml": self.esv_ml,
            "sv_ml": self.sv_ml,
            "ef_pct": self.ef_pct,
            "lv_mass_g": self.lv_mass_g,
            "rv_edv_ml": self.rv_edv_ml,
            "rv_esv_ml": self.rv_esv_ml,
            "rv_sv_ml": self.rv_sv_ml,
            "rv_ef_pct": self.rv_ef_pct,
            "stack_gaps_mm": self.stack_gaps_mm,
            "software_version": self.software_version,
            "algorithm_version": self.algorithm_version,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }
        if self.selection_id is not None:
            payload["selection_id"] = self.selection_id
        if self.inclusion_set_id is not None:
            payload["inclusion_set_id"] = self.inclusion_set_id
        if self.range_table_version is not None:
            payload["range_table_version"] = self.range_table_version
        return payload


# The repository QC verdict vocabulary, exactly as produced by
# ``cmr_qc.report.QCReport.verdict`` ("PASS" when no flags, else "REVIEW REQUIRED").
_QC_VERDICTS = ("PASS", "REVIEW REQUIRED")


@dataclass(frozen=True)
class ApprovalSnapshot:
    series_uid: str
    source_revision_id: str
    ed_frame: int
    es_frame: int
    measurement_id: str
    qc_verdict: str
    approved_by: str
    approved_at: str
    provenance: Provenance
    software_version: str
    algorithm_version: str = "node1-record-only"
    selection_id: str | None = None
    qc_report_digest: str | None = None
    inclusion_set_id: str | None = None
    #: Astra F2/F5 (2026-09-09): the content-addressed digest of the
    #: topology-disposition SET this approval relied on. Omitted from the
    #: canonical payload when None so every legacy approval payload stays
    #: byte-identical.
    topology_disposition_digest: str | None = None
    approval_id: str = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.ed_frame < 0 or self.es_frame < 0:
            raise ValueError("ED/ES frame indices must be non-negative")
        if self.qc_verdict not in _QC_VERDICTS:
            raise ValueError(f"qc_verdict must be one of {', '.join(_QC_VERDICTS)}")
        _validate_optional_hex64("selection_id", self.selection_id)
        _validate_optional_hex64("qc_report_digest", self.qc_report_digest)
        _validate_optional_hex64("inclusion_set_id", self.inclusion_set_id)
        _validate_optional_hex64(
            "topology_disposition_digest", self.topology_disposition_digest
        )
        _utc(self.approved_at)
        _validate_provenance(self.provenance, self.series_uid)
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        payload = self.to_canonical()
        digest = _digest(payload)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "approval_id", digest)

    def to_canonical(self) -> dict[str, Any]:
        # S2C-001 additive binding fields are omitted when None so every
        # legacy approval payload stays byte-identical.
        payload = {
            "domain": "ApprovalSnapshot",
            "series_uid": self.series_uid,
            "source_revision_id": self.source_revision_id,
            "ed_frame": self.ed_frame,
            "es_frame": self.es_frame,
            "measurement_id": self.measurement_id,
            "qc_verdict": self.qc_verdict,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "provenance": self.provenance,
            "software_version": self.software_version,
            "algorithm_version": self.algorithm_version,
        }
        if self.selection_id is not None:
            payload["selection_id"] = self.selection_id
        if self.qc_report_digest is not None:
            payload["qc_report_digest"] = self.qc_report_digest
        if self.inclusion_set_id is not None:
            payload["inclusion_set_id"] = self.inclusion_set_id
        if self.topology_disposition_digest is not None:
            payload["topology_disposition_digest"] = self.topology_disposition_digest
        return payload


# --- S3-001 whole-cine hole-aware records -----------------------------------

Point2D = tuple[float, float]
_UUID_V4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _s3_hex64(name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64-character lowercase hex, got {value!r}")


_S3_COORD_LIMIT = 1e150

def _s3_coordinate(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an int or float, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if abs(float(value)) > _S3_COORD_LIMIT:
        raise ValueError(
            f"{name} magnitude exceeds the exact-predicate safety limit "
            f"(abs must be <= {_S3_COORD_LIMIT:.0e}), got {value!r}"
        )


@dataclass(frozen=True)
class PolygonRing:
    """Simple pixel-coordinate ring; closure is implicit."""

    points: tuple[Point2D, ...]

    def __post_init__(self) -> None:
        normalized: list[Point2D] = []
        for point in self.points:
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError(f"polygon vertices must be (x, y) pairs, got {point!r}")
            _s3_coordinate("polygon x coordinate", point[0])
            _s3_coordinate("polygon y coordinate", point[1])
            normalized.append((point[0], point[1]))
        object.__setattr__(self, "points", tuple(normalized))
        if len(self.points) < 3:
            raise ValueError("PolygonRing must have at least 3 points")
        if any(a == b for a, b in zip(self.points, self.points[1:])):
            raise ValueError("consecutive duplicate vertices forbidden")
        if self.points[0] == self.points[-1]:
            raise ValueError("do not repeat the first vertex at the end; closure is implicit")
        _validate_no_self_intersection(self.points)
        if abs(_signed_area(self.points)) < _AREA_EPSILON:
            raise ValueError("degenerate ring: signed area below epsilon")

    def to_canonical(self) -> dict[str, Any]:
        return {"points": self.points}


@dataclass(frozen=True)
class HoleAwareContour:
    """One exterior polygon plus zero or more strictly-contained holes."""

    exterior: PolygonRing
    holes: tuple[PolygonRing, ...] = ()
    structure: Structure = Structure.LV_MYOCARDIUM

    def __post_init__(self) -> None:
        if not isinstance(self.structure, Structure):
            raise ValueError(f"structure must be a Structure member, got {self.structure!r}")
        holes = tuple(self.holes)
        if any(not isinstance(hole, PolygonRing) for hole in holes):
            raise ValueError("holes must be PolygonRing instances")
        exterior = PolygonRing(_ring_winding_normalized(self.exterior.points, ccw=True))
        normalized_holes = tuple(
            PolygonRing(_ring_winding_normalized(hole.points, ccw=False)) for hole in holes
        )
        _validate_hole_topology(exterior.points, [hole.points for hole in normalized_holes])
        object.__setattr__(self, "exterior", exterior)
        object.__setattr__(self, "holes", normalized_holes)

    @property
    def net_area(self) -> float:
        return abs(_signed_area(self.exterior.points)) - sum(
            abs(_signed_area(hole.points)) for hole in self.holes
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "structure": self.structure.value,
            "exterior": self.exterior,
            "holes": list(self.holes),
        }


def _s3_key_string(key: tuple[int, int, Structure]) -> str:
    return f"{key[0]}:{key[1]}:{key[2].value}"


def _s3_validate_key(key: Any) -> None:
    if not isinstance(key, tuple) or len(key) != 3:
        raise ValueError(
            "contour mapping keys must be (slice_index, frame_index, structure) tuples"
        )
    slice_index, frame_index, structure = key
    if isinstance(slice_index, bool) or not isinstance(slice_index, int) or slice_index < 0:
        raise ValueError(f"slice_index must be a non-negative integer, got {slice_index!r}")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError(f"frame_index must be a non-negative integer, got {frame_index!r}")
    if not isinstance(structure, Structure):
        raise ValueError(f"key structure must be a Structure member, got {structure!r}")


CineContourMapping = Mapping[tuple[int, int, Structure], tuple[HoleAwareContour, ...]]


def _s3_freeze_contours(contours: CineContourMapping) -> Mapping[
    tuple[int, int, Structure], tuple[HoleAwareContour, ...]
]:
    frozen: dict[tuple[int, int, Structure], tuple[HoleAwareContour, ...]] = {}
    for key, values in contours.items():
        _s3_validate_key(key)
        values_tuple = tuple(values)
        if any(not isinstance(value, HoleAwareContour) for value in values_tuple):
            raise ValueError("contour values must be HoleAwareContour instances")
        if any(value.structure != key[2] for value in values_tuple):
            raise ValueError(
                f"contours key {_s3_key_string(key)!r} does not match contour structure"
            )
        frozen[key] = values_tuple
    return MappingProxyType(frozen)


def _s3_canonical_contours(contours: CineContourMapping) -> dict[str, list[Any]]:
    return {
        _s3_key_string(key): [_canonical(value) for value in values]
        for key, values in sorted(contours.items(), key=lambda item: _s3_key_string(item[0]))
    }


@dataclass(frozen=True)
class WholeCinePrediction:
    """Immutable content-addressed whole-cine prediction artifact."""

    study_uid: str
    series_uid: str
    source: str
    contours: CineContourMapping
    slice_count: int
    frame_count: int
    model_run_id: str | None
    model: ModelMetadata | None
    provenance: Provenance
    created_at: str
    prediction_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.source not in ("ai", "manual-seed"):
            raise ValueError("prediction source must be 'ai' or 'manual-seed'")
        if self.source == "ai":
            if self.model_run_id is None:
                raise ValueError("ai predictions require a non-null model_run_id")
            if self.provenance.model is None:
                raise ValueError("ai predictions require model identity in provenance.model")
            if self.model is not None and self.model != self.provenance.model:
                raise ValueError("prediction model must agree with provenance.model")
        elif self.model_run_id is not None:
            raise ValueError("manual-seed predictions must carry model_run_id=None")
        if (
            isinstance(self.slice_count, bool)
            or not isinstance(self.slice_count, int)
            or self.slice_count < 1
        ):
            raise ValueError(f"slice_count must be an integer >= 1, got {self.slice_count!r}")
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count < 1
        ):
            raise ValueError(f"frame_count must be an integer >= 1, got {self.frame_count!r}")
        _utc(self.created_at)
        _validate_provenance(self.provenance, self.series_uid)
        contours = _s3_freeze_contours(self.contours)
        distinct_slices = {key[0] for key in contours}
        distinct_frames = {key[1] for key in contours}
        if len(distinct_slices) != self.slice_count:
            raise ValueError("slice_count does not cover exactly the distinct slice indices")
        if len(distinct_frames) != self.frame_count:
            raise ValueError("frame_count does not cover exactly the distinct frame indices")
        outside_slices = [
            index for index in sorted(distinct_slices)
            if not 0 <= index < self.slice_count
        ]
        if outside_slices:
            raise ValueError(
                "slice indices must fall within [0, slice_count); outside: "
                f"{outside_slices!r} (slice_count={self.slice_count})"
            )
        outside_frames = [
            index for index in sorted(distinct_frames)
            if not 0 <= index < self.frame_count
        ]
        if outside_frames:
            raise ValueError(
                "frame indices must fall within [0, frame_count); outside: "
                f"{outside_frames!r} (frame_count={self.frame_count})"
            )
        object.__setattr__(self, "contours", contours)
        object.__setattr__(self, "model", _freeze_model(self.model))
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(self, "prediction_id", _digest(self.to_canonical()))

    def to_canonical(self) -> dict[str, Any]:
        return {
            "domain": "WholeCinePrediction",
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "source": self.source,
            "contours": _s3_canonical_contours(self.contours),
            "slice_count": self.slice_count,
            "frame_count": self.frame_count,
            "model_run_id": self.model_run_id,
            "model": self.model,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class WholeCineRevision:
    """Immutable physician-edit revision carrying the full post-edit state."""

    study_uid: str
    series_uid: str
    version: int
    parent_prediction_id: str
    parent_revision_id: str | None
    editor: str
    edited_keys: tuple[tuple[int, int, Structure], ...]
    contours: CineContourMapping
    provenance: Provenance
    created_at: str
    revision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("revision version must be >= 1")
        if self.version == 1 and self.parent_revision_id is not None:
            raise ValueError("version 1 cannot have a parent revision")
        if self.version > 1 and self.parent_revision_id is None:
            raise ValueError("versions after 1 require a parent revision")
        # Sol S6: provenance.revision is bound to the chain position:
        # version 1 starts at revision 0; every later version N carries
        # exactly N - 1.
        expected_revision = self.version - 1
        if self.provenance.revision != expected_revision:
            raise ValueError(
                "provenance.revision must bind the chain position "
                f"(version {self.version} requires revision "
                f"{expected_revision}, got {self.provenance.revision})"
            )
        _s3_hex64("parent_prediction_id", self.parent_prediction_id)
        if self.parent_revision_id is not None:
            _s3_hex64("parent_revision_id", self.parent_revision_id)
        _utc(self.created_at)
        _validate_provenance(self.provenance, self.series_uid)
        contours = _s3_freeze_contours(self.contours)
        keys = tuple(self.edited_keys)
        if not keys:
            raise ValueError("edited_keys must be non-empty")
        seen: set[tuple[int, int, Structure]] = set()
        for key in keys:
            _s3_validate_key(key)
            if key in seen:
                raise ValueError(f"duplicate edited key {_s3_key_string(key)!r}")
            if key not in contours:
                raise ValueError(f"edited key {_s3_key_string(key)!r} does not exist")
            seen.add(key)
        object.__setattr__(self, "edited_keys", tuple(sorted(seen, key=_s3_key_string)))
        object.__setattr__(self, "contours", contours)
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(self, "revision_id", _digest(self.to_canonical()))

    def to_canonical(self) -> dict[str, Any]:
        return {
            "domain": "WholeCineRevision",
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "version": self.version,
            "parent_prediction_id": self.parent_prediction_id,
            "parent_revision_id": self.parent_revision_id,
            "editor": self.editor,
            "edited_keys": [_s3_key_string(key) for key in self.edited_keys],
            "contours": _s3_canonical_contours(self.contours),
            "provenance": self.provenance,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# FNC-C additive sparse (selected-phase) prediction artifact (P1 provenance).
#
# ``WholeCinePrediction`` CANNOT represent a sparse artifact honestly: it
# requires the distinct source frame indices of ``contours`` to cover exactly
# ``[0, frame_count)``, so a selective ED/ES run over frames {0, 12} of a
# 25-frame cine is invalid there (and forcing it in would re-key SOURCE frames
# onto LOCAL slots 0/1, which is a silent phase lie — PREIMPL review,
# "Sparse/all-25 compatibility and provenance").
#
# ``SelectedPhasePrediction`` is therefore a SEPARATE, narrowly named,
# additive artifact for explicit selected-phase coverage. Its identity
# (``prediction_id``) is content-addressed over a canonical payload whose
# ``domain`` is distinct AND which carries ``source_frame_indices``: the
# exact ordered source frame indices that were inferred. A sparse artifact
# with ``source_frame_indices=(0, 12)`` therefore can never collide with an
# all-phase artifact whose local slots happen to be ``(0, 1)``, even when the
# masks and model are byte-identical.
#
# Contour keys are ALWAYS ``(slice_index, SOURCE frame_index, structure)``.
# Local slot indices are never used as keys anywhere in this artifact.
# ---------------------------------------------------------------------------

_SELECTED_PHASE_DOMAIN = "SelectedPhasePrediction"


def _sp_validate_source_frame_indices(
    value: Any, frame_count: int
) -> tuple[int, ...]:
    """Freeze and validate ``source_frame_indices`` (fail closed, no coercion).

    Accepted input: a sequence of non-negative integers, non-empty, STRICTLY
    ASCENDING with NO duplicates, every element within ``[0, frame_count)``.

    Documented choice (order-sensitivity): the declared tuple is REQUIRED to
    already be in canonical ascending order; an unsorted or duplicated input
    is REJECTED rather than silently normalized. Deterministic normalization
    would make two different spellings denote one artifact, and the review
    requires the coverage tuple to be identity-significant and unambiguous.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ValueError(
            "source_frame_indices must be a tuple/list of integers, got "
            f"{value!r}"
        )
    indices: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"source_frame_indices entries must be integers, got {item!r}"
            )
        indices.append(item)
    if not indices:
        raise ValueError("source_frame_indices must be non-empty")
    if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
        raise ValueError(
            "source_frame_indices must be strictly ascending and unique, got "
            f"{tuple(indices)!r}"
        )
    outside = [index for index in indices if not 0 <= index < frame_count]
    if outside:
        raise ValueError(
            "source_frame_indices must fall within [0, frame_count); outside: "
            f"{outside!r} (frame_count={frame_count})"
        )
    return tuple(indices)


@dataclass(frozen=True)
class SelectedPhasePrediction:
    """Immutable content-addressed SELECTED-PHASE prediction artifact (FNC-C).

    Represents AI (or manual-seed) geometry produced for an EXPLICIT subset of
    the cine source frames — e.g. the physician-confirmed ED/ES phases —
    without claiming whole-cine coverage. ``slice_count`` and ``frame_count``
    are the CINE GRID extents the artifact was produced against (not the
    covered counts); the covered frames are exactly
    ``source_frame_indices``.

    No clinical arithmetic is introduced here: this is provenance/identity
    only. Quantification of a sparse artifact remains the FNC-B
    selected-phase path's business, and a sparse artifact can never satisfy a
    full-cycle/curve task (see :mod:`cmr_workflow.coverage`).
    """

    study_uid: str
    series_uid: str
    source: str
    contours: CineContourMapping
    slice_count: int
    frame_count: int
    source_frame_indices: tuple[int, ...]
    model_run_id: str | None
    model: ModelMetadata | None
    provenance: Provenance
    created_at: str
    prediction_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.source not in ("ai", "manual-seed"):
            raise ValueError("prediction source must be 'ai' or 'manual-seed'")
        if self.source == "ai":
            if self.model_run_id is None:
                raise ValueError("ai predictions require a non-null model_run_id")
            if self.provenance.model is None:
                raise ValueError("ai predictions require model identity in provenance.model")
            if self.model is not None and self.model != self.provenance.model:
                raise ValueError("prediction model must agree with provenance.model")
        elif self.model_run_id is not None:
            raise ValueError("manual-seed predictions must carry model_run_id=None")
        if (
            isinstance(self.slice_count, bool)
            or not isinstance(self.slice_count, int)
            or self.slice_count < 1
        ):
            raise ValueError(f"slice_count must be an integer >= 1, got {self.slice_count!r}")
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count < 1
        ):
            raise ValueError(f"frame_count must be an integer >= 1, got {self.frame_count!r}")
        _utc(self.created_at)
        _validate_provenance(self.provenance, self.series_uid)
        contours = _s3_freeze_contours(self.contours)
        frame_indices = _sp_validate_source_frame_indices(
            self.source_frame_indices, self.frame_count
        )
        slice_indices = sorted({key[0] for key in contours})
        contour_frames = sorted({key[1] for key in contours})
        # Every contour must be keyed by a frame the artifact declares it
        # covers: a stray source frame is a coverage lie, not extra data.
        undeclared = [index for index in contour_frames if index not in set(frame_indices)]
        if undeclared:
            raise ValueError(
                "contour frame indices must be declared in source_frame_indices; "
                f"undeclared: {undeclared!r} (source_frame_indices={frame_indices!r})"
            )
        # ...and the declared coverage must be EXACTLY the covered frames
        # (no promise of a phase that carries no geometry).
        if tuple(contour_frames) != frame_indices:
            raise ValueError(
                "source_frame_indices must equal exactly the distinct contour "
                f"frame indices; declared {frame_indices!r} != derived "
                f"{tuple(contour_frames)!r}"
            )
        outside_slices = [
            index for index in slice_indices if not 0 <= index < self.slice_count
        ]
        if outside_slices:
            raise ValueError(
                "slice indices must fall within [0, slice_count); outside: "
                f"{outside_slices!r} (slice_count={self.slice_count})"
            )
        object.__setattr__(self, "contours", contours)
        object.__setattr__(self, "source_frame_indices", frame_indices)
        object.__setattr__(self, "model", _freeze_model(self.model))
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(self, "prediction_id", _digest(self.to_canonical()))

    def to_canonical(self) -> dict[str, Any]:
        return {
            "domain": _SELECTED_PHASE_DOMAIN,
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "source": self.source,
            "contours": _s3_canonical_contours(self.contours),
            "slice_count": self.slice_count,
            "frame_count": self.frame_count,
            "source_frame_indices": list(self.source_frame_indices),
            "model_run_id": self.model_run_id,
            "model": self.model,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }


def _s3_json_only(name: str, value: Any) -> None:
    if type(value).__module__.split(".")[0] == "numpy":
        raise ValueError(f"{name} accepts JSON-primitive values only; numpy fails closed")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite numbers")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _s3_json_only(name, item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} mapping keys must be strings")
            _s3_json_only(name, item)
        return
    raise ValueError(f"{name} accepts JSON-primitive values only")


def _s3_deep_freeze(value: Any) -> Any:
    """Recursively freeze JSON-primitive structures (review F2 fix).

    Mappings become MappingProxyType and lists/tuples become tuples,
    recursively, so caller-held nested objects can never mutate the
    canonical output after construction while ``record_digest`` stays
    stale. Scalars pass through verbatim.
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _s3_deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_s3_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_s3_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ModelRunRecord:
    """Immutable provenance record of one AI inference execution."""

    model_run_id: str
    model_name: str
    model_version: str
    model_sha256: str
    framework: str
    study_uid: str
    series_uid: str
    input_sop_uids: tuple[str, ...]
    input_content_hash: str
    spatial_transforms: Mapping[str, Any]
    uq_method: str | None
    uq_summary_metrics: Mapping[str, float] | None
    mask_artifact_uri: str
    mask_artifact_sha256: str
    uncertainty_artifact_uri: str | None
    execution_time_s: float
    created_at: str
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_run_id, str) or not _UUID_V4_RE.match(self.model_run_id):
            raise ValueError("model_run_id must be a canonical UUIDv4 string")
        _s3_hex64("model_sha256", self.model_sha256)
        _s3_hex64("input_content_hash", self.input_content_hash)
        _s3_hex64("mask_artifact_sha256", self.mask_artifact_sha256)
        if not isinstance(self.spatial_transforms, Mapping):
            raise ValueError("spatial_transforms must be a mapping")
        _s3_json_only("spatial_transforms", dict(self.spatial_transforms))
        if self.uq_summary_metrics is not None:
            if not isinstance(self.uq_summary_metrics, Mapping):
                raise ValueError("uq_summary_metrics must be a mapping or None")
            _s3_json_only("uq_summary_metrics", dict(self.uq_summary_metrics))
        if isinstance(self.execution_time_s, bool) or not isinstance(
            self.execution_time_s, (int, float)
        ):
            raise ValueError("execution_time_s must be a real number")
        if not math.isfinite(float(self.execution_time_s)) or self.execution_time_s < 0:
            raise ValueError("execution_time_s must be finite and non-negative")
        _utc(self.created_at)
        # Sol S7: a bare string would silently iterate into per-character
        # elements; require a real list/tuple container up front.
        if not isinstance(self.input_sop_uids, (list, tuple)):
            raise ValueError(
                "input_sop_uids must be a list or tuple of strings, got "
                f"{type(self.input_sop_uids).__name__}"
            )
        sop_uids = tuple(self.input_sop_uids)
        for sop_uid in sop_uids:
            if not isinstance(sop_uid, str):
                raise ValueError(
                    "input_sop_uids elements must be strings, got "
                    f"{type(sop_uid).__name__}"
                )
        object.__setattr__(self, "input_sop_uids", _s3_deep_freeze(sop_uids))
        object.__setattr__(
            self,
            "spatial_transforms",
            _s3_deep_freeze(dict(self.spatial_transforms)),
        )
        object.__setattr__(
            self,
            "uq_summary_metrics",
            None
            if self.uq_summary_metrics is None
            else _s3_deep_freeze(dict(self.uq_summary_metrics)),
        )
        object.__setattr__(self, "record_digest", _digest(self.to_canonical()))

    def to_canonical(self) -> dict[str, Any]:
        return {
            "domain": "ModelRunRecord",
            "model_run_id": self.model_run_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "framework": self.framework,
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "input_sop_uids": list(self.input_sop_uids),
            "input_content_hash": self.input_content_hash,
            "spatial_transforms": self.spatial_transforms,
            "uq_method": self.uq_method,
            "uq_summary_metrics": self.uq_summary_metrics,
            "mask_artifact_uri": self.mask_artifact_uri,
            "mask_artifact_sha256": self.mask_artifact_sha256,
            "uncertainty_artifact_uri": self.uncertainty_artifact_uri,
            "execution_time_s": self.execution_time_s,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TopologyBlock:
    """Durable record of a non-projectable topology condition for a cine plane."""

    series_uid: str
    slice_index: int
    frame_index: int
    structure: str
    reason: str
    created_at: str
    block_id: str | None = None

    def __post_init__(self) -> None:
        if self.slice_index < 0 or self.frame_index < 0:
            raise ValueError("slice_index and frame_index must be non-negative")
        _utc(self.created_at)
        payload = self.to_canonical()
        digest = _digest(payload)
        if self.block_id is not None and self.block_id != digest:
            raise ValueError(f"block_id mismatch: expected {digest}, got {self.block_id}")
        object.__setattr__(self, "block_id", digest)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "domain": "TopologyBlock",
            "series_uid": self.series_uid,
            "slice_index": self.slice_index,
            "frame_index": self.frame_index,
            "structure": self.structure,
            "reason": self.reason,
            "created_at": self.created_at,
        }


__all__ = [
    "ApprovalSnapshot",
    "ContourRevision",
    "EDESSelection",
    "HoleAwareContour",
    "Measurement",
    "ModelRunRecord",
    "Point2D",
    "PolygonRing",
    "Prediction",
    "TopologyBlock",
    "WholeCinePrediction",
    "WholeCineRevision",
    "canonical_digest",
    "canonical_json",
]

