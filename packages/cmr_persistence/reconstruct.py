"""Payload reconstruction and the persistence-local EDESSelection projection.

Two responsibilities live here (Node 2, contract R1 amendments A + C):

1. ``*_from_payload`` rebuilders turn parsed canonical-JSON payloads back into
   typed Node 1 records. Parsed values pass through VERBATIM into the Node 1
   constructors — no ``float()``/``int()`` coercion anywhere, so ints stay
   ints, floats stay floats (``-0.0`` preserved) and tuples/lists are handled
   by the existing ``_canonical`` normalization inside ``cmr_core``. The store
   re-verifies every load byte-for-byte, so any drift fails closed.

2. ``edes_selection_payload`` / ``edes_selection_payload_text`` /
   ``selection_id_from_payload_text`` remain the storage-side compatibility
   wrappers for :class:`~cmr_core.domain_types.EDESSelection`. Since S2C-001
   the type itself carries ``to_canonical()`` and the derived
   ``selection_id``; the wrappers DELEGATE to that core implementation so
   there is exactly one projection definition, while historical callers keep
   working unchanged. The delegated projection is byte-identical to the v2
   projection by construction (same field set, same serializer), so no
   stored selection id can churn.

S2C-001 additive fields: every rebuilder reads new optional payload keys via
``dict.get`` so legacy payloads without them reconstruct into records whose
canonical form omits those keys — byte-for-byte round trips stay exact.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from cmr_contours.model import Contour
from cmr_core.domain_types import (
    ApprovalSnapshot,
    ContourRevision,
    EDESSelection,
    HoleAwareContour,
    Measurement,
    ModelRunRecord,
    Point2D,
    PolygonRing,
    Prediction,
    SelectedPhasePrediction,
    WholeCinePrediction,
    WholeCineRevision,
    canonical_json,
)
from cmr_core.errors import CMRWarning, Severity
from cmr_core.provenance import ModelMetadata, Provenance
from cmr_core.segmentation import Structure
from cmr_qc.report import QCReport
from cmr_quantification.units import Quantity, Unit

from cmr_persistence.records import ModelActivationRecord, ModelBundleRecord

__all__ = [
    "edes_selection_payload",
    "edes_selection_payload_text",
    "selection_id_from_payload_text",
    "qc_report_from_payload",
    "approval_snapshot_from_payload",
    "contour_revision_from_payload",
    "edes_selection_from_payload",
    "measurement_from_payload",
    "prediction_from_payload",
    "model_run_from_payload",
    "model_bundle_from_payload",
    "model_activation_from_payload",
    "whole_cine_prediction_from_payload",
    "whole_cine_revision_from_payload",
    "whole_cine_edited_keys_json",
    "selected_phase_prediction_from_payload",
    "selected_phase_frame_indices_json",
]

# Exact json.dumps parameter set of cmr_core.domain_types.canonical_json.
# Keep identical by construction: test_persistence_immutability.py asserts the
# cross-consistency invariant against the imported serializer.
_CANONICAL_JSON_KWARGS: dict[str, Any] = {
    "sort_keys": True,
    "separators": (",", ":"),
    "ensure_ascii": False,
    "allow_nan": False,
}


def edes_selection_payload(selection: EDESSelection) -> dict[str, Any]:
    """Frozen canonical projection dict for an EDESSelection (amendment A).

    S2C-001 compatibility wrapper: delegates to the core
    ``EDESSelection.to_canonical()`` whose field set is pinned to the v2
    persistence projection, normalized through ``canonical_json`` so every
    nested domain leaf is a plain JSON value exactly as in schema v2.
    """

    return cast(dict[str, Any], json.loads(canonical_json(selection.to_canonical())))


def edes_selection_payload_text(selection: EDESSelection) -> str:
    """Serialize the projection with the exact canonical parameter set."""

    return json.dumps(edes_selection_payload(selection), **_CANONICAL_JSON_KWARGS)


def selection_id_from_payload_text(payload_text: str) -> str:
    """Storage-computed EDESSelection identity: sha256 of the payload TEXT.

    Retained for historical callers; equals the core-derived
    ``EDESSelection.selection_id`` for the same projection bytes.
    """

    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()


def _model_from_payload(value: dict[str, Any] | None) -> ModelMetadata | None:
    if value is None:
        return None
    return ModelMetadata(
        name=value["name"],
        version=value["version"],
        label_map=dict(value["label_map"]),
        preprocessing_version=value["preprocessing_version"],
        checksum=value["checksum"],
    )


def _provenance_from_payload(value: dict[str, Any]) -> Provenance:
    # Values pass through verbatim: ints stay ints, floats stay floats.
    # S2C-001: workflow_id is read via .get so legacy payloads without the
    # key reconstruct into unbound provenance (canonical form omits it).
    return Provenance(
        study_uid=value["study_uid"],
        series_uid=value["series_uid"],
        software_version=value["software_version"],
        model=_model_from_payload(value["model"]),
        created_at=value["created_at"],
        revision=value["revision"],
        workflow_id=value.get("workflow_id"),
    )


def _contour_from_payload(value: dict[str, Any]) -> Contour:
    return Contour(
        slice_index=value["slice_index"],
        frame_index=value["frame_index"],
        structure=Structure(value["structure"]),
        points=value["points"],
        source=value["source"],
        edited=value["edited"],
        approved=value["approved"],
        revision=value["revision"],
        model=_model_from_payload(value["model"]),
        provenance=None
        if value["provenance"] is None
        else _provenance_from_payload(value["provenance"]),
    )


def _contours_from_payload(value: dict[str, Any]) -> dict[Structure, Contour]:
    return {Structure(key): _contour_from_payload(contour) for key, contour in value.items()}


def _qc_report_from_payload(value: dict[str, Any] | None) -> QCReport | None:
    if value is None:
        return None
    flags = [
        CMRWarning(
            code=flag["code"],
            message=flag["message"],
            source=flag["source"],
            severity=Severity(flag["severity"]),
        )
        for flag in value["flags"]
    ]
    return QCReport(flags=flags)


def qc_report_from_payload(value: dict[str, Any] | None) -> QCReport | None:
    """Public wrapper over the private report rebuilder.

    S2C-001: the store's checked-approval transaction digests the PERSISTED
    head revision's gateway ``qc_report`` payload; it needs the same verbatim
    reconstruction rule the loaders use, without reaching into privates.
    """
    return _qc_report_from_payload(value)


def prediction_from_payload(data: dict[str, Any]) -> Prediction:
    """Rebuild a :class:`Prediction` from its parsed canonical payload."""

    return Prediction(
        study_uid=data["study_uid"],
        series_uid=data["series_uid"],
        source=data["source"],
        contours=_contours_from_payload(data["contours"]),
        model=_model_from_payload(data["model"]),
        provenance=_provenance_from_payload(data["provenance"]),
        created_at=data["created_at"],
    )


def contour_revision_from_payload(data: dict[str, Any]) -> ContourRevision:
    """Rebuild a :class:`ContourRevision` from its parsed canonical payload."""

    return ContourRevision(
        series_uid=data["series_uid"],
        parent_prediction_id=data["parent_prediction_id"],
        parent_revision_id=data["parent_revision_id"],
        version=data["version"],
        contours=_contours_from_payload(data["contours"]),
        editor=data["editor"],
        created_at=data["created_at"],
        provenance=_provenance_from_payload(data["provenance"]),
        qc_report=_qc_report_from_payload(data["qc_report"]),
        # Additive create/remove operation identity (CMR_CREATE_A_001, R2
        # C1): read via .get so legacy payloads without the keys
        # reconstruct into records whose canonical form omits them
        # (byte-for-byte round trips stay exact).
        edit_op=data.get("edit_op"),
        edit_key=_edit_key_from_payload(data.get("edit_key")),
        inverse_of=data.get("inverse_of"),
    )


def _edit_key_from_payload(
    value: dict[str, Any] | None,
) -> tuple[int, int, Structure] | None:
    """Rebuild one ``(slice, frame, structure)`` operation key, else ``None``."""
    if value is None:
        return None
    return (
        int(value["slice_index"]),
        int(value["frame_index"]),
        Structure(value["structure"]),
    )


def edes_selection_from_payload(data: dict[str, Any]) -> EDESSelection:
    """Rebuild an :class:`EDESSelection` from its parsed projection payload."""

    return EDESSelection(
        series_uid=data["series_uid"],
        ed_frame=data["ed_frame"],
        es_frame=data["es_frame"],
        selected_by=data["selected_by"],
        selected_at=data["selected_at"],
        provenance=_provenance_from_payload(data["provenance"]),
        n_frames=data["n_frames"],
    )


def measurement_from_payload(data: dict[str, Any]) -> Measurement:
    """Rebuild a :class:`Measurement` from its parsed canonical payload."""

    def quantity(value: dict[str, Any]) -> Quantity:
        return Quantity(value=value["value"], unit=Unit(value["unit"]))

    return Measurement(
        series_uid=data["series_uid"],
        ed_frame=data["ed_frame"],
        es_frame=data["es_frame"],
        source_revision_id=data["source_revision_id"],
        edv_ml=quantity(data["edv_ml"]),
        esv_ml=quantity(data["esv_ml"]),
        sv_ml=quantity(data["sv_ml"]),
        ef_pct=quantity(data["ef_pct"]),
        lv_mass_g=quantity(data["lv_mass_g"]),
        rv_edv_ml=quantity(data["rv_edv_ml"]),
        rv_esv_ml=quantity(data["rv_esv_ml"]),
        rv_sv_ml=quantity(data["rv_sv_ml"]),
        rv_ef_pct=quantity(data["rv_ef_pct"]),
        stack_gaps_mm=data["stack_gaps_mm"],
        software_version=data["software_version"],
        provenance=_provenance_from_payload(data["provenance"]),
        created_at=data["created_at"],
        algorithm_version=data["algorithm_version"],
        selection_id=data.get("selection_id"),
        inclusion_set_id=data.get("inclusion_set_id"),
        # CFM-001 §2.6: read via .get so legacy payloads without the key
        # reconstruct into records whose canonical form omits it
        # (byte-for-byte round trips stay exact).
        range_table_version=data.get("range_table_version"),
    )


def approval_snapshot_from_payload(data: dict[str, Any]) -> ApprovalSnapshot:
    """Rebuild an :class:`ApprovalSnapshot` from its parsed canonical payload."""

    return ApprovalSnapshot(
        series_uid=data["series_uid"],
        source_revision_id=data["source_revision_id"],
        ed_frame=data["ed_frame"],
        es_frame=data["es_frame"],
        measurement_id=data["measurement_id"],
        qc_verdict=data["qc_verdict"],
        approved_by=data["approved_by"],
        approved_at=data["approved_at"],
        provenance=_provenance_from_payload(data["provenance"]),
        software_version=data["software_version"],
        algorithm_version=data["algorithm_version"],
        selection_id=data.get("selection_id"),
        qc_report_digest=data.get("qc_report_digest"),
        inclusion_set_id=data.get("inclusion_set_id"),
        # Astra F2/F5: the bound topology-disposition-set digest round-trips
        # with the snapshot (absent for every legacy payload).
        topology_disposition_digest=data.get("topology_disposition_digest"),
    )


# --- S3-001 whole-cine / model-run deserializers (schema v5) ---------------------


def _cine_key_to_string(key: tuple[int, int, Structure]) -> str:
    """Deterministic contour-key string ``"{slice}:{frame}:{structure}"``."""
    return f"{key[0]}:{key[1]}:{key[2].value}"


def _cine_key_from_string(key_string: str) -> tuple[int, int, Structure]:
    """Parse a canonical contour-key string back into its tuple key.

    Fail-closed on malformed strings; structure values are simple identifiers
    without colons, so a three-part split is exact.
    """
    parts = key_string.split(":")
    if len(parts) != 3:
        raise ValueError(f"malformed whole-cine contour key {key_string!r}")
    try:
        return (int(parts[0]), int(parts[1]), Structure(parts[2]))
    except ValueError as exc:
        raise ValueError(f"malformed whole-cine contour key {key_string!r}") from exc


def _polygon_ring_from_payload(value: dict[str, Any]) -> PolygonRing:
    """Rebuild a :class:`PolygonRing` verbatim — no numeric coercion anywhere."""
    points = cast(
        "tuple[Point2D, ...]",
        tuple((point[0], point[1]) for point in value["points"]),
    )
    return PolygonRing(points=points)


def _hole_aware_contour_from_payload(value: dict[str, Any]) -> HoleAwareContour:
    """Rebuild one hole-aware contour from its parsed canonical payload."""
    return HoleAwareContour(
        exterior=_polygon_ring_from_payload(value["exterior"]),
        holes=tuple(_polygon_ring_from_payload(hole) for hole in value.get("holes", ())),
        structure=Structure(value["structure"]),
    )


def _cine_contours_from_payload(
    value: dict[str, Any],
) -> dict[tuple[int, int, Structure], tuple[HoleAwareContour, ...]]:
    """Rebuild the ``(slice, frame, structure) -> contours`` mapping from the
    canonical string-keyed payload projection."""
    contours: dict[tuple[int, int, Structure], tuple[HoleAwareContour, ...]] = {}
    for key_string, contour_values in value.items():
        key = _cine_key_from_string(key_string)
        contours[key] = tuple(
            _hole_aware_contour_from_payload(contour) for contour in contour_values
        )
    return contours


def selected_phase_frame_indices_json(frame_indices: tuple[int, ...]) -> str:
    """Canonical TEXT form of the identity-significant ``source_frame_indices``.

    The FNC-C schema v8 column stores exactly this compact
    comma-separated bracketed list (``[0,12]``), and the store asserts the
    column equals this projection of the payload field on every load. The
    artifact's own validation already guarantees non-empty, strictly
    ascending, unique, in-range integers; nothing is coerced here.
    """

    return f"[{','.join(str(int(index)) for index in frame_indices)}]"


def selected_phase_prediction_from_payload(
    data: dict[str, Any],
) -> SelectedPhasePrediction:
    """Rebuild a :class:`SelectedPhasePrediction` from its canonical payload.

    ``source_frame_indices`` is read as a REQUIRED key (the FNC-C artifact has
    no legacy predecessor, so there is never a payload without it) and passed
    through verbatim as a tuple of ints — no coercion.
    """

    return SelectedPhasePrediction(
        study_uid=data["study_uid"],
        series_uid=data["series_uid"],
        source=data["source"],
        contours=_cine_contours_from_payload(data["contours"]),
        slice_count=data["slice_count"],
        frame_count=data["frame_count"],
        source_frame_indices=tuple(data["source_frame_indices"]),
        model_run_id=data["model_run_id"],
        model=_model_from_payload(data["model"]),
        provenance=_provenance_from_payload(data["provenance"]),
        created_at=data["created_at"],
    )


def whole_cine_prediction_from_payload(data: dict[str, Any]) -> WholeCinePrediction:
    """Rebuild a :class:`WholeCinePrediction` from its parsed canonical payload."""

    return WholeCinePrediction(
        study_uid=data["study_uid"],
        series_uid=data["series_uid"],
        source=data["source"],
        contours=_cine_contours_from_payload(data["contours"]),
        slice_count=data["slice_count"],
        frame_count=data["frame_count"],
        model_run_id=data["model_run_id"],
        model=_model_from_payload(data["model"]),
        provenance=_provenance_from_payload(data["provenance"]),
        created_at=data["created_at"],
    )


def whole_cine_revision_from_payload(data: dict[str, Any]) -> WholeCineRevision:
    """Rebuild a :class:`WholeCineRevision` from its parsed canonical payload."""

    return WholeCineRevision(
        study_uid=data["study_uid"],
        series_uid=data["series_uid"],
        version=data["version"],
        parent_prediction_id=data["parent_prediction_id"],
        parent_revision_id=data["parent_revision_id"],
        editor=data["editor"],
        edited_keys=tuple(_cine_key_from_string(key_string) for key_string in data["edited_keys"]),
        contours=_cine_contours_from_payload(data["contours"]),
        provenance=_provenance_from_payload(data["provenance"]),
        created_at=data["created_at"],
    )


def model_run_from_payload(data: dict[str, Any]) -> ModelRunRecord:
    """Rebuild a :class:`ModelRunRecord` from its parsed canonical payload."""

    uq_summary_metrics = data["uq_summary_metrics"]
    return ModelRunRecord(
        model_run_id=data["model_run_id"],
        model_name=data["model_name"],
        model_version=data["model_version"],
        model_sha256=data["model_sha256"],
        framework=data["framework"],
        study_uid=data["study_uid"],
        series_uid=data["series_uid"],
        input_sop_uids=tuple(data["input_sop_uids"]),
        input_content_hash=data["input_content_hash"],
        spatial_transforms=dict(data["spatial_transforms"]),
        uq_method=data["uq_method"],
        uq_summary_metrics=(None if uq_summary_metrics is None else dict(uq_summary_metrics)),
        mask_artifact_uri=data["mask_artifact_uri"],
        mask_artifact_sha256=data["mask_artifact_sha256"],
        uncertainty_artifact_uri=data["uncertainty_artifact_uri"],
        execution_time_s=data["execution_time_s"],
        created_at=data["created_at"],
    )


def model_bundle_from_payload(data: dict[str, Any]) -> ModelBundleRecord:
    """Rebuild a v6 model-bundle record without coercing payload values."""

    return ModelBundleRecord(
        bundle_id=data["bundle_id"],
        model_id=data["model_id"],
        semantic_version=data["semantic_version"],
        weights_format=data["weights_format"],
        payload_sha256=data["payload_sha256"],
        manifest_json=data["manifest_json"],
        install_root_relpath=data["install_root_relpath"],
        installed_at=data["installed_at"],
        software_version=data["software_version"],
    )


def model_activation_from_payload(data: dict[str, Any]) -> ModelActivationRecord:
    """Rebuild a v6 activation record and verify its content identity."""

    record = ModelActivationRecord(
        bundle_id=data["bundle_id"],
        previous_activation_id=data.get("previous_activation_id"),
        reason=data["reason"],
        activated_at=data["activated_at"],
    )
    if record.activation_id != data.get("activation_id", record.activation_id):
        raise ValueError("activation id does not match canonical activation payload")
    return record


def whole_cine_edited_keys_json(revision: WholeCineRevision) -> str:
    """Canonical ``edited_keys_json`` column TEXT derived from the record.

    Single derivation source for BOTH the write path and the load-time
    column/payload equality verification, so the additive store metadata can
    never drift from the record identity.
    """
    return canonical_json([_cine_key_to_string(key) for key in revision.edited_keys])
