"""Append-only embedded SQLite persistence for Node 1 clinical records.

Single-engine, single-connection store over :mod:`sqlite3` (Python stdlib
only — no ORM, no server, no cloud, no new dependency). Raw DICOM remains the
imaging source of truth: this store persists workflow identity, references,
provenance, digests, revisions, selections, measurements, QC and approvals —
never source pixels and never patient data.

Immutability model
    Five structurally separate record tables (``predictions``,
    ``contour_revisions``, ``edes_selections``, ``measurements``,
    ``approval_snapshots``) plus the N7.1 decision table
    ``qc_decision_records``. ``BEFORE UPDATE``/``BEFORE DELETE`` triggers
    raise ``RAISE(ABORT, 'append-only')`` on every table, so raw-SQL mutation
    also fails closed. Because SQLite only fires delete triggers under
    ``INSERT OR REPLACE`` when recursive triggers are enabled, additional
    ``BEFORE INSERT`` guard triggers abort any insert whose row id (or, for
    revisions, whose ``(series_uid, version)`` pair) already exists — closing
    the replace path at schema level. No ``UPDATE``/``DELETE``/``INSERT OR
    REPLACE`` code path exists in package code. The immutable clinical
    approvals table keeps its foreign key to ``measurements``: B-LITE
    sentinel decisions (physician overrides where Node 5 legitimately
    produced no Measurement) are persisted SEPARATELY in
    ``qc_decision_records``, which anchors to ``contour_revisions``
    instead and never weakens the approval FK (milestone N7.1).

Payload integrity
    Every ``payload TEXT`` column holds exactly
    ``cmr_core.domain_types.canonical_json()`` output (sorted keys, compact
    separators, ``ensure_ascii=False``, ``allow_nan=False``).
    ``prediction_id``/``revision_id``/``measurement_id``/``approval_id`` are
    the Node 1 content-addressed identities and equal
    ``sha256(payload TEXT)``; ``edes_selections.selection_id`` is
    storage-computed as ``sha256(canonical projection TEXT)`` because that
    type carries no identity field. Writers assert the digest identity BEFORE
    every insert and fail closed otherwise; loaders verify
    ``sha256(stored TEXT) == stored id`` AND ``canonical_json(reconstructed)
    == stored TEXT`` byte-for-byte without ever returning partial data.

Transactions
    Writers use explicit ``BEGIN IMMEDIATE`` transactions with rollback on
    any failure; series-coherence guards run inside the same transaction;
    every referenced parent link (``parent_prediction_id`` /
    ``parent_revision_id``) is resolved to its stored series INSIDE that same
    transaction BEFORE the revision INSERT, so a cross-series parent splice is
    rejected with zero staged rows (one coherent revision chain per series);
    ``save_revision_with_measurement`` inserts the revision BEFORE the
    measurement in ONE transaction so an FK violation leaves zero partial
    rows. ``PRAGMA foreign_keys=ON`` executes on every connection before any
    other statement (read connections included). Journal mode stays at the
    SQLite default (no WAL sidecars).

Migration hatch protocol (contract R1 amendment D; exercised by N7.1, S2C-001)
    ``PRAGMA user_version`` is the SOLE migration mechanism. Fresh databases
    apply deterministic stepwise DDL inside one transaction during schema
    init: the v1 DDL first (stamping ``user_version = 1``), then the v1 -> v2
    hatch (additive ``qc_decision_records``), then the v2 -> v3 hatch — fresh,
    v1 and v2 databases take ONE code path, so every end state is structurally
    identical by construction. Each version step migrates INSIDE the same
    schema-init transaction following the fixed hatch order:
    ``DROP TRIGGER IF EXISTS`` -> transform tables -> recreate triggers ->
    recreate NEW-table triggers -> set ``user_version``. The N7.1 step is
    purely additive. The S2C-001 step (v2 -> v3) adds the three append-only
    identity tables ``dicom_source_manifests`` / ``dicom_source_files`` /
    ``slice_inclusion_sets``, nullable workflow/selection/digest binding
    columns on the five clinical tables, per-series store-sequence backfill
    (original ``rowid`` order) with unique indexes, and sequence-guard insert
    triggers; no original column, payload byte, id or trigger semantic is
    dropped or altered — any failure rolls the whole hatch (schema, data,
    triggers, version stamp) back together. The CFM-001 step (v3 -> v4) is
    strictly additive: ``slice_inclusion_sets.inclusion_sequence`` store
    metadata backfilled by original insertion order within ``(workflow_id,
    series_uid)``, the unique ``slice_inclusion_sets_workflow_sequence``
    index plus a matching sequence-guard insert trigger, and the
    ``measurements.range_table_version`` TEXT column; every pre-existing
    payload byte, id and original column stays byte-identical. The S3-001
    step (v4 -> v5) is strictly additive per the Additive Hatch Doctrine: it
    creates the three append-only tables ``model_runs`` /
    ``whole_cine_predictions`` / ``whole_cine_revisions`` with their full
    guard sets, source/run coupling CHECK and linear-chain unique index.
    The S3-004E step (v6 -> v7) is strictly additive: it creates the
    append-only registration table ``cine_workflow_registrations`` with the
    composite ``(workflow_id, sequence)`` primary key, hex64 CHECKs, the
    foreign key into ``dicom_source_manifests`` and the full no-update /
    no-delete / no-replace guard set. Sequential explicit stamping (spec
    §3.2): every intermediate version is stamped in order — legacy
    databases walk 1 -> ... -> 4 explicitly BEFORE the v5 hatch executes,
    then the additive v6 hatch stamps 6 and the additive v7 hatch stamps 7;
    fresh installs run the whole sequence in one transaction ending stamped
    at 12. The F14 v11 -> v12 hatch is strictly additive as well: it creates
    the append-only ``selected_phase_analyses`` binding table and its guard
    triggers. Re-running init against a v12 database is a no-op. Runtime code
    paths never drop triggers: opening a database stamped with an unknown
    future ``user_version > 12``
    fails closed here until that future migration is authored.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar, cast

from cmr_core.domain_types import (
    ApprovalSnapshot,
    ContourRevision,
    EDESSelection,
    Measurement,
    ModelRunRecord,
    Prediction,
    SelectedPhasePrediction,
    TopologyBlock,
    WholeCinePrediction,
    WholeCineRevision,
    canonical_digest,
    canonical_json,
)
from cmr_core.errors import INVALID_ARGUMENT, TOPOLOGY_BLOCKED, CMRError
from cmr_core.provenance import Provenance
from cmr_core.segmentation import Structure
from cmr_core.source_manifest import DicomSourceFile, DicomSourceManifest
from cmr_quantification.measurements import validate_complete_grid_entries
from inference.failures import (
    REGISTRY_BUNDLE_UNKNOWN,
    REGISTRY_CONFLICT,
    REGISTRY_ROLLBACK_UNAVAILABLE,
    REGISTRY_STATE_CONFLICT,
)

from cmr_persistence.reconstruct import (
    approval_snapshot_from_payload,
    contour_revision_from_payload,
    edes_selection_from_payload,
    edes_selection_payload,
    edes_selection_payload_text,
    measurement_from_payload,
    model_run_from_payload,
    qc_report_from_payload,
    selected_phase_frame_indices_json,
    selected_phase_prediction_from_payload,
    selection_id_from_payload_text,
    whole_cine_edited_keys_json,
    whole_cine_prediction_from_payload,
    whole_cine_revision_from_payload,
)
from cmr_persistence.records import (
    CINE_WORKFLOW_REGISTRATION_RECORD_SCHEMA,
    CineWorkflowRegistrationV1,
    ModelActivationRecord,
    ModelBundleRecord,
    ModelRunRow,
    QCDecisionRecord,
    RegistryState,
    SelectedPhaseAnalysisRecord,
    SelectedPhasePredictionRow,
    SliceInclusionFrameRole,
    SliceInclusionSet,
    TopologyDispositionRecord,
    WholeCinePredictionRow,
    WholeCineRevisionRow,
    analysis_record_from_payload,
    applicable_topology_dispositions,
    chain_key_authoring_times,
    qc_decision_from_payload,
    registration_issuance_token,
    slice_inclusion_set_from_payload,
    topology_disposition_from_payload,
    topology_disposition_set_digest,
)

SCHEMA_USER_VERSION = 12

ANALYSIS_JOB_KINDS = ("selected_phase_recompute",)
ANALYSIS_JOB_STATES = ("queued", "running", "succeeded", "failed", "cancelled")
INFERENCE_JOB_STATES = ANALYSIS_JOB_STATES


def _normalize_analysis_timestamp(value: str) -> str:
    """Normalize an analysis-job instant to fixed-format UTC ISO-8601."""
    if not isinstance(value, str):
        raise ValueError("analysis timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("analysis timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("analysis timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class AnalysisJobRecord:
    job_id: str
    series_uid: str
    kind: str
    dedup_key: str
    request_payload: str
    state: str
    error_code: str | None
    artifact_measurement_id: str | None
    timings: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class InferenceJobRecord:
    job_id: str
    series_uid: str
    state: str
    error_code: str | None
    stage1_checkpoint: str
    stage2_checkpoint: str
    device: str
    selected_frames: str
    prediction_id: str | None
    model_run_id: str | None
    request_payload: str
    timings: str
    created_at: str
    updated_at: str


T = TypeVar("T")

_APPEND_ONLY = "append-only"

TABLES = (
    "predictions",
    "contour_revisions",
    "edes_selections",
    "measurements",
    "approval_snapshots",
)
# NOTE (N7.1): ``qc_decision_records`` is deliberately NOT part of ``TABLES``.
# That tuple enumerates the five Node 1/Node 2 CLINICAL record tables whose
# payloads carry Measurement-bound identities; existing immutability tests
# enumerate exactly those five. The decision table is separate by design and
# is covered by tests/unit/test_persistence_qc_decisions.py.

# Deterministic, stepwise DDL applied in order during schema init.
_SCHEMA_V1_DDL: tuple[str, ...] = (
    # --- predictions ---------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id TEXT PRIMARY KEY CHECK(length(prediction_id) = 64),
        payload TEXT NOT NULL
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS predictions_no_update
    BEFORE UPDATE ON predictions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS predictions_no_delete
    BEFORE DELETE ON predictions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS predictions_no_replace
    BEFORE INSERT ON predictions
    WHEN EXISTS (
        SELECT 1 FROM predictions WHERE prediction_id = NEW.prediction_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # --- contour_revisions ----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS contour_revisions (
        revision_id TEXT PRIMARY KEY CHECK(length(revision_id) = 64),
        series_uid TEXT NOT NULL,
        version INTEGER NOT NULL CHECK(version >= 1),
        parent_prediction_id TEXT REFERENCES predictions(prediction_id),
        parent_revision_id TEXT REFERENCES contour_revisions(revision_id),
        payload TEXT NOT NULL,
        UNIQUE(series_uid, version)
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS contour_revisions_no_update
    BEFORE UPDATE ON contour_revisions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS contour_revisions_no_delete
    BEFORE DELETE ON contour_revisions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS contour_revisions_no_replace_id
    BEFORE INSERT ON contour_revisions
    WHEN EXISTS (
        SELECT 1 FROM contour_revisions WHERE revision_id = NEW.revision_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # One linear chain per series BY DESIGN (amendment E): re-using a version
    # slot aborts even under INSERT OR REPLACE.
    f"""
    CREATE TRIGGER IF NOT EXISTS contour_revisions_no_fork_version
    BEFORE INSERT ON contour_revisions
    WHEN EXISTS (
        SELECT 1 FROM contour_revisions
        WHERE series_uid = NEW.series_uid AND version = NEW.version
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # --- edes_selections --------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS edes_selections (
        selection_id TEXT PRIMARY KEY CHECK(length(selection_id) = 64),
        series_uid TEXT NOT NULL,
        selected_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS edes_selections_no_update
    BEFORE UPDATE ON edes_selections
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS edes_selections_no_delete
    BEFORE DELETE ON edes_selections
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS edes_selections_no_replace
    BEFORE INSERT ON edes_selections
    WHEN EXISTS (
        SELECT 1 FROM edes_selections WHERE selection_id = NEW.selection_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # --- measurements -------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS measurements (
        measurement_id TEXT PRIMARY KEY CHECK(length(measurement_id) = 64),
        series_uid TEXT NOT NULL,
        source_revision_id TEXT NOT NULL REFERENCES contour_revisions(revision_id),
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS measurements_no_update
    BEFORE UPDATE ON measurements
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS measurements_no_delete
    BEFORE DELETE ON measurements
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS measurements_no_replace
    BEFORE INSERT ON measurements
    WHEN EXISTS (
        SELECT 1 FROM measurements WHERE measurement_id = NEW.measurement_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # --- approval_snapshots -----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS approval_snapshots (
        approval_id TEXT PRIMARY KEY CHECK(length(approval_id) = 64),
        series_uid TEXT NOT NULL,
        source_revision_id TEXT NOT NULL REFERENCES contour_revisions(revision_id),
        measurement_id TEXT NOT NULL REFERENCES measurements(measurement_id),
        approved_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS approval_snapshots_no_update
    BEFORE UPDATE ON approval_snapshots
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS approval_snapshots_no_delete
    BEFORE DELETE ON approval_snapshots
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS approval_snapshots_no_replace
    BEFORE INSERT ON approval_snapshots
    WHEN EXISTS (
        SELECT 1 FROM approval_snapshots WHERE approval_id = NEW.approval_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
)

# N7.1 migration hatch (v1 -> v2), applied in order inside the schema-init
# transaction. Fixed hatch order: DROP TRIGGER IF EXISTS -> transform tables
# -> recreate triggers; the caller stamps ``user_version`` afterwards.
#
# The step is purely ADDITIVE: a separate decision table for B-LITE sentinel
# approval decisions plus its own three append-only triggers (same pattern as
# every other record table). The immutable clinical ``approval_snapshots``
# table and its foreign key to ``measurements`` are untouched, so no existing
# trigger references an object this step creates and the DROP step is a
# no-op by construction — it exists to keep the hatch protocol literal for
# future steps that DO transform existing objects.
_SCHEMA_V2_HATCH_DDL: tuple[str, ...] = (
    # --- hatch step 1: drop this step's triggers if re-run -------------------
    """
    DROP TRIGGER IF EXISTS qc_decision_records_no_update
    """,
    """
    DROP TRIGGER IF EXISTS qc_decision_records_no_delete
    """,
    """
    DROP TRIGGER IF EXISTS qc_decision_records_no_replace
    """,
    # --- hatch step 2: transform (additive table creation) -------------------
    """
    CREATE TABLE IF NOT EXISTS qc_decision_records (
        decision_id TEXT PRIMARY KEY CHECK(length(decision_id) = 64),
        series_uid TEXT NOT NULL,
        source_revision_id TEXT NOT NULL REFERENCES contour_revisions(revision_id),
        ed_frame INTEGER NOT NULL,
        es_frame INTEGER NOT NULL,
        qc_verdict TEXT NOT NULL CHECK(qc_verdict IN ('PASS','REVIEW REQUIRED')),
        approved_by TEXT NOT NULL,
        approved_at TEXT NOT NULL,
        override_reason TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    # --- hatch step 3: recreate this step's append-only triggers -------------
    f"""
    CREATE TRIGGER IF NOT EXISTS qc_decision_records_no_update
    BEFORE UPDATE ON qc_decision_records
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS qc_decision_records_no_delete
    BEFORE DELETE ON qc_decision_records
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS qc_decision_records_no_replace
    BEFORE INSERT ON qc_decision_records
    WHEN EXISTS (
        SELECT 1 FROM qc_decision_records WHERE decision_id = NEW.decision_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
)

# S2C-001 migration hatch (v2 -> v3), applied in order inside the schema-init
# transaction. Fixed hatch order per design spec §3.4:
#   1. drop the existing edes_selections_*/measurements_* append-only triggers
#      (recreated verbatim in step 7);
#   2. defensively drop any v3 trigger names (idempotent hatch re-entry);
#   3. create the three new append-only identity tables;
#   4. add the nullable binding columns (nullable SOLELY for migrated v2 rows;
#      store APIs require them for manifest-backed v3 writes);
#   5. backfill per-series store sequences from original rowid insertion order;
#   6. create the unique sequence indexes;
#   7. recreate every dropped trigger plus the new-table and sequence-guard
#      triggers;
# then the caller stamps ``user_version = 3`` and commits. Any failure rolls
# back schema, data, trigger and version changes together.
_SCHEMA_V3_HATCH_DDL: tuple[str, ...] = (
    # --- hatch step 1: drop existing triggers that step 7 recreates ----------
    """
    DROP TRIGGER IF EXISTS edes_selections_no_update
    """,
    """
    DROP TRIGGER IF EXISTS edes_selections_no_delete
    """,
    """
    DROP TRIGGER IF EXISTS edes_selections_no_replace
    """,
    """
    DROP TRIGGER IF EXISTS measurements_no_update
    """,
    """
    DROP TRIGGER IF EXISTS measurements_no_delete
    """,
    """
    DROP TRIGGER IF EXISTS measurements_no_replace
    """,
    # --- hatch step 2: defensively drop this step's own trigger names --------
    """
    DROP TRIGGER IF EXISTS dicom_source_manifests_no_update
    """,
    """
    DROP TRIGGER IF EXISTS dicom_source_manifests_no_delete
    """,
    """
    DROP TRIGGER IF EXISTS dicom_source_manifests_no_replace
    """,
    """
    DROP TRIGGER IF EXISTS dicom_source_files_no_update
    """,
    """
    DROP TRIGGER IF EXISTS dicom_source_files_no_delete
    """,
    """
    DROP TRIGGER IF EXISTS dicom_source_files_no_replace_id
    """,
    """
    DROP TRIGGER IF EXISTS dicom_source_files_no_replace_path
    """,
    """
    DROP TRIGGER IF EXISTS dicom_source_files_no_replace_sop
    """,
    """
    DROP TRIGGER IF EXISTS slice_inclusion_sets_no_update
    """,
    """
    DROP TRIGGER IF EXISTS slice_inclusion_sets_no_delete
    """,
    """
    DROP TRIGGER IF EXISTS slice_inclusion_sets_no_replace
    """,
    """
    DROP TRIGGER IF EXISTS edes_selections_no_replace_sequence
    """,
    """
    DROP TRIGGER IF EXISTS measurements_no_replace_sequence
    """,
    # --- hatch step 3: new identity tables ------------------------------------
    """
    CREATE TABLE IF NOT EXISTS dicom_source_manifests (
        workflow_id TEXT PRIMARY KEY
            CHECK(length(workflow_id) = 64
                  AND workflow_id NOT GLOB '*[^0-9a-f]*'),
        study_uid TEXT NOT NULL CHECK(length(trim(study_uid)) > 0),
        series_uid TEXT NOT NULL CHECK(length(trim(series_uid)) > 0),
        cine_volume_digest TEXT NOT NULL
            CHECK(length(cine_volume_digest) = 64
                  AND cine_volume_digest NOT GLOB '*[^0-9a-f]*'),
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dicom_source_files (
        workflow_id TEXT NOT NULL
            REFERENCES dicom_source_manifests(workflow_id),
        ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
        relative_path TEXT NOT NULL CHECK(length(trim(relative_path)) > 0),
        sop_instance_uid TEXT NOT NULL CHECK(length(trim(sop_instance_uid)) > 0),
        file_sha256 TEXT NOT NULL
            CHECK(length(file_sha256) = 64
                  AND file_sha256 NOT GLOB '*[^0-9a-f]*'),
        PRIMARY KEY(workflow_id, ordinal),
        UNIQUE(workflow_id, relative_path),
        UNIQUE(workflow_id, sop_instance_uid)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS slice_inclusion_sets (
        inclusion_set_id TEXT PRIMARY KEY
            CHECK(length(inclusion_set_id) = 64
                  AND inclusion_set_id NOT GLOB '*[^0-9a-f]*'),
        workflow_id TEXT
            REFERENCES dicom_source_manifests(workflow_id),
        series_uid TEXT NOT NULL,
        source_revision_id TEXT NOT NULL
            REFERENCES contour_revisions(revision_id),
        selection_id TEXT NOT NULL
            REFERENCES edes_selections(selection_id),
        selected_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    # --- hatch step 4: additive nullable binding columns -----------------------
    """
    ALTER TABLE predictions
    ADD COLUMN workflow_id TEXT
        REFERENCES dicom_source_manifests(workflow_id)
        CHECK(workflow_id IS NULL OR length(workflow_id) = 64)
    """,
    """
    ALTER TABLE contour_revisions
    ADD COLUMN workflow_id TEXT
        REFERENCES dicom_source_manifests(workflow_id)
        CHECK(workflow_id IS NULL OR length(workflow_id) = 64)
    """,
    """
    ALTER TABLE edes_selections
    ADD COLUMN workflow_id TEXT
        REFERENCES dicom_source_manifests(workflow_id)
        CHECK(workflow_id IS NULL OR length(workflow_id) = 64)
    """,
    """
    ALTER TABLE edes_selections
    ADD COLUMN selection_sequence INTEGER
        CHECK(selection_sequence IS NULL OR selection_sequence >= 1)
    """,
    """
    ALTER TABLE measurements
    ADD COLUMN workflow_id TEXT
        REFERENCES dicom_source_manifests(workflow_id)
        CHECK(workflow_id IS NULL OR length(workflow_id) = 64)
    """,
    """
    ALTER TABLE measurements
    ADD COLUMN selection_id TEXT
        REFERENCES edes_selections(selection_id)
        CHECK(selection_id IS NULL OR length(selection_id) = 64)
    """,
    """
    ALTER TABLE measurements
    ADD COLUMN inclusion_set_id TEXT
        REFERENCES slice_inclusion_sets(inclusion_set_id)
        CHECK(inclusion_set_id IS NULL OR length(inclusion_set_id) = 64)
    """,
    """
    ALTER TABLE measurements
    ADD COLUMN measurement_sequence INTEGER
        CHECK(measurement_sequence IS NULL OR measurement_sequence >= 1)
    """,
    """
    ALTER TABLE approval_snapshots
    ADD COLUMN workflow_id TEXT
        REFERENCES dicom_source_manifests(workflow_id)
        CHECK(workflow_id IS NULL OR length(workflow_id) = 64)
    """,
    """
    ALTER TABLE approval_snapshots
    ADD COLUMN selection_id TEXT
        REFERENCES edes_selections(selection_id)
        CHECK(selection_id IS NULL OR length(selection_id) = 64)
    """,
    """
    ALTER TABLE approval_snapshots
    ADD COLUMN qc_report_digest TEXT
        CHECK(qc_report_digest IS NULL OR length(qc_report_digest) = 64)
    """,
    """
    ALTER TABLE approval_snapshots
    ADD COLUMN inclusion_set_id TEXT
        REFERENCES slice_inclusion_sets(inclusion_set_id)
        CHECK(inclusion_set_id IS NULL OR length(inclusion_set_id) = 64)
    """,
    """
    ALTER TABLE qc_decision_records
    ADD COLUMN workflow_id TEXT
        REFERENCES dicom_source_manifests(workflow_id)
        CHECK(workflow_id IS NULL OR length(workflow_id) = 64)
    """,
    """
    ALTER TABLE qc_decision_records
    ADD COLUMN selection_id TEXT
        REFERENCES edes_selections(selection_id)
        CHECK(selection_id IS NULL OR length(selection_id) = 64)
    """,
    # --- hatch step 5: store-sequence backfill (original rowid order) ---------
    """
    UPDATE edes_selections SET selection_sequence = (
        SELECT COUNT(*) FROM edes_selections AS earlier
        WHERE earlier.series_uid = edes_selections.series_uid
              AND earlier.rowid <= edes_selections.rowid
    )
    """,
    """
    UPDATE measurements SET measurement_sequence = (
        SELECT COUNT(*) FROM measurements AS earlier
        WHERE earlier.series_uid = measurements.series_uid
              AND earlier.rowid <= measurements.rowid
    )
    """,
    # --- hatch step 6: unique sequence indexes ---------------------------------
    """
    CREATE UNIQUE INDEX IF NOT EXISTS edes_selections_series_sequence
    ON edes_selections(series_uid, selection_sequence)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS measurements_series_sequence
    ON measurements(series_uid, measurement_sequence)
    """,
    # --- hatch step 7: recreate dropped + new-table + guard triggers -----------
    f"""
    CREATE TRIGGER IF NOT EXISTS edes_selections_no_update
    BEFORE UPDATE ON edes_selections
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS edes_selections_no_delete
    BEFORE DELETE ON edes_selections
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS edes_selections_no_replace
    BEFORE INSERT ON edes_selections
    WHEN EXISTS (
        SELECT 1 FROM edes_selections WHERE selection_id = NEW.selection_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS measurements_no_update
    BEFORE UPDATE ON measurements
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS measurements_no_delete
    BEFORE DELETE ON measurements
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS measurements_no_replace
    BEFORE INSERT ON measurements
    WHEN EXISTS (
        SELECT 1 FROM measurements WHERE measurement_id = NEW.measurement_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # INSERT OR REPLACE must not delete the row occupying a store sequence
    # (mirrors contour_revisions_no_fork_version).
    f"""
    CREATE TRIGGER IF NOT EXISTS edes_selections_no_replace_sequence
    BEFORE INSERT ON edes_selections
    WHEN EXISTS (
        SELECT 1 FROM edes_selections
        WHERE series_uid = NEW.series_uid
              AND selection_sequence = NEW.selection_sequence
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS measurements_no_replace_sequence
    BEFORE INSERT ON measurements
    WHEN EXISTS (
        SELECT 1 FROM measurements
        WHERE series_uid = NEW.series_uid
              AND measurement_sequence = NEW.measurement_sequence
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS dicom_source_manifests_no_update
    BEFORE UPDATE ON dicom_source_manifests
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS dicom_source_manifests_no_delete
    BEFORE DELETE ON dicom_source_manifests
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # Identity predicate: an existing workflow_id aborts even under
    # INSERT OR REPLACE.
    f"""
    CREATE TRIGGER IF NOT EXISTS dicom_source_manifests_no_replace
    BEFORE INSERT ON dicom_source_manifests
    WHEN EXISTS (
        SELECT 1 FROM dicom_source_manifests
        WHERE workflow_id = NEW.workflow_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS dicom_source_files_no_update
    BEFORE UPDATE ON dicom_source_files
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS dicom_source_files_no_delete
    BEFORE DELETE ON dicom_source_files
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # Identity predicates: (workflow_id, ordinal), (workflow_id,
    # relative_path), (workflow_id, sop_instance_uid).
    f"""
    CREATE TRIGGER IF NOT EXISTS dicom_source_files_no_replace_id
    BEFORE INSERT ON dicom_source_files
    WHEN EXISTS (
        SELECT 1 FROM dicom_source_files
        WHERE workflow_id = NEW.workflow_id AND ordinal = NEW.ordinal
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS dicom_source_files_no_replace_path
    BEFORE INSERT ON dicom_source_files
    WHEN EXISTS (
        SELECT 1 FROM dicom_source_files
        WHERE workflow_id = NEW.workflow_id AND relative_path = NEW.relative_path
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS dicom_source_files_no_replace_sop
    BEFORE INSERT ON dicom_source_files
    WHEN EXISTS (
        SELECT 1 FROM dicom_source_files
        WHERE workflow_id = NEW.workflow_id
              AND sop_instance_uid = NEW.sop_instance_uid
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS slice_inclusion_sets_no_update
    BEFORE UPDATE ON slice_inclusion_sets
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS slice_inclusion_sets_no_delete
    BEFORE DELETE ON slice_inclusion_sets
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # Identity predicate: an existing inclusion_set_id aborts.
    f"""
    CREATE TRIGGER IF NOT EXISTS slice_inclusion_sets_no_replace
    BEFORE INSERT ON slice_inclusion_sets
    WHEN EXISTS (
        SELECT 1 FROM slice_inclusion_sets
        WHERE inclusion_set_id = NEW.inclusion_set_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
)

# CFM-001 migration hatch (v3 -> v4), applied in order inside the
# schema-init transaction. Fixed hatch order: DROP TRIGGER IF EXISTS ->
# transform tables -> recreate triggers; the caller stamps ``user_version``
# afterwards. Strictly additive per spec §1.7:
#   1. defensively drop this step's own index/trigger names (hatch re-entry);
#   2. add ``slice_inclusion_sets.inclusion_sequence`` (store metadata,
#      NULL legal only transiently during backfill) and the trailing
#      ``measurements.range_table_version`` TEXT column;
#   3. backfill ``inclusion_sequence`` by ORIGINAL INSERTION ORDER within
#      ``(workflow_id, series_uid)`` — NULL-safe via SQLite's ``IS``
#      operator so unbound rows group together;
#   4. create the unique ``slice_inclusion_sets_workflow_sequence`` index
#      and the sequence-guard insert trigger.
# Original columns, payload bytes and ids are untouched; any failure rolls
# schema, data, trigger and version changes back together.
_SCHEMA_V4_HATCH_DDL: tuple[str, ...] = (
    # --- hatch step 1: drop this step's own names if re-run ------------------
    """
    DROP INDEX IF EXISTS slice_inclusion_sets_workflow_sequence
    """,
    """
    DROP TRIGGER IF EXISTS slice_inclusion_sets_no_replace_sequence
    """,
    # The range_table_version sync below is an UPDATE; the append-only
    # trigger is dropped first and recreated VERBATIM at the end (the same
    # protocol the v2->v3 hatch uses for its sequence backfills).
    """
    DROP TRIGGER IF EXISTS measurements_no_update
    """,
    """
    DROP TRIGGER IF EXISTS slice_inclusion_sets_no_update
    """,
    # --- hatch step 2: additive columns ---------------------------------------
    """
    ALTER TABLE slice_inclusion_sets
    ADD COLUMN inclusion_sequence INTEGER
        CHECK(inclusion_sequence IS NULL OR inclusion_sequence >= 1)
    """,
    # Backfill by ORIGINAL INSERTION ORDER within (workflow_id, series_uid)
    # — NULL-safe via SQLite's ``IS`` operator so unbound rows group
    # together (spec §1.7). The table carries no UPDATE trigger.
    """
    UPDATE slice_inclusion_sets SET inclusion_sequence = (
        SELECT COUNT(*) FROM slice_inclusion_sets AS earlier
        WHERE earlier.series_uid = slice_inclusion_sets.series_uid
              AND (earlier.workflow_id IS slice_inclusion_sets.workflow_id)
              AND earlier.rowid <= slice_inclusion_sets.rowid
    )
    """,
    """
    ALTER TABLE measurements
    ADD COLUMN range_table_version TEXT
    """,
    # Column/payload equality (spec §2.6) is established AT MIGRATION TIME
    # from the canonical bytes themselves: legacy payloads without the key
    # stay NULL; any payload already carrying the field syncs exactly.
    """
    UPDATE measurements SET range_table_version =
        json_extract(measurements.payload, '$.range_table_version')
    """,
    # --- hatch step 3: unique sequence index + guard triggers -----------------
    """
    CREATE UNIQUE INDEX slice_inclusion_sets_workflow_sequence
    ON slice_inclusion_sets(workflow_id, series_uid, inclusion_sequence)
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS slice_inclusion_sets_no_replace_sequence
    BEFORE INSERT ON slice_inclusion_sets
    WHEN EXISTS (
        SELECT 1 FROM slice_inclusion_sets
        WHERE series_uid = NEW.series_uid
              AND (workflow_id IS NEW.workflow_id)
              AND inclusion_sequence = NEW.inclusion_sequence
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS measurements_no_update
    BEFORE UPDATE ON measurements
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS slice_inclusion_sets_no_update
    BEFORE UPDATE ON slice_inclusion_sets
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
)

# S3-001 migration hatch (v4 -> v5), applied in order inside the schema-init
# transaction. Fixed hatch order: idempotent DROP guards for every v5-owned
# object at the head -> additive table creation -> triggers -> indices; the
# caller stamps ``user_version`` afterwards. STRICTLY ADDITIVE per design
# spec §3.1/§3.2 (Additive Hatch Doctrine): three new append-only tables
# (``model_runs``, ``whole_cine_predictions``, ``whole_cine_revisions``) with
# the full no-update/no-delete/no-replace guard set, the source/run coupling
# CHECK mirroring the v4 Prediction rule at SQL level, 36/64-length CHECKs,
# the linear-chain unique ``(series_uid, version)`` index, and series lookup
# indices. NO v4 object is altered or dropped: every pre-existing payload
# byte, id, column and trigger semantic stays byte-identical; any failure
# rolls schema, trigger and version changes back together.
_SCHEMA_V5_HATCH_DDL: tuple[str, ...] = (
    # --- hatch step 0: idempotent DROP guards for v5-owned objects -----------
    """
    DROP INDEX IF EXISTS whole_cine_revisions_series_version
    """,
    """
    DROP TRIGGER IF EXISTS whole_cine_revisions_no_replace_version
    """,
    """
    DROP TRIGGER IF EXISTS whole_cine_revisions_no_replace_id
    """,
    """
    DROP TRIGGER IF EXISTS whole_cine_revisions_no_update
    """,
    """
    DROP TRIGGER IF EXISTS whole_cine_revisions_no_delete
    """,
    """
    DROP INDEX IF EXISTS idx_whole_cine_predictions_series
    """,
    """
    DROP TRIGGER IF EXISTS whole_cine_predictions_no_replace
    """,
    """
    DROP TRIGGER IF EXISTS whole_cine_predictions_no_update
    """,
    """
    DROP TRIGGER IF EXISTS whole_cine_predictions_no_delete
    """,
    """
    DROP INDEX IF EXISTS idx_model_runs_series
    """,
    """
    DROP TRIGGER IF EXISTS model_runs_no_replace
    """,
    """
    DROP TRIGGER IF EXISTS model_runs_no_update
    """,
    """
    DROP TRIGGER IF EXISTS model_runs_no_delete
    """,
    # --- hatch step 1: additive tables ----------------------------------------
    # 1. Model runs registry table (append-only)
    """
    CREATE TABLE IF NOT EXISTS model_runs (
        model_run_id TEXT PRIMARY KEY CHECK(length(model_run_id) = 36),
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        model_sha256 TEXT NOT NULL CHECK(length(model_sha256) = 64),
        framework TEXT NOT NULL,
        study_uid TEXT NOT NULL,
        series_uid TEXT NOT NULL,
        input_content_hash TEXT NOT NULL CHECK(length(input_content_hash) = 64),
        mask_artifact_sha256 TEXT NOT NULL CHECK(length(mask_artifact_sha256) = 64),
        uq_method TEXT,
        payload TEXT NOT NULL,
        record_digest TEXT NOT NULL CHECK(length(record_digest) = 64),
        created_at TEXT NOT NULL
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS model_runs_no_update
    BEFORE UPDATE ON model_runs
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS model_runs_no_delete
    BEFORE DELETE ON model_runs
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # no-replace guard: INSERT OR REPLACE must never silently overwrite
    f"""
    CREATE TRIGGER IF NOT EXISTS model_runs_no_replace
    BEFORE INSERT ON model_runs
    WHEN EXISTS (
        SELECT 1 FROM model_runs WHERE model_run_id = NEW.model_run_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_model_runs_series
    ON model_runs(study_uid, series_uid, created_at)
    """,
    # 2. Whole-cine prediction artifacts table (append-only)
    # NOTE (S3-001 documented deviation): design spec §3.1 writes the
    # table-level source/run coupling CHECK BETWEEN column definitions
    # (before ``slice_count``). SQLite's grammar rejects any column
    # definition that follows a table constraint ("near \"slice_count\":
    # syntax error", VERIFIED_BY_EXECUTION), so the table-level CHECK is
    # placed AFTER all column definitions below. Column set, constraint set,
    # FK targets and trigger semantics are otherwise transcribed exactly;
    # no semantic property changes.
    """
    CREATE TABLE IF NOT EXISTS whole_cine_predictions (
        prediction_id TEXT PRIMARY KEY CHECK(length(prediction_id) = 64),
        study_uid TEXT NOT NULL,
        series_uid TEXT NOT NULL,
        source TEXT NOT NULL CHECK(source IN ('ai', 'manual-seed')),
        model_run_id TEXT REFERENCES model_runs(model_run_id),
        slice_count INTEGER NOT NULL CHECK(slice_count >= 1),
        frame_count INTEGER NOT NULL CHECK(frame_count >= 1),
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        -- source/run coupling mirrors the v4 Prediction rule at SQL level (F3 AGY)
        CHECK(
            (source = 'ai' AND model_run_id IS NOT NULL)
            OR (source = 'manual-seed' AND model_run_id IS NULL)
        )
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS whole_cine_predictions_no_update
    BEFORE UPDATE ON whole_cine_predictions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS whole_cine_predictions_no_delete
    BEFORE DELETE ON whole_cine_predictions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS whole_cine_predictions_no_replace
    BEFORE INSERT ON whole_cine_predictions
    WHEN EXISTS (
        SELECT 1 FROM whole_cine_predictions WHERE prediction_id = NEW.prediction_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_whole_cine_predictions_series
    ON whole_cine_predictions(series_uid, created_at)
    """,
    # 3. Whole-cine revisions (append-only, linear chain per series)
    """
    CREATE TABLE IF NOT EXISTS whole_cine_revisions (
        revision_id TEXT PRIMARY KEY CHECK(length(revision_id) = 64),
        study_uid TEXT NOT NULL,
        series_uid TEXT NOT NULL,
        version INTEGER NOT NULL CHECK(version >= 1),
        parent_prediction_id TEXT NOT NULL REFERENCES whole_cine_predictions(prediction_id),
        parent_revision_id TEXT REFERENCES whole_cine_revisions(revision_id),
        editor TEXT NOT NULL,
        edited_keys_json TEXT NOT NULL, -- canonical JSON array of "slice:frame:structure" strings
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS whole_cine_revisions_no_update
    BEFORE UPDATE ON whole_cine_revisions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS whole_cine_revisions_no_delete
    BEFORE DELETE ON whole_cine_revisions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS whole_cine_revisions_no_replace_id
    BEFORE INSERT ON whole_cine_revisions
    WHEN EXISTS (
        SELECT 1 FROM whole_cine_revisions WHERE revision_id = NEW.revision_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # Review F1 fix: the id-only no-replace guard misses the case where
    # INSERT OR REPLACE supplies a NEW revision_id for an OCCUPIED
    # (series_uid, version) slot — the unique index then performs a
    # REPLACE-delete of the persisted row instead of aborting. This fork
    # guard (mirroring contour_revisions_no_fork_version) fails such
    # replace-style inserts closed at schema level.
    f"""
    CREATE TRIGGER IF NOT EXISTS whole_cine_revisions_no_replace_version
    BEFORE INSERT ON whole_cine_revisions
    WHEN EXISTS (
        SELECT 1 FROM whole_cine_revisions
        WHERE series_uid = NEW.series_uid AND version = NEW.version
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # linear-chain guard: same (series_uid, version) can never be re-issued
    # via a replace-style insert.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS whole_cine_revisions_series_version
    ON whole_cine_revisions(series_uid, version)
    """,
)

# S3-003 migration hatch (v5 -> v6).  This is additive by construction: the
# v1..v5 objects are not altered, and every new registry table has explicit
# no-update/no-delete/no-replace protection.  The registry-state row is the
# sole intentionally updated row and its writer performs the expected-pointer
# CAS in Python inside one transaction.
_SCHEMA_V6_HATCH_DDL: tuple[str, ...] = (
    "DROP TRIGGER IF EXISTS model_bundles_no_update",
    "DROP TRIGGER IF EXISTS model_bundles_no_delete",
    "DROP TRIGGER IF EXISTS model_bundles_no_replace",
    "DROP TRIGGER IF EXISTS model_activations_no_update",
    "DROP TRIGGER IF EXISTS model_activations_no_delete",
    "DROP TRIGGER IF EXISTS model_activations_no_replace",
    "DROP TRIGGER IF EXISTS model_registry_state_no_delete",
    "DROP TRIGGER IF EXISTS model_registry_state_no_replace",
    """
    CREATE TABLE IF NOT EXISTS model_bundles (
        bundle_id TEXT PRIMARY KEY CHECK(length(bundle_id) = 64),
        model_id TEXT NOT NULL,
        semantic_version TEXT NOT NULL,
        weights_format TEXT NOT NULL CHECK(weights_format IN ('safetensors-v1','onnx-v1')),
        payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256) = 64),
        manifest_json TEXT NOT NULL,
        install_root_relpath TEXT NOT NULL,
        installed_at TEXT NOT NULL,
        software_version TEXT NOT NULL,
        digest TEXT NOT NULL CHECK(length(digest) = 64)
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS model_bundles_no_update
    BEFORE UPDATE ON model_bundles BEGIN SELECT RAISE(ABORT, '{_APPEND_ONLY}'); END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS model_bundles_no_delete
    BEFORE DELETE ON model_bundles BEGIN SELECT RAISE(ABORT, '{_APPEND_ONLY}'); END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS model_bundles_no_replace
    BEFORE INSERT ON model_bundles WHEN EXISTS
    (SELECT 1 FROM model_bundles WHERE bundle_id = NEW.bundle_id)
    BEGIN SELECT RAISE(ABORT, '{_APPEND_ONLY}'); END
    """,
    """
    CREATE TABLE IF NOT EXISTS model_activations (
        activation_id TEXT PRIMARY KEY CHECK(length(activation_id) = 64),
        bundle_id TEXT NOT NULL REFERENCES model_bundles(bundle_id),
        previous_activation_id TEXT REFERENCES model_activations(activation_id),
        reason TEXT NOT NULL CHECK(length(reason) > 0),
        activated_at TEXT NOT NULL,
        digest TEXT NOT NULL CHECK(length(digest) = 64)
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS model_activations_no_update
    BEFORE UPDATE ON model_activations BEGIN SELECT RAISE(ABORT, '{_APPEND_ONLY}'); END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS model_activations_no_delete
    BEFORE DELETE ON model_activations BEGIN SELECT RAISE(ABORT, '{_APPEND_ONLY}'); END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS model_activations_no_replace
    BEFORE INSERT ON model_activations WHEN EXISTS
    (SELECT 1 FROM model_activations WHERE activation_id = NEW.activation_id)
    BEGIN SELECT RAISE(ABORT, '{_APPEND_ONLY}'); END
    """,
    """
    CREATE TABLE IF NOT EXISTS model_registry_state (
        id INTEGER PRIMARY KEY CHECK(id = 1),
        active_activation_id TEXT NOT NULL REFERENCES model_activations(activation_id),
        updated_at TEXT NOT NULL
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS model_registry_state_no_delete
    BEFORE DELETE ON model_registry_state BEGIN SELECT RAISE(ABORT, '{_APPEND_ONLY}'); END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS model_registry_state_no_replace
    BEFORE INSERT ON model_registry_state WHEN EXISTS
    (SELECT 1 FROM model_registry_state WHERE id = NEW.id)
    BEGIN SELECT RAISE(ABORT, '{_APPEND_ONLY}'); END
    """,
)


# S3-004E migration hatch (v6 -> v7).  Additive by construction: the v1..v6
# objects are not altered, and the new registration table has explicit
# no-update/no-delete/no-replace protection (composite (workflow_id, sequence)
# identity, hex64 CHECKs, FK into dicom_source_manifests).  S3-003 owns the
# v5 -> v6 hatch; this module owns ONLY v6 -> v7.
_SCHEMA_V7_HATCH_DDL: tuple[str, ...] = (
    "DROP TRIGGER IF EXISTS cine_workflow_registrations_no_update",
    "DROP TRIGGER IF EXISTS cine_workflow_registrations_no_delete",
    "DROP TRIGGER IF EXISTS cine_workflow_registrations_no_replace",
    """
    CREATE TABLE IF NOT EXISTS cine_workflow_registrations (
        workflow_id TEXT NOT NULL
            REFERENCES dicom_source_manifests(workflow_id)
            CHECK(length(workflow_id) = 64
                  AND workflow_id NOT GLOB '*[^0-9a-f]*'),
        sequence INTEGER NOT NULL,
        store_record_uri TEXT NOT NULL,
        record_sha256 TEXT NOT NULL
            CHECK(length(record_sha256) = 64
                  AND record_sha256 NOT GLOB '*[^0-9a-f]*'),
        record_bytes BLOB NOT NULL,
        PRIMARY KEY (workflow_id, sequence)
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS cine_workflow_registrations_no_update
    BEFORE UPDATE ON cine_workflow_registrations
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS cine_workflow_registrations_no_delete
    BEFORE DELETE ON cine_workflow_registrations
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS cine_workflow_registrations_no_replace
    BEFORE INSERT ON cine_workflow_registrations
    WHEN EXISTS (
        SELECT 1 FROM cine_workflow_registrations
        WHERE workflow_id = NEW.workflow_id AND sequence = NEW.sequence
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
)

# FNC-C migration hatch (v7 -> v8).  STRICTLY ADDITIVE by construction: no
# v1..v7 object is altered, dropped or re-created.  It adds ONE append-only
# table for the FNC-C sparse (selected-phase) prediction artifact, with the
# SAME append-only trigger protection pattern as ``whole_cine_predictions``
# (no UPDATE, no DELETE, no REPLACE), hex64 CHECKs on the identity and the
# source/run coupling CHECK, plus a canonical-JSON CHECK on the
# identity-significant ``source_frame_indices`` column.
#
# The canonical JSON form of ``source_frame_indices`` is the compact
# comma-separated bracketed list produced by
# :func:`selected_phase_frame_indices_json` (e.g. ``[0,12]``).  The SQL
# CHECKs pin that EXACT shape (valid JSON array of at least one element,
# only digits/commas inside the brackets); ordering, uniqueness, range and
# the "declared == derived contour frames" law are enforced by the
# ``SelectedPhasePrediction`` dataclass BEFORE any row can be staged and
# re-verified on every load, exactly as the rest of the schema splits
# SQL-level shape checks from payload-level invariants.
#
# Correction C1 (P2-5/P2-6), VERIFIED_BY_EXECUTION:
# * ``prediction_id`` is declared NOT NULL and pinned to lowercase hex64 with
#   the same GLOB style the other identity columns use. A bare
#   ``TEXT PRIMARY KEY`` in SQLite is nullable, so the previous length-only
#   CHECK accepted NULL and ``'z'*64``.
# * The payload-side coverage member is checked EXPLICITLY. Plain equality
#   against ``json_extract(payload, ...)`` is not enough: when the member is
#   absent (or JSON null) ``json_extract`` returns NULL and the comparison is
#   NULL, which is NOT a CHECK violation in SQLite. The added
#   ``IS NOT NULL`` (plus array-type and length checks on the EXTRACTED
#   value) rejects ``{}`` and ``{"source_frame_indices": null}`` outright.
# Hatch discipline: this is the same additive ``CREATE TABLE IF NOT EXISTS``
# step. A database that already has the v8 table (created before C1) keeps
# its original DDL; the payload reconstruction still fails closed there and
# new databases get the corrected constraints. No existing v8 object is
# altered or dropped.
_SCHEMA_V8_HATCH_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS selected_phase_predictions (
        prediction_id TEXT NOT NULL PRIMARY KEY
            CHECK(length(prediction_id) = 64
                  AND prediction_id NOT GLOB '*[^0-9a-f]*'),
        study_uid TEXT NOT NULL,
        series_uid TEXT NOT NULL,
        source TEXT NOT NULL CHECK(source IN ('ai', 'manual-seed')),
        model_run_id TEXT REFERENCES model_runs(model_run_id),
        slice_count INTEGER NOT NULL CHECK(slice_count >= 1),
        frame_count INTEGER NOT NULL CHECK(frame_count >= 1),
        source_frame_indices TEXT NOT NULL
            CHECK(json_valid(source_frame_indices) = 1)
            CHECK(json_type(source_frame_indices) = 'array')
            CHECK(json_array_length(source_frame_indices) >= 1),
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        -- source/run coupling mirrors the WholeCinePrediction rule at SQL level
        CHECK(
            (source = 'ai' AND model_run_id IS NOT NULL)
            OR (source = 'manual-seed' AND model_run_id IS NULL)
        ),
        -- the identity-significant coverage column can never be absent from
        -- the payload (NULL comparison loophole closed) and can never drift
        -- from it (exact canonical JSON equality)
        CHECK(json_extract(payload, '$.source_frame_indices') IS NOT NULL),
        CHECK(json_type(payload, '$.source_frame_indices') = 'array'),
        CHECK(json_array_length(payload, '$.source_frame_indices') >= 1),
        CHECK(source_frame_indices = json_extract(payload, '$.source_frame_indices'))
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS selected_phase_predictions_no_update
    BEFORE UPDATE ON selected_phase_predictions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS selected_phase_predictions_no_delete
    BEFORE DELETE ON selected_phase_predictions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS selected_phase_predictions_no_replace
    BEFORE INSERT ON selected_phase_predictions
    WHEN EXISTS (
        SELECT 1 FROM selected_phase_predictions WHERE prediction_id = NEW.prediction_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_selected_phase_predictions_series
    ON selected_phase_predictions(series_uid, created_at)
    """,
)

# FNC-E2b-2 C1: separate ADDITIVE table for user-requested sparse X1
# inference jobs (owner ruling 2026-09-12: NO analysis_jobs rebuild, NO v9
# kind-CHECK change). Lifecycle is queued->running->succeeded|failed|cancelled
# plus queued->cancelled. Terminal rows NEVER resurrect (A3): audit fields
# may be attached once via attach_inference_result without changing state.
# Current-eligibility (A1): a sparse artifact is current-eligible iff a
# SUCCEEDED inference_jobs row binds (prediction_id[, model_run_id],
# head_revision_id + selection_id) to the current head/selection. The
# request_payload carries the bound identity; the absolute weights root is
# NEVER persisted (A8: relative pinned names + digests only).
_SCHEMA_V10_HATCH_DDL: tuple[str, ...] = (
    "DROP TRIGGER IF EXISTS inference_jobs_no_update_identity",
    "DROP TRIGGER IF EXISTS inference_jobs_no_delete",
    "DROP TRIGGER IF EXISTS inference_jobs_no_replace",
    """
    CREATE TABLE IF NOT EXISTS inference_jobs (
        job_id TEXT NOT NULL PRIMARY KEY
            CHECK(length(job_id) = 64 AND job_id NOT GLOB '*[^0-9a-f]*'),
        series_uid TEXT NOT NULL,
        dedup_key TEXT NOT NULL
            CHECK(length(dedup_key) = 64 AND dedup_key NOT GLOB '*[^0-9a-f]*'),
        state TEXT NOT NULL
            CHECK(state IN ('queued','running','succeeded','failed','cancelled')),
        error_code TEXT,
        stage1_checkpoint TEXT NOT NULL,
        stage2_checkpoint TEXT NOT NULL,
        device TEXT NOT NULL DEFAULT 'cpu',
        selected_frames TEXT NOT NULL
            CHECK(json_valid(selected_frames) = 1 AND json_type(selected_frames) = 'array'),
        prediction_id TEXT,
        model_run_id TEXT,
        request_payload TEXT NOT NULL CHECK(json_valid(request_payload) = 1),
        timings TEXT NOT NULL CHECK(json_valid(timings) = 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS inference_jobs_no_update_identity
    BEFORE UPDATE ON inference_jobs
    WHEN OLD.job_id <> NEW.job_id
      OR OLD.series_uid <> NEW.series_uid
      OR OLD.dedup_key <> NEW.dedup_key
      OR OLD.stage1_checkpoint <> NEW.stage1_checkpoint
      OR OLD.stage2_checkpoint <> NEW.stage2_checkpoint
      OR OLD.device <> NEW.device
      OR OLD.selected_frames <> NEW.selected_frames
      OR OLD.request_payload <> NEW.request_payload
      OR OLD.created_at <> NEW.created_at
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS inference_jobs_no_delete
    BEFORE DELETE ON inference_jobs
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS inference_jobs_no_replace
    BEFORE INSERT ON inference_jobs
    WHEN EXISTS (SELECT 1 FROM inference_jobs WHERE job_id = NEW.job_id)
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_inference_jobs_queue
    ON inference_jobs(state, created_at, job_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_inference_jobs_series
    ON inference_jobs(series_uid, created_at, job_id)
    """,
)

_SCHEMA_V11_HATCH_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS adoption_bindings (
        binding_id TEXT PRIMARY KEY CHECK(length(binding_id) = 64),
        series_uid TEXT NOT NULL,
        head_revision_id TEXT NOT NULL REFERENCES contour_revisions(revision_id),
        selection_id TEXT NOT NULL REFERENCES edes_selections(selection_id),
        prediction_id TEXT NOT NULL,
        model_run_id TEXT,
        revision_ids TEXT NOT NULL CHECK(json_valid(revision_ids) = 1),
        adopted_by TEXT NOT NULL,
        adopted_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS adoption_bindings_no_update
    BEFORE UPDATE ON adoption_bindings
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS adoption_bindings_no_delete
    BEFORE DELETE ON adoption_bindings
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS adoption_bindings_no_replace
    BEFORE INSERT ON adoption_bindings
    WHEN EXISTS (
        SELECT 1 FROM adoption_bindings WHERE binding_id = NEW.binding_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
)

# F14 / CMR_FUNCTION_V1_ANALYSIS_001 migration hatch (v11 -> v12). STRICTLY
# ADDITIVE per the Additive Hatch Doctrine: no v1..v11 object is altered,
# dropped or re-created. It creates ONE append-only table for the coherent
# selected-phase analysis binding (metrics + QC + exact inputs), with the full
# no-update / no-delete guard set. Two additional guards make the binding
# honest:
#   * ``..._no_replace_binding`` refuses a second row for the SAME clinical
#     binding (head, selection, inclusion, measurement, qc digest) even under
#     INSERT OR REPLACE, so an approval resolves exactly ONE analysis row;
#   * ``..._no_replace_sequence`` refuses a store-sequence slot collision.
# ``workflow_id``/``inclusion_set_id`` stay NULLABLE (disclosed legacy
# unbound/null-inclusion states remain readable); the seal path is what
# requires a non-null inclusion binding for a NEW measured approval.
_SCHEMA_V12_HATCH_DDL: tuple[str, ...] = (
    # --- hatch step 1: idempotent DROP guards for v12-owned objects ----------
    """
    DROP INDEX IF EXISTS selected_phase_analyses_series_sequence
    """,
    """
    DROP TRIGGER IF EXISTS selected_phase_analyses_no_replace_sequence
    """,
    """
    DROP TRIGGER IF EXISTS selected_phase_analyses_no_replace_binding
    """,
    """
    DROP TRIGGER IF EXISTS selected_phase_analyses_no_replace
    """,
    """
    DROP TRIGGER IF EXISTS selected_phase_analyses_no_delete
    """,
    """
    DROP TRIGGER IF EXISTS selected_phase_analyses_no_update
    """,
    # --- hatch step 2: additive table -----------------------------------------
    """
    CREATE TABLE IF NOT EXISTS selected_phase_analyses (
        analysis_id TEXT PRIMARY KEY
            CHECK(length(analysis_id) = 64
                  AND analysis_id NOT GLOB '*[^0-9a-f]*'),
        series_uid TEXT NOT NULL,
        study_uid TEXT NOT NULL,
        workflow_id TEXT CHECK(workflow_id IS NULL OR length(workflow_id) = 64),
        head_revision_id TEXT NOT NULL REFERENCES contour_revisions(revision_id),
        selection_id TEXT NOT NULL REFERENCES edes_selections(selection_id),
        ed_frame INTEGER NOT NULL CHECK(ed_frame >= 0),
        es_frame INTEGER NOT NULL CHECK(es_frame >= 0),
        inclusion_set_id TEXT REFERENCES slice_inclusion_sets(inclusion_set_id),
        measurement_id TEXT NOT NULL REFERENCES measurements(measurement_id),
        qc_report_digest TEXT NOT NULL
            CHECK(length(qc_report_digest) = 64
                  AND qc_report_digest NOT GLOB '*[^0-9a-f]*'),
        qc_verdict TEXT NOT NULL CHECK(qc_verdict IN ('PASS','REVIEW REQUIRED')),
        algorithm_version TEXT NOT NULL,
        software_version TEXT NOT NULL,
        range_table_version TEXT,
        analysis_sequence INTEGER NOT NULL CHECK(analysis_sequence >= 1),
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    # --- hatch step 3: guard triggers ------------------------------------------
    f"""
    CREATE TRIGGER IF NOT EXISTS selected_phase_analyses_no_update
    BEFORE UPDATE ON selected_phase_analyses
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS selected_phase_analyses_no_delete
    BEFORE DELETE ON selected_phase_analyses
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS selected_phase_analyses_no_replace
    BEFORE INSERT ON selected_phase_analyses
    WHEN EXISTS (
        SELECT 1 FROM selected_phase_analyses WHERE analysis_id = NEW.analysis_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # One clinical binding -> at most ONE analysis row (NULL-safe comparison).
    f"""
    CREATE TRIGGER IF NOT EXISTS selected_phase_analyses_no_replace_binding
    BEFORE INSERT ON selected_phase_analyses
    WHEN EXISTS (
        SELECT 1 FROM selected_phase_analyses
        WHERE head_revision_id = NEW.head_revision_id
          AND selection_id = NEW.selection_id
          AND (inclusion_set_id IS NEW.inclusion_set_id)
          AND measurement_id = NEW.measurement_id
          AND qc_report_digest = NEW.qc_report_digest
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS selected_phase_analyses_no_replace_sequence
    BEFORE INSERT ON selected_phase_analyses
    WHEN EXISTS (
        SELECT 1 FROM selected_phase_analyses
        WHERE series_uid = NEW.series_uid
          AND analysis_sequence = NEW.analysis_sequence
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    # --- hatch step 4: series lookup index -------------------------------------
    """
    CREATE UNIQUE INDEX IF NOT EXISTS selected_phase_analyses_series_sequence
    ON selected_phase_analyses(series_uid, analysis_sequence)
    """,
)

# FNC-E R1 migration hatch (v8 -> v9). Analysis jobs are an append-only
_SCHEMA_V9_HATCH_DDL: tuple[str, ...] = (
    "DROP TRIGGER IF EXISTS analysis_jobs_no_update_identity",
    "DROP TRIGGER IF EXISTS analysis_jobs_no_delete",
    "DROP TRIGGER IF EXISTS analysis_jobs_no_replace",
    """
    CREATE TABLE IF NOT EXISTS analysis_jobs (
        job_id TEXT NOT NULL PRIMARY KEY
            CHECK(length(job_id) = 64 AND job_id NOT GLOB '*[^0-9a-f]*'),
        series_uid TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('selected_phase_recompute')),
        dedup_key TEXT NOT NULL
            CHECK(length(dedup_key) = 64 AND dedup_key NOT GLOB '*[^0-9a-f]*'),
        request_payload TEXT NOT NULL CHECK(json_valid(request_payload) = 1),
        state TEXT NOT NULL
            CHECK(state IN ('queued','running','succeeded','failed','cancelled')),
        error_code TEXT,
        artifact_measurement_id TEXT,
        timings TEXT NOT NULL CHECK(json_valid(timings) = 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(series_uid, kind, dedup_key)
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS analysis_jobs_no_update_identity
    BEFORE UPDATE ON analysis_jobs
    WHEN OLD.job_id <> NEW.job_id
      OR OLD.series_uid <> NEW.series_uid
      OR OLD.kind <> NEW.kind
      OR OLD.dedup_key <> NEW.dedup_key
      OR OLD.request_payload <> NEW.request_payload
      OR OLD.created_at <> NEW.created_at
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS analysis_jobs_no_delete
    BEFORE DELETE ON analysis_jobs
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS analysis_jobs_no_replace
    BEFORE INSERT ON analysis_jobs
    WHEN EXISTS (SELECT 1 FROM analysis_jobs WHERE job_id = NEW.job_id)
      OR EXISTS (
          SELECT 1 FROM analysis_jobs
          WHERE series_uid = NEW.series_uid
            AND kind = NEW.kind
            AND dedup_key = NEW.dedup_key
      )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analysis_jobs_queue
    ON analysis_jobs(state, created_at, job_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analysis_jobs_series
    ON analysis_jobs(series_uid, created_at, job_id)
    """,
)

_SERIES_TOPOLOGY_BLOCKS_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS series_topology_blocks (
        block_id TEXT PRIMARY KEY CHECK(length(block_id) = 64),
        series_uid TEXT NOT NULL,
        slice_index INTEGER NOT NULL,
        frame_index INTEGER NOT NULL,
        structure TEXT NOT NULL,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL,
        UNIQUE(series_uid, slice_index, frame_index, structure)
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS series_topology_blocks_no_update
    BEFORE UPDATE ON series_topology_blocks
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS series_topology_blocks_no_delete
    BEFORE DELETE ON series_topology_blocks
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS series_topology_blocks_no_replace
    BEFORE INSERT ON series_topology_blocks
    WHEN EXISTS (
        SELECT 1 FROM series_topology_blocks
        WHERE series_uid = NEW.series_uid
          AND slice_index = NEW.slice_index
          AND frame_index = NEW.frame_index
          AND structure = NEW.structure
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
)

#: Additive denormalized approval column (Astra F2/F5): the bound
#: topology-disposition-set digest of a sealed approval.
TOPOLOGY_DISPOSITION_COLUMN = "topology_disposition_digest"

# --- Topology disposition table (Astra F1-F5, 2026-09-09) ----------------------
#
# Its OWN append-only DDL, executed alongside the existing
# ``series_topology_blocks`` DDL after the versioned schema hatches. Nothing
# here touches a pre-existing table or trigger: only the new table is created,
# with its own complete no-update / no-delete / no-replace guard set.
_TOPOLOGY_DISPOSITIONS_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS topology_dispositions (
        disposition_id TEXT PRIMARY KEY CHECK(length(disposition_id) = 64),
        series_uid TEXT NOT NULL,
        slice_index INTEGER NOT NULL,
        frame_index INTEGER NOT NULL,
        structure TEXT NOT NULL,
        source_block_id TEXT NOT NULL CHECK(length(source_block_id) = 64),
        source_revision_id TEXT NOT NULL CHECK(length(source_revision_id) = 64),
        selection_id TEXT NOT NULL CHECK(length(selection_id) = 64),
        frame_role TEXT NOT NULL CHECK(frame_role IN ('ED', 'ES')),
        disposition TEXT NOT NULL CHECK(disposition IN ('REPLACED', 'EXCLUDED')),
        physician TEXT NOT NULL,
        created_at TEXT NOT NULL,
        reason TEXT NOT NULL,
        workflow_id TEXT CHECK(length(workflow_id) = 64),
        payload TEXT NOT NULL
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS topology_dispositions_no_update
    BEFORE UPDATE ON topology_dispositions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS topology_dispositions_no_delete
    BEFORE DELETE ON topology_dispositions
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS topology_dispositions_no_replace
    BEFORE INSERT ON topology_dispositions
    WHEN EXISTS (
        SELECT 1 FROM topology_dispositions
        WHERE disposition_id = NEW.disposition_id
    )
    BEGIN
        SELECT RAISE(ABORT, '{_APPEND_ONLY}');
    END
    """,
)


class PayloadIntegrityError(ValueError):
    """A payload digest/canonical-form invariant failed (write or load)."""


class SeriesCoherenceError(ValueError):
    """A cross-record series/frame coherence splice was attempted."""


class StaleHeadError(SeriesCoherenceError):
    """Deterministic stale-head failure inside a checked sealing transaction.

    Raised by :meth:`SqliteCmrStore.save_approval_checked`,
    :meth:`SqliteCmrStore.save_qc_decision_checked` and
    :meth:`SqliteCmrStore.save_analysis_checked` when the persisted current
    head (or latest selection) no longer matches the expected identity the
    caller sealed against. The transaction rolls back with ZERO staged rows.
    """


class JobNotPublishableError(StaleHeadError):
    """A worker tried to publish a job that is no longer running."""


class AnalysisAuthorityError(StaleHeadError):
    """The coherent selected-phase analysis authority cannot be formed.

    Raised by the checked writers when the state they are asked to publish
    does not carry the QC report/selection identity the analysis binding
    requires (CMR_FUNCTION_V1_ANALYSIS_001), and by
    :meth:`SqliteCmrStore.save_approval_checked` when the seal does not resolve
    the CURRENT coherent analysis. The transaction rolls back with ZERO staged
    rows; the caller must refresh the analysis against the current state. It
    deliberately extends :class:`StaleHeadError` so every existing
    stale-conflict boundary (HTTP 409 ``stale_view``, worker stale handling)
    treats it as the stale state it is.
    """


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_registration_record_bytes(record: dict[str, Any]) -> bytes:
    """Canonical bytes of one cine-workflow registration record.

    Frozen S3-004E encoding (identical to the committed sample
    ``tests/ai/fixtures/cine_registration_record_schema_v1.json``):
    ``json.dumps(record, sort_keys=True, separators=(",", ":"),
    ensure_ascii=False)`` encoded UTF-8 with LF and ONE trailing newline.
    """
    return (
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )


def _next_sequence(
    connection: sqlite3.Connection, table: str, sequence_column: str, series_uid: str
) -> int:
    """Next per-series store insertion sequence (MAX+1 inside the caller's
    transaction). Store sequence is persistence metadata ordered by actual
    insert time — never by caller-controlled timestamps (S2C-001 disclosed
    amendment)."""
    row = connection.execute(
        f"SELECT COALESCE(MAX({sequence_column}), 0) + 1 FROM {table} WHERE series_uid = ?",
        (series_uid,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _next_inclusion_sequence(
    connection: sqlite3.Connection, series_uid: str, workflow_id: str | None
) -> int:
    """Next ``inclusion_sequence`` within ``(workflow_id, series_uid)``.

    CFM-001 spec §1.7: store metadata ordered by ACTUAL INSERTION TIME,
    never by ``selected_at``. The NULL-safe ``IS`` comparison groups
    unbound rows with each other exactly like the v4 backfill does.
    """
    row = connection.execute(
        "SELECT COALESCE(MAX(inclusion_sequence), 0) + 1 FROM slice_inclusion_sets"
        " WHERE series_uid = ? AND (workflow_id IS ?)",
        (series_uid, workflow_id),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _workflow_id_from_payload(payload_text: str) -> str | None:
    """Read ``provenance.workflow_id`` from a stored canonical payload."""
    try:
        data = json.loads(payload_text)
    except ValueError as exc:
        raise PayloadIntegrityError("stored payload is not valid JSON") from exc
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        return None
    workflow_id = provenance.get("workflow_id")
    return None if workflow_id is None else str(workflow_id)


def _workflow_id_of_provenance(provenance: Any) -> str | None:
    """Read ``workflow_id`` from an already-parsed provenance dict."""
    if not isinstance(provenance, dict):
        return None
    workflow_id = provenance.get("workflow_id")
    return None if workflow_id is None else str(workflow_id)


def _parse_utc(value: str) -> datetime:
    """Parse a timezone-aware ISO-8601 instant (fail closed when naive)."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp is not timezone-aware: {value!r}")
    return parsed.astimezone(timezone.utc)


def _timestamp_after(left: str, right: str) -> bool:
    """Strict UTC order; an unorderable/blank pair fails closed (False)."""
    if not left or not right:
        return False
    try:
        return _parse_utc(left) > _parse_utc(right)
    except ValueError:
        return False


def _block_is_metric_relevant(
    *,
    slice_index: int,
    frame_index: int,
    structure: str,
    n_slices: int,
    n_frames: int | None,
    ed_frame: int,
) -> bool:
    """Whether one persisted block key is required by the selected metrics.

    EXACT mirror of the relevance leg of
    :func:`cmr_workflow.topology_disposition.required_metric_keys` (the single
    source of truth the controller consumes): ``LV_CAVITY``/``RV_CAVITY`` at
    every in-grid ``(slice, frame)`` plus ``LV_MYOCARDIUM`` at every slice of
    the selected ED frame. An unknown frame count cannot prove irrelevance, so
    every key stays relevant (fail closed); a key provably outside the grid or
    a non-ED ``LV_MYOCARDIUM`` is irrelevant exactly as in the controller.
    """
    if n_frames is None:
        return True
    if not (0 <= slice_index < n_slices and 0 <= frame_index < n_frames):
        return False
    if structure in (Structure.LV_CAVITY.value, Structure.RV_CAVITY.value):
        return True
    if structure == Structure.LV_MYOCARDIUM.value:
        return frame_index == ed_frame
    return False

def _temporal_axis_provenance_of_payload(payload_text: str) -> str:
    """Read ``temporal_axis_provenance`` from a stored manifest payload.

    The field is identity-relevant (it participates in ``workflow_id``
    derivation), so a stored manifest can only round-trip when the load path
    restores it. Payloads written before the field existed (dicom loader r3,
    2026-09-06) omit the key entirely: those rows were necessarily built with
    the constructor default, so ``"measured"`` is the correct — and only
    possible — reconstruction. No schema migration is required: the value
    already lives in the stored canonical ``payload`` TEXT.
    """
    try:
        data = json.loads(payload_text)
    except ValueError as exc:
        raise PayloadIntegrityError("stored payload is not valid JSON") from exc
    if not isinstance(data, dict):
        raise PayloadIntegrityError("stored manifest payload is not a JSON object")
    value = data.get("temporal_axis_provenance")
    if value is None:
        # Backward compatibility: pre-field payloads default to "measured".
        return "measured"
    if not isinstance(value, str) or value not in ("measured", "derived"):
        raise PayloadIntegrityError(
            "stored manifest payload carries an invalid "
            f"temporal_axis_provenance {value!r}"
        )
    return value


def _range_table_version_of_payload(payload_text: str) -> str | None:
    """Read the canonical ``range_table_version`` of a measurement payload.

    CFM-001 §2.6: the additive v4 column must equal the payload field.
    Reading it through THIS single helper on every write keeps column and
    canonical bytes in lockstep (the value is derived from the payload
    TEXT itself, never from a parallel argument that could drift).
    """
    try:
        data = json.loads(payload_text)
    except ValueError as exc:
        raise PayloadIntegrityError("stored payload is not valid JSON") from exc
    value = data.get("range_table_version")
    return None if value is None else str(value)


def _assert_range_column_matches_payload(column_value: Any, reconstructed: Measurement) -> None:
    """Fail closed unless the v4 column equals the payload field.

    CFM-001 §2.6: SQLite stores ``range_table_version`` in the additive
    column AND verifies column/payload equality on load. A drifted column
    is a corrupt store, never a display hint.
    """
    expected = reconstructed.range_table_version
    actual = None if column_value is None else str(column_value)
    if actual != expected:
        raise PayloadIntegrityError(
            "measurements.range_table_version column "
            f"{actual!r} does not equal the payload field {expected!r} for "
            f"measurement {reconstructed.measurement_id!r}"
        )


def _assert_bound_workflow_agreement(**named: str | None) -> None:
    """Reject mixing bound and unbound workflow identities (S2C-001).

    Every named value must be pairwise equal; ``None`` is legal ONLY when all
    values are ``None`` (a fully legacy unbound lineage). Any mix of a bound
    id with ``None``, or two different ids, fails closed.
    """
    values = list(named.values())
    bound_values = [value for value in values if value is not None]
    if len(set(bound_values)) > 1:
        detail = ", ".join(f"{name}={value!r}" for name, value in sorted(named.items()))
        raise SeriesCoherenceError(
            f"mixed workflow identities in one lineage: {detail}; all records "
            "in one coherence chain must carry the same manifest-bound "
            "workflow identity"
        )
    if bound_values and len(bound_values) < len(values):
        missing = sorted(name for name, value in named.items() if value is None)
        raise SeriesCoherenceError(
            f"mixed bound/unbound workflow identities: unbound {missing}; when "
            "any record in the chain is manifest-bound every participating "
            "record must carry the same workflow identity"
        )


# --- Display-path reconstruction memo (additive, read-only) ------------------
#
# WholeCinePrediction rows are IMMUTABLE (append-only store; prediction_id is
# the content digest of the payload). Reconstruct+verify of one row is
# therefore a pure function of the row bytes and is expensive for large
# cine-scale payloads (the display path loads the latest prediction on every
# /contours request). This memo caches the reconstructed object per
# (db path, prediction_id); a NEW prediction (new id) simply misses. It never
# serves stale data: the id IS the payload digest. Bounded LRU.
_WC_PREDICTION_MEMO: dict[tuple[str, str], WholeCinePrediction] = {}
_WC_PREDICTION_MEMO_LIMIT = 8


def _wc_prediction_memo_get(path: Path, prediction_id: str) -> WholeCinePrediction | None:
    key = (str(path), prediction_id)
    hit = _WC_PREDICTION_MEMO.get(key)
    if hit is not None:
        _WC_PREDICTION_MEMO[key] = _WC_PREDICTION_MEMO.pop(key)
    return hit


def _wc_prediction_memo_put(path: Path, prediction: WholeCinePrediction) -> None:
    key = (str(path), prediction.prediction_id)
    _WC_PREDICTION_MEMO[key] = prediction
    while len(_WC_PREDICTION_MEMO) > _WC_PREDICTION_MEMO_LIMIT:
        oldest = next(iter(_WC_PREDICTION_MEMO))
        del _WC_PREDICTION_MEMO[oldest]


class SqliteCmrStore:
    """Append-only embedded SQLite persistence for Node 1 clinical records."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._connection: sqlite3.Connection | None = self._connect()
        try:
            self._initialize_schema()
        except BaseException:
            self.close()
            raise

    # --- connection / schema ----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None -> autocommit mode: transactions are managed
        # explicitly via BEGIN IMMEDIATE / COMMIT / ROLLBACK.
        connection = sqlite3.connect(str(self._path), isolation_level=None)
        # Contract scope item 4: executed on EVERY connection before any other
        # statement (read connections included).
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> SqliteCmrStore:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("store connection is closed")
        return self._connection

    def _rollback_quietly(self, connection: sqlite3.Connection) -> None:
        # A failed statement can leave the transaction auto-aborted; a
        # rollback attempt may then raise "cannot rollback - no transaction is
        # active", which must not mask the original error.
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    def _initialize_schema(self) -> None:
        connection = self._require_connection()
        current = self._read_user_version()
        if current > SCHEMA_USER_VERSION:
            raise RuntimeError(
                f"database user_version {current} is newer than supported version "
                f"{SCHEMA_USER_VERSION}; per the migration hatch protocol a forward "
                "migration must be authored inside this schema-init transaction "
                "(DROP TRIGGER IF EXISTS -> transform -> recreate triggers -> set "
                "user_version); runtime never drops triggers"
            )
        if current < SCHEMA_USER_VERSION:
            connection.execute("BEGIN IMMEDIATE")
            try:
                # Stepwise, deterministic, single transaction. Fresh databases and
                # legacy databases of ANY prior stamped version (0..6) take exactly
                # this one path: each hatch below is conditional on the stamped
                # version, so every input lands at an identical v7 end state.
                # Sequential explicit stamping doctrine (S3-001 spec §3.2): the v4
                # hatch is stamped ``user_version = 4`` BEFORE the v5 hatch runs,
                # v5 is stamped before v6 and v6 before v7, so a legacy database
                # walks every intermediate version explicitly instead of jumping
                # to the head.
                if current < 1:
                    for statement in _SCHEMA_V1_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=1")
                if current < 2:
                    for statement in _SCHEMA_V2_HATCH_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=2")
                if current < 3:
                    for statement in _SCHEMA_V3_HATCH_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=3")
                if current < 4:
                    for statement in _SCHEMA_V4_HATCH_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=4")
                if current < 5:
                    for statement in _SCHEMA_V5_HATCH_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=5")
                if current < 6:
                    for statement in _SCHEMA_V6_HATCH_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=6")
                if current < 7:
                    for statement in _SCHEMA_V7_HATCH_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=7")
                if current < 8:
                    for statement in _SCHEMA_V8_HATCH_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=8")
                if current < 9:
                    for statement in _SCHEMA_V9_HATCH_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=9")
                if current < 10:
                    for statement in _SCHEMA_V10_HATCH_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=10")
                if current < 11:
                    for statement in _SCHEMA_V11_HATCH_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=11")
                if current < 12:
                    for statement in _SCHEMA_V12_HATCH_DDL:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version=12")
                connection.execute("COMMIT")
            except BaseException:
                self._rollback_quietly(connection)
                raise
        for statement in _SERIES_TOPOLOGY_BLOCKS_DDL:
            connection.execute(statement)
        for statement in _TOPOLOGY_DISPOSITIONS_DDL:
            connection.execute(statement)
        # Astra F2/F5: an ADDITIVE denormalized column on ``approval_snapshots``
        # carrying the bound topology-disposition-set digest. It is added here,
        # OUTSIDE the versioned hatches, so no existing hatch step (and no
        # pre-existing column, payload byte or trigger) is touched; the
        # statement is guarded by a pragma check so re-init is a no-op and a
        # legacy database of ANY stamped version ends in the same state.
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(approval_snapshots)")
        }
        if TOPOLOGY_DISPOSITION_COLUMN not in columns:
            connection.execute(
                f"ALTER TABLE approval_snapshots ADD COLUMN "
                f"{TOPOLOGY_DISPOSITION_COLUMN} TEXT "
                "CHECK(topology_disposition_digest IS NULL OR "
                "length(topology_disposition_digest) = 64)"
            )

    def _read_user_version(self) -> int:
        connection = self._require_connection()
        row = connection.execute("PRAGMA user_version").fetchone()
        assert row is not None  # sqlite always returns one row for this pragma
        return int(row[0])

    def _in_write_transaction(self, body: Callable[[sqlite3.Connection], T]) -> T:
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            result = body(connection)
            connection.execute("COMMIT")
        except BaseException:
            self._rollback_quietly(connection)
            raise
        return result

    # --- write-path integrity (amendment F) --------------------------------------

    @staticmethod
    def _content_addressed_payload(record: Any, expected_id: str) -> str:
        """Canonical TEXT whose sha256 equals the record's public id."""

        payload_text = canonical_json(record)
        computed = _sha256_hex(payload_text)
        if computed != expected_id:
            raise PayloadIntegrityError(
                f"payload digest {computed!r} does not match record id {expected_id!r}"
            )
        return payload_text

    @staticmethod
    def _selection_payload(selection: EDESSelection) -> tuple[str, str]:
        payload_dict = edes_selection_payload(selection)
        payload_text = edes_selection_payload_text(selection)
        # Cross-consistency (amendment A): the persistence-local projection
        # serializer must produce exactly what the imported Node 1 serializer
        # produces over the same plain dict.
        if canonical_json(payload_dict) != payload_text:
            raise PayloadIntegrityError(
                "EDESSelection projection serialization diverges from canonical_json"
            )
        return payload_text, selection_id_from_payload_text(payload_text)

    # --- saves -----------------------------------------------------------------

    def save_prediction(self, prediction: Prediction) -> str:
        payload_text = self._content_addressed_payload(prediction, prediction.prediction_id)
        bound_workflow = prediction.provenance.workflow_id

        def body(connection: sqlite3.Connection) -> str:
            # S2C-001 correction F2: a manifest-bound prediction must anchor
            # to a PERSISTED source manifest; unbound predictions stay legal.
            if bound_workflow is not None:
                manifest_row = connection.execute(
                    "SELECT 1 FROM dicom_source_manifests WHERE workflow_id = ?",
                    (bound_workflow,),
                ).fetchone()
                if manifest_row is None:
                    raise SeriesCoherenceError(
                        f"prediction carries workflow {bound_workflow!r} but no "
                        "such DICOM source manifest is persisted; save the "
                        "manifest before any workflow-bound record"
                    )
            connection.execute(
                "INSERT INTO predictions (prediction_id, payload, workflow_id) VALUES (?, ?, ?)",
                (prediction.prediction_id, payload_text, bound_workflow),
            )
            return prediction.prediction_id

        return self._in_write_transaction(body)

    def save_revision(self, revision: ContourRevision) -> str:
        payload_text = self._content_addressed_payload(revision, revision.revision_id)

        def body(connection: sqlite3.Connection) -> str:
            # Parent-link coherence guard (review DEFECT-1), inside this same
            # BEGIN IMMEDIATE transaction and BEFORE the revision INSERT.
            _assert_parent_links_share_series(connection, revision)
            _insert_revision_row(connection, revision, payload_text)
            return revision.revision_id

        return self._in_write_transaction(body)

    def save_revision_with_measurement(
        self, revision: ContourRevision, measurement: Measurement
    ) -> str:
        """Atomically persist one revision plus its associated measurement."""

        revision_payload = self._content_addressed_payload(revision, revision.revision_id)
        measurement_payload = self._content_addressed_payload(
            measurement, measurement.measurement_id
        )

        def body(connection: sqlite3.Connection) -> str:
            # Series-coherence guard (amendment B), before every INSERT.
            if measurement.series_uid != revision.series_uid:
                raise SeriesCoherenceError(
                    "measurement series_uid "
                    f"{measurement.series_uid!r} does not match revision series_uid "
                    f"{revision.series_uid!r}"
                )
            # Parent-link coherence guard (review DEFECT-1): resolve every
            # referenced parent's series BEFORE any row is staged, so a
            # rejection leaves zero staged rows after rollback.
            _assert_parent_links_share_series(connection, revision)
            # Exact association guard (Nodes 2-4 reconciliation correction):
            # this atomic save binds the measurement to the revision saved in
            # THIS transaction. A measurement pointing at any other revision
            # (even an existing same-series one) is rejected before either row
            # can survive; reusing an older measurement requires its own
            # explicit API instead of an ambiguous atomic pair-save.
            if measurement.source_revision_id != revision.revision_id:
                raise SeriesCoherenceError(
                    "measurement source_revision_id "
                    f"{measurement.source_revision_id!r} does not match the "
                    f"revision saved in this transaction "
                    f"{revision.revision_id!r}"
                )
            # S2C-001 correction F2: one workflow identity across the paired
            # revision and measurement (bound/unbound mixing fails closed
            # before any row is staged).
            _assert_bound_workflow_agreement(
                revision=revision.provenance.workflow_id,
                measurement=measurement.provenance.workflow_id,
            )
            # CFM-001 correction F6: this primitive no longer bypasses the
            # inclusion invariants. When an applicable set exists for the
            # persisted head + the measurement's selection, the measurement
            # MUST bind exactly it; a null binding stays legal ONLY when no
            # applicable set exists (explicit legacy-import scope).
            self._verify_inclusion_binding(
                connection,
                series_uid=measurement.series_uid,
                head_revision_id=revision.revision_id,
                workflow_id=measurement.provenance.workflow_id,
                selection_id=measurement.selection_id,
                bound_inclusion_set_id=measurement.inclusion_set_id,
                allow_null_set=True,
                gaps_length=len(measurement.stack_gaps_mm),
            )
            # Amendment G: the revision row is inserted BEFORE the measurement
            # row, both inside this single transaction.
            _insert_revision_row(connection, revision, revision_payload)
            referenced_series = connection.execute(
                "SELECT series_uid FROM contour_revisions WHERE revision_id = ?",
                (measurement.source_revision_id,),
            ).fetchone()
            if referenced_series is None:
                raise SeriesCoherenceError(
                    "measurement source_revision_id "
                    f"{measurement.source_revision_id!r} has no revision row"
                )
            if referenced_series[0] != measurement.series_uid:
                raise SeriesCoherenceError(
                    "measurement series_uid "
                    f"{measurement.series_uid!r} does not match source revision "
                    f"series_uid {referenced_series[0]!r}"
                )
            connection.execute(
                "INSERT INTO measurements (measurement_id, series_uid,"
                " source_revision_id, created_at, payload, workflow_id,"
                " selection_id, inclusion_set_id, measurement_sequence,"
                " range_table_version)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    measurement.measurement_id,
                    measurement.series_uid,
                    measurement.source_revision_id,
                    measurement.created_at,
                    measurement_payload,
                    measurement.provenance.workflow_id,
                    measurement.selection_id,
                    measurement.inclusion_set_id,
                    _next_sequence(
                        connection,
                        "measurements",
                        "measurement_sequence",
                        measurement.series_uid,
                    ),
                    _range_table_version_of_payload(measurement_payload),
                ),
            )
            # F14: the paired revision+measurement immediately derive the ONE
            # coherent analysis binding (metrics + QC + exact inputs). A child
            # without a QC report leaves the measurement readable but can
            # never satisfy the seal's analysis requirement.
            self._insert_analysis_binding(
                connection, measurement=measurement, require_qc=False
            )
            return revision.revision_id

        return self._in_write_transaction(body)

    def save_measurement_for_head(
        self,
        measurement: Measurement,
        *,
        expected_head_revision_id: str,
        expected_selection_id: str | None,
    ) -> str:
        """Insert ONE measurement whose revision row is ALREADY persisted.

        CFM-001 correction-round helper for seeding/composition flows that
        persist the contour head first (e.g. with its gateway QCReport) and
        only afterwards bind a Measurement — including the explicit
        inclusion binding verification against the CURRENT head, identical
        to :meth:`save_analysis_checked`.
        """
        payload_text = self._content_addressed_payload(measurement, measurement.measurement_id)

        def body(connection: sqlite3.Connection) -> str:
            head_row = self._current_head_row(connection, measurement.series_uid)
            if head_row is None or str(head_row[0]) != expected_head_revision_id:
                raise StaleHeadError(f"stale head: expected {expected_head_revision_id!r}")
            if measurement.source_revision_id != expected_head_revision_id:
                raise SeriesCoherenceError(
                    "measurement source_revision_id "
                    f"{measurement.source_revision_id!r} does not match this "
                    f"transaction's head {expected_head_revision_id!r}"
                )
            if (measurement.selection_id or None) != (expected_selection_id or None):
                raise SeriesCoherenceError(
                    "measurement binds selection "
                    f"{measurement.selection_id!r} but this transaction "
                    f"seals {expected_selection_id!r}"
                )
            self._verify_inclusion_binding(
                connection,
                series_uid=measurement.series_uid,
                head_revision_id=expected_head_revision_id,
                workflow_id=measurement.provenance.workflow_id,
                selection_id=measurement.selection_id,
                bound_inclusion_set_id=measurement.inclusion_set_id,
                allow_null_set=True,
                gaps_length=len(measurement.stack_gaps_mm),
            )
            connection.execute(
                "INSERT INTO measurements (measurement_id, series_uid,"
                " source_revision_id, created_at, payload, workflow_id,"
                " selection_id, inclusion_set_id, measurement_sequence,"
                " range_table_version)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    measurement.measurement_id,
                    measurement.series_uid,
                    measurement.source_revision_id,
                    measurement.created_at,
                    payload_text,
                    measurement.provenance.workflow_id,
                    measurement.selection_id,
                    measurement.inclusion_set_id,
                    _next_sequence(
                        connection,
                        "measurements",
                        "measurement_sequence",
                        measurement.series_uid,
                    ),
                    _range_table_version_of_payload(payload_text),
                ),
            )
            # F14: derive the coherent analysis binding for this measurement.
            self._insert_analysis_binding(
                connection, measurement=measurement, require_qc=False
            )
            return measurement.measurement_id

        return self._in_write_transaction(body)

    def save_edes_selection(self, selection: EDESSelection) -> str:
        """Persist a selection idempotently; returns its storage-computed id.

        Identical payloads resolve to the existing row (check-then-insert
        inside one BEGIN IMMEDIATE transaction); distinct logical selections
        are distinct rows (amendment H).
        """

        payload_text, selection_id = self._selection_payload(selection)

        def body(connection: sqlite3.Connection) -> str:
            existing = connection.execute(
                "SELECT selection_id FROM edes_selections WHERE selection_id = ?",
                (selection_id,),
            ).fetchone()
            if existing is not None:
                return str(existing[0])
            connection.execute(
                "INSERT INTO edes_selections (selection_id, series_uid, selected_at,"
                " payload, workflow_id, selection_sequence) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    selection_id,
                    selection.series_uid,
                    selection.selected_at,
                    payload_text,
                    selection.provenance.workflow_id,
                    _next_sequence(
                        connection,
                        "edes_selections",
                        "selection_sequence",
                        selection.series_uid,
                    ),
                ),
            )
            return selection_id

        return self._in_write_transaction(body)

    def _resolve_current_nullable_lineage(
        self,
        connection: sqlite3.Connection,
        *,
        series_uid: str,
        expected_n_slices: int,
    ) -> tuple[str | None, str | None, str | None]:
        """Resolve the three nullable identities used by the workstation CAS.

        This helper is intentionally called by the checked writer *after* its
        ``BEGIN IMMEDIATE`` has been acquired.  The current head, latest
        selection and latest applicable inclusion set therefore form one
        SQLite snapshot; a caller cannot observe one identity from one state
        and another identity from a later state.
        """
        head_row = self._current_head_row(connection, series_uid)
        head_id = None if head_row is None else str(head_row[0])
        head_workflow = None if head_row is None or head_row[3] is None else str(head_row[3])

        selection_row = connection.execute(
            "SELECT selection_id, workflow_id FROM edes_selections WHERE series_uid = ?"
            " ORDER BY selection_sequence DESC LIMIT 1",
            (series_uid,),
        ).fetchone()
        selection_id = None if selection_row is None else str(selection_row[0])
        selection_workflow = (
            None
            if selection_row is None or selection_row[1] is None
            else str(selection_row[1])
        )
        # A missing head or selection is a genuine nullable lineage value, but
        # it is not an existing workflow participant.  A checked ED/ES write
        # must be able to create the first bound selection beside a bound
        # source-backed contour head; only persisted rows take part in this
        # workflow-mixing check.
        workflow_participants: dict[str, str | None] = {}
        if head_row is not None:
            workflow_participants["head"] = head_workflow
        if selection_row is not None:
            workflow_participants["selection"] = selection_workflow
        if workflow_participants:
            _assert_bound_workflow_agreement(**workflow_participants)

        inclusion_id: str | None = None
        if head_id is not None and selection_id is not None:
            resolved = self._resolve_latest_applicable_set(
                connection,
                series_uid=series_uid,
                head_revision_id=head_id,
                selection_id=selection_id,
                workflow_id=head_workflow,
                expected_n_slices=expected_n_slices,
            )
            if resolved is not None:
                inclusion_id = resolved[0].inclusion_set_id
        return head_id, selection_id, inclusion_id

    @staticmethod
    def _validate_checked_identity(name: str, value: str | None) -> str | None:
        """Validate one nullable content identity without treating null as skip."""
        if value is not None and (
            not isinstance(value, str) or not value
        ):
            raise SeriesCoherenceError(
                f"{name} must be a non-empty string or None; got {value!r}"
            )
        return value

    @staticmethod
    def _validate_expected_n_slices(expected_n_slices: int) -> int:
        if (
            isinstance(expected_n_slices, bool)
            or not isinstance(expected_n_slices, int)
            or expected_n_slices < 1
        ):
            raise SeriesCoherenceError(
                "expected_n_slices must be a positive integer (the cine's "
                f"actual slice count); got {expected_n_slices!r}"
            )
        return expected_n_slices

    @staticmethod
    def _assert_checked_lineage(
        *,
        actual: tuple[str | None, str | None, str | None],
        expected_head_revision_id: str | None,
        expected_selection_id: str | None,
        expected_inclusion_set_id: str | None,
    ) -> None:
        expected = (
            expected_head_revision_id,
            expected_selection_id,
            expected_inclusion_set_id,
        )
        if actual != expected:
            labels = ("head", "selection", "inclusion")
            differences = ", ".join(
                f"{label}: expected {wanted!r}, current {found!r}"
                for label, wanted, found in zip(labels, expected, actual)
                if wanted != found
            )
            raise StaleHeadError(
                "stale_view: nullable lineage changed before the checked write; "
                f"{differences}; zero rows written"
            )

    def save_edes_selection_checked(
        self,
        selection: EDESSelection,
        *,
        expected_head_revision_id: str | None,
        expected_selection_id: str | None,
        expected_inclusion_set_id: str | None,
        expected_n_slices: int,
    ) -> str:
        """Persist an ED/ES selection after an atomic nullable-lineage CAS.

        The CAS is deliberately performed before the idempotency lookup.  In
        particular, replaying an identical selection payload with a changed
        head, selection or applicable inclusion set raises
        :class:`StaleHeadError` instead of silently returning the old id.
        """
        expected_n_slices = self._validate_expected_n_slices(expected_n_slices)
        expected_head_revision_id = self._validate_checked_identity(
            "expected_head_revision_id", expected_head_revision_id
        )
        expected_selection_id = self._validate_checked_identity(
            "expected_selection_id", expected_selection_id
        )
        expected_inclusion_set_id = self._validate_checked_identity(
            "expected_inclusion_set_id", expected_inclusion_set_id
        )
        payload_text, selection_id = self._selection_payload(selection)

        def body(connection: sqlite3.Connection) -> str:
            actual = self._resolve_current_nullable_lineage(
                connection,
                series_uid=selection.series_uid,
                expected_n_slices=expected_n_slices,
            )
            self._assert_checked_lineage(
                actual=actual,
                expected_head_revision_id=expected_head_revision_id,
                expected_selection_id=expected_selection_id,
                expected_inclusion_set_id=expected_inclusion_set_id,
            )
            # This lookup is intentionally AFTER the complete nullable CAS.
            existing = connection.execute(
                "SELECT selection_id FROM edes_selections WHERE selection_id = ?",
                (selection_id,),
            ).fetchone()
            if existing is not None:
                return str(existing[0])
            connection.execute(
                "INSERT INTO edes_selections (selection_id, series_uid, selected_at,"
                " payload, workflow_id, selection_sequence) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    selection_id,
                    selection.series_uid,
                    selection.selected_at,
                    payload_text,
                    selection.provenance.workflow_id,
                    _next_sequence(
                        connection,
                        "edes_selections",
                        "selection_sequence",
                        selection.series_uid,
                    ),
                ),
            )
            return selection_id

        return self._in_write_transaction(body)

    def save_approval(self, snapshot: ApprovalSnapshot) -> str:
        """Atomic insert primitive for one immutable approval snapshot.

        This node ships the primitive only; the dependency-graph phrase
        ``approve(approved_by)`` is Node 7 controller composition on top of it
        (recorded interpretation, contract scope item 6).

        CFM-001 correction round 2 (finding 3): this writer REFUSES
        null-set measured approvals unconditionally — legacy null-bound
        records remain readable via the load paths, never re-sealable.
        """

        payload_text = self._content_addressed_payload(snapshot, snapshot.approval_id)

        def body(connection: sqlite3.Connection) -> str:
            # CFM-001 correction F6: the unchecked primitive no longer
            # bypasses inclusion invariants. It remains scoped to LEGACY
            # IMPORT (a null binding is tolerated ONLY where no applicable
            # set exists at all); wherever an applicable set resolves, the
            # verdict matches the checked path exactly.
            measurement_probe_row = connection.execute(
                "SELECT series_uid, json_extract(payload,'$.inclusion_set_id'),"
                " payload FROM measurements WHERE measurement_id = ?",
                (snapshot.measurement_id,),
            ).fetchone()
            if measurement_probe_row is not None:
                m_inclusion = (
                    None if measurement_probe_row[1] is None else str(measurement_probe_row[1])
                )
                m_payload_text = str(measurement_probe_row[2])
                if (snapshot.inclusion_set_id or None) != (m_inclusion or None):
                    raise SeriesCoherenceError(
                        "approval inclusion_set_id "
                        f"{snapshot.inclusion_set_id!r} does not match the "
                        f"measurement's binding {m_inclusion!r}"
                    )
                m_gaps_raw = json.loads(str(m_payload_text)).get("stack_gaps_mm")
                self._verify_inclusion_binding(
                    connection,
                    series_uid=snapshot.series_uid,
                    head_revision_id=snapshot.source_revision_id,
                    workflow_id=snapshot.provenance.workflow_id,
                    selection_id=snapshot.selection_id,
                    bound_inclusion_set_id=m_inclusion,
                    allow_null_set=False,
                    gaps_length=(len(m_gaps_raw) if isinstance(m_gaps_raw, list) else None),
                )
            # Series-coherence guards (amendment B), all before the INSERT.
            measurement_row = connection.execute(
                "SELECT series_uid, source_revision_id, payload FROM measurements"
                " WHERE measurement_id = ?",
                (snapshot.measurement_id,),
            ).fetchone()
            if measurement_row is None:
                raise SeriesCoherenceError(
                    f"measurement {snapshot.measurement_id!r} is not present"
                )
            m_series, m_source_revision, m_payload = measurement_row
            if snapshot.series_uid != m_series:
                raise SeriesCoherenceError(
                    "approval series_uid "
                    f"{snapshot.series_uid!r} does not match measurement series_uid "
                    f"{m_series!r}"
                )
            measurement_data = json.loads(m_payload)
            if (
                snapshot.ed_frame != measurement_data["ed_frame"]
                or snapshot.es_frame != measurement_data["es_frame"]
            ):
                raise SeriesCoherenceError(
                    f"approval ED/ES frames ({snapshot.ed_frame}, {snapshot.es_frame})"
                    " do not match measurement frames "
                    f"({measurement_data['ed_frame']}, {measurement_data['es_frame']})"
                )
            if snapshot.source_revision_id != m_source_revision:
                raise SeriesCoherenceError(
                    "approval source_revision_id "
                    f"{snapshot.source_revision_id!r} does not match measurement "
                    f"source_revision_id {m_source_revision!r}"
                )
            revision_row = connection.execute(
                "SELECT series_uid FROM contour_revisions WHERE revision_id = ?",
                (snapshot.source_revision_id,),
            ).fetchone()
            if revision_row is None:
                raise SeriesCoherenceError(
                    f"source revision {snapshot.source_revision_id!r} is not present"
                )
            if snapshot.series_uid != revision_row[0]:
                raise SeriesCoherenceError(
                    "approval series_uid "
                    f"{snapshot.series_uid!r} does not match source revision series_uid "
                    f"{revision_row[0]!r}"
                )
            connection.execute(
                "INSERT INTO approval_snapshots (approval_id, series_uid,"
                " source_revision_id, measurement_id, approved_at, payload,"
                " workflow_id, selection_id, qc_report_digest, inclusion_set_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.approval_id,
                    snapshot.series_uid,
                    snapshot.source_revision_id,
                    snapshot.measurement_id,
                    snapshot.approved_at,
                    payload_text,
                    snapshot.provenance.workflow_id,
                    snapshot.selection_id,
                    snapshot.qc_report_digest,
                    snapshot.inclusion_set_id,
                ),
            )
            return snapshot.approval_id

        return self._in_write_transaction(body)

    def save_qc_decision(self, record: QCDecisionRecord) -> str:
        """Atomic insert primitive for one immutable QC decision record.

        Milestone N7.1: durable persistence for B-LITE sentinel approval
        decisions WITHOUT weakening ``approval_snapshots`` or its foreign key
        to ``measurements`` — the decision table anchors to
        ``contour_revisions`` instead (a sentinel decision carries no numeric
        clinical metric, hence no Measurement row exists to reference).

        Series-coherence guards run inside the same ``BEGIN IMMEDIATE``
        transaction and BEFORE the INSERT, mirroring :meth:`save_approval`:
        the referenced revision must exist and belong to the record's series.
        The column-level foreign key plus the three append-only triggers
        enforce the same rules at schema level (defense in depth).
        """

        payload_text = self._content_addressed_payload(record, record.decision_id)

        def body(connection: sqlite3.Connection) -> str:
            # Series-coherence guard, before the INSERT: resolve the anchored
            # revision to its stored series inside this transaction.
            revision_row = connection.execute(
                "SELECT series_uid FROM contour_revisions WHERE revision_id = ?",
                (record.source_revision_id,),
            ).fetchone()
            if revision_row is None:
                raise SeriesCoherenceError(
                    f"source revision {record.source_revision_id!r} is not present"
                )
            if record.series_uid != revision_row[0]:
                raise SeriesCoherenceError(
                    "decision series_uid "
                    f"{record.series_uid!r} does not match source revision series_uid "
                    f"{revision_row[0]!r}"
                )
            connection.execute(
                "INSERT INTO qc_decision_records (decision_id, series_uid,"
                " source_revision_id, ed_frame, es_frame, qc_verdict, approved_by,"
                " approved_at, override_reason, payload, workflow_id, selection_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.decision_id,
                    record.series_uid,
                    record.source_revision_id,
                    record.ed_frame,
                    record.es_frame,
                    record.qc_verdict,
                    record.approved_by,
                    record.approved_at,
                    record.override_reason,
                    payload_text,
                    record.provenance.workflow_id,
                    record.selection_id,
                ),
            )
            return record.decision_id

        return self._in_write_transaction(body)

    # --- manifest / inclusion-set identity tables (S2C-001 clusters A + G) ---

    def save_source_manifest(self, manifest: DicomSourceManifest) -> str:
        """Persist one DICOM source manifest atomically (header + child rows).

        Idempotent ONLY when an existing row matches on header fields, exact
        canonical payload TEXT and every child file row; any divergence from
        a same-``workflow_id`` row fails closed (:class:`PayloadIntegrityError`)
        instead of mutating history.
        """
        payload_text = canonical_json(manifest)

        def body(connection: sqlite3.Connection) -> str:
            existing = connection.execute(
                "SELECT study_uid, series_uid, cine_volume_digest, payload"
                " FROM dicom_source_manifests WHERE workflow_id = ?",
                (manifest.workflow_id,),
            ).fetchone()
            if existing is not None:
                expected_header = (
                    manifest.study_uid,
                    manifest.series_uid,
                    manifest.cine_volume_digest,
                    payload_text,
                )
                if tuple(existing) != expected_header:
                    raise PayloadIntegrityError(
                        f"source manifest {manifest.workflow_id!r} already exists "
                        "with different header/payload bytes; workflow identities "
                        "are content-addressed and immutable"
                    )
                stored_children = connection.execute(
                    "SELECT ordinal, relative_path, sop_instance_uid, file_sha256"
                    " FROM dicom_source_files WHERE workflow_id = ? ORDER BY ordinal",
                    (manifest.workflow_id,),
                ).fetchall()
                expected_children = [
                    (
                        ordinal,
                        source_file.relative_path,
                        source_file.sop_instance_uid,
                        source_file.file_sha256,
                    )
                    for ordinal, source_file in enumerate(manifest.files)
                ]
                if [tuple(row) for row in stored_children] != expected_children:
                    raise PayloadIntegrityError(
                        f"source manifest {manifest.workflow_id!r} already exists "
                        "with different contributing-file rows"
                    )
                return manifest.workflow_id
            connection.execute(
                "INSERT INTO dicom_source_manifests (workflow_id, study_uid,"
                " series_uid, cine_volume_digest, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    manifest.workflow_id,
                    manifest.study_uid,
                    manifest.series_uid,
                    manifest.cine_volume_digest,
                    payload_text,
                ),
            )
            for ordinal, source_file in enumerate(manifest.files):
                connection.execute(
                    "INSERT INTO dicom_source_files (workflow_id, ordinal,"
                    " relative_path, sop_instance_uid, file_sha256)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        manifest.workflow_id,
                        ordinal,
                        source_file.relative_path,
                        source_file.sop_instance_uid,
                        source_file.file_sha256,
                    ),
                )
            return manifest.workflow_id

        return self._in_write_transaction(body)

    def load_source_manifest(self, workflow_id: str) -> DicomSourceManifest | None:
        """Load and fully re-verify one source manifest, else ``None``.

        Re-verifies the derived manifest digest against the stored
        ``workflow_id``, byte-canonical storage of the payload TEXT and
        child-row equivalence before returning.
        """
        row = (
            self._require_connection()
            .execute(
                "SELECT study_uid, series_uid, cine_volume_digest, payload"
                " FROM dicom_source_manifests WHERE workflow_id = ?",
                (workflow_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        study_uid, series_uid, cine_volume_digest, payload_text = (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
        )
        child_rows = (
            self._require_connection()
            .execute(
                "SELECT ordinal, relative_path, sop_instance_uid, file_sha256"
                " FROM dicom_source_files WHERE workflow_id = ? ORDER BY ordinal",
                (workflow_id,),
            )
            .fetchall()
        )
        files = tuple(
            DicomSourceFile(
                relative_path=str(child[1]),
                sop_instance_uid=str(child[2]),
                file_sha256=str(child[3]),
            )
            for child in child_rows
        )
        reconstructed = DicomSourceManifest(
            study_uid=study_uid,
            series_uid=series_uid,
            cine_volume_digest=cine_volume_digest,
            files=files,
            # ``temporal_axis_provenance`` is identity-relevant, so it MUST be
            # restored from the stored canonical payload — not defaulted — or a
            # "derived" series re-derives a different workflow_id and every
            # load fails the id check below. Pre-field payloads (written before
            # dicom loader r3) legitimately lack the key and default to
            # "measured" (see ``_temporal_axis_provenance_of_payload``).
            temporal_axis_provenance=_temporal_axis_provenance_of_payload(payload_text),
        )
        if reconstructed.workflow_id != workflow_id:
            raise PayloadIntegrityError(
                f"stored manifest rows derive workflow id "
                f"{reconstructed.workflow_id!r}, not stored id {workflow_id!r}"
            )
        if canonical_json(reconstructed) != payload_text:
            raise PayloadIntegrityError(
                f"stored source manifest {workflow_id!r} is not byte-canonical"
            )
        return reconstructed

    # --- cine-workflow registration table (S3-004E D8, schema v7) --------------

    def append_registration(
        self,
        workflow_id: str,
        cine_volume_digest: str,
        store_record_uri: str,
    ) -> tuple[bytes, str, int]:
        """Append one cine-workflow registration record (append-only, v7).

        Allocates ``sequence`` for ``workflow_id`` INSIDE one
        ``BEGIN IMMEDIATE`` transaction (``MAX(sequence)+1``), serializes the
        canonical registration-record bytes (frozen schema, ``sequence``
        included), inserts the row and returns
        ``(record_bytes, record_sha256, sequence)``.

        Fails closed with :class:`ValueError` when:

        * ``workflow_id`` is not persisted in ``dicom_source_manifests``
          (the FK fails — no registration may reference an unregistered
          workflow);
        * ``cine_volume_digest`` disagrees with the persisted manifest;
        * ``store_record_uri`` is not a well-formed ``cas://`` URI;
        * the composite ``(workflow_id, sequence)`` key already exists
          (idempotent conflict — the table is append-only, never replaced).
        """
        for name, value in (
            ("workflow_id", workflow_id),
            ("cine_volume_digest", cine_volume_digest),
        ):
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be 64-character lowercase hex")
        if not isinstance(store_record_uri, str) or not store_record_uri.startswith("cas://"):
            raise ValueError(
                f"store_record_uri must be a well-formed cas:// URI, got {store_record_uri!r}"
            )
        if store_record_uri[len("cas://") :].strip() == "":
            raise ValueError("store_record_uri must carry a non-empty cas:// authority/path")

        def body(connection: sqlite3.Connection) -> tuple[bytes, str, int]:
            manifest_row = connection.execute(
                "SELECT cine_volume_digest FROM dicom_source_manifests"
                " WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
            if manifest_row is None:
                raise ValueError(
                    f"workflow {workflow_id!r} is not present in "
                    "dicom_source_manifests; persist the source manifest before "
                    "registering a cine workflow"
                )
            if str(manifest_row[0]) != cine_volume_digest:
                raise PayloadIntegrityError(
                    f"cine_volume_digest {cine_volume_digest!r} disagrees with the "
                    f"persisted manifest digest {str(manifest_row[0])!r} for "
                    f"workflow {workflow_id!r}"
                )
            sequence_row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM cine_workflow_registrations"
                " WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
            assert sequence_row is not None
            sequence = int(sequence_row[0])
            existing = connection.execute(
                "SELECT 1 FROM cine_workflow_registrations"
                " WHERE workflow_id = ? AND sequence = ?",
                (workflow_id, sequence),
            ).fetchone()
            if existing is not None:
                raise ValueError(
                    f"cine-workflow registration ({workflow_id!r}, {sequence}) "
                    "already exists; the table is append-only and never replaced"
                )
            record = {
                "schema": CINE_WORKFLOW_REGISTRATION_RECORD_SCHEMA,
                "sequence": sequence,
                "manifest_digest": workflow_id,
                "cine_volume_digest": cine_volume_digest,
                "store_record_uri": store_record_uri,
                "workflow_id": workflow_id,
            }
            record_bytes = _canonical_registration_record_bytes(record)
            record_sha256 = hashlib.sha256(record_bytes).hexdigest()
            connection.execute(
                "INSERT INTO cine_workflow_registrations (workflow_id, sequence,"
                " store_record_uri, record_sha256, record_bytes)"
                " VALUES (?, ?, ?, ?, ?)",
                (workflow_id, sequence, store_record_uri, record_sha256, record_bytes),
            )
            return record_bytes, record_sha256, sequence

        result = self._in_write_transaction_typed(body)
        assert isinstance(result, tuple) and len(result) == 3
        return result[0], result[1], result[2]

    def read_registration(
        self, workflow_id: str, sequence: int
    ) -> CineWorkflowRegistrationV1:
        """Read one persisted registration and return the store-issued authority.

        Raises ``KeyError`` when no row exists for ``(workflow_id, sequence)``.
        Performs the frozen integrity checks and raises
        :class:`PayloadIntegrityError` on any mismatch:

        * ``sha256(record_bytes)`` == the stored ``record_sha256``;
        * the canonical re-serialization of the parsed record is byte-equal to
          the stored bytes (guards against non-canonical storage);
        * the parsed ``schema`` is EXACT
          ``"cine-workflow-registration-record-v1"``;
        * the parsed ``workflow_id`` / ``sequence`` / ``manifest_digest`` /
          ``cine_volume_digest`` / ``store_record_uri`` agree with the row
          columns.

        This is the SINGLE construction path of
        :class:`~cmr_persistence.records.CineWorkflowRegistrationV1`.
        """
        row = (
            self._require_connection()
            .execute(
                "SELECT workflow_id, sequence, store_record_uri, record_sha256,"
                " record_bytes FROM cine_workflow_registrations"
                " WHERE workflow_id = ? AND sequence = ?",
                (workflow_id, sequence),
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(
                f"no cine-workflow registration for ({workflow_id!r}, {sequence!r})"
            )
        stored_workflow_id = str(row[0])
        stored_sequence = int(row[1])
        stored_uri = str(row[2])
        stored_sha256 = str(row[3])
        stored_bytes = bytes(row[4])

        if hashlib.sha256(stored_bytes).hexdigest() != stored_sha256:
            raise PayloadIntegrityError(
                f"cine-workflow registration ({stored_workflow_id!r}, "
                f"{stored_sequence}) record bytes do not match their stored digest"
            )
        try:
            parsed = json.loads(stored_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PayloadIntegrityError(
                f"cine-workflow registration ({stored_workflow_id!r}, "
                f"{stored_sequence}) record bytes are not valid canonical JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise PayloadIntegrityError(
                "cine-workflow registration record must be a JSON object"
            )
        if _canonical_registration_record_bytes(parsed) != stored_bytes:
            raise PayloadIntegrityError(
                f"cine-workflow registration ({stored_workflow_id!r}, "
                f"{stored_sequence}) is not canonically serialized"
            )
        if parsed.get("schema") != CINE_WORKFLOW_REGISTRATION_RECORD_SCHEMA:
            raise PayloadIntegrityError(
                "cine-workflow registration record schema must be exactly "
                f"{CINE_WORKFLOW_REGISTRATION_RECORD_SCHEMA!r}, got "
                f"{parsed.get('schema')!r}"
            )
        if not stored_uri.startswith("cas://") or not stored_uri[len("cas://") :].strip():
            raise PayloadIntegrityError(
                f"cine-workflow registration store_record_uri {stored_uri!r} "
                "is not a well-formed cas:// URI"
            )
        expected_columns = (
            ("workflow_id", parsed.get("workflow_id"), stored_workflow_id),
            ("sequence", parsed.get("sequence"), stored_sequence),
            ("manifest_digest", parsed.get("manifest_digest"), stored_workflow_id),
            ("store_record_uri", parsed.get("store_record_uri"), stored_uri),
        )
        for name, parsed_value, column_value in expected_columns:
            if parsed_value != column_value:
                raise PayloadIntegrityError(
                    f"cine-workflow registration record field {name!r} "
                    f"{parsed_value!r} diverges from the stored column "
                    f"{column_value!r}"
                )
        manifest_row = (
            self._require_connection()
            .execute(
                "SELECT cine_volume_digest FROM dicom_source_manifests"
                " WHERE workflow_id = ?",
                (stored_workflow_id,),
            )
            .fetchone()
        )
        if manifest_row is None:
            raise PayloadIntegrityError(
                f"cine-workflow registration {stored_workflow_id!r} references a "
                "workflow with no persisted DICOM source manifest"
            )
        if parsed.get("cine_volume_digest") != str(manifest_row[0]):
            raise PayloadIntegrityError(
                f"cine-workflow registration record field 'cine_volume_digest' "
                f"{parsed.get('cine_volume_digest')!r} diverges from the persisted "
                f"manifest digest {str(manifest_row[0])!r}"
            )
        return CineWorkflowRegistrationV1(
            workflow_id=stored_workflow_id,
            manifest_digest=str(parsed["manifest_digest"]),
            cine_volume_digest=str(parsed["cine_volume_digest"]),
            sequence=stored_sequence,
            store_record_bytes=stored_bytes,
            store_record_sha256=stored_sha256,
            persisted_record_bytes=stored_bytes,
            # The single construction path's issuance proof (correction6 /
            # R3 finding 1): ONLY this API may construct the authority, and
            # the object it returns carries the store-issued marker the
            # kernel seam re-verifies.
            _issuance_token=registration_issuance_token(),
        )

    def save_slice_inclusion_set(
        self,
        inclusion_set: SliceInclusionSet,
        *,
        expected_n_slices: int,
    ) -> str:
        """Append-only persist of one schema-ready :class:`SliceInclusionSet`.

        Content-addressed like every record table. Coherence guards inside
        this transaction: the anchored revision and selection must exist and
        belong to the set's series; frame roles were validated at record
        construction (ED|ES vocabulary, duplicate-key rejection). CFM-001:
        the row additionally receives its ``inclusion_sequence`` store
        metadata (MAX+1 within ``(workflow_id, series_uid)`` under this
        transaction).

        Correction round 3 (R4): ``expected_n_slices`` is MANDATORY — every
        new write must prove its grid covers exactly the cine's slice count.
        """
        if (
            isinstance(expected_n_slices, bool)
            or not isinstance(expected_n_slices, int)
            or expected_n_slices < 1
        ):
            raise SeriesCoherenceError(
                "expected_n_slices must be a positive integer (the cine's "
                f"actual slice count); got {expected_n_slices!r}"
            )
        payload_text = self._content_addressed_payload(
            inclusion_set, inclusion_set.inclusion_set_id
        )

        def body(connection: sqlite3.Connection) -> str:
            _insert_slice_inclusion_set_row(
                connection, inclusion_set, payload_text, expected_n_slices
            )
            return inclusion_set.inclusion_set_id

        return self._in_write_transaction(body)

    def save_slice_inclusion_set_checked(
        self,
        inclusion_set: SliceInclusionSet,
        *,
        expected_head_revision_id: str | None,
        expected_selection_id: str | None,
        expected_inclusion_set_id: str | None,
        expected_n_slices: int,
    ) -> str:
        """Checked persist of one physician-authored inclusion decision grid.

        FUNCV1 physician applicability loop C1 (P1-1): the decision-persist
        path is INDEPENDENT of recompute success. It reuses the SAME
        SliceInclusionSet authority model (complete 5xN grid, binary weights,
        coherence guards in :func:`_insert_slice_inclusion_set_row`) and the
        SAME three-way nullable lineage CAS as the compound checked refresh
        (head/selection/inclusion resolved atomically inside ONE
        ``BEGIN IMMEDIATE`` transaction, BEFORE any INSERT). Because a new
        applicable grid supersedes the previous grid, the previous coherent
        analysis/approval can no longer resolve as current
        (``_resolve_current_analysis`` binds the LATEST APPLICABLE set), so
        the old result is invalidated WITHOUT any new measurement row: the
        series reopens with ``analysis=None`` / ``not_current`` and the seal
        refuses ``AnalysisAuthorityError``/``StaleHeadError`` (zero staged
        rows) until a recompute against the new grid succeeds. Missing
        geometry keeps refusing recompute/seal fail-closed — persistence
        never fabricates contours or zeroes.

        No new table, model, or state machine: the row lands in the EXISTING
        ``slice_inclusion_sets`` table. Idempotency: a grid whose content
        equals the current applicable set returns the current id WITHOUT
        staging a duplicate row (no ``selected_at``/authorship participation
        in the signature — the physician-meaningful grid content only).
        """
        expected_n_slices = self._validate_expected_n_slices(expected_n_slices)
        expected_head_revision_id = self._validate_checked_identity(
            "expected_head_revision_id", expected_head_revision_id
        )
        expected_selection_id = self._validate_checked_identity(
            "expected_selection_id", expected_selection_id
        )
        expected_inclusion_set_id = self._validate_checked_identity(
            "expected_inclusion_set_id", expected_inclusion_set_id
        )
        set_payload = self._content_addressed_payload(
            inclusion_set, inclusion_set.inclusion_set_id
        )

        def _grid_signature(record: SliceInclusionSet) -> str:
            return canonical_digest(
                sorted(
                    (
                        int(entry.slice_index),
                        entry.frame_role.value,
                        entry.structure.value,
                        bool(entry.included),
                    )
                    for entry in record.entries
                )
            )

        def body(connection: sqlite3.Connection) -> str:
            # The complete CAS is the first lineage-sensitive operation in
            # this transaction. It must precede every unchanged/duplicate
            # decision and every INSERT.
            actual = self._resolve_current_nullable_lineage(
                connection,
                series_uid=inclusion_set.series_uid,
                expected_n_slices=expected_n_slices,
            )
            self._assert_checked_lineage(
                actual=actual,
                expected_head_revision_id=expected_head_revision_id,
                expected_selection_id=expected_selection_id,
                expected_inclusion_set_id=expected_inclusion_set_id,
            )
            if actual[0] is None or actual[1] is None:
                raise SeriesCoherenceError(
                    "checked applicability persist requires a current head "
                    "and a persisted ED/ES selection"
                )
            current_set: SliceInclusionSet | None = None
            if actual[2] is not None:
                head_row = self._current_head_row(connection, inclusion_set.series_uid)
                resolved = self._resolve_latest_applicable_set(
                    connection,
                    series_uid=inclusion_set.series_uid,
                    head_revision_id=actual[0],
                    selection_id=actual[1],
                    workflow_id=(
                        None if head_row is None or head_row[3] is None
                        else str(head_row[3])
                    ),
                    expected_n_slices=expected_n_slices,
                )
                if resolved is not None:
                    current_set = resolved[0]
            if (
                current_set is not None
                and _grid_signature(current_set) == _grid_signature(inclusion_set)
            ):
                return current_set.inclusion_set_id
            _insert_slice_inclusion_set_row(
                connection, inclusion_set, set_payload, expected_n_slices
            )
            return inclusion_set.inclusion_set_id

        return self._in_write_transaction(body)

    def load_slice_inclusion_sets(self, series_uid: str) -> list[SliceInclusionSet]:
        """All persisted inclusion sets for a series, digest-verified.

        CFM-001: ordered by STORE SEQUENCE (``inclusion_sequence``), never
        by ``selected_at`` — "latest set" means highest store sequence.
        """
        rows = (
            self._require_connection()
            .execute(
                "SELECT inclusion_set_id, payload FROM slice_inclusion_sets"
                " WHERE series_uid = ?"
                " ORDER BY inclusion_sequence ASC, inclusion_set_id ASC",
                (series_uid,),
            )
            .fetchall()
        )
        sets: list[SliceInclusionSet] = []
        for stored_id, payload_text in rows:
            reconstructed = self._verify_and_reconstruct(
                str(stored_id), str(payload_text), slice_inclusion_set_from_payload
            )
            sets.append(reconstructed)
        return sets

    # --- S3-001 whole-cine / model-run records (schema v5 additive hatch) -----

    # --- S3-003 model registry records (schema v6 additive hatch) -------------

    @staticmethod
    def _model_bundle_from_row(row: tuple[Any, ...]) -> ModelBundleRecord:
        record = ModelBundleRecord(
            bundle_id=str(row[0]),
            model_id=str(row[1]),
            semantic_version=str(row[2]),
            weights_format=str(row[3]),
            payload_sha256=str(row[4]),
            manifest_json=str(row[5]),
            install_root_relpath=str(row[6]),
            installed_at=str(row[7]),
            software_version=str(row[8]),
        )
        if record.digest != str(row[9]):
            raise PayloadIntegrityError(
                f"stored model bundle digest does not match {record.bundle_id!r}"
            )
        return record

    @staticmethod
    def _model_activation_from_row(row: tuple[Any, ...]) -> ModelActivationRecord:
        record = ModelActivationRecord(
            bundle_id=str(row[1]),
            previous_activation_id=None if row[2] is None else str(row[2]),
            reason=str(row[3]),
            activated_at=str(row[4]),
        )
        if record.activation_id != str(row[0]) or record.digest != str(row[5]):
            raise PayloadIntegrityError(
                f"stored model activation digest does not match {record.activation_id!r}"
            )
        return record

    def save_model_bundle(self, record: ModelBundleRecord) -> ModelBundleRecord:
        """Insert one bundle row, returning an identical existing row."""

        self._content_addressed_payload(record, record.digest)

        def body(connection: sqlite3.Connection) -> ModelBundleRecord:
            row = connection.execute(
                "SELECT bundle_id, model_id, semantic_version, weights_format,"
                " payload_sha256, manifest_json, install_root_relpath, installed_at,"
                " software_version, digest FROM model_bundles WHERE bundle_id = ?",
                (record.bundle_id,),
            ).fetchone()
            if row is not None:
                existing = self._model_bundle_from_row(tuple(row))
                immutable_fields = (
                    "bundle_id",
                    "model_id",
                    "semantic_version",
                    "weights_format",
                    "payload_sha256",
                    "manifest_json",
                    "install_root_relpath",
                )
                if any(
                    getattr(existing, field) != getattr(record, field)
                    for field in immutable_fields
                ):
                    raise CMRError(
                        REGISTRY_CONFLICT,
                        f"bundle id {record.bundle_id!r} is already bound to different content",
                    )
                return existing
            connection.execute(
                "INSERT INTO model_bundles (bundle_id, model_id, semantic_version,"
                " weights_format, payload_sha256, manifest_json, install_root_relpath,"
                " installed_at, software_version, digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.bundle_id,
                    record.model_id,
                    record.semantic_version,
                    record.weights_format,
                    record.payload_sha256,
                    record.manifest_json,
                    record.install_root_relpath,
                    record.installed_at,
                    record.software_version,
                    record.digest,
                ),
            )
            return record

        return self._in_write_transaction(body)

    def load_model_bundle(self, bundle_id: str) -> ModelBundleRecord | None:
        row = (
            self._require_connection()
            .execute(
                "SELECT bundle_id, model_id, semantic_version, weights_format,"
                " payload_sha256, manifest_json, install_root_relpath, installed_at,"
                " software_version, digest FROM model_bundles WHERE bundle_id = ?",
                (bundle_id,),
            )
            .fetchone()
        )
        return None if row is None else self._model_bundle_from_row(tuple(row))

    def list_model_bundles(self) -> tuple[ModelBundleRecord, ...]:
        rows = (
            self._require_connection()
            .execute(
                "SELECT bundle_id, model_id, semantic_version, weights_format,"
                " payload_sha256, manifest_json, install_root_relpath, installed_at,"
                " software_version, digest FROM model_bundles ORDER BY installed_at, digest"
            )
            .fetchall()
        )
        return tuple(self._model_bundle_from_row(tuple(row)) for row in rows)

    def load_registry_state(self) -> RegistryState | None:
        row = (
            self._require_connection()
            .execute(
                "SELECT active_activation_id, updated_at FROM model_registry_state WHERE id = 1"
            )
            .fetchone()
        )
        return None if row is None else RegistryState(str(row[0]), str(row[1]))

    def load_model_activation(self, activation_id: str) -> ModelActivationRecord | None:
        row = (
            self._require_connection()
            .execute(
                "SELECT activation_id, bundle_id, previous_activation_id, reason,"
                " activated_at, digest FROM model_activations WHERE activation_id = ?",
                (activation_id,),
            )
            .fetchone()
        )
        return None if row is None else self._model_activation_from_row(tuple(row))

    def current_active_activation(self) -> ModelActivationRecord | None:
        state = self.load_registry_state()
        if state is None:
            return None
        activation = self.load_model_activation(state.active_activation_id)
        if activation is None:
            raise CMRError(REGISTRY_STATE_CONFLICT, "registry state points to a missing activation")
        return activation

    def activate_model(
        self,
        bundle_id: str,
        reason: str,
        expected_active_activation_id: str | None,
        activated_at: str,
    ) -> ModelActivationRecord:
        def body(connection: sqlite3.Connection) -> ModelActivationRecord:
            if (
                connection.execute(
                    "SELECT 1 FROM model_bundles WHERE bundle_id = ?", (bundle_id,)
                ).fetchone()
                is None
            ):
                raise CMRError(REGISTRY_BUNDLE_UNKNOWN, f"unknown bundle id {bundle_id!r}")
            state_row = connection.execute(
                "SELECT active_activation_id FROM model_registry_state WHERE id = 1"
            ).fetchone()
            current = None if state_row is None else str(state_row[0])
            if expected_active_activation_id != current:
                raise CMRError(
                    REGISTRY_STATE_CONFLICT,
                    "expected active activation does not match the registry state",
                )
            activation = ModelActivationRecord(
                bundle_id=bundle_id,
                previous_activation_id=current,
                reason=reason,
                activated_at=activated_at,
            )
            connection.execute(
                "INSERT INTO model_activations (activation_id, bundle_id,"
                " previous_activation_id, reason, activated_at, digest) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    activation.activation_id,
                    activation.bundle_id,
                    activation.previous_activation_id,
                    activation.reason,
                    activation.activated_at,
                    activation.digest,
                ),
            )
            if current is None:
                connection.execute(
                    "INSERT INTO model_registry_state (id, active_activation_id, updated_at)"
                    " VALUES (1, ?, ?)",
                    (activation.activation_id, activated_at),
                )
            else:
                cursor = connection.execute(
                    "UPDATE model_registry_state SET active_activation_id = ?, updated_at = ?"
                    " WHERE id = 1 AND active_activation_id = ?",
                    (activation.activation_id, activated_at, expected_active_activation_id),
                )
                if cursor.rowcount != 1:
                    raise CMRError(
                        REGISTRY_STATE_CONFLICT, "registry state changed during activation"
                    )
            return activation

        return self._in_write_transaction(body)

    def rollback_model(
        self, expected_active_activation_id: str | None, activated_at: str
    ) -> ModelActivationRecord:
        state = self.load_registry_state()
        if state is None or state.active_activation_id != expected_active_activation_id:
            raise CMRError(
                REGISTRY_STATE_CONFLICT, "expected active activation does not match state"
            )
        current = self.current_active_activation()
        if current is None or current.previous_activation_id is None:
            raise CMRError(
                REGISTRY_ROLLBACK_UNAVAILABLE, "the active activation has no previous activation"
            )
        previous = self.load_model_activation(current.previous_activation_id)
        if previous is None:
            raise CMRError(REGISTRY_STATE_CONFLICT, "activation history is dangling")
        if self.load_model_bundle(previous.bundle_id) is None:
            raise CMRError(REGISTRY_BUNDLE_UNKNOWN, "rollback target bundle is missing")
        return self.activate_model(
            previous.bundle_id, "rollback-v1", expected_active_activation_id, activated_at
        )

    def save_model_run(self, record: ModelRunRecord) -> str:
        """Append-only persist of one immutable model-run provenance record.

        Content-addressed per the store convention: the writer asserts
        ``sha256(canonical payload TEXT) == record.record_digest`` BEFORE the
        INSERT; the schema-level append-only trigger set (no update / no
        delete / no replace) closes every mutation path afterwards.
        """
        payload_text = self._content_addressed_payload(record, record.record_digest)

        def body(connection: sqlite3.Connection) -> str:
            connection.execute(
                "INSERT INTO model_runs (model_run_id, model_name, model_version,"
                " model_sha256, framework, study_uid, series_uid,"
                " input_content_hash, mask_artifact_sha256, uq_method, payload,"
                " record_digest, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.model_run_id,
                    record.model_name,
                    record.model_version,
                    record.model_sha256,
                    record.framework,
                    record.study_uid,
                    record.series_uid,
                    record.input_content_hash,
                    record.mask_artifact_sha256,
                    record.uq_method,
                    payload_text,
                    record.record_digest,
                    record.created_at,
                ),
            )
            return record.model_run_id

        return self._in_write_transaction(body)

    def load_model_run(self, model_run_id: str) -> ModelRunRecord | None:
        """Load and fully re-verify one model run, else ``None``.

        Verifies the payload digest against the record's ``record_digest``, the
        primary-key agreement with the payload field and byte-canonical
        storage before returning.
        """
        row = (
            self._require_connection()
            .execute(
                "SELECT model_run_id, model_name, model_version, model_sha256,"
                " framework, study_uid, series_uid, input_content_hash,"
                " mask_artifact_sha256, uq_method, payload, record_digest,"
                " created_at"
                " FROM model_runs WHERE model_run_id = ?",
                (model_run_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        stored = ModelRunRow(
            model_run_id=str(row[0]),
            model_name=str(row[1]),
            model_version=str(row[2]),
            model_sha256=str(row[3]),
            framework=str(row[4]),
            study_uid=str(row[5]),
            series_uid=str(row[6]),
            input_content_hash=str(row[7]),
            mask_artifact_sha256=str(row[8]),
            uq_method=None if row[9] is None else str(row[9]),
            payload=str(row[10]),
            record_digest=str(row[11]),
            created_at=str(row[12]),
        )
        # Sol S2: the stored digest column must equal sha256(payload TEXT)
        # AND the recomputed record digest — a drifted column can never
        # masquerade as provenance.
        payload_sha256 = _sha256_hex(stored.payload)
        if stored.record_digest != payload_sha256:
            raise PayloadIntegrityError(
                f"stored model_runs.record_digest column {stored.record_digest!r}"
                f" does not equal sha256(payload) {payload_sha256!r} for model"
                f" run {stored.model_run_id!r}"
            )
        try:
            data = json.loads(stored.payload)
        except ValueError as exc:
            raise PayloadIntegrityError(
                f"stored model-run payload for {stored.model_run_id!r} is not valid JSON"
            ) from exc
        reconstructed = model_run_from_payload(dict(data))
        if _sha256_hex(stored.payload) != reconstructed.record_digest:
            raise PayloadIntegrityError(
                f"stored model-run payload digest does not match the record "
                f"digest {reconstructed.record_digest!r} of "
                f"{stored.model_run_id!r}"
            )
        if reconstructed.model_run_id != stored.model_run_id:
            raise PayloadIntegrityError(
                f"stored model_runs row id {stored.model_run_id!r} does not equal "
                f"the payload field {reconstructed.model_run_id!r}"
            )
        if canonical_json(reconstructed) != stored.payload:
            raise PayloadIntegrityError(
                f"stored model run {stored.model_run_id!r} is not byte-canonical"
            )
        _assert_model_run_columns_match_payload(stored, reconstructed)
        return reconstructed

    def save_whole_cine_prediction(self, prediction: WholeCinePrediction) -> str:
        """Append-only persist of one whole-cine prediction artifact.

        The SQL-level source/run coupling CHECK plus the ``model_runs`` foreign
        key enforce the same ai/manual-seed invariants the dataclass enforces;
        the payload digest identity is asserted before the INSERT.
        """
        payload_text = self._content_addressed_payload(prediction, prediction.prediction_id)

        def body(connection: sqlite3.Connection) -> str:
            # Sol S5: resolve the model-run row INSIDE this transaction and
            # require full lineage/identity agreement before any row is staged.
            if prediction.model_run_id is not None:
                run_row = connection.execute(
                    "SELECT study_uid, series_uid, model_name, model_version"
                    " FROM model_runs WHERE model_run_id = ?",
                    (prediction.model_run_id,),
                ).fetchone()
                if run_row is None:
                    raise SeriesCoherenceError(
                        f"model run {prediction.model_run_id!r} is not present"
                    )
                if str(run_row[0]) != prediction.study_uid:
                    raise SeriesCoherenceError(
                        "prediction study_uid "
                        f"{prediction.study_uid!r} does not match model run "
                        f"study_uid {str(run_row[0])!r}"
                    )
                if str(run_row[1]) != prediction.series_uid:
                    raise SeriesCoherenceError(
                        "prediction series_uid "
                        f"{prediction.series_uid!r} does not match model run "
                        f"series_uid {str(run_row[1])!r}"
                    )
                if prediction.model is None:
                    raise SeriesCoherenceError(
                        "model_run_id requires model identity on the prediction "
                        "record for agreement verification"
                    )
                if prediction.model.name != str(run_row[2]) or prediction.model.version != str(
                    run_row[3]
                ):
                    raise SeriesCoherenceError(
                        "prediction model identity "
                        f"{prediction.model.name}@{prediction.model.version} "
                        "does not match model run identity "
                        f"{str(run_row[2])}@{str(run_row[3])}"
                    )
            connection.execute(
                "INSERT INTO whole_cine_predictions (prediction_id, study_uid,"
                " series_uid, source, model_run_id, slice_count, frame_count,"
                " payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    prediction.prediction_id,
                    prediction.study_uid,
                    prediction.series_uid,
                    prediction.source,
                    prediction.model_run_id,
                    prediction.slice_count,
                    prediction.frame_count,
                    payload_text,
                    prediction.created_at,
                ),
            )
            return prediction.prediction_id

        return self._in_write_transaction(body)

    def load_whole_cine_prediction(self, prediction_id: str) -> WholeCinePrediction | None:
        """Load and fully re-verify one whole-cine prediction, else ``None``.

        A module-level reconstruction memo may short-circuit the rebuild when
        this exact row was already reconstructed in-process: rows are
        immutable and content-addressed (prediction_id == payload digest), so
        a hit returns the same verified object with zero fidelity loss. The
        full digest/canonical verification still runs on every miss.
        """
        memo_hit = _wc_prediction_memo_get(self._path, prediction_id)
        if memo_hit is not None:
            return memo_hit
        row = (
            self._require_connection()
            .execute(
                "SELECT prediction_id, study_uid, series_uid, source, model_run_id,"
                " slice_count, frame_count, payload, created_at"
                " FROM whole_cine_predictions WHERE prediction_id = ?",
                (prediction_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        stored = WholeCinePredictionRow(
            prediction_id=str(row[0]),
            study_uid=str(row[1]),
            series_uid=str(row[2]),
            source=str(row[3]),
            model_run_id=None if row[4] is None else str(row[4]),
            slice_count=int(row[5]),
            frame_count=int(row[6]),
            payload=str(row[7]),
            created_at=str(row[8]),
        )
        reconstructed = self._verify_and_reconstruct(
            stored.prediction_id, stored.payload, whole_cine_prediction_from_payload
        )
        _assert_whole_cine_prediction_columns_match_payload(stored, reconstructed)
        _wc_prediction_memo_put(self._path, reconstructed)
        return reconstructed

    # --- FNC-C selected-phase (sparse) prediction artifacts (schema v8) --------

    def save_selected_phase_prediction(self, prediction: SelectedPhasePrediction) -> str:
        """Append-only persist of one FNC-C sparse prediction artifact.

        Mirrors :meth:`save_whole_cine_prediction`: the payload digest is
        asserted before the INSERT, the model-run lineage/identity checks run
        INSIDE the write transaction (study/series agreement plus model
        name/version agreement), and the table's own triggers reject UPDATE,
        DELETE and INSERT-OR-REPLACE. The identity-significant
        ``source_frame_indices`` is stored as its canonical JSON TEXT and
        re-verified against the payload on every load.
        """
        payload_text = self._content_addressed_payload(prediction, prediction.prediction_id)

        def body(connection: sqlite3.Connection) -> str:
            if prediction.model_run_id is not None:
                run_row = connection.execute(
                    "SELECT study_uid, series_uid, model_name, model_version"
                    " FROM model_runs WHERE model_run_id = ?",
                    (prediction.model_run_id,),
                ).fetchone()
                if run_row is None:
                    raise SeriesCoherenceError(
                        f"model run {prediction.model_run_id!r} is not present"
                    )
                if str(run_row[0]) != prediction.study_uid:
                    raise SeriesCoherenceError(
                        "prediction study_uid "
                        f"{prediction.study_uid!r} does not match model run "
                        f"study_uid {str(run_row[0])!r}"
                    )
                if str(run_row[1]) != prediction.series_uid:
                    raise SeriesCoherenceError(
                        "prediction series_uid "
                        f"{prediction.series_uid!r} does not match model run "
                        f"series_uid {str(run_row[1])!r}"
                    )
                if prediction.model is None:
                    raise SeriesCoherenceError(
                        "model_run_id requires model identity on the prediction "
                        "record for agreement verification"
                    )
                if prediction.model.name != str(run_row[2]) or prediction.model.version != str(
                    run_row[3]
                ):
                    raise SeriesCoherenceError(
                        "prediction model identity "
                        f"{prediction.model.name}@{prediction.model.version} "
                        "does not match model run identity "
                        f"{str(run_row[2])}@{str(run_row[3])}"
                    )
            connection.execute(
                "INSERT INTO selected_phase_predictions (prediction_id, study_uid,"
                " series_uid, source, model_run_id, slice_count, frame_count,"
                " source_frame_indices, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    prediction.prediction_id,
                    prediction.study_uid,
                    prediction.series_uid,
                    prediction.source,
                    prediction.model_run_id,
                    prediction.slice_count,
                    prediction.frame_count,
                    selected_phase_frame_indices_json(prediction.source_frame_indices),
                    payload_text,
                    prediction.created_at,
                ),
            )
            return prediction.prediction_id

        return self._in_write_transaction(body)

    def load_selected_phase_prediction(
        self, prediction_id: str
    ) -> SelectedPhasePrediction | None:
        """Load and fully re-verify one sparse prediction artifact, else ``None``.

        The digest, byte-canonical and denormalized-column (including
        ``source_frame_indices``) checks are the same fail-closed set the
        whole-cine path applies.
        """
        row = (
            self._require_connection()
            .execute(
                "SELECT prediction_id, study_uid, series_uid, source, model_run_id,"
                " slice_count, frame_count, source_frame_indices, payload, created_at"
                " FROM selected_phase_predictions WHERE prediction_id = ?",
                (prediction_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        stored = SelectedPhasePredictionRow(
            prediction_id=str(row[0]),
            study_uid=str(row[1]),
            series_uid=str(row[2]),
            source=str(row[3]),
            model_run_id=None if row[4] is None else str(row[4]),
            slice_count=int(row[5]),
            frame_count=int(row[6]),
            source_frame_indices=str(row[7]),
            payload=str(row[8]),
            created_at=str(row[9]),
        )
        reconstructed = self._verify_and_reconstruct(
            stored.prediction_id, stored.payload, selected_phase_prediction_from_payload
        )
        _assert_selected_phase_prediction_columns_match_payload(stored, reconstructed)
        return reconstructed

    def load_latest_selected_phase_prediction(
        self, series_uid: str
    ) -> SelectedPhasePrediction | None:
        """Latest persisted sparse prediction for a series by store insertion order.

        Strictly queries ``selected_phase_predictions`` (never
        ``whole_cine_predictions``), mirroring
        :meth:`load_latest_whole_cine_prediction`.
        """
        row = (
            self._require_connection()
            .execute(
                "SELECT prediction_id FROM selected_phase_predictions "
                "WHERE series_uid = ? ORDER BY rowid DESC LIMIT 1",
                (series_uid,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return self.load_selected_phase_prediction(str(row[0]))

    def save_whole_cine_revision(self, revision: WholeCineRevision) -> str:
        """Append-only persist of one whole-cine physician-edit revision.

        Coherence guards run inside this transaction BEFORE the INSERT: any
        PRESENT parent (prediction or prior revision) must resolve to the
        child's series; absent parents fail closed on the schema-level foreign
        keys (the established orphan-insert semantics).
        """
        payload_text = self._content_addressed_payload(revision, revision.revision_id)

        def body(connection: sqlite3.Connection) -> str:
            prediction_row = connection.execute(
                "SELECT series_uid, study_uid FROM whole_cine_predictions WHERE prediction_id = ?",
                (revision.parent_prediction_id,),
            ).fetchone()
            if prediction_row is not None:
                if str(prediction_row[0]) != revision.series_uid:
                    raise SeriesCoherenceError(
                        "revision series_uid "
                        f"{revision.series_uid!r} does not match parent prediction "
                        f"series_uid {str(prediction_row[0])!r}"
                    )
                # Sol S1: study_uid coherence against the base prediction.
                if str(prediction_row[1]) != revision.study_uid:
                    raise SeriesCoherenceError(
                        "revision study_uid "
                        f"{revision.study_uid!r} does not match parent prediction "
                        f"study_uid {str(prediction_row[1])!r}"
                    )
            if revision.parent_revision_id is not None:
                parent_row = connection.execute(
                    "SELECT series_uid, study_uid, version, parent_prediction_id"
                    " FROM whole_cine_revisions WHERE revision_id = ?",
                    (revision.parent_revision_id,),
                ).fetchone()
                if parent_row is not None:
                    if str(parent_row[0]) != revision.series_uid:
                        raise SeriesCoherenceError(
                            "revision series_uid "
                            f"{revision.series_uid!r} does not match parent revision "
                            f"series_uid {str(parent_row[0])!r}"
                        )
                    # Sol S1: study_uid coherence along the chain.
                    if str(parent_row[1]) != revision.study_uid:
                        raise SeriesCoherenceError(
                            "revision study_uid "
                            f"{revision.study_uid!r} does not match parent revision "
                            f"study_uid {str(parent_row[1])!r}"
                        )
                    # Sol S1: strict version continuity — skips are rejected.
                    if int(parent_row[2]) != revision.version - 1:
                        raise SeriesCoherenceError(
                            f"revision version {revision.version} does not continue "
                            f"the parent chain (parent version {int(parent_row[2])};"
                            " child must be parent_version + 1)"
                        )
                    # Sol S1: one lineage per chain — the base prediction is fixed.
                    if str(parent_row[3]) != revision.parent_prediction_id:
                        raise SeriesCoherenceError(
                            "parent_prediction_id switch rejected: child anchors "
                            f"{revision.parent_prediction_id!r} but parent revision "
                            f"anchors {str(parent_row[3])!r}"
                        )
            connection.execute(
                "INSERT INTO whole_cine_revisions (revision_id, study_uid,"
                " series_uid, version, parent_prediction_id, parent_revision_id,"
                " editor, edited_keys_json, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    revision.revision_id,
                    revision.study_uid,
                    revision.series_uid,
                    revision.version,
                    revision.parent_prediction_id,
                    revision.parent_revision_id,
                    revision.editor,
                    whole_cine_edited_keys_json(revision),
                    payload_text,
                    revision.created_at,
                ),
            )
            return revision.revision_id

        return self._in_write_transaction(body)

    def load_whole_cine_revision(self, revision_id: str) -> WholeCineRevision | None:
        """Load and fully re-verify one whole-cine revision, else ``None``."""
        row = (
            self._require_connection()
            .execute(
                "SELECT revision_id, study_uid, series_uid, version,"
                " parent_prediction_id, parent_revision_id, editor,"
                " edited_keys_json, payload, created_at"
                " FROM whole_cine_revisions WHERE revision_id = ?",
                (revision_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        stored = WholeCineRevisionRow(
            revision_id=str(row[0]),
            study_uid=str(row[1]),
            series_uid=str(row[2]),
            version=int(row[3]),
            parent_prediction_id=str(row[4]),
            parent_revision_id=None if row[5] is None else str(row[5]),
            editor=str(row[6]),
            edited_keys_json=str(row[7]),
            payload=str(row[8]),
            created_at=str(row[9]),
        )
        reconstructed = self._verify_and_reconstruct(
            stored.revision_id, stored.payload, whole_cine_revision_from_payload
        )
        _assert_whole_cine_revision_columns_match_payload(stored, reconstructed)
        return reconstructed

    def load_latest_whole_cine_head(
        self, series_uid: str
    ) -> WholeCinePrediction | WholeCineRevision | None:
        """The current whole-cine lineage head for one series, else ``None``.

        Resolution order: the highest-version persisted
        ``whole_cine_revisions`` row, else the newest prediction by store
        insertion sequence (``rowid``), else ``None``. Read-only; the
        checked writer re-resolves the head INSIDE its own transaction.
        """
        row = (
            self._require_connection()
            .execute(
                "SELECT revision_id FROM whole_cine_revisions WHERE series_uid = ?"
                " ORDER BY version DESC LIMIT 1",
                (series_uid,),
            )
            .fetchone()
        )
        if row is not None:
            return self.load_whole_cine_revision(str(row[0]))
        row = (
            self._require_connection()
            .execute(
                "SELECT prediction_id FROM whole_cine_predictions"
                " WHERE series_uid = ? ORDER BY rowid DESC LIMIT 1",
                (series_uid,),
            )
            .fetchone()
        )
        if row is not None:
            return self.load_whole_cine_prediction(str(row[0]))
        return None

    def save_whole_cine_revision_checked(
        self,
        revision: WholeCineRevision,
        *,
        expected_parent_head_id: str | None,
    ) -> str:
        """S3-009 D7: append ONE whole-cine contour-save revision under CAS.

        Inside ONE ``BEGIN IMMEDIATE`` transaction this checked writer:

        1. resolves the CURRENT whole-cine lineage head for the series
           (highest-version ``whole_cine_revisions`` row, else the newest
           ``whole_cine_predictions`` row);
        2. requires the expected parent identity to be non-null and to equal
           that head (``StaleHeadError`` otherwise, zero rows staged);
        3. requires the child's lineage to anchor exactly to that head
           (``parent_prediction_id`` always; ``parent_revision_id`` for a
           version > 1 — the SUPERSEDES lineage is explicit);
        4. re-runs the frozen append-only coherence guards of
           :meth:`save_whole_cine_revision` (series/study/version/lineage
           continuity, ``edited_keys_json`` content addressing) and INSERTs
           the row in the SAME transaction.

        The physician correction and the original AI prediction therefore
        stay SEPARATE records (invariant 10): the prediction row is never
        touched, and every correction is traceable to the exact parent it
        supersedes.
        """
        if expected_parent_head_id is None:
            raise StaleHeadError(
                "save_contour_edits requires a non-null expected whole-cine "
                "head identity; a null lineage cannot be superseded"
            )
        payload_text = self._content_addressed_payload(revision, revision.revision_id)

        def head_identity(connection: sqlite3.Connection) -> tuple[str, str | None, str | None]:
            revision_row = connection.execute(
                "SELECT revision_id, parent_prediction_id FROM whole_cine_revisions"
                " WHERE series_uid = ? ORDER BY version DESC LIMIT 1",
                (revision.series_uid,),
            ).fetchone()
            if revision_row is not None:
                return str(revision_row[0]), "revision", str(revision_row[1])
            prediction_row = connection.execute(
                "SELECT prediction_id FROM whole_cine_predictions"
                " WHERE series_uid = ? ORDER BY rowid DESC LIMIT 1",
                (revision.series_uid,),
            ).fetchone()
            if prediction_row is not None:
                return str(prediction_row[0]), "prediction", None
            return "", None, None

        def body(connection: sqlite3.Connection) -> str:
            current_head, head_kind, head_prediction_id = head_identity(connection)
            if current_head == "" or head_kind is None:
                raise StaleHeadError(
                    f"no whole-cine lineage exists for series "
                    f"{revision.series_uid!r}; nothing to save contour edits against"
                )
            if current_head != expected_parent_head_id:
                raise StaleHeadError(
                    "stale whole-cine head: expected "
                    f"{expected_parent_head_id!r} but the current head is "
                    f"{current_head!r}; zero rows written"
                )
            if head_kind == "revision":
                if revision.parent_revision_id != current_head:
                    raise StaleHeadError(
                        "the child's supersedes lineage (parent_revision_id) "
                        "does not anchor to the verified current head "
                        f"{current_head!r}"
                    )
                if revision.parent_prediction_id != head_prediction_id:
                    raise StaleHeadError(
                        "the child's parent_prediction_id does not anchor to "
                        f"the current head's lineage {head_prediction_id!r}"
                    )
            else:
                if revision.parent_prediction_id != current_head:
                    raise StaleHeadError(
                        "the child's parent_prediction_id does not anchor to "
                        f"the verified current head {current_head!r}"
                    )
            # The frozen append-only coherence guards run in the SAME
            # transaction (series/study/version/lineage continuity), then
            # the INSERT stages against the verified head.
            prediction_row = connection.execute(
                "SELECT series_uid, study_uid FROM whole_cine_predictions"
                " WHERE prediction_id = ?",
                (revision.parent_prediction_id,),
            ).fetchone()
            if prediction_row is not None:
                if str(prediction_row[0]) != revision.series_uid:
                    raise SeriesCoherenceError(
                        "revision series_uid "
                        f"{revision.series_uid!r} does not match parent prediction "
                        f"series_uid {str(prediction_row[0])!r}"
                    )
                if str(prediction_row[1]) != revision.study_uid:
                    raise SeriesCoherenceError(
                        "revision study_uid "
                        f"{revision.study_uid!r} does not match parent prediction "
                        f"study_uid {str(prediction_row[1])!r}"
                    )
            if revision.parent_revision_id is not None:
                parent_row = connection.execute(
                    "SELECT series_uid, study_uid, version, parent_prediction_id"
                    " FROM whole_cine_revisions WHERE revision_id = ?",
                    (revision.parent_revision_id,),
                ).fetchone()
                if parent_row is not None:
                    if str(parent_row[0]) != revision.series_uid:
                        raise SeriesCoherenceError(
                            "revision series_uid "
                            f"{revision.series_uid!r} does not match parent revision "
                            f"series_uid {str(parent_row[0])!r}"
                        )
                    if str(parent_row[1]) != revision.study_uid:
                        raise SeriesCoherenceError(
                            "revision study_uid "
                            f"{revision.study_uid!r} does not match parent revision "
                            f"study_uid {str(parent_row[1])!r}"
                        )
                    if int(parent_row[2]) != revision.version - 1:
                        raise SeriesCoherenceError(
                            f"revision version {revision.version} does not continue "
                            f"the parent chain (parent version {int(parent_row[2])};"
                            " child must be parent_version + 1)"
                        )
                    if str(parent_row[3]) != revision.parent_prediction_id:
                        raise SeriesCoherenceError(
                            "parent_prediction_id switch rejected: child anchors "
                            f"{revision.parent_prediction_id!r} but parent revision "
                            f"anchors {str(parent_row[3])!r}"
                        )
            connection.execute(
                "INSERT INTO whole_cine_revisions (revision_id, study_uid,"
                " series_uid, version, parent_prediction_id, parent_revision_id,"
                " editor, edited_keys_json, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    revision.revision_id,
                    revision.study_uid,
                    revision.series_uid,
                    revision.version,
                    revision.parent_prediction_id,
                    revision.parent_revision_id,
                    revision.editor,
                    whole_cine_edited_keys_json(revision),
                    payload_text,
                    revision.created_at,
                ),
            )
            return revision.revision_id

        return self._in_write_transaction(body)

    # --- CFM-001 applicable-inclusion-set resolution ---------------------------

    @staticmethod
    def _contour_preserving_lineage_ids(
        connection: sqlite3.Connection, series_uid: str, head_revision_id: str
    ) -> set[str]:
        """Head-anchored prefix of CONTOUR-PRESERVING transitions (correction F4).

        Starting from the head, a parent revision joins the scope ONLY when
        its canonical ``contours`` payload fragment is byte-identical to the
        child's — i.e. every step towards it changed NO contour geometry
        (QC-snapshot forks qualify; physician contour edits stop the walk).
        Only ids in THIS scope may anchor an applicable inclusion set, so a
        contour-head change resets inclusion instead of carrying it forward.
        """
        scope: set[str] = {head_revision_id}

        def _row(revision_id: str) -> Any | None:
            return connection.execute(
                "SELECT parent_revision_id,"
                " json_extract(payload, '$.contours')"
                " FROM contour_revisions WHERE revision_id = ? AND series_uid = ?",
                (revision_id, series_uid),
            ).fetchone()

        current = head_revision_id
        head_row = _row(current)
        if head_row is None or head_row[1] is None:
            return scope
        current_contours = head_row[1]
        # Correction round 2 (finding 2): the PARENT'S OWN contour payload is
        # loaded and compared BEFORE the parent joins the scope. The previous
        # walk compared the current row against itself and admitted the
        # immediate parent unconditionally, letting a set anchor across a
        # physician contour edit.
        while True:
            parent_row = _row(current)
            if parent_row is None or parent_row[0] is None:
                break
            parent_id = str(parent_row[0])
            full_parent = _row(parent_id)
            if full_parent is None or full_parent[1] is None:
                break
            if full_parent[1] != current_contours:
                break
            scope.add(parent_id)
            current = parent_id
        return scope

    def _resolve_latest_applicable_set(
        self,
        connection: sqlite3.Connection,
        *,
        series_uid: str,
        head_revision_id: str,
        selection_id: str | None,
        workflow_id: str | None,
        expected_n_slices: int | None = None,
    ) -> tuple[SliceInclusionSet, int] | None:
        """Highest-sequence APPLICABLE inclusion set for a sealing context.

        Applicable (CFM-001 §1.7/§1.8): same series; same explicit selection
        id; workflow agreeing under the bound/unbound rule with the sealing
        context; anchored (``source_revision_id``) to the current head or
        ANY of its ancestors (analysis refresh forks QC children without
        changing contour geometry, so a set authored on the parent stays
        clinically bound across such forks); and carrying a COMPLETE 5xN
        entry grid. Candidates are examined in DESCENDING store sequence so
        the FIRST hit is the latest applicable set; an incomplete grid on
        that candidate is a deterministic stale-state failure, never a
        silent fall-back to an older set.

        Returns ``(set, sequence)`` or ``None`` when no applicable set
        exists.
        """
        if selection_id is None:
            return None
        lineage = self._contour_preserving_lineage_ids(connection, series_uid, head_revision_id)
        rows = connection.execute(
            "SELECT inclusion_set_id, payload, workflow_id FROM slice_inclusion_sets"
            " WHERE series_uid = ? AND selection_id = ?"
            " ORDER BY inclusion_sequence DESC, inclusion_set_id ASC",
            (series_uid, selection_id),
        ).fetchall()
        for stored_id, payload_text, row_workflow in rows:
            if workflow_id is None:
                if row_workflow is not None:
                    continue  # bound set beside unbound lineage: not applicable
            elif row_workflow != workflow_id:
                continue
            try:
                candidate = self._verify_and_reconstruct(
                    str(stored_id),
                    str(payload_text),
                    slice_inclusion_set_from_payload,
                )
            except PayloadIntegrityError:
                raise
            if candidate.source_revision_id not in lineage:
                continue
            # Complete-entry-grid verification (CFM-001 §1.7). A malformed
            # grid can never silently bind a measurement or approval.
            try:
                parsed = dict(json.loads(str(payload_text)))
                covered = validate_complete_grid_entries(parsed["entries"])
            except PayloadIntegrityError:
                raise
            except (ValueError, KeyError, TypeError) as exc:
                if isinstance(exc, ValueError):
                    raise SeriesCoherenceError(
                        f"applicable inclusion set {stored_id!r} does not "
                        f"carry a complete entry grid: {exc}"
                    ) from exc
                raise PayloadIntegrityError(
                    f"applicable inclusion set payload {stored_id!r} is unreadable"
                ) from exc
            # Correction round 2 (finding 4): coverage must equal the cine's
            # ACTUAL slice count when the caller can supply it — a complete
            # 5xk PREFIX over a longer cine is not applicable.
            if expected_n_slices is not None and covered != expected_n_slices:
                raise SeriesCoherenceError(
                    f"applicable inclusion set {stored_id!r} covers "
                    f"{covered} slices but the cine stack has "
                    f"{expected_n_slices}; truncated-tail grids are not "
                    "applicable"
                )
            sequence_row = connection.execute(
                "SELECT inclusion_sequence FROM slice_inclusion_sets WHERE inclusion_set_id = ?",
                (stored_id,),
            ).fetchone()
            assert sequence_row is not None
            return candidate, int(sequence_row[0])
        return None

    def load_latest_applicable_inclusion_set(
        self,
        series_uid: str,
        *,
        head_revision_id: str,
        selection_id: str | None,
        workflow_id: str | None,
        expected_n_slices: int,
    ) -> tuple[SliceInclusionSet, int] | None:
        """Public read-only resolution of the latest APPLICABLE set.

        Same semantics as :meth:`_resolve_latest_applicable_set`; used by
        the workflow/controller and the viewer to display and verify the
        current applicable inclusion set.

        Correction round 4 (R4-final): ``expected_n_slices`` is MANDATORY —
        applicable-set coverage is always bound to the cine's actual slice
        count.
        """
        if (
            isinstance(expected_n_slices, bool)
            or not isinstance(expected_n_slices, int)
            or expected_n_slices < 1
        ):
            raise SeriesCoherenceError(
                "expected_n_slices must be a positive integer (the cine's "
                f"actual slice count); got {expected_n_slices!r}"
            )
        return self._resolve_latest_applicable_set(
            self._require_connection(),
            series_uid=series_uid,
            head_revision_id=head_revision_id,
            selection_id=selection_id,
            workflow_id=workflow_id,
            expected_n_slices=expected_n_slices,
        )

    # --- atomic head-check-and-seal (S2C-001 clusters C + D + E) --------------

    @staticmethod
    def _current_head_row(connection: sqlite3.Connection, series_uid: str) -> sqlite3.Row | None:
        """Highest-version revision row of the persisted chain, else None.

        Columns: ``revision_id``, ``version``, ``payload``, ``workflow_id``.
        """
        return cast(
            sqlite3.Row | None,
            connection.execute(
                "SELECT revision_id, version, payload, workflow_id FROM contour_revisions"
                " WHERE series_uid = ? ORDER BY version DESC LIMIT 1",
                (series_uid,),
            ).fetchone(),
        )

    @staticmethod
    def _verify_persisted_head_payload(
        head_row: sqlite3.Row,
    ) -> dict[str, Any]:
        """Digest-verify the stored head payload and parse it fail-closed."""
        head_id, payload_text = str(head_row[0]), str(head_row[2])
        if _sha256_hex(payload_text) != head_id:
            raise PayloadIntegrityError(
                f"persisted head revision payload digest does not match id {head_id!r}"
            )
        try:
            return dict(json.loads(payload_text))
        except ValueError as exc:
            raise PayloadIntegrityError(
                f"persisted head revision payload {head_id!r} is not valid JSON"
            ) from exc

    def _assert_selection_binding_coherence(
        self,
        connection: sqlite3.Connection,
        *,
        series_uid: str,
        sealing_workflow_id: str | None,
        head_workflow_id: str | None,
        measurement_payload: dict[str, Any] | None,
        selection_id: str | None,
    ) -> None:
        """Verify selection/workflow binding coherence before sealing.

        - a populated ``selection_id`` must reference a persisted selection
          row of the SAME series;
        - when the measurement carries a selection id it must equal the
          sealing record's id (mixed bound/unbound binding rejected);
        - bound/unbound workflow identities must agree across the sealing
          record provenance, the measurement provenance, the referenced
          selection row and the current head revision row.
        """
        measurement_workflow = (
            None
            if measurement_payload is None
            else _workflow_id_of_provenance(measurement_payload.get("provenance"))
        )
        selection_workflow: str | None = None
        if selection_id is not None:
            selection_row = connection.execute(
                "SELECT series_uid, workflow_id FROM edes_selections WHERE selection_id = ?",
                (selection_id,),
            ).fetchone()
            if selection_row is None:
                raise SeriesCoherenceError(
                    f"selection {selection_id!r} is not present in the persisted "
                    "store; approvals and decisions must reference explicit "
                    "persisted selections"
                )
            if str(selection_row[0]) != series_uid:
                raise SeriesCoherenceError(
                    f"selection {selection_id!r} belongs to series "
                    f"{str(selection_row[0])!r}, not {series_uid!r}"
                )
            selection_workflow = None if selection_row[1] is None else str(selection_row[1])
        if measurement_payload is not None:
            measurement_selection = measurement_payload.get("selection_id")
            if (measurement_selection or None) != (selection_id or None):
                raise SeriesCoherenceError(
                    "mixed selection binding: sealing id "
                    f"{selection_id!r} vs measurement selection_id "
                    f"{measurement_selection!r}; bound and unbound records may "
                    "never be mixed in one lineage"
                )
        _assert_bound_workflow_agreement(
            sealing=sealing_workflow_id,
            measurement=measurement_workflow,
            selection=selection_workflow,
            head=head_workflow_id,
        )

    def _assert_store_topology_not_blocking(
        self,
        connection: sqlite3.Connection,
        *,
        series_uid: str,
        head_revision_id: str,
        selection_id: str,
        n_slices: int,
        ed_frame: int,
        es_frame: int,
    ) -> None:
        """Store-side metric-relevant topology last line (F14 defense in depth).

        This re-runs the SAME relevance-and-resolution predicate shape the
        controller owns in :mod:`cmr_workflow.topology_disposition`, from
        persisted rows only (the chain-authoring-time derivation comes from the
        shared :func:`~cmr_persistence.records.chain_key_authoring_times`), so
        a persisted block key that is metric relevant AND neither authored by
        the current chain after the block nor covered by an APPLICABLE
        disposition fails the seal here with the same typed
        :data:`~cmr_core.errors.TOPOLOGY_BLOCKED` visible refusal. A key the
        controller can prove irrelevant (outside the selected grid, or a
        non-ED LV_MYOCARDIUM) is ignored exactly like the controller ignores
        it; an unknown grid frame count keeps every key relevant (fail
        closed).
        """
        block_rows = connection.execute(
            "SELECT slice_index, frame_index, structure, created_at FROM"
            " series_topology_blocks WHERE series_uid = ?",
            (series_uid,),
        ).fetchall()
        if not block_rows:
            return
        selection_row = connection.execute(
            "SELECT payload FROM edes_selections WHERE selection_id = ?",
            (selection_id,),
        ).fetchone()
        selection_n_frames: int | None = None
        if selection_row is not None:
            selection_n_frames = edes_selection_from_payload(
                json.loads(str(selection_row[0]))
            ).n_frames
        disposition_rows = connection.execute(
            "SELECT payload FROM topology_dispositions WHERE series_uid = ?",
            (series_uid,),
        ).fetchall()
        applicable = applicable_topology_dispositions(
            [
                topology_disposition_from_payload(json.loads(str(row[0])))
                for row in disposition_rows
            ],
            head_revision_id=head_revision_id,
            selection_id=selection_id,
        )
        resolved_keys = {
            (record.slice_index, record.frame_index, record.structure)
            for record in applicable
        }
        # R1 review F1: a key counts as REPLACED only when the current chain
        # AUTHORED (changed) its geometry AFTER the block was recorded; a
        # verbatim QC refresh is not an authoring event. Same shared helper the
        # controller's one resolver uses.
        chain_key_times = chain_key_authoring_times(self.load_chain(series_uid))
        blocking: list[tuple[int, int, str]] = []
        for row in block_rows:
            slice_index = int(row[0])
            frame_index = int(row[1])
            structure = str(row[2])
            if not _block_is_metric_relevant(
                slice_index=slice_index,
                frame_index=frame_index,
                structure=structure,
                n_slices=n_slices,
                n_frames=selection_n_frames,
                ed_frame=ed_frame,
            ):
                continue
            key = (slice_index, frame_index, structure)
            if key in resolved_keys:
                continue
            try:
                structure_member = Structure(structure)
            except ValueError:
                structure_member = None
            if structure_member is not None and _timestamp_after(
                chain_key_times.get((slice_index, frame_index, structure_member), ""),
                str(row[3]),
            ):
                continue
            blocking.append(key)
        if blocking:
            names = ", ".join(
                f"(slice={s}, frame={f}, structure={st})" for s, f, st in blocking[:4]
            )
            extra = len(blocking) - 4
            suffix = f" (+{extra} more)" if extra > 0 else ""
            raise CMRError(
                TOPOLOGY_BLOCKED,
                f"cannot seal series {series_uid!r}: {len(blocking)} metric-relevant "
                f"topology key(s) remain unresolved for head {head_revision_id!r} / "
                f"selection {selection_id!r}: {names}{suffix}; each such key must be "
                "replaced by physician geometry in the current chain head or "
                "explicitly excluded by a physician topology disposition bound to "
                "this exact head and selection",
            )

    def record_topology_block(
        self,
        block_or_series_uid: TopologyBlock | str | None = None,
        slice_index: int | None = None,
        frame_index: int | None = None,
        structure: str | None = None,
        reason: str | None = None,
        created_at: str | None = None,
        *,
        series_uid: str | None = None,
    ) -> str:
        """Record a durable non-projectable topology block for a series plane.

        Per-plane idempotent (owner P0 decision 2026-09-09): ``block_id`` is a
        digest of ``reason`` + ``created_at`` among other fields, so ANY
        re-encounter of the same plane key produces a different digest by
        construction. Re-recording an already-blocked
        ``(series_uid, slice_index, frame_index, structure)`` key therefore
        returns the EXISTING ``block_id`` instead of raising a conflict: one
        non-projectable encounter must not permanently poison every later
        operation on that plane. The append-only triggers
        (``series_topology_blocks_no_update`` / ``_no_delete`` /
        ``_no_replace``) stay in force — the first record for a key is
        immutable and is never rewritten or superseded.

        First-observation-wins: the re-record returns the FIRST block's id and
        keeps the FIRST ``reason`` verbatim, so a later encounter's diagnostic
        text is DISCARDED (deliberate: rewriting reason text would mutate an
        approved-prohibition-relevant record). Capturing the later
        diagnostics as a separate per-encounter history is FUTURE WORK and is
        deferred — nothing here records it.
        """
        resolved_uid = block_or_series_uid if block_or_series_uid is not None else series_uid
        if isinstance(resolved_uid, TopologyBlock):
            block = resolved_uid
        else:
            if (
                resolved_uid is None
                or slice_index is None
                or frame_index is None
                or structure is None
                or reason is None
                or created_at is None
            ):
                raise CMRError(
                    INVALID_ARGUMENT,
                    "record_topology_block requires all fields when passing individual arguments",
                )
            block = TopologyBlock(
                series_uid=resolved_uid,
                slice_index=slice_index,
                frame_index=frame_index,
                structure=structure,
                reason=reason,
                created_at=created_at,
            )

        def body(connection: sqlite3.Connection) -> str:
            existing = connection.execute(
                "SELECT block_id FROM series_topology_blocks "
                "WHERE series_uid = ? AND slice_index = ? AND frame_index = ? AND structure = ?",
                (block.series_uid, block.slice_index, block.frame_index, block.structure),
            ).fetchone()
            if existing is not None:
                # Per-plane idempotency (owner P0 2026-09-09): the key is
                # already blocked, so the durable decision stands and the
                # EXISTING block_id is returned. The row is append-only; no
                # update, no delete, no replace.
                return str(existing[0])
            payload_text = canonical_json(block)
            connection.execute(
                "INSERT INTO series_topology_blocks "
                "(block_id, series_uid, slice_index, frame_index, "
                "structure, reason, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    block.block_id,
                    block.series_uid,
                    block.slice_index,
                    block.frame_index,
                    block.structure,
                    block.reason,
                    block.created_at,
                    payload_text,
                ),
            )
            block_id = block.block_id
            assert block_id is not None
            return block_id

        return self._in_write_transaction(body)

    def load_topology_blocks(self, series_uid: str) -> list[TopologyBlock]:
        """Load durable topology blocks for a series, ordered deterministically."""
        connection = self._require_connection()
        rows = connection.execute(
            "SELECT block_id, series_uid, slice_index, frame_index, "
            "structure, reason, created_at, payload "
            "FROM series_topology_blocks WHERE series_uid = ? "
            "ORDER BY created_at ASC, slice_index ASC, frame_index ASC, structure ASC",
            (series_uid,),
        ).fetchall()
        blocks: list[TopologyBlock] = []
        for row in rows:
            block_id = str(row[0])
            series_uid_val = str(row[1])
            slice_index = int(row[2])
            frame_index = int(row[3])
            structure = str(row[4])
            reason = str(row[5])
            created_at = str(row[6])
            payload_text = str(row[7])
            computed_payload_digest = _sha256_hex(payload_text)
            if computed_payload_digest != block_id:
                raise PayloadIntegrityError(
                    f"payload digest {computed_payload_digest!r} "
                    f"does not match block_id {block_id!r}"
                )
            block = TopologyBlock(
                series_uid=series_uid_val,
                slice_index=slice_index,
                frame_index=frame_index,
                structure=structure,
                reason=reason,
                created_at=created_at,
                block_id=block_id,
            )
            blocks.append(block)
        return blocks

    # --- Topology dispositions (Astra F1-F5, 2026-09-09) -----------------------

    def save_topology_disposition_set_checked(
        self,
        dispositions: Sequence[TopologyDispositionRecord],
        *,
        expected_head_revision_id: str,
        selection_id: str,
    ) -> tuple[str, ...]:
        """Append-only insert of ONE disposition set, CAS-bound at write time.

        Every record must (a) carry the SAME ``series_uid``, (b) bind the
        exact ``expected_head_revision_id`` supplied here, (c) bind the exact
        ``selection_id`` supplied here, and (d) resolve a block that is
        ACTUALLY persisted for that series. The insert runs inside ONE
        ``BEGIN IMMEDIATE`` transaction together with a re-verification of the
        CURRENT persisted chain head, so a disposition can never be authored
        against a stale head. Returns the inserted ``disposition_id`` values
        in canonical (slice, frame, structure) order.

        Duplicate-free per (slice, frame, structure, head, selection)
        BINDING (R1 review F2, 2026-09-10): a second disposition for the
        SAME key bound to the SAME head and selection fails closed
        (append-only provenance — a disposition is never rewritten or
        superseded in place). A key MAY however carry one disposition per
        binding: after a head or selection change the physician records a
        NEW disposition for the new state while every historical row stays
        intact. Exactly one record per binding is admitted, so the
        APPLICABLE record for a key — the one bound to the CURRENT head +
        selection — is always unique (see
        :func:`~cmr_persistence.records.applicable_topology_dispositions`).
        """
        records = tuple(dispositions)
        if not records:
            raise CMRError(
                INVALID_ARGUMENT,
                "save_topology_disposition_set_checked requires at least one "
                "topology disposition record",
            )
        series_uid = records[0].series_uid
        payloads: list[tuple[TopologyDispositionRecord, str]] = []
        for record in records:
            if not isinstance(record, TopologyDispositionRecord):
                raise CMRError(
                    INVALID_ARGUMENT,
                    "dispositions must be TopologyDispositionRecord instances, got "
                    f"{type(record).__name__}",
                )
            if record.series_uid != series_uid:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"disposition series {record.series_uid!r} does not match the "
                    f"set's series {series_uid!r}; one set belongs to one series",
                )
            if record.source_revision_id != expected_head_revision_id:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"disposition for key {record.identity_key!r} binds head "
                    f"{record.source_revision_id!r} but the expected current head "
                    f"is {expected_head_revision_id!r}",
                )
            if record.selection_id != selection_id:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"disposition for key {record.identity_key!r} binds selection "
                    f"{record.selection_id!r} but the expected selection is "
                    f"{selection_id!r}",
                )
            payloads.append(
                (record, self._content_addressed_payload(record, record.disposition_id))
            )

        def body(connection: sqlite3.Connection) -> tuple[str, ...]:
            head_row = self._current_head_row(connection, series_uid)
            if head_row is None:
                raise StaleHeadError(
                    f"no persisted revision chain for series {series_uid!r}; a "
                    "topology disposition requires a head to bind to"
                )
            head_id = str(head_row[0])
            if head_id != expected_head_revision_id:
                raise StaleHeadError(
                    f"stale head: expected {expected_head_revision_id!r} but the "
                    f"persisted chain head is {head_id!r}; zero rows written"
                )
            selection_row = connection.execute(
                "SELECT series_uid FROM edes_selections WHERE selection_id = ?",
                (selection_id,),
            ).fetchone()
            if selection_row is None:
                raise SeriesCoherenceError(
                    f"selection {selection_id!r} is not present in the persisted "
                    "store; a topology disposition must bind a persisted selection"
                )
            if str(selection_row[0]) != series_uid:
                raise SeriesCoherenceError(
                    f"selection {selection_id!r} belongs to series "
                    f"{str(selection_row[0])!r}, not {series_uid!r}"
                )
            inserted: list[str] = []
            for record, payload_text in payloads:
                block_row = connection.execute(
                    "SELECT block_id FROM series_topology_blocks WHERE series_uid = ?"
                    " AND slice_index = ? AND frame_index = ? AND structure = ?",
                    (
                        record.series_uid,
                        record.slice_index,
                        record.frame_index,
                        record.structure,
                    ),
                ).fetchone()
                if block_row is None:
                    raise CMRError(
                        INVALID_ARGUMENT,
                        f"no persisted topology block for key "
                        f"{record.identity_key!r} in series {series_uid!r}; there "
                        "is nothing to resolve",
                    )
                if str(block_row[0]) != record.source_block_id:
                    raise CMRError(
                        INVALID_ARGUMENT,
                        f"disposition for key {record.identity_key!r} cites block "
                        f"{record.source_block_id!r} but the persisted block is "
                        f"{str(block_row[0])!r}",
                    )
                # R1 REVIEW F2 (P1, 2026-09-10): the duplicate check is
                # BINDING-SCOPED, not key-scoped. The table stays append-only
                # (nothing is ever updated or deleted), but a key may carry
                # one disposition per (head_revision, selection) binding:
                # after a head or selection change the physician MUST be able
                # to record a NEW disposition for the current state — the old
                # row is stale history, never a reason to dead-end the
                # workflow. An identical record (same key AND same binding
                # AND the same content-addressed disposition id) is still
                # refused: the CAS digest rejects any re-insert of the same
                # bytes, and the content check below names it explicitly.
                existing = connection.execute(
                    "SELECT disposition_id FROM topology_dispositions "
                    "WHERE series_uid = ? AND slice_index = ? AND frame_index = ? "
                    "AND structure = ? AND source_revision_id = ? "
                    "AND selection_id = ?",
                    (
                        record.series_uid,
                        record.slice_index,
                        record.frame_index,
                        record.structure,
                        record.source_revision_id,
                        record.selection_id,
                    ),
                ).fetchone()
                if existing is not None:
                    same_bytes = str(existing[0]) == record.disposition_id
                    raise CMRError(
                        INVALID_ARGUMENT,
                        f"key {record.identity_key!r} already carries "
                        + (
                            "the identical durable topology disposition "
                            if same_bytes
                            else "a durable topology disposition "
                        )
                        + f"{str(existing[0])!r} bound to this exact head "
                        f"{record.source_revision_id!r} and selection "
                        f"{record.selection_id!r}; dispositions are append-only "
                        "and are never superseded in place"
                        + (
                            ""
                            if same_bytes
                            else " — record a NEW disposition against the NEW "
                            "head/selection to re-dispose this key after a "
                            "clinical state change"
                        ),
                    )
                connection.execute(
                    "INSERT INTO topology_dispositions (disposition_id, series_uid,"
                    " slice_index, frame_index, structure, source_block_id,"
                    " source_revision_id, selection_id, frame_role, disposition,"
                    " physician, created_at, reason, workflow_id, payload)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.disposition_id,
                        record.series_uid,
                        record.slice_index,
                        record.frame_index,
                        record.structure,
                        record.source_block_id,
                        record.source_revision_id,
                        record.selection_id,
                        record.frame_role.value,
                        record.disposition.value,
                        record.physician,
                        record.created_at,
                        record.reason,
                        record.workflow_id,
                        payload_text,
                    ),
                )
                inserted.append(record.disposition_id)
            return tuple(inserted)

        return self._in_write_transaction(body)

    def load_topology_dispositions(
        self, series_uid: str
    ) -> list[TopologyDispositionRecord]:
        """Load durable topology dispositions for a series, deterministically.

        HISTORY, not applicability: every row ever recorded for the series is
        returned, including rows whose (head, selection) binding has since
        been superseded. Append-only provenance means those rows are never
        removed; use :meth:`load_applicable_topology_dispositions` for the
        set that speaks about a CURRENT clinical state.

        Order: ``(slice_index, frame_index, structure, created_at,
        disposition_id)`` ascending, so the same stored set always yields the
        same list (and therefore the same set digest) regardless of insertion
        order. Every payload is digest-verified against its ``disposition_id``
        before reconstruction.
        """
        connection = self._require_connection()
        rows = connection.execute(
            "SELECT disposition_id, payload FROM topology_dispositions "
            "WHERE series_uid = ? "
            "ORDER BY slice_index ASC, frame_index ASC, structure ASC, "
            "created_at ASC, disposition_id ASC",
            (series_uid,),
        ).fetchall()
        dispositions: list[TopologyDispositionRecord] = []
        for disposition_id, payload_text in rows:
            stored_id, text = str(disposition_id), str(payload_text)
            computed = _sha256_hex(text)
            if computed != stored_id:
                raise PayloadIntegrityError(
                    f"payload digest {computed!r} does not match disposition_id "
                    f"{stored_id!r}"
                )
            dispositions.append(
                topology_disposition_from_payload(dict(json.loads(text)))
            )
        return dispositions

    def load_applicable_topology_dispositions(
        self,
        series_uid: str,
        *,
        head_revision_id: str,
        selection_id: str,
    ) -> list[TopologyDispositionRecord]:
        """The APPLICABLE dispositions of a series for ONE clinical state.

        R1 review F2 (2026-09-10): a key may carry one disposition per
        (head, selection) binding, so the historical table can hold several
        rows for the same ``(slice, frame, structure)`` key. The APPLICABLE
        one — and the only one that may resolve a block — is the record
        bound to the CURRENT head and selection. This loader is the
        read-path counterpart of the write rule and of the approval digest
        binding; it delegates to
        :func:`~cmr_persistence.records.applicable_topology_dispositions`
        so there is exactly one definition.
        """
        return applicable_topology_dispositions(
            self.load_topology_dispositions(series_uid),
            head_revision_id=head_revision_id,
            selection_id=selection_id,
        )

    def save_revision_checked(
        self,
        expected_head_revision_id: str,
        revision: ContourRevision,
        *,
        expected_selection_id: str | None = None,
        expected_inclusion_set_id: str | None = None,
        expected_sparse_prediction_id: str | None = None,
        expected_whole_prediction_id: str | None = None,
        expected_topology_block_ids: tuple[str, ...] | list[str] | None = None,
        expected_topology_disposition_ids: tuple[str, ...] | list[str] | None = None,
        expected_n_slices: int | None = None,
    ) -> str:
        """Atomically verify the CURRENT persisted head, then persist ONE revision.

        Inside ONE BEGIN IMMEDIATE transaction:
        1. Resolves highest-version revision row of revision.series_uid.
        2. Requires head_id == expected_head_revision_id (StaleHeadError otherwise).
        3. Requires revision.parent_revision_id == head_id.
           Requires revision.version == int(head_row[1]) + 1.
           Asserts _assert_parent_links_share_series(connection, revision).
        4. (CREATE-A C1, opt-in) When ``expected_n_slices`` is supplied, ALSO
           re-resolves the mutable create-eligibility bindings — latest
           selection, latest applicable inclusion set, current-eligible sparse
           and latest whole-cine source identities, plus the series topology
           block/disposition identities — INSIDE this same transaction and
           requires each to equal its expectation EXACTLY (``None`` means
           "expect absent", never "skip"). Any drift aborts with
           ``StaleHeadError`` and zero rows written, so a binding committed
           after the caller's preflight can never commit under a stale
           verdict. When ``expected_n_slices`` is omitted this block is
           skipped entirely: behavior is byte-identical to the historical
           head-only CAS, so existing callers are unaffected (no CAS semantic
           change, no new public writer).
        5. Inserts revision row into contour_revisions and commits.
        6. Returns revision.revision_id.
        """
        payload_text = self._content_addressed_payload(revision, revision.revision_id)
        enforce_eligibility = expected_n_slices is not None
        if enforce_eligibility:
            expected_n_slices = self._validate_expected_n_slices(expected_n_slices)
            expected_selection_id = self._validate_checked_identity(
                "expected_selection_id", expected_selection_id
            )
            expected_inclusion_set_id = self._validate_checked_identity(
                "expected_inclusion_set_id", expected_inclusion_set_id
            )
            expected_sparse_prediction_id = self._validate_checked_identity(
                "expected_sparse_prediction_id", expected_sparse_prediction_id
            )
            expected_whole_prediction_id = self._validate_checked_identity(
                "expected_whole_prediction_id", expected_whole_prediction_id
            )

            def _wanted_ids(
                name: str, value: tuple[str, ...] | list[str] | None
            ) -> tuple[str, ...]:
                if value is None:
                    return ()
                if isinstance(value, str) or not isinstance(value, (tuple, list)):
                    raise SeriesCoherenceError(
                        f"{name} must be a tuple/list of ids or None; got {value!r}"
                    )
                items = tuple(str(item) for item in value)
                if any(not item for item in items):
                    raise SeriesCoherenceError(
                        f"{name} must contain non-empty string ids; got {value!r}"
                    )
                return tuple(sorted(items))

            wanted_block_ids = _wanted_ids(
                "expected_topology_block_ids", expected_topology_block_ids
            )
            wanted_disposition_ids = _wanted_ids(
                "expected_topology_disposition_ids",
                expected_topology_disposition_ids,
            )
        else:
            wanted_block_ids = ()
            wanted_disposition_ids = ()

        def body(connection: sqlite3.Connection) -> str:
            head_row = self._current_head_row(connection, revision.series_uid)
            if head_row is None:
                raise StaleHeadError(
                    f"no persisted revision chain for series {revision.series_uid!r}; "
                    "nothing to supersede"
                )
            head_id = str(head_row[0])
            if head_id != expected_head_revision_id:
                raise StaleHeadError(
                    f"stale head: expected {expected_head_revision_id!r} but the "
                    f"persisted chain head is {head_id!r}; zero rows written"
                )
            if revision.parent_revision_id != head_id:
                raise SeriesCoherenceError(
                    f"revision parent_revision_id {revision.parent_revision_id!r} "
                    f"does not match current head {head_id!r}"
                )
            expected_version = int(head_row[1]) + 1
            if revision.version != expected_version:
                raise SeriesCoherenceError(
                    f"revision version {revision.version} does not match "
                    f"expected version {expected_version}"
                )
            if enforce_eligibility:
                # Every read below executes on this transaction's own
                # connection (``self._require_connection()`` IS
                # ``connection`` here), so the eligibility snapshot is taken
                # AFTER this writer's BEGIN IMMEDIATE was acquired: a binding
                # committed after the caller's preflight is observed here and
                # refuses with zero rows written.
                actual = self._resolve_current_nullable_lineage(
                    connection,
                    series_uid=revision.series_uid,
                    expected_n_slices=expected_n_slices,
                )
                self._assert_checked_lineage(
                    actual=actual,
                    expected_head_revision_id=expected_head_revision_id,
                    expected_selection_id=expected_selection_id,
                    expected_inclusion_set_id=expected_inclusion_set_id,
                )
                live_head_id, live_selection_id = actual[0], actual[1]
                live_sparse = self.load_current_selected_phase_prediction(
                    revision.series_uid,
                    head_revision_id=live_head_id,
                    selection_id=live_selection_id,
                )
                live_sparse_id = (
                    None if live_sparse is None else live_sparse.prediction_id
                )
                if live_sparse_id != expected_sparse_prediction_id:
                    raise StaleHeadError(
                        "create eligibility changed during save: sparse source "
                        "binding moved (expected "
                        f"{expected_sparse_prediction_id!r}, current "
                        f"{live_sparse_id!r}); zero rows written"
                    )
                live_whole = self.load_latest_whole_cine_prediction(
                    revision.series_uid
                )
                live_whole_id = (
                    None if live_whole is None else live_whole.prediction_id
                )
                if live_whole_id != expected_whole_prediction_id:
                    raise StaleHeadError(
                        "create eligibility changed during save: whole-cine "
                        "source binding moved (expected "
                        f"{expected_whole_prediction_id!r}, current "
                        f"{live_whole_id!r}); zero rows written"
                    )
                live_block_ids = tuple(
                    sorted(
                        str(block.block_id)
                        for block in self.load_topology_blocks(revision.series_uid)
                    )
                )
                if live_block_ids != wanted_block_ids:
                    raise StaleHeadError(
                        "create eligibility changed during save: durable "
                        "topology blocks moved; zero rows written"
                    )
                live_disposition_ids = tuple(
                    sorted(
                        str(record.disposition_id)
                        for record in self.load_topology_dispositions(
                            revision.series_uid
                        )
                    )
                )
                if live_disposition_ids != wanted_disposition_ids:
                    raise StaleHeadError(
                        "create eligibility changed during save: durable "
                        "topology dispositions moved; zero rows written"
                    )
            _assert_parent_links_share_series(connection, revision)
            _insert_revision_row(connection, revision, payload_text)
            return revision.revision_id

        return self._in_write_transaction(body)

    def save_adopted_revision_batch_checked(
        self,
        *,
        expected_head_revision_id: str,
        expected_selection_id: str,
        revisions: list[ContourRevision],
        binding: dict[str, Any],
        expected_inclusion_set_id: str | None = None,
        expected_inclusion_waivers: (
            tuple[tuple[int, str, str], ...] | list[tuple[int, str, str]] | None
        ) = None,
        expected_n_slices: int | None = None,
    ) -> list[str]:
        """Atomically verify lineage + persist a multi-plane adoption batch.

        AMENDMENT-001 (CMR_FUNCV1_ADOPT_001 C2): the single-owner
        single-revision checked writer cannot express an all-or-nothing
        multi-plane adoption. This batch writer runs inside ONE
        ``BEGIN IMMEDIATE`` transaction:

        1. resolves the current chain head AND the latest selection in the
           SAME snapshot; requires head == ``expected_head_revision_id``
           AND selection == ``expected_selection_id`` (``StaleHeadError``
           otherwise — a selection change racing the adoption refuses even
           if the head did not move);
        2. (F22 C1 P1-3) re-resolves the latest APPLICABLE inclusion set
           for that same head/selection/workflow snapshot with the SAME
           ``expected_n_slices`` the caller consumed, and requires BOTH
           its set id AND its exact waiver-role coverage to equal
           ``expected_inclusion_set_id``/``expected_inclusion_waivers``
           (``StaleHeadError`` otherwise — a superseding inclusion persist
           racing the adoption refuses with zero writes, even when head
           and selection did not move). The waiver vocabulary is the
           persisted ``(slice, role, structure)`` triple, in deterministic
           sorted order. The authority is read with the EXISTING
           ``_resolve_latest_applicable_set`` resolver — no second
           authority is created. The production adopt path always supplies
           all three expectations; direct-store callers that omit
           ``expected_n_slices`` skip the inclusion re-verification
           (legacy behavior, no production caller does this);
        3. appends every revision in order (each parent-linked to the
           previous, version-sequential, series-coherent — the same guards
           as :meth:`save_revision_checked`);
        4. inserts ONE ``adoption_bindings`` row binding the batch to the
           exact sparse artifact (prediction/model-run, head, selection,
           revision ids, actor/timestamp);
        5. any failure rolls back the WHOLE batch: zero revisions, zero
           binding rows (never a partial adoption).

        No second contour authority is created: the revisions live in the
        existing physician lineage; only the binding record is new.
        """
        if not revisions:
            raise ValueError("revisions must be a non-empty list")
        series_uid = revisions[0].series_uid
        payload_texts = [
            self._content_addressed_payload(revision, revision.revision_id)
            for revision in revisions
        ]

        def body(connection: sqlite3.Connection) -> list[str]:
            head_row = self._current_head_row(connection, series_uid)
            if head_row is None:
                raise StaleHeadError(
                    f"no persisted revision chain for series {series_uid!r}; "
                    "nothing to supersede"
                )
            if str(head_row[0]) != expected_head_revision_id:
                raise StaleHeadError(
                    f"stale head: expected {expected_head_revision_id!r} but the "
                    f"persisted chain head is {str(head_row[0])!r}; zero rows written"
                )
            selection_row = connection.execute(
                "SELECT selection_id FROM edes_selections WHERE series_uid = ?"
                " ORDER BY selection_sequence DESC LIMIT 1",
                (series_uid,),
            ).fetchone()
            live_selection_id = None if selection_row is None else str(selection_row[0])
            if live_selection_id != expected_selection_id:
                raise StaleHeadError(
                    f"selection changed during adoption: expected "
                    f"{expected_selection_id!r}, current {live_selection_id!r}; "
                    "zero rows written"
                )
            # F22 C1 P1-3: re-verify the applicability authority consumed
            # for the waiver exemption INSIDE this same transaction. The
            # resolver is the EXISTING latest-applicable authority (same
            # head/selection/workflow snapshot as the CAS above); only the
            # comparison target comes from the caller.
            head_workflow: str | None = (
                None if head_row[3] is None else str(head_row[3])
            )
            if expected_n_slices is not None:
                if (
                    isinstance(expected_n_slices, bool)
                    or not isinstance(expected_n_slices, int)
                    or expected_n_slices < 1
                ):
                    raise SeriesCoherenceError(
                        "expected_n_slices must be a positive integer (the cine's "
                        f"actual slice count); got {expected_n_slices!r}"
                    )
                live_applicable = self._resolve_latest_applicable_set(
                    connection,
                    series_uid=series_uid,
                    head_revision_id=expected_head_revision_id,
                    selection_id=live_selection_id,
                    workflow_id=head_workflow,
                    expected_n_slices=expected_n_slices,
                )
                live_set_id = (
                    None if live_applicable is None else live_applicable[0].inclusion_set_id
                )
                if live_set_id != expected_inclusion_set_id:
                    raise StaleHeadError(
                        "applicability authority changed during adoption: expected "
                        f"inclusion set {expected_inclusion_set_id!r}, current "
                        f"{live_set_id!r}; zero rows written"
                    )
                if expected_inclusion_waivers is not None and live_applicable is not None:
                    live_waivers = tuple(
                        sorted(
                            (
                                int(entry.slice_index),
                                str(entry.frame_role.value),
                                str(entry.structure.value),
                            )
                            for entry in live_applicable[0].entries
                            if not bool(entry.included)
                        )
                    )
                    wanted = tuple(
                        tuple(item) for item in expected_inclusion_waivers
                    )
                    if live_waivers != tuple(wanted):
                        raise StaleHeadError(
                            "applicability authority changed during adoption: waiver "
                            f"coverage {list(wanted)!r} != current "
                            f"{list(live_waivers)!r}; zero rows written"
                        )
            expected_parent_id = expected_head_revision_id
            expected_version = int(head_row[1]) + 1
            for revision, payload_text in zip(revisions, payload_texts):
                if revision.parent_revision_id != expected_parent_id:
                    raise SeriesCoherenceError(
                        f"batch revision parent {revision.parent_revision_id!r} "
                        f"does not chain from {expected_parent_id!r}"
                    )
                if revision.version != expected_version:
                    raise SeriesCoherenceError(
                        f"batch revision version {revision.version} does not match "
                        f"expected {expected_version}"
                    )
                _assert_parent_links_share_series(connection, revision)
                _insert_revision_row(connection, revision, payload_text)
                expected_parent_id = revision.revision_id
                expected_version += 1
            binding_id = _sha256_hex(canonical_json(binding))
            connection.execute(
                "INSERT INTO adoption_bindings (binding_id, series_uid,"
                " head_revision_id, selection_id, prediction_id, model_run_id,"
                " revision_ids, adopted_by, adopted_at, payload)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    binding_id,
                    series_uid,
                    expected_head_revision_id,
                    expected_selection_id,
                    str(binding["prediction_id"]),
                    binding.get("model_run_id"),
                    json.dumps(binding["revision_ids"]),
                    str(binding["adopted_by"]),
                    str(binding["adopted_at"]),
                    canonical_json(binding),
                ),
            )
            return [revision.revision_id for revision in revisions]

        return self._in_write_transaction(body)

    def load_latest_whole_cine_prediction(
        self, series_uid: str
    ) -> WholeCinePrediction | None:
        """Load the latest persisted whole-cine prediction for a series, or None.

        Strictly queries whole_cine_predictions (never whole_cine_revisions).
        """
        connection = self._require_connection()
        row = connection.execute(
            "SELECT prediction_id FROM whole_cine_predictions "
            "WHERE series_uid = ? ORDER BY rowid DESC LIMIT 1",
            (series_uid,),
        ).fetchone()
        if row is None:
            return None
        return self.load_whole_cine_prediction(str(row[0]))

    def bootstrap_legacy_chain_checked(
        self,
        series_uid: str,
        prediction: Prediction,
        revisions: Sequence[ContourRevision],
    ) -> str:
        """Atomically bootstrap a legacy chain with P0 and R1..Rk in a single transaction.

        Writes exactly 1 row to predictions and k rows to contour_revisions.
        """
        if not revisions:
            raise SeriesCoherenceError("bootstrap revisions must contain at least one revision")
        if prediction.series_uid != series_uid:
            raise SeriesCoherenceError(
                f"prediction series_uid {prediction.series_uid!r} "
                f"does not match series_uid {series_uid!r}"
            )
        pred_payload_text = self._content_addressed_payload(
            prediction, prediction.prediction_id
        )

        def body(connection: sqlite3.Connection) -> str:
            # Exactly-once guard
            head_row = self._current_head_row(connection, series_uid)
            existing_pred = connection.execute(
                "SELECT 1 FROM predictions WHERE prediction_id = ?",
                (prediction.prediction_id,),
            ).fetchone()
            if head_row is not None or existing_pred is not None:
                raise SeriesCoherenceError(
                    f"series {series_uid!r} already bootstrapped; duplicate bootstrap forbidden"
                )

            # In-transaction chain verification
            # Genesis revision R1
            r0 = revisions[0]
            if r0.parent_prediction_id != prediction.prediction_id:
                raise SeriesCoherenceError(
                    f"genesis revision parent_prediction_id {r0.parent_prediction_id!r} "
                    f"does not match prediction_id {prediction.prediction_id!r}"
                )
            if r0.parent_revision_id is not None:
                raise SeriesCoherenceError(
                    f"genesis revision parent_revision_id must be None, "
                    f"got {r0.parent_revision_id!r}"
                )
            if r0.version != 1:
                raise SeriesCoherenceError(
                    f"genesis revision version must be 1, got {r0.version}"
                )
            if r0.series_uid != series_uid:
                raise SeriesCoherenceError(
                    f"genesis revision series_uid {r0.series_uid!r} does not match {series_uid!r}"
                )

            # Sequential revisions R2..Rk
            for i in range(1, len(revisions)):
                r_prev = revisions[i - 1]
                r_curr = revisions[i]
                if r_curr.parent_revision_id != r_prev.revision_id:
                    raise SeriesCoherenceError(
                        f"revision[{i}] parent_revision_id {r_curr.parent_revision_id!r} "
                        f"does not match previous revision_id {r_prev.revision_id!r}"
                    )
                if r_curr.version != r_prev.version + 1:
                    raise SeriesCoherenceError(
                        f"revision[{i}] version {r_curr.version} does not match "
                        f"expected {r_prev.version + 1}"
                    )
                if r_curr.series_uid != series_uid:
                    raise SeriesCoherenceError(
                        f"revision[{i}] series_uid {r_curr.series_uid!r} "
                        f"does not match {series_uid!r}"
                    )

            # Lineage coherence for all revisions
            for r in revisions:
                _assert_parent_links_share_series(connection, r)

            # Manifest check for prediction if workflow-bound
            bound_workflow = prediction.provenance.workflow_id
            if bound_workflow is not None:
                manifest_row = connection.execute(
                    "SELECT 1 FROM dicom_source_manifests WHERE workflow_id = ?",
                    (bound_workflow,),
                ).fetchone()
                if manifest_row is None:
                    raise SeriesCoherenceError(
                        f"prediction carries workflow {bound_workflow!r} but no "
                        "such DICOM source manifest is persisted; save the "
                        "manifest before any workflow-bound record"
                    )

            # Persist P0
            connection.execute(
                "INSERT INTO predictions (prediction_id, payload, workflow_id) VALUES (?, ?, ?)",
                (prediction.prediction_id, pred_payload_text, bound_workflow),
            )

            # Persist R1..Rk
            for r in revisions:
                r_payload_text = self._content_addressed_payload(r, r.revision_id)
                _insert_revision_row(connection, r, r_payload_text)

            return revisions[-1].revision_id

        return self._in_write_transaction(body)

    def save_approval_checked(
        self,
        expected_head_revision_id: str,
        snapshot: ApprovalSnapshot,
        supplied_qc_report: Any,
        topology_dispositions: Sequence[TopologyDispositionRecord] = (),
    ) -> str:
        """Atomically verify the CURRENT persisted head, then seal ONE approval.

        Inside ONE ``BEGIN IMMEDIATE`` transaction this store API:

        1. resolves the highest-version revision of the snapshot's series;
        2. requires it to equal ``expected_head_revision_id``
           (:class:`StaleHeadError` otherwise, zero staged rows);
        3. requires ``snapshot.source_revision_id`` to equal that head;
        4. digest-verifies the persisted head payload and requires its
           gateway ``qc_report`` to be present;
        5. requires ``canonical_digest(supplied_qc_report)``
           == ``canonical_digest(persisted head qc_report)``
           == ``snapshot.qc_report_digest`` (verdict equality alone is NOT
           sufficient);
        6. verifies measurement, selection, workflow, frames, algorithm and
           revision coherence;
        7. inserts the approval and commits.

        Astra F2/F5 (2026-09-09): the caller passes the topology-disposition
        records the approval was resolved against; this transaction
        RE-VERIFIES that ``snapshot.topology_disposition_digest`` equals the
        digest of the CURRENTLY persisted APPLICABLE disposition set for the
        series (the records bound to the sealed head and selection — not
        every historical row, see below), so an approval can never seal a
        stale or fabricated disposition binding (required test 5).

        R1 review F2 (2026-09-10): ``topology_dispositions`` is append-only
        and a key may now carry one disposition per (head, selection)
        binding. The sealed digest therefore binds the APPLICABLE set —
        every persisted record bound to ``snapshot.source_revision_id`` and
        ``snapshot.selection_id`` — so a superseded historical row never
        changes the identity an approval proves it relied on, while the
        current binding is still proven exactly.

        Production approval paths MUST use this method; the unchecked
        :meth:`save_approval` remains only as a compatibility primitive.
        """
        payload_text = self._content_addressed_payload(snapshot, snapshot.approval_id)
        supplied_digest = canonical_digest(supplied_qc_report)
        expected_dispositions = tuple(topology_dispositions)

        def body(connection: sqlite3.Connection) -> str:
            # Step 0: Pre-approval topology disposition binding check. The
            # relevance/resolution predicate itself runs in the controller
            # (one place, no drift); here the durable binding is verified:
            # the sealed digest must match the persisted APPLICABLE
            # disposition set (F2: the rows bound to the sealed head and
            # selection, not every historical row of the series) exactly,
            # and every disposition it names must still be bound to the
            # CURRENT head and selection.
            persisted_digest: str | None = None
            rows = connection.execute(
                "SELECT disposition_id FROM topology_dispositions WHERE series_uid = ?",
                (snapshot.series_uid,),
            ).fetchall()
            if rows:
                applicable = applicable_topology_dispositions(
                    self.load_topology_dispositions(snapshot.series_uid),
                    head_revision_id=snapshot.source_revision_id,
                    selection_id=snapshot.selection_id or "",
                )
                persisted_digest = (
                    topology_disposition_set_digest(applicable) if applicable else None
                )
            if persisted_digest != snapshot.topology_disposition_digest:
                raise CMRError(
                    TOPOLOGY_BLOCKED,
                    f"cannot approve series {snapshot.series_uid}: the sealed "
                    "topology-disposition digest "
                    f"{snapshot.topology_disposition_digest!r} does not match the "
                    f"persisted disposition set digest {persisted_digest!r} "
                    "(the APPLICABLE set: the records bound to this sealed head "
                    "and selection, not superseded history); an approval must "
                    "bind exactly the dispositions it relied on",
                )
            for record in expected_dispositions:
                if record.source_revision_id != snapshot.source_revision_id:
                    raise CMRError(
                        TOPOLOGY_BLOCKED,
                        f"cannot approve series {snapshot.series_uid}: disposition "
                        f"for key {record.identity_key!r} binds head "
                        f"{record.source_revision_id!r} which is not the sealed "
                        f"head {snapshot.source_revision_id!r}",
                    )
                if snapshot.selection_id is not None and (
                    record.selection_id != snapshot.selection_id
                ):
                    raise CMRError(
                        TOPOLOGY_BLOCKED,
                        f"cannot approve series {snapshot.series_uid}: disposition "
                        f"for key {record.identity_key!r} binds selection "
                        f"{record.selection_id!r} which is not the sealed "
                        f"selection {snapshot.selection_id!r}",
                    )
            # (1)+(2) current persisted head truth, inside this transaction.
            head_row = self._current_head_row(connection, snapshot.series_uid)
            if head_row is None:
                raise StaleHeadError(
                    f"no persisted revision chain for series "
                    f"{snapshot.series_uid!r}; nothing to seal against"
                )
            head_id = str(head_row[0])
            if head_id != expected_head_revision_id:
                raise StaleHeadError(
                    f"stale head: expected {expected_head_revision_id!r} but the "
                    f"persisted chain head is {head_id!r}; zero rows written"
                )
            # (3) the snapshot anchors exactly to the verified head.
            if snapshot.source_revision_id != head_id:
                raise StaleHeadError(
                    f"snapshot source_revision_id {snapshot.source_revision_id!r} "
                    f"does not anchor to the verified current head {head_id!r}"
                )
            # (4) persisted QC truth.
            head_payload = self._verify_persisted_head_payload(head_row)
            persisted_qc = qc_report_from_payload(head_payload.get("qc_report"))
            if persisted_qc is None:
                raise SeriesCoherenceError(
                    f"persisted head revision {head_id!r} carries no QC report; "
                    "measured approval requires the sealed report to equal the "
                    "persisted head's report"
                )
            # (5) exact QC payload digest equality (order-sensitive).
            persisted_digest = canonical_digest(persisted_qc)
            if supplied_digest != persisted_digest:
                raise SeriesCoherenceError(
                    "supplied QC report digest does not match the persisted head "
                    f"revision's QC payload ({supplied_digest[:12]}.. vs "
                    f"{persisted_digest[:12]}..); verdict equality alone is "
                    "insufficient"
                )
            if snapshot.qc_report_digest != supplied_digest:
                raise SeriesCoherenceError(
                    f"snapshot qc_report_digest {snapshot.qc_report_digest!r} does "
                    f"not match the supplied report digest {supplied_digest!r}"
                )
            # (6) measurement / frames / revision / selection / workflow /
            # algorithm coherence, mirroring the compatibility primitive.
            measurement_row = connection.execute(
                "SELECT series_uid, source_revision_id, payload"
                " FROM measurements WHERE measurement_id = ?",
                (snapshot.measurement_id,),
            ).fetchone()
            if measurement_row is None:
                raise SeriesCoherenceError(
                    f"measurement {snapshot.measurement_id!r} is not present"
                )
            m_series, m_source_revision, m_payload_text = measurement_row
            m_payload = dict(json.loads(str(m_payload_text)))
            m_algorithm = m_payload.get("algorithm_version")
            if snapshot.series_uid != m_series:
                raise SeriesCoherenceError(
                    "approval series_uid "
                    f"{snapshot.series_uid!r} does not match measurement series_uid "
                    f"{m_series!r}"
                )
            if (
                snapshot.ed_frame != m_payload["ed_frame"]
                or snapshot.es_frame != m_payload["es_frame"]
            ):
                raise SeriesCoherenceError(
                    f"approval ED/ES frames ({snapshot.ed_frame}, {snapshot.es_frame})"
                    " do not match measurement frames "
                    f"({m_payload['ed_frame']}, {m_payload['es_frame']})"
                )
            if snapshot.source_revision_id != m_source_revision:
                raise SeriesCoherenceError(
                    "approval source_revision_id "
                    f"{snapshot.source_revision_id!r} does not match measurement "
                    f"source_revision_id {m_source_revision!r}"
                )
            if m_algorithm is not None and snapshot.algorithm_version != str(m_algorithm):
                raise SeriesCoherenceError(
                    "approval algorithm_version "
                    f"{snapshot.algorithm_version!r} does not match measurement "
                    f"algorithm_version {str(m_algorithm)!r}"
                )
            self._assert_selection_binding_coherence(
                connection,
                series_uid=snapshot.series_uid,
                sealing_workflow_id=snapshot.provenance.workflow_id,
                head_workflow_id=None if head_row[3] is None else str(head_row[3]),
                measurement_payload=m_payload,
                selection_id=snapshot.selection_id,
            )
            # --- CFM-001 §1.3/§1.7 inclusion binding ----------------------
            # The snapshot must bind EXACTLY the measurement's set identity,
            # which itself must equal the LATEST APPLICABLE set. A null-bound
            # measurement can never seal a new measured approval.
            m_inclusion = (
                None
                if m_payload.get("inclusion_set_id") is None
                else str(m_payload["inclusion_set_id"])
            )
            if (snapshot.inclusion_set_id or None) != (m_inclusion or None):
                raise SeriesCoherenceError(
                    "approval inclusion_set_id "
                    f"{snapshot.inclusion_set_id!r} does not match the "
                    f"measurement's binding {m_inclusion!r}; one coherent "
                    "clinical state required"
                )
            self._verify_inclusion_binding(
                connection,
                series_uid=snapshot.series_uid,
                head_revision_id=head_id,
                workflow_id=snapshot.provenance.workflow_id,
                selection_id=snapshot.selection_id,
                bound_inclusion_set_id=m_inclusion,
                allow_null_set=False,
                gaps_length=len(m_payload["stack_gaps_mm"]),
            )
            # --- F14: the ONE coherent analysis authority ------------------
            # The seal must bind the CURRENT coherent analysis: the same head,
            # the latest selection, the latest applicable inclusion set, the
            # exact 9-metric measurement AND the QC payload digest that
            # measurement was paired with. A measurement that was never
            # published through the analysis authority (or whose inputs have
            # since changed) can never seal "as current".
            current_analysis, analysis_measurement, analysis_reason = (
                self._resolve_current_analysis(
                    connection,
                    series_uid=snapshot.series_uid,
                    expected_n_slices=len(m_payload["stack_gaps_mm"]),
                )
            )
            if current_analysis is None or analysis_measurement is None:
                raise AnalysisAuthorityError(
                    "no CURRENT coherent analysis for series "
                    f"{snapshot.series_uid!r} (reason {analysis_reason!r}); a "
                    "measured approval requires the refreshed selected-phase "
                    "analysis whose metrics AND QC were derived from the current "
                    "head, selection and inclusion state"
                )
            if not current_analysis.binds(
                head_revision_id=snapshot.source_revision_id,
                selection_id=snapshot.selection_id,
                inclusion_set_id=snapshot.inclusion_set_id,
                measurement_id=snapshot.measurement_id,
                qc_report_digest=supplied_digest,
            ):
                raise AnalysisAuthorityError(
                    "stale analysis: the current coherent analysis binds head "
                    f"{current_analysis.head_revision_id!r} / selection "
                    f"{current_analysis.selection_id!r} / inclusion "
                    f"{current_analysis.inclusion_set_id!r} / measurement "
                    f"{current_analysis.measurement_id!r} / qc "
                    f"{current_analysis.qc_report_digest!r} but this approval "
                    f"binds {snapshot.source_revision_id!r} / "
                    f"{snapshot.selection_id!r} / {snapshot.inclusion_set_id!r} / "
                    f"{snapshot.measurement_id!r} / {supplied_digest!r}; re-run "
                    "the analysis against the current state before sealing"
                )
            # --- metric-relevant topology, store-side last line -----------
            # The controller runs the FULL relevance-and-resolution predicate
            # (it owns the one implementation); this transaction re-runs the
            # SAME predicate shape from persisted rows (shared authoring-time
            # helper) so a direct store call can never seal a metric-relevant
            # block whose key is neither authored by the current chain after
            # the block nor covered by an applicable disposition. An unknown
            # grid extent stays conservative (fail closed).
            self._assert_store_topology_not_blocking(
                connection,
                series_uid=snapshot.series_uid,
                head_revision_id=head_id,
                selection_id=snapshot.selection_id or "",
                n_slices=len(m_payload["stack_gaps_mm"]),
                ed_frame=int(m_payload["ed_frame"]),
                es_frame=int(m_payload["es_frame"]),
            )
            # (7) insert. The topology-disposition digest is denormalized so
            # the durable binding is queryable without parsing the payload.
            connection.execute(
                "INSERT INTO approval_snapshots (approval_id, series_uid,"
                " source_revision_id, measurement_id, approved_at, payload,"
                " workflow_id, selection_id, qc_report_digest, inclusion_set_id,"
                " topology_disposition_digest)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.approval_id,
                    snapshot.series_uid,
                    snapshot.source_revision_id,
                    snapshot.measurement_id,
                    snapshot.approved_at,
                    payload_text,
                    snapshot.provenance.workflow_id,
                    snapshot.selection_id,
                    snapshot.qc_report_digest,
                    snapshot.inclusion_set_id,
                    snapshot.topology_disposition_digest,
                ),
            )
            return snapshot.approval_id

        return self._in_write_transaction(body)

    def save_qc_decision_checked(
        self,
        expected_head_revision_id: str,
        record: QCDecisionRecord,
    ) -> str:
        """Atomically verify the CURRENT persisted head, then seal ONE decision.

        Same head/revision/series/workflow/selection checks as
        :meth:`save_approval_checked` but NO QC payload digest comparison —
        the B-LITE sentinel decision remains digest-exempt in S2C-001 while
        still failing closed on a stale head.
        """
        payload_text = self._content_addressed_payload(record, record.decision_id)

        def body(connection: sqlite3.Connection) -> str:
            head_row = self._current_head_row(connection, record.series_uid)
            if head_row is None:
                raise StaleHeadError(
                    f"no persisted revision chain for series {record.series_uid!r}; "
                    "nothing to seal against"
                )
            head_id = str(head_row[0])
            if head_id != expected_head_revision_id:
                raise StaleHeadError(
                    f"stale head: expected {expected_head_revision_id!r} but the "
                    f"persisted chain head is {head_id!r}; zero rows written"
                )
            if record.source_revision_id != head_id:
                raise StaleHeadError(
                    f"decision source_revision_id {record.source_revision_id!r} "
                    f"does not anchor to the verified current head {head_id!r}"
                )
            self._verify_persisted_head_payload(head_row)
            self._assert_selection_binding_coherence(
                connection,
                series_uid=record.series_uid,
                sealing_workflow_id=record.provenance.workflow_id,
                head_workflow_id=None if head_row[3] is None else str(head_row[3]),
                measurement_payload=None,
                selection_id=record.selection_id,
            )
            connection.execute(
                "INSERT INTO qc_decision_records (decision_id, series_uid,"
                " source_revision_id, ed_frame, es_frame, qc_verdict, approved_by,"
                " approved_at, override_reason, payload, workflow_id, selection_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.decision_id,
                    record.series_uid,
                    record.source_revision_id,
                    record.ed_frame,
                    record.es_frame,
                    record.qc_verdict,
                    record.approved_by,
                    record.approved_at,
                    record.override_reason,
                    payload_text,
                    record.provenance.workflow_id,
                    record.selection_id,
                ),
            )
            return record.decision_id

        return self._in_write_transaction(body)

    # --- F14: coherent selected-phase analysis authority (v12) ----------------

    def _insert_analysis_binding(
        self,
        connection: sqlite3.Connection,
        *,
        measurement: Measurement,
        require_qc: bool,
    ) -> SelectedPhaseAnalysisRecord | None:
        """Derive and insert ONE analysis binding for a persisted measurement.

        CMR_FUNCTION_V1_ANALYSIS_001: every measurement-writing path (the
        checked inclusion/analysis transactions, the legacy compatibility pair
        saves and the selected-phase job publication) routes through here, so
        the 9 metrics and the QC report they are paired with become ONE
        persisted, content-addressed record instead of two independently
        readable rows.

        Derivation is deterministic from the CURRENT transaction state:

        * the measurement's ``source_revision_id`` row must exist and carry the
          gateway ``qc_report`` (the exact field ``save_approval_checked``
          re-verifies) — nothing is derived from a caller-supplied report;
        * the measurement's ``selection_id`` must reference a persisted
          selection of the same series whose frames equal the measurement's;
        * the LATEST APPLICABLE inclusion set for ``(head, selection)`` must
          equal the measurement's inclusion binding (both may be null in the
          disclosed legacy null-set states);
        * study/workflow identity comes from the measurement/head provenance.

        Returns ``None`` when the state cannot form a coherent analysis and
        ``require_qc`` is False (the measurement row stays readable but can
        never satisfy the seal's analysis requirement). Raises
        :class:`AnalysisAuthorityError` when ``require_qc`` is True: the
        selected-phase job publication MUST produce metrics + QC or publish
        nothing.
        """
        revision_row = connection.execute(
            "SELECT revision_id, version, payload, workflow_id FROM contour_revisions"
            " WHERE revision_id = ?",
            (measurement.source_revision_id,),
        ).fetchone()
        if revision_row is None:
            raise SeriesCoherenceError(
                "measurement source_revision_id "
                f"{measurement.source_revision_id!r} has no revision row; a "
                "coherent analysis requires the exact contour head it was "
                "computed from"
            )
        if _sha256_hex(str(revision_row[2])) != measurement.source_revision_id:
            raise PayloadIntegrityError(
                "persisted head revision payload digest does not match id "
                f"{measurement.source_revision_id!r}"
            )
        head_payload = dict(json.loads(str(revision_row[2])))
        head_qc = qc_report_from_payload(head_payload.get("qc_report"))
        if head_qc is None:
            if require_qc:
                raise AnalysisAuthorityError(
                    f"head revision {measurement.source_revision_id!r} carries no "
                    "persisted QC report; a coherent selected-phase analysis "
                    "requires metrics AND QC derived from the same exact state — "
                    "apply the confirmed inclusion grid to refresh the analysis "
                    "first"
                )
            return None
        if measurement.selection_id is None:
            if require_qc:
                raise AnalysisAuthorityError(
                    "measurement binds no ED/ES selection; a coherent "
                    "selected-phase analysis requires the explicit selection it "
                    "was computed against"
                )
            return None
        selection_row = connection.execute(
            "SELECT payload FROM edes_selections WHERE selection_id = ?",
            (measurement.selection_id,),
        ).fetchone()
        if selection_row is None:
            raise SeriesCoherenceError(
                f"measurement binds selection {measurement.selection_id!r} but no "
                "such selection row is persisted; a coherent analysis requires "
                "the exact selection it was computed against"
            )
        selection = self._verify_and_reconstruct_selection(
            measurement.selection_id, str(selection_row[0])
        )
        if (selection.ed_frame, selection.es_frame) != (
            measurement.ed_frame,
            measurement.es_frame,
        ):
            raise SeriesCoherenceError(
                "measurement frames "
                f"({measurement.ed_frame}, {measurement.es_frame}) disagree with "
                f"the bound selection frames "
                f"({selection.ed_frame}, {selection.es_frame}); one coherent "
                "clinical state required"
            )
        resolved = self._resolve_latest_applicable_set(
            connection,
            series_uid=measurement.series_uid,
            head_revision_id=measurement.source_revision_id,
            selection_id=measurement.selection_id,
            workflow_id=measurement.provenance.workflow_id,
            expected_n_slices=len(measurement.stack_gaps_mm),
        )
        applicable_id = None if resolved is None else resolved[0].inclusion_set_id
        if (applicable_id or None) != (measurement.inclusion_set_id or None):
            raise SeriesCoherenceError(
                "measurement binds inclusion set "
                f"{measurement.inclusion_set_id!r} but the latest APPLICABLE set "
                f"for head {measurement.source_revision_id!r} / selection "
                f"{measurement.selection_id!r} is {applicable_id!r}; a coherent "
                "analysis must bind the set it was computed with"
            )
        head_provenance = head_payload.get("provenance") or {}
        study_uid = measurement.provenance.study_uid or head_provenance.get("study_uid")
        if not isinstance(study_uid, str) or not study_uid.strip():
            if require_qc:
                raise AnalysisAuthorityError(
                    "the analysis state carries no study identity (study_uid); a "
                    "coherent selected-phase analysis requires study/series/"
                    "workflow identity"
                )
            return None
        record = SelectedPhaseAnalysisRecord(
            series_uid=measurement.series_uid,
            study_uid=study_uid,
            head_revision_id=measurement.source_revision_id,
            selection_id=measurement.selection_id,
            ed_frame=measurement.ed_frame,
            es_frame=measurement.es_frame,
            measurement_id=measurement.measurement_id,
            qc_report_digest=canonical_digest(head_qc),
            qc_verdict=head_qc.verdict,
            algorithm_version=measurement.algorithm_version,
            software_version=measurement.software_version,
            provenance=Provenance(
                study_uid=study_uid,
                series_uid=measurement.series_uid,
                software_version=measurement.software_version,
                model=None,
                created_at=measurement.created_at,
                revision=measurement.provenance.revision,
                workflow_id=measurement.provenance.workflow_id,
            ),
            workflow_id=measurement.provenance.workflow_id,
            inclusion_set_id=measurement.inclusion_set_id,
            range_table_version=measurement.range_table_version,
        )
        # Idempotent replay: the SAME clinical binding can never be persisted
        # twice (the v12 no-replace-binding trigger enforces it at schema
        # level); an identical replay returns the already-persisted row, and a
        # divergent row for the same binding is payload corruption.
        existing = connection.execute(
            "SELECT analysis_id, payload FROM selected_phase_analyses"
            " WHERE head_revision_id = ? AND selection_id = ?"
            "   AND (inclusion_set_id IS ?) AND measurement_id = ?"
            "   AND qc_report_digest = ?",
            (
                record.head_revision_id,
                record.selection_id,
                record.inclusion_set_id,
                record.measurement_id,
                record.qc_report_digest,
            ),
        ).fetchone()
        if existing is not None:
            persisted = self._verify_and_reconstruct(
                str(existing[0]), str(existing[1]), analysis_record_from_payload
            )
            if persisted.analysis_id != record.analysis_id:
                raise PayloadIntegrityError(
                    "two different analysis records claim the same clinical "
                    f"binding; persisted {persisted.analysis_id!r} vs derived "
                    f"{record.analysis_id!r}"
                )
            return persisted
        analysis_payload = self._content_addressed_payload(record, record.analysis_id)
        connection.execute(
            "INSERT INTO selected_phase_analyses (analysis_id, series_uid,"
            " study_uid, workflow_id, head_revision_id, selection_id, ed_frame,"
            " es_frame, inclusion_set_id, measurement_id, qc_report_digest,"
            " qc_verdict, algorithm_version, software_version, range_table_version,"
            " analysis_sequence, created_at, payload)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.analysis_id,
                record.series_uid,
                record.study_uid,
                record.workflow_id,
                record.head_revision_id,
                record.selection_id,
                record.ed_frame,
                record.es_frame,
                record.inclusion_set_id,
                record.measurement_id,
                record.qc_report_digest,
                record.qc_verdict,
                record.algorithm_version,
                record.software_version,
                record.range_table_version,
                _next_sequence(
                    connection,
                    "selected_phase_analyses",
                    "analysis_sequence",
                    measurement.series_uid,
                ),
                record.provenance.created_at,
                analysis_payload,
            ),
        )
        return record

    def save_analysis_checked(
        self,
        expected_parent_head_id: str,
        expected_selection_id: str,
        child: ContourRevision,
        measurement: Measurement,
    ) -> str:
        """Atomically persist the analysis-refresh child AND its measurement.

        One ``BEGIN IMMEDIATE`` transaction: recheck the current head equals
        ``expected_parent_head_id`` and the latest persisted selection equals
        ``expected_selection_id`` (both stale-fail with zero staged rows),
        then insert the revision FIRST and the measurement SECOND so FK
        violations leave no partial state (amendment G ordering preserved).

        CFM-001 inclusion binding: the measurement must bind EXACTLY the
        latest APPLICABLE inclusion set resolved against the expected parent
        head plus the expected selection. When such a set exists, a null or
        different binding fails closed (a concurrent newer set therefore
        rolls the transaction back with ZERO staged rows); when none exists,
        only the legacy null-set compatibility shape is accepted.
        """
        revision_payload = self._content_addressed_payload(child, child.revision_id)
        measurement_payload = self._content_addressed_payload(
            measurement, measurement.measurement_id
        )

        def body(connection: sqlite3.Connection) -> str:
            return self._checked_analysis_body(
                connection,
                expected_parent_head_id=expected_parent_head_id,
                expected_selection_id=expected_selection_id,
                child=child,
                measurement=measurement,
                revision_payload=revision_payload,
                measurement_payload=measurement_payload,
            )

        return self._in_write_transaction(body)

    def save_inclusion_set_with_analysis_checked(
        self,
        inclusion_set: SliceInclusionSet,
        expected_parent_head_id: str,
        expected_selection_id: str,
        child: ContourRevision,
        measurement: Measurement,
    ) -> tuple[str, str]:
        """Atomically persist the applicable set PLUS the refreshed analysis.

        CFM-001 §1.8 "Apply Inclusion & Refresh Analysis": one
        ``BEGIN IMMEDIATE`` transaction inserts the physician-authored
        :class:`SliceInclusionSet` row (coherence guards + store sequence)
        and THEN the QC-bearing child revision and its measurement, whose
        binding is verified against the just-inserted set through the same
        checked path as :meth:`save_analysis_checked`. Any failure —
        including a concurrent newer set landing before the commit — rolls
        EVERYTHING back so zero partial rows survive and the previous
        durable state stays intact.

        Returns ``(inclusion_set_id, child_revision_id)``.
        """
        set_payload = self._content_addressed_payload(inclusion_set, inclusion_set.inclusion_set_id)
        revision_payload = self._content_addressed_payload(child, child.revision_id)
        measurement_payload = self._content_addressed_payload(
            measurement, measurement.measurement_id
        )

        def body(connection: sqlite3.Connection) -> tuple[str, str]:
            # Correction round 2 (finding 4): bind the cine's ACTUAL slice
            # count via the measurement's complete gap-tuple length.
            _insert_slice_inclusion_set_row(
                connection,
                inclusion_set,
                set_payload,
                len(measurement.stack_gaps_mm),
            )
            self._checked_analysis_body(
                connection,
                expected_parent_head_id=expected_parent_head_id,
                expected_selection_id=expected_selection_id,
                child=child,
                measurement=measurement,
                revision_payload=revision_payload,
                measurement_payload=measurement_payload,
            )
            return inclusion_set.inclusion_set_id, child.revision_id

        result = self._in_write_transaction_typed(body)
        assert isinstance(result, tuple)
        return result[0], result[1]

    def save_inclusion_set_with_analysis_expected_checked(
        self,
        inclusion_set: SliceInclusionSet,
        expected_parent_head_id: str | None,
        expected_selection_id: str | None,
        expected_inclusion_set_id: str | None,
        expected_n_slices: int,
        child: ContourRevision,
        measurement: Measurement,
    ) -> tuple[str, str]:
        """Checked inclusion refresh with a nullable three-way lineage CAS.

        Unlike the compatibility method above, this surface resolves the
        current head, latest selection and latest applicable inclusion set
        inside this method's single ``BEGIN IMMEDIATE`` transaction and
        compares all three identities before it considers duplicate content.
        A successful unchanged replay returns ``(current_set_id, head_id)``;
        the second value is deliberately the current head rather than the
        prepared child id, which lets the controller report ``unchanged``
        without a pre-transaction idempotency shortcut.
        """
        expected_n_slices = self._validate_expected_n_slices(expected_n_slices)
        expected_parent_head_id = self._validate_checked_identity(
            "expected_parent_head_id", expected_parent_head_id
        )
        expected_selection_id = self._validate_checked_identity(
            "expected_selection_id", expected_selection_id
        )
        expected_inclusion_set_id = self._validate_checked_identity(
            "expected_inclusion_set_id", expected_inclusion_set_id
        )
        if len(measurement.stack_gaps_mm) != expected_n_slices:
            raise SeriesCoherenceError(
                "measurement stack_gaps_mm length must equal expected_n_slices; "
                f"got {len(measurement.stack_gaps_mm)} and {expected_n_slices}"
            )
        set_payload = self._content_addressed_payload(
            inclusion_set, inclusion_set.inclusion_set_id
        )
        revision_payload = self._content_addressed_payload(child, child.revision_id)
        measurement_payload = self._content_addressed_payload(
            measurement, measurement.measurement_id
        )

        def _grid_signature(record: SliceInclusionSet) -> str:
            return canonical_digest(
                sorted(
                    (
                        int(entry.slice_index),
                        entry.frame_role.value,
                        entry.structure.value,
                        bool(entry.included),
                    )
                    for entry in record.entries
                )
            )

        def body(connection: sqlite3.Connection) -> tuple[str, str]:
            # The complete CAS is the first lineage-sensitive operation in
            # this transaction.  It must precede every unchanged/duplicate
            # decision and every INSERT.
            actual = self._resolve_current_nullable_lineage(
                connection,
                series_uid=inclusion_set.series_uid,
                expected_n_slices=expected_n_slices,
            )
            self._assert_checked_lineage(
                actual=actual,
                expected_head_revision_id=expected_parent_head_id,
                expected_selection_id=expected_selection_id,
                expected_inclusion_set_id=expected_inclusion_set_id,
            )

            # Idempotency is resolved only after the nullable CAS.  It uses
            # the same clinically meaningful content checks as the frozen
            # controller path: current head QC, current applicable grid and a
            # coherent measurement bound to that grid.
            head_row = self._current_head_row(connection, inclusion_set.series_uid)
            if head_row is None or actual[0] is None or actual[1] is None:
                raise SeriesCoherenceError(
                    "checked inclusion refresh requires a current head and a "
                    "persisted ED/ES selection"
                )
            # Nullable CAS equality above guarantees these values are the
            # non-null current identities.  Narrow them before delegating to
            # the legacy checked-analysis body, whose contract intentionally
            # remains non-null for an analysis-bearing child.
            if expected_parent_head_id is None or expected_selection_id is None:
                raise SeriesCoherenceError(
                    "checked inclusion refresh requires non-null head and "
                    "selection identities once persisted state exists"
                )
            current_set: SliceInclusionSet | None = None
            if actual[2] is not None:
                resolved = self._resolve_latest_applicable_set(
                    connection,
                    series_uid=inclusion_set.series_uid,
                    head_revision_id=actual[0],
                    selection_id=actual[1],
                    workflow_id=None if head_row[3] is None else str(head_row[3]),
                    expected_n_slices=expected_n_slices,
                )
                if resolved is not None:
                    current_set = resolved[0]

            head_payload = self._verify_persisted_head_payload(head_row)
            current_qc = qc_report_from_payload(head_payload.get("qc_report"))
            current_measurement: Measurement | None = None
            measurement_row = connection.execute(
                "SELECT measurement_id, payload, range_table_version"
                " FROM measurements WHERE series_uid = ?"
                "   AND source_revision_id = ?"
                "   AND json_extract(payload, '$.selection_id') = ?"
                " ORDER BY measurement_sequence DESC LIMIT 1",
                (inclusion_set.series_uid, actual[0], actual[1]),
            ).fetchone()
            if measurement_row is not None:
                current_measurement = self._verify_and_reconstruct(
                    measurement_row[0], measurement_row[1], measurement_from_payload
                )
                _assert_range_column_matches_payload(
                    measurement_row[2], current_measurement
                )

            qc_same = (
                current_qc is None
                and child.qc_report is None
            ) or (
                current_qc is not None
                and child.qc_report is not None
                and canonical_digest(current_qc) == canonical_digest(child.qc_report)
            )
            if (
                current_set is not None
                and actual[2] == current_set.inclusion_set_id
                and _grid_signature(current_set) == _grid_signature(inclusion_set)
                and qc_same
                and current_measurement is not None
                and current_measurement.source_revision_id == actual[0]
                and current_measurement.selection_id == actual[1]
                and current_measurement.inclusion_set_id == actual[2]
            ):
                # F14 heal: a state that is already content-identical still
                # needs its ONE coherent analysis binding (legacy rows and
                # states written before v12 carry none). The insert is
                # idempotent and commits with this same transaction.
                self._insert_analysis_binding(
                    connection, measurement=current_measurement, require_qc=False
                )
                # No row is staged.  This marker is consumed by the checked
                # controller after the transaction has committed.
                return current_set.inclusion_set_id, actual[0]

            _insert_slice_inclusion_set_row(
                connection,
                inclusion_set,
                set_payload,
                expected_n_slices,
            )
            self._checked_analysis_body(
                connection,
                expected_parent_head_id=expected_parent_head_id,
                expected_selection_id=expected_selection_id,
                child=child,
                measurement=measurement,
                revision_payload=revision_payload,
                measurement_payload=measurement_payload,
            )
            return inclusion_set.inclusion_set_id, child.revision_id

        result = self._in_write_transaction_typed(body)
        assert isinstance(result, tuple)
        return result[0], result[1]

    def _in_write_transaction_typed(self, body: Callable[[sqlite3.Connection], Any]) -> Any:
        """Same protocol as :meth:`_in_write_transaction` without forcing a
        string result (the compound CFM-001 save returns a tuple)."""
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            result = body(connection)
            connection.execute("COMMIT")
        except BaseException:
            self._rollback_quietly(connection)
            raise
        return result

    def _checked_analysis_body(
        self,
        connection: sqlite3.Connection,
        *,
        expected_parent_head_id: str,
        expected_selection_id: str,
        child: ContourRevision,
        measurement: Measurement,
        revision_payload: str,
        measurement_payload: str,
    ) -> str:
        """Shared checked-analysis transaction body (S2C-001 + CFM-001)."""
        head_row = self._current_head_row(connection, child.series_uid)
        if head_row is None:
            raise StaleHeadError(f"no persisted revision chain for series {child.series_uid!r}")
        head_id = str(head_row[0])
        if head_id != expected_parent_head_id:
            raise StaleHeadError(
                f"stale head: expected parent {expected_parent_head_id!r} but "
                f"the persisted chain head is {head_id!r}; revision and "
                "measurement roll back together"
            )
        latest_selection_row = connection.execute(
            "SELECT selection_id FROM edes_selections WHERE series_uid = ?"
            " ORDER BY selection_sequence DESC LIMIT 1",
            (child.series_uid,),
        ).fetchone()
        latest_selection_id = None if latest_selection_row is None else str(latest_selection_row[0])
        if latest_selection_id != expected_selection_id:
            raise StaleHeadError(
                f"stale selection: expected {expected_selection_id!r} but the "
                f"latest persisted selection is {latest_selection_id!r}; "
                "revision and measurement roll back together"
            )
        # S2C-001 correction F3: the sealed analysis must bind EXACTLY the
        # expected selection and compose onto the expected parent head.
        if (measurement.selection_id or None) != (expected_selection_id or None):
            raise SeriesCoherenceError(
                "measurement binds selection "
                f"{measurement.selection_id!r} but this transaction seals "
                f"selection {expected_selection_id!r}"
            )
        if child.parent_revision_id != expected_parent_head_id:
            raise SeriesCoherenceError(
                "child revision composes onto parent "
                f"{child.parent_revision_id!r} but this transaction expects "
                f"parent head {expected_parent_head_id!r}"
            )
        if measurement.series_uid != child.series_uid:
            raise SeriesCoherenceError(
                "measurement series_uid "
                f"{measurement.series_uid!r} does not match revision series_uid "
                f"{child.series_uid!r}"
            )
        if measurement.source_revision_id != child.revision_id:
            raise SeriesCoherenceError(
                "measurement source_revision_id "
                f"{measurement.source_revision_id!r} does not match the "
                f"revision saved in this transaction {child.revision_id!r}"
            )
        # S2C-001 correction F3: one workflow identity across verified
        # head, composed child, measurement and latest persisted
        # selection; bound/unbound mixing fails closed before staging.
        selection_workflow_row = connection.execute(
            "SELECT workflow_id FROM edes_selections WHERE selection_id = ?",
            (expected_selection_id,),
        ).fetchone()
        if selection_workflow_row is None and expected_selection_id is not None:
            raise SeriesCoherenceError(f"selection {expected_selection_id!r} is not present")
        _assert_bound_workflow_agreement(
            head=None if head_row[3] is None else str(head_row[3]),
            child=child.provenance.workflow_id,
            measurement=measurement.provenance.workflow_id,
            selection=(
                None
                if selection_workflow_row is None or selection_workflow_row[0] is None
                else str(selection_workflow_row[0])
            ),
        )
        # --- CFM-001 inclusion binding ------------------------------------
        self._verify_inclusion_binding(
            connection,
            series_uid=child.series_uid,
            head_revision_id=head_id,
            workflow_id=child.provenance.workflow_id,
            selection_id=expected_selection_id or None,
            bound_inclusion_set_id=measurement.inclusion_set_id,
            allow_null_set=True,
            gaps_length=len(measurement.stack_gaps_mm),
        )
        _assert_parent_links_share_series(connection, child)
        # Revision row FIRST, measurement row SECOND (FK order).
        _insert_revision_row(connection, child, revision_payload)
        referenced_series = connection.execute(
            "SELECT series_uid, workflow_id FROM contour_revisions WHERE revision_id = ?",
            (measurement.source_revision_id,),
        ).fetchone()
        if referenced_series is None:
            raise SeriesCoherenceError(
                "measurement source_revision_id "
                f"{measurement.source_revision_id!r} has no revision row"
            )
        if referenced_series[0] != measurement.series_uid:
            raise SeriesCoherenceError(
                "measurement series_uid "
                f"{measurement.series_uid!r} does not match source revision "
                f"series_uid {referenced_series[0]!r}"
            )
        connection.execute(
            "INSERT INTO measurements (measurement_id, series_uid,"
            " source_revision_id, created_at, payload, workflow_id,"
            " selection_id, inclusion_set_id, measurement_sequence,"
            " range_table_version)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                measurement.measurement_id,
                measurement.series_uid,
                measurement.source_revision_id,
                measurement.created_at,
                measurement_payload,
                measurement.provenance.workflow_id,
                measurement.selection_id,
                measurement.inclusion_set_id,
                _next_sequence(
                    connection,
                    "measurements",
                    "measurement_sequence",
                    measurement.series_uid,
                ),
                _range_table_version_of_payload(measurement_payload),
            ),
        )
        # F14: the child (QC-bearing head) + measurement derive the ONE
        # coherent analysis binding inside this same transaction, so metrics,
        # QC and their exact inputs commit or roll back together.
        self._insert_analysis_binding(
            connection, measurement=measurement, require_qc=False
        )
        return child.revision_id

    def _verify_inclusion_binding(
        self,
        connection: sqlite3.Connection,
        *,
        series_uid: str,
        head_revision_id: str,
        workflow_id: str | None,
        selection_id: str | None,
        bound_inclusion_set_id: str | None,
        allow_null_set: bool,
        gaps_length: int | None = None,
    ) -> None:
        """CFM-001 §1.7 binding verification inside a sealing transaction.

        - resolve the LATEST APPLICABLE set for the sealing context;
        - when one exists, the sealed record MUST bind exactly its id —
          a concurrent newer set therefore fails deterministically here
          with ZERO staged rows (:class:`StaleHeadError`);
        - when none exists, a non-null binding is stale-state and fails;
          a null binding is legal ONLY where explicitly allowed (the
          analysis compatibility path). Post-CFM measured APPROVALS never
          allow it (spec §1.3).
        """
        resolved = self._resolve_latest_applicable_set(
            connection,
            series_uid=series_uid,
            head_revision_id=head_revision_id,
            selection_id=selection_id,
            workflow_id=workflow_id,
            expected_n_slices=gaps_length,
        )
        if resolved is None:
            if bound_inclusion_set_id is not None:
                raise StaleHeadError(
                    f"stale inclusion state: record binds inclusion set "
                    f"{bound_inclusion_set_id!r} but NO applicable inclusion "
                    f"set exists for series {series_uid!r} at head "
                    f"{head_revision_id!r} (selection {selection_id!r}); "
                    "zero rows written"
                )
            if not allow_null_set:
                raise SeriesCoherenceError(
                    "post-CFM measured approval requires the current "
                    "applicable inclusion set: a measurement with "
                    "inclusion_set_id=None cannot be sealed; confirm and "
                    "persist the complete inclusion grid first (spec CFM-001 "
                    "§1.3)"
                )
            return
        candidate, sequence = resolved
        if bound_inclusion_set_id is None:
            raise StaleHeadError(
                f"stale inclusion state: applicable inclusion set "
                f"{candidate.inclusion_set_id!r} (sequence {sequence}) exists "
                "but the measurement binds none; refresh the analysis with "
                "the confirmed set before sealing; zero rows written"
            )
        if bound_inclusion_set_id != candidate.inclusion_set_id:
            raise StaleHeadError(
                f"stale inclusion state: record binds inclusion set "
                f"{bound_inclusion_set_id!r} but the latest APPLICABLE set is "
                f"{candidate.inclusion_set_id!r} (sequence {sequence}); a "
                "concurrent newer set invalidates the sealed state; zero "
                "rows written"
            )

    # --- FNC-E analysis jobs ------------------------------------------------------

    @staticmethod
    def _analysis_job_record(row: sqlite3.Row | tuple[Any, ...]) -> AnalysisJobRecord:
        return AnalysisJobRecord(
            job_id=str(row[0]),
            series_uid=str(row[1]),
            kind=str(row[2]),
            dedup_key=str(row[3]),
            request_payload=str(row[4]),
            state=str(row[5]),
            error_code=None if row[6] is None else str(row[6]),
            artifact_measurement_id=None if row[7] is None else str(row[7]),
            timings=str(row[8]),
            created_at=str(row[9]),
            updated_at=str(row[10]),
        )

    def enqueue_analysis_job(
        self,
        *,
        series_uid: str,
        kind: str,
        request_payload: Mapping[str, Any],
        dedup_identity: Mapping[str, Any],
        created_at: str,
    ) -> AnalysisJobRecord:
        """Insert one canonical job, returning the existing row on dedup.

        Analysis timestamps are stored as UTC ISO-8601 with six fractional
        digits and an explicit ``+00:00`` suffix.  Normalizing at this
        boundary prevents offset spellings from changing debounce order.
        """
        if kind not in ANALYSIS_JOB_KINDS:
            raise ValueError(f"unsupported analysis job kind: {kind!r}")
        if not isinstance(series_uid, str) or not series_uid:
            raise ValueError("series_uid must be a non-empty string")
        created_at = _normalize_analysis_timestamp(created_at)
        payload_text = canonical_json(request_payload)
        dedup_key = canonical_digest(dedup_identity)
        job_id = _sha256_hex(payload_text)
        timings = canonical_json(
            {
                "enqueued_at": created_at,
                "debounce_wait_s": None,
                "claimed_at": None,
                "coverage_resolution_s": None,
                "qc_s": None,
                "quantification_s": None,
                "publish_s": None,
                "finished_at": None,
            }
        )

        def body(connection: sqlite3.Connection) -> AnalysisJobRecord:
            connection.execute(
                "INSERT INTO analysis_jobs (job_id, series_uid, kind, dedup_key, "
                "request_payload, state, error_code, artifact_measurement_id, "
                "timings, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    series_uid,
                    kind,
                    dedup_key,
                    payload_text,
                    "queued",
                    None,
                    None,
                    timings,
                    created_at,
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT job_id, series_uid, kind, dedup_key, request_payload, state, "
                "error_code, artifact_measurement_id, timings, created_at, updated_at "
                "FROM analysis_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            assert row is not None
            return self._analysis_job_record(row)

        try:
            return self._in_write_transaction_typed(body)
        except sqlite3.IntegrityError:
            # UNIQUE(series_uid, kind, dedup_key) is the authoritative debounce
            # identity. A conflicting job_id is never silently re-bound.
            row = self._require_connection().execute(
                "SELECT job_id, series_uid, kind, dedup_key, request_payload, state, "
                "error_code, artifact_measurement_id, timings, created_at, updated_at "
                "FROM analysis_jobs WHERE series_uid = ? AND kind = ? AND dedup_key = ?",
                (series_uid, kind, dedup_key),
            ).fetchone()
            if row is None:
                raise
            return self._analysis_job_record(row)

    def load_analysis_jobs(
        self, series_uid: str, *, limit: int | None = None
    ) -> list[AnalysisJobRecord]:
        if limit is not None and (isinstance(limit, bool) or limit < 1):
            raise ValueError("limit must be a positive integer when supplied")
        sql = (
            "SELECT job_id, series_uid, kind, dedup_key, request_payload, state, "
            "error_code, artifact_measurement_id, timings, created_at, updated_at "
            "FROM analysis_jobs WHERE series_uid = ? ORDER BY created_at ASC, job_id ASC"
        )
        params: list[Any] = [series_uid]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._require_connection().execute(sql, params).fetchall()
        return [self._analysis_job_record(row) for row in rows]

    def claim_next_queued_analysis_job(
        self,
        *,
        series_uid: str | None = None,
        now: str,
        debounce_s: float,
    ) -> AnalysisJobRecord | None:
        """Atomically claim the oldest eligible queued job."""
        if debounce_s < 0:
            raise ValueError("debounce_s must be non-negative")
        now_text = _normalize_analysis_timestamp(now)
        now_dt = datetime.fromisoformat(now_text)
        cutoff = now_dt - timedelta(seconds=float(debounce_s))
        cutoff_text = cutoff.isoformat(timespec="microseconds")

        def body(connection: sqlite3.Connection) -> AnalysisJobRecord | None:
            query = (
                "SELECT job_id, series_uid, kind, dedup_key, request_payload, state, "
                "error_code, artifact_measurement_id, timings, created_at, updated_at "
                "FROM analysis_jobs WHERE state = 'queued' AND created_at <= ?"
            )
            params: list[Any] = [cutoff_text]
            if series_uid is not None:
                query += " AND series_uid = ?"
                params.append(series_uid)
            query += " ORDER BY created_at ASC, job_id ASC LIMIT 1"
            row = connection.execute(query, params).fetchone()
            if row is None:
                return None
            job_id = str(row[0])
            timings = json.loads(str(row[8]))
            timings["claimed_at"] = now_text
            timings["debounce_wait_s"] = max(
                0.0,
                (
                    now_dt
                    - datetime.fromisoformat(_normalize_analysis_timestamp(str(row[9])))
                ).total_seconds(),
            )
            timings_text = canonical_json(timings)
            connection.execute(
                "UPDATE analysis_jobs SET state = 'running', updated_at = ?, timings = ? "
                "WHERE job_id = ? AND state = 'queued'",
                (now_text, timings_text, job_id),
            )
            updated = connection.execute(
                "SELECT job_id, series_uid, kind, dedup_key, request_payload, state, "
                "error_code, artifact_measurement_id, timings, created_at, updated_at "
                "FROM analysis_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            assert updated is not None
            return self._analysis_job_record(updated)

        return self._in_write_transaction_typed(body)

    def reconcile_stale_analysis_jobs(
        self, *, now: str, stale_after_s: float,
        exclude_job_ids: Collection[str] = (),
    ) -> int:
        """Fail old running jobs orphaned by a prior worker process.

        This is restart recovery only: stale jobs are never recomputed because
        their in-flight lineage and worker ownership are no longer trustworthy.
        ``exclude_job_ids`` protects jobs actively owned by a live worker in
        the same process (the periodic loop passes its own live tokens).
        """
        if stale_after_s < 0:
            raise ValueError("stale_after_s must be non-negative")
        now_text = _normalize_analysis_timestamp(now)
        now_dt = datetime.fromisoformat(now_text)
        excluded = set(exclude_job_ids)

        def body(connection: sqlite3.Connection) -> int:
            rows = connection.execute(
                "SELECT job_id, updated_at FROM analysis_jobs WHERE state = 'running'"
            ).fetchall()
            stale_ids = [
                str(row[0])
                for row in rows
                if str(row[0]) not in excluded
                and (
                    now_dt
                    - datetime.fromisoformat(_normalize_analysis_timestamp(str(row[1])))
                ).total_seconds()
                >= float(stale_after_s)
            ]
            for job_id in stale_ids:
                connection.execute(
                    "UPDATE analysis_jobs SET state = 'failed', "
                    "error_code = 'interrupted_worker_restart', updated_at = ? "
                    "WHERE job_id = ? AND state = 'running'",
                    (now_text, job_id),
                )
            return len(stale_ids)

        return self._in_write_transaction_typed(body)

    def finish_analysis_job(
        self,
        job_id: str,
        *,
        state: str,
        error_code: str | None,
        artifact_measurement_id: str | None,
        timings: Mapping[str, Any],
        updated_at: str | None = None,
    ) -> AnalysisJobRecord:
        if state not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("finish state must be terminal")
        timings_text = canonical_json(timings)
        updated_at = updated_at or str(timings.get("finished_at") or "")

        def body(connection: sqlite3.Connection) -> AnalysisJobRecord:
            row = connection.execute(
                "SELECT job_id, series_uid, kind, dedup_key, request_payload, state, "
                "error_code, artifact_measurement_id, timings, created_at, updated_at "
                "FROM analysis_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            if str(row[5]) in {"succeeded", "failed", "cancelled"}:
                return self._analysis_job_record(row)
            connection.execute(
                "UPDATE analysis_jobs SET state = ?, error_code = ?, "
                "artifact_measurement_id = ?, timings = ?, updated_at = ? WHERE job_id = ?",
                (state, error_code, artifact_measurement_id, timings_text, updated_at, job_id),
            )
            updated = connection.execute(
                "SELECT job_id, series_uid, kind, dedup_key, request_payload, state, "
                "error_code, artifact_measurement_id, timings, created_at, updated_at "
                "FROM analysis_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            assert updated is not None
            return self._analysis_job_record(updated)

        return self._in_write_transaction_typed(body)

    def cancel_analysis_job(
        self, job_id: str, *, updated_at: str, series_uid: str | None = None
    ) -> AnalysisJobRecord:
        def body(connection: sqlite3.Connection) -> AnalysisJobRecord:
            row = connection.execute(
                "SELECT job_id, series_uid, kind, dedup_key, request_payload, state, "
                "error_code, artifact_measurement_id, timings, created_at, updated_at "
                "FROM analysis_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None or (series_uid is not None and str(row[1]) != series_uid):
                raise KeyError(job_id)
            if str(row[5]) in {"queued", "running"}:
                connection.execute(
                    "UPDATE analysis_jobs SET state = 'cancelled', error_code = 'cancelled', "
                    "updated_at = ? WHERE job_id = ? AND state IN ('queued','running')",
                    (updated_at, job_id),
                )
            updated = connection.execute(
                "SELECT job_id, series_uid, kind, dedup_key, request_payload, state, "
                "error_code, artifact_measurement_id, timings, created_at, updated_at "
                "FROM analysis_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            assert updated is not None
            return self._analysis_job_record(updated)

        return self._in_write_transaction_typed(body)

    @staticmethod
    def _inference_job_record(row: sqlite3.Row | tuple[Any, ...]) -> InferenceJobRecord:
        return InferenceJobRecord(
            job_id=str(row[0]),
            series_uid=str(row[1]),
            state=str(row[3]),
            error_code=None if row[4] is None else str(row[4]),
            stage1_checkpoint=str(row[5]),
            stage2_checkpoint=str(row[6]),
            device=str(row[7]),
            selected_frames=str(row[8]),
            prediction_id=None if row[9] is None else str(row[9]),
            model_run_id=None if row[10] is None else str(row[10]),
            request_payload=str(row[11]),
            timings=str(row[12]),
            created_at=str(row[13]),
            updated_at=str(row[14]),
        )

    @staticmethod
    def _inference_job_select(
        connection: sqlite3.Connection, *, job_id: str
    ) -> tuple[Any, ...] | None:
        return connection.execute(
            "SELECT job_id, series_uid, dedup_key, state, error_code, stage1_checkpoint, "
            "stage2_checkpoint, device, selected_frames, prediction_id, model_run_id, "
            "request_payload, timings, created_at, updated_at "
            "FROM inference_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()

    def enqueue_inference_job(
        self,
        *,
        series_uid: str,
        stage1_checkpoint: str,
        stage2_checkpoint: str,
        selected_frames: Sequence[int],
        request_payload: Mapping[str, Any],
        created_at: str,
        device: str = "cpu",
    ) -> InferenceJobRecord:
        """Append a user-requested sparse inference job.

        Logical vs attempt identity (A4): the LOGICAL dedup identity is the
        canonical request semantics (series_uid, selected_frames, bound
        head_revision_id + selection_id, device, both checkpoint digests,
        provider + algorithm version — NEVER absolute host paths, A8);
        callers pass it as ``request_payload``. The ATTEMPT identity is
        ``job_id = sha256(dedup_key + canonical attempt discriminator)``
        where the discriminator is ``request_attempt`` (count of existing
        rows with the same dedup_key + 1) computed INSIDE the same
        ``BEGIN IMMEDIATE`` transaction as the live lookup + insert.

        Live-only coalescing: a same-dedup_key row in ``queued|running`` is
        returned unchanged; terminal rows never coalesce — a re-request
        (even with a fixed clock) produces a different ``job_id``. This
        intentionally avoids a uniqueness constraint: physicians can
        explicitly re-request inference. No v9 all-history
        uniqueness/replacement guard is copied onto ``dedup_key``.
        """
        if not isinstance(series_uid, str) or not series_uid:
            raise ValueError("series_uid must be a non-empty string")
        if not isinstance(stage1_checkpoint, str) or not stage1_checkpoint:
            raise ValueError("stage1_checkpoint must be a non-empty string")
        if not isinstance(stage2_checkpoint, str) or not stage2_checkpoint:
            raise ValueError("stage2_checkpoint must be a non-empty string")
        if isinstance(selected_frames, (str, bytes)):
            raise ValueError("selected_frames must be a sequence of integers")
        frames = list(selected_frames)
        if not frames or any(
            isinstance(frame, bool) or not isinstance(frame, int) for frame in frames
        ):
            raise ValueError("selected_frames must be a non-empty sequence of integers")
        if frames != sorted(set(frames)):
            raise ValueError("selected_frames must be ascending and unique")
        created_at = _normalize_analysis_timestamp(created_at)
        # A4: logical dedup identity covers ONLY portable semantics. The
        # caller-supplied request_payload is the logical identity EXCEPT the
        # attempt discriminator and any absolute-path checkpoint spellings
        # (A8: absolute host paths are local runtime config, never identity).
        logical_identity = {
            key: value
            for key, value in dict(request_payload).items()
            if key not in {"request_attempt", "stage1_checkpoint", "stage2_checkpoint"}
        }
        base_payload = canonical_json(logical_identity)
        dedup_key = _sha256_hex(base_payload)
        selected_frames_text = canonical_json(frames)
        timings = canonical_json(
            {
                "enqueued_at": created_at,
                "runtime_resolution_s": None,
                "inference_s": None,
                "publish_s": None,
                "finished_at": None,
            }
        )

        def body(connection: sqlite3.Connection) -> InferenceJobRecord:
            live = connection.execute(
                "SELECT job_id, series_uid, dedup_key, state, error_code, "
                "stage1_checkpoint, stage2_checkpoint, device, selected_frames, "
                "prediction_id, model_run_id, request_payload, timings, created_at, "
                "updated_at FROM inference_jobs "
                "WHERE series_uid = ? AND dedup_key = ? "
                "AND state IN ('queued','running') ORDER BY created_at ASC LIMIT 1",
                (series_uid, dedup_key),
            ).fetchone()
            if live is not None:
                return self._inference_job_record(live)
            # A4: attempt discriminator. request_attempt = (# existing rows
            # with this dedup_key) + 1, computed INSIDE this same
            # BEGIN IMMEDIATE transaction as the live lookup + insert, so two
            # concurrent enqueues of the identical request serialize: exactly
            # one live row survives (the loser either coalesces on the
            # winner's row or fails on the job_id guard — never two live
            # rows). job_id = sha256(dedup_key + canonical discriminator).
            payload = dict(request_payload)
            payload.pop("request_attempt", None)
            prior_count = connection.execute(
                "SELECT COUNT(*) FROM inference_jobs WHERE series_uid = ? AND dedup_key = ?",
                (series_uid, dedup_key),
            ).fetchone()
            attempt = int(prior_count[0]) + 1 if prior_count is not None else 1
            payload["request_attempt"] = attempt
            payload_text = canonical_json(payload)
            job_id = _sha256_hex(dedup_key + payload_text)
            try:
                connection.execute(
                    "INSERT INTO inference_jobs (job_id, series_uid, dedup_key, state, error_code, "
                    "stage1_checkpoint, stage2_checkpoint, device, selected_frames, "
                    "prediction_id, model_run_id, request_payload, timings, created_at,"
                    " updated_at) "
                    "VALUES (?, ?, ?, 'queued', NULL, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)",
                    (
                        job_id,
                        series_uid,
                        dedup_key,
                        stage1_checkpoint,
                        stage2_checkpoint,
                        str(device),
                        selected_frames_text,
                        payload_text,
                        timings,
                        created_at,
                        created_at,
                    ),
                )
            except sqlite3.IntegrityError:
                # Concurrent duplicate attempt insert (same dedup_key +
                # same attempt): re-read the live row under this
                # transaction's snapshot — exactly one live row survives.
                live_retry = connection.execute(
                    "SELECT job_id, series_uid, dedup_key, state, error_code, "
                    "stage1_checkpoint, stage2_checkpoint, device, selected_frames, "
                    "prediction_id, model_run_id, request_payload, timings, created_at, "
                    "updated_at FROM inference_jobs "
                    "WHERE series_uid = ? AND dedup_key = ? "
                    "AND state IN ('queued','running') ORDER BY created_at ASC LIMIT 1",
                    (series_uid, dedup_key),
                ).fetchone()
                if live_retry is not None:
                    return self._inference_job_record(live_retry)
                raise
            row = self._inference_job_select(connection, job_id=job_id)
            assert row is not None
            return self._inference_job_record(row)

        return self._in_write_transaction_typed(body)

    def load_inference_jobs(self, series_uid: str) -> list[InferenceJobRecord]:
        rows = self._require_connection().execute(
            "SELECT job_id, series_uid, dedup_key, state, error_code, "
            "stage1_checkpoint, stage2_checkpoint, device, selected_frames, "
            "prediction_id, model_run_id, request_payload, timings, created_at, "
            "updated_at FROM inference_jobs "
            "WHERE series_uid = ? ORDER BY created_at ASC, job_id ASC",
            (series_uid,),
        ).fetchall()
        return [self._inference_job_record(row) for row in rows]

    def claim_next_queued_inference_job(self, *, now: str) -> InferenceJobRecord | None:
        now_text = _normalize_analysis_timestamp(now)

        def body(connection: sqlite3.Connection) -> InferenceJobRecord | None:
            row = connection.execute(
                "SELECT job_id, series_uid, dedup_key, state, error_code, "
                "stage1_checkpoint, stage2_checkpoint, device, selected_frames, "
                "prediction_id, model_run_id, request_payload, timings, created_at, "
                "updated_at FROM inference_jobs "
                "WHERE state = 'queued' ORDER BY created_at ASC, job_id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            timings = json.loads(str(row[12]))
            timings["claimed_at"] = now_text
            connection.execute(
                "UPDATE inference_jobs SET state = 'running', updated_at = ?, timings = ? "
                "WHERE job_id = ? AND state = 'queued'",
                (now_text, canonical_json(timings), str(row[0])),
            )
            updated = self._inference_job_select(connection, job_id=str(row[0]))
            assert updated is not None
            return self._inference_job_record(updated)

        return self._in_write_transaction_typed(body)

    def reconcile_stale_inference_jobs(
        self, *, now: str, stale_after_s: float, exclude_job_ids: Collection[str] = ()
    ) -> int:
        """Mark orphaned ``running`` inference rows terminal (A3).

        Crash/restart recovery: a row left ``running`` with no live owner is
        terminal ``failed/interrupted`` (shared ``RUNNING_STALE_S=600``
        bound; the runner passes its live-token job ids as
        ``exclude_job_ids`` so active computation is protected). A crash
        between glue persist and job attach leaves the artifact unbound, so
        A1 keeps it ineligible. Single-worker ownership is assumed: two live
        workers claiming from the same database is unsupported.
        """
        if stale_after_s < 0:
            raise ValueError("stale_after_s must be non-negative")
        now_text = _normalize_analysis_timestamp(now)
        now_dt = datetime.fromisoformat(now_text)
        excluded = set(exclude_job_ids)

        def body(connection: sqlite3.Connection) -> int:
            rows = connection.execute(
                "SELECT job_id, updated_at FROM inference_jobs WHERE state = 'running'"
            ).fetchall()
            stale = [
                str(row[0]) for row in rows
                if str(row[0]) not in excluded
                and (now_dt - datetime.fromisoformat(
                    _normalize_analysis_timestamp(str(row[1]))
                )).total_seconds() >= float(stale_after_s)
            ]
            for job_id in stale:
                connection.execute(
                    "UPDATE inference_jobs SET state = 'failed', "
                    "error_code = 'interrupted', updated_at = ? "
                    "WHERE job_id = ? AND state = 'running'",
                    (now_text, job_id),
                )
            return len(stale)

        return self._in_write_transaction_typed(body)

    def finish_inference_job(
        self,
        job_id: str,
        *,
        state: str,
        error_code: str | None,
        prediction_id: str | None = None,
        model_run_id: str | None = None,
        timings: Mapping[str, Any],
        updated_at: str | None = None,
    ) -> InferenceJobRecord:
        """Move a ``running`` inference row to terminal (A3/F3).

        Frozen transitions: ``queued->running`` happens only in
        :meth:`claim_next_queued_inference_job`; ``queued->cancelled`` only
        in :meth:`cancel_inference_job`; here ``running->succeeded|failed|
        cancelled`` ONLY. A ``queued`` finish (including
        ``queued->succeeded``) raises ``ValueError`` — the runner claims
        first, so queued success can never bypass the lifecycle. The
        running->terminal write carries the outcome identities atomically
        with the state (late glue completion after stop()/cancel is never
        dropped). Already-terminal rows are returned unchanged — a
        ``cancelled`` row NEVER becomes ``succeeded``. Error vocabulary
        (exact strings): ``stale_lineage``, ``cancelled_before_start``,
        ``verification_failed``, ``interrupted``, ``execution_failed``
        (plus the terminal ``cancelled`` marker for cancellations).
        """
        if state not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("finish state must be terminal")
        timings_text = canonical_json(timings)
        updated_at = updated_at or str(timings.get("finished_at") or "")

        def body(connection: sqlite3.Connection) -> InferenceJobRecord:
            row = self._inference_job_select(connection, job_id=job_id)
            if row is None:
                raise KeyError(job_id)
            if str(row[3]) in {"succeeded", "failed", "cancelled"}:
                return self._inference_job_record(row)
            if str(row[3]) != "running":
                raise ValueError(f"cannot finish inference job from state {str(row[3])!r}")
            connection.execute(
                "UPDATE inference_jobs SET state = ?, error_code = ?, prediction_id = ?, "
                "model_run_id = ?, timings = ?, updated_at = ? WHERE job_id = ? AND state = ?",
                (
                    state,
                    error_code,
                    prediction_id,
                    model_run_id,
                    timings_text,
                    updated_at,
                    job_id,
                    str(row[3]),
                ),
            )
            updated = self._inference_job_select(connection, job_id=job_id)
            assert updated is not None
            return self._inference_job_record(updated)

        return self._in_write_transaction_typed(body)

    def cancel_inference_job(
        self, job_id: str, *, updated_at: str, series_uid: str | None = None
    ) -> InferenceJobRecord:
        """Cancel a live inference row (A3: ``queued->cancelled`` or
        ``running->cancelled`` only; terminal rows are returned unchanged)."""
        def body(connection: sqlite3.Connection) -> InferenceJobRecord:
            row = self._inference_job_select(connection, job_id=job_id)
            if row is None or (series_uid is not None and str(row[1]) != series_uid):
                raise KeyError(job_id)
            if str(row[3]) in {"queued", "running"}:
                connection.execute(
                    "UPDATE inference_jobs SET state = 'cancelled', error_code = 'cancelled', "
                    "updated_at = ? WHERE job_id = ? AND state IN ('queued','running')",
                    (updated_at, job_id),
                )
            updated = self._inference_job_select(connection, job_id=job_id)
            assert updated is not None
            return self._inference_job_record(updated)

        return self._in_write_transaction_typed(body)

    def attach_inference_result(
        self,
        job_id: str,
        *,
        prediction_id: str | None,
        model_run_id: str | None,
        timings: Mapping[str, Any] | None = None,
        updated_at: str | None = None,
    ) -> InferenceJobRecord:
        """Attach audit bindings to an ALREADY-TERMINAL inference row (A3/F3).

        Attaches ONCE: only NULL ``prediction_id``/``model_run_id`` fields
        are populated, and ``timings``/``updated_at`` are written only when
        the row's audit binding is first attached (a repeat attach with the
        fields already set is an idempotent no-op that leaves
        ``timings``/``updated_at`` untouched). The row's ``state`` is NEVER
        changed: a ``cancelled`` row stays ``cancelled`` (late glue
        completion is audit-only and NOT current-eligible per A1), a
        ``failed`` row never becomes ``succeeded``. Rows still
        ``queued``/``running`` raise ``ValueError`` — use
        :meth:`finish_inference_job` for the running->terminal transition.
        Unknown ``job_id`` raises ``KeyError``.
        """
        timings_text = None if timings is None else canonical_json(timings)

        def body(connection: sqlite3.Connection) -> InferenceJobRecord:
            row = self._inference_job_select(connection, job_id=job_id)
            if row is None:
                raise KeyError(job_id)
            if str(row[3]) not in {"succeeded", "failed", "cancelled"}:
                raise ValueError(
                    "attach_inference_result requires an already-terminal row; "
                    f"job {job_id!r} is {str(row[3])!r}"
                )
            updates: list[str] = []
            params: list[Any] = []
            had_binding = row[9] is not None or row[10] is not None
            binding_attached = False
            if prediction_id is not None and row[9] is None:
                updates.append("prediction_id = ?")
                params.append(prediction_id)
                binding_attached = True
            if model_run_id is not None and row[10] is None:
                updates.append("model_run_id = ?")
                params.append(model_run_id)
                binding_attached = True
            # Attach-once (F3/C3): timings/updated_at ride ONLY the FIRST
            # binding attach of the row. A partial first attach (prediction
            # only) followed by a later model-run attach must NOT overwrite
            # the first attach's timings/timestamp — a partial attach is
            # still "no binding before this call".
            if binding_attached and not had_binding:
                if timings_text is not None:
                    updates.append("timings = ?")
                    params.append(timings_text)
                if updated_at is not None:
                    updates.append("updated_at = ?")
                    params.append(updated_at)
            if updates:
                params.append(job_id)
                connection.execute(
                    f"UPDATE inference_jobs SET {', '.join(updates)} WHERE job_id = ?",
                    params,
                )
            updated = self._inference_job_select(connection, job_id=job_id)
            assert updated is not None
            return self._inference_job_record(updated)

        return self._in_write_transaction_typed(body)

    def finalize_inference_outcome(
        self,
        job_id: str,
        *,
        prediction_id: str | None,
        model_run_id: str | None,
        timings: Mapping[str, Any],
        updated_at: str | None = None,
        decision: str | None = None,
    ) -> InferenceJobRecord:
        """Atomically finalize a post-glue outcome against the ROW (C3/F3).

        ONE transaction, ONE authoritative read: if the row is still
        ``running`` the decision is applied — ``None``/``"succeeded"`` ->
        ``succeeded``, ``"cancelled"`` -> ``cancelled``, ``"stale_lineage"``
        -> ``failed``/``stale_lineage`` — always writing the returned ids
        atomically (a cancellation that won the DB transaction before token
        notification is preserved — never resurrected — and the audit ids
        are never dropped). If the row is already terminal, missing audit
        fields are attached once (same attach-once rules as
        :meth:`attach_inference_result`; state never changes). ``queued``
        rows raise ``ValueError``. Unknown ``job_id`` raises ``KeyError``.
        """
        if decision not in {None, "succeeded", "cancelled", "stale_lineage"}:
            raise ValueError(f"unknown finalize decision {decision!r}")
        timings_text = canonical_json(timings)
        updated_at = updated_at or str(timings.get("finished_at") or "")

        def body(connection: sqlite3.Connection) -> InferenceJobRecord:
            row = self._inference_job_select(connection, job_id=job_id)
            if row is None:
                raise KeyError(job_id)
            current_state = str(row[3])
            if current_state == "running":
                if decision in {None, "succeeded"}:
                    state, error_code = "succeeded", None
                elif decision == "cancelled":
                    state, error_code = "cancelled", "cancelled"
                else:
                    state, error_code = "failed", "stale_lineage"
                connection.execute(
                    "UPDATE inference_jobs SET state = ?, error_code = ?, "
                    "prediction_id = ?, model_run_id = ?, timings = ?, updated_at = ? "
                    "WHERE job_id = ? AND state = 'running'",
                    (
                        state,
                        error_code,
                        prediction_id,
                        model_run_id,
                        timings_text,
                        updated_at,
                        job_id,
                    ),
                )
            elif current_state in {"succeeded", "failed", "cancelled"}:
                updates: list[str] = []
                params: list[Any] = []
                had_binding = row[9] is not None or row[10] is not None
                if prediction_id is not None and row[9] is None:
                    updates.append("prediction_id = ?")
                    params.append(prediction_id)
                if model_run_id is not None and row[10] is None:
                    updates.append("model_run_id = ?")
                    params.append(model_run_id)
                if updates and not had_binding:
                    updates.append("timings = ?")
                    params.append(timings_text)
                    updates.append("updated_at = ?")
                    params.append(updated_at)
                if updates:
                    params.append(job_id)
                    connection.execute(
                        f"UPDATE inference_jobs SET {', '.join(updates)} WHERE job_id = ?",
                        params,
                    )
            else:
                raise ValueError(f"cannot finalize inference job from state {current_state!r}")
            updated = self._inference_job_select(connection, job_id=job_id)
            assert updated is not None
            return self._inference_job_record(updated)

        return self._in_write_transaction_typed(body)

    def load_current_selected_phase_prediction(
        self,
        series_uid: str,
        *,
        head_revision_id: str | None,
        selection_id: str | None,
    ) -> SelectedPhasePrediction | None:
        """Load the CURRENT-ELIGIBLE sparse artifact for one series (A1).

        A sparse artifact P is CURRENT-ELIGIBLE iff there exists an
        ``inference_jobs`` row J with ``J.state == 'succeeded'``,
        ``J.prediction_id == P.prediction_id`` (and, when ``J.model_run_id``
        is not NULL, ``J.model_run_id == P.model_run_id``), and J's bound
        request identity (``head_revision_id`` + ``selection_id`` recorded in
        ``J.request_payload``) equals the supplied current identity. Among
        eligible bindings the newest by ``(created_at, job_id)`` wins;
        anything else (cancelled/failed/interrupted/unbound artifacts and
        artifacts bound to a stale head/selection) is audit-only and yields
        ``None`` here. The unqualified
        :meth:`load_latest_selected_phase_prediction` remains for
        audit/history readers only and MUST NOT feed current
        viewer/coverage consumers. Newest eligible wins; an invalid newer
        binding is skipped, not terminating (F4).
        """
        jobs = self._require_connection().execute(
            "SELECT prediction_id, model_run_id, request_payload, created_at, job_id "
            "FROM inference_jobs WHERE series_uid = ? AND state = 'succeeded' "
            "ORDER BY created_at DESC, job_id DESC",
            (series_uid,),
        ).fetchall()
        # F4: validate EVERY candidate pair BEFORE selecting the newest
        # eligible. A newer invalid binding (missing prediction row,
        # model-run mismatch, series incoherence) is SKIPPED, never
        # terminating — an older eligible binding still resolves.
        for row in jobs:
            bound_prediction_id = None if row[0] is None else str(row[0])
            if not bound_prediction_id:
                continue
            bound_model_run_id = None if row[1] is None else str(row[1])
            try:
                bound_payload = json.loads(str(row[2]))
            except (ValueError, TypeError):
                continue
            if not isinstance(bound_payload, dict):
                continue
            if bound_payload.get("head_revision_id") != head_revision_id:
                continue
            if bound_payload.get("selection_id") != selection_id:
                continue
            artifact = self.load_selected_phase_prediction(bound_prediction_id)
            if artifact is None:
                continue
            if bound_model_run_id and artifact.model_run_id != bound_model_run_id:
                continue
            if artifact.series_uid != series_uid:
                continue
            return artifact
        return None

    def save_selected_phase_analysis_checked(
        self,
        measurement: Measurement,
        *,
        expected_head_revision_id: str | None,
        expected_selection_id: str | None,
        expected_inclusion_set_id: str | None,
        expected_n_slices: int,
        expected_job_id: str | None = None,
        job_timings: Mapping[str, Any] | None = None,
        job_updated_at: str | None = None,
    ) -> str:
        """Publish against lineage, optionally completing the worker job atomically.

        When ``expected_job_id`` is supplied, the job must still be ``running``
        at the start of this ``BEGIN IMMEDIATE`` transaction.  The measurement
        insert, the coherent analysis binding insert and
        ``running -> succeeded`` transition then commit together; cancellation
        and publication are therefore mutually exclusive.

        F14 (CMR_FUNCTION_V1_ANALYSIS_001): the publication goes through the
        SAME coherent analysis authority the explicit inclusion refresh uses —
        the measurement is bound to the current head, the latest selection and
        the latest applicable inclusion set, and the head MUST carry the QC
        report derived from that exact state. A state without that QC report
        raises :class:`AnalysisAuthorityError`, so the job publishes nine
        metrics AND their QC or NOTHING (the whole transaction, including the
        job transition, rolls back).
        """
        measurement_payload = self._content_addressed_payload(
            measurement, measurement.measurement_id
        )
        job_timings_text = (
            None if job_timings is None else canonical_json(job_timings)
        )
        job_updated_at = job_updated_at or measurement.created_at

        def _assert_publishable_job(connection: sqlite3.Connection) -> None:
            if expected_job_id is None:
                return
            row = connection.execute(
                "SELECT series_uid, state FROM analysis_jobs WHERE job_id = ?",
                (expected_job_id,),
            ).fetchone()
            if row is None or str(row[0]) != measurement.series_uid:
                raise JobNotPublishableError(
                    f"analysis job {expected_job_id!r} is missing or belongs to "
                    f"another series; zero rows written"
                )
            if str(row[1]) != "running":
                raise JobNotPublishableError(
                    f"analysis job {expected_job_id!r} is {row[1]!r}, not running; "
                    "zero rows written"
                )


        def body(connection: sqlite3.Connection) -> str:
            _assert_publishable_job(connection)
            actual = self._resolve_current_nullable_lineage(
                connection,
                series_uid=measurement.series_uid,
                expected_n_slices=expected_n_slices,
            )
            self._assert_checked_lineage(
                actual=actual,
                expected_head_revision_id=expected_head_revision_id,
                expected_selection_id=expected_selection_id,
                expected_inclusion_set_id=expected_inclusion_set_id,
            )
            if measurement.source_revision_id != expected_head_revision_id:
                raise SeriesCoherenceError(
                    "selected-phase measurement must bind the expected current head"
                )
            if measurement.selection_id != expected_selection_id:
                raise SeriesCoherenceError(
                    "selected-phase measurement must bind the expected selection"
                )
            if measurement.inclusion_set_id != expected_inclusion_set_id:
                raise SeriesCoherenceError(
                    "selected-phase measurement must bind the expected inclusion set"
                )
            self._verify_inclusion_binding(
                connection,
                series_uid=measurement.series_uid,
                head_revision_id=expected_head_revision_id or "",
                workflow_id=measurement.provenance.workflow_id,
                selection_id=expected_selection_id,
                bound_inclusion_set_id=measurement.inclusion_set_id,
                allow_null_set=True,
                gaps_length=len(measurement.stack_gaps_mm),
            )
            connection.execute(
                "INSERT INTO measurements (measurement_id, series_uid, source_revision_id, "
                "created_at, payload, workflow_id, selection_id, inclusion_set_id, "
                "measurement_sequence, range_table_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    measurement.measurement_id,
                    measurement.series_uid,
                    measurement.source_revision_id,
                    measurement.created_at,
                    measurement_payload,
                    measurement.provenance.workflow_id,
                    measurement.selection_id,
                    measurement.inclusion_set_id,
                    _next_sequence(
                        connection, "measurements", "measurement_sequence", measurement.series_uid
                    ),
                    _range_table_version_of_payload(measurement_payload),
                ),
            )
            # F14: the job publication is the ONE selected-phase analysis
            # authority for the current state — metrics + QC + exact inputs
            # commit together or nothing does. A head without the matching QC
            # report raises AnalysisAuthorityError here, rolling the whole
            # transaction (measurement AND job transition) back.
            self._insert_analysis_binding(
                connection, measurement=measurement, require_qc=True
            )
            if expected_job_id is not None:
                if job_timings_text is None:
                    raise ValueError("job_timings is required for atomic job publication")
                connection.execute(
                    "UPDATE analysis_jobs SET state = 'succeeded', error_code = NULL, "
                    "artifact_measurement_id = ?, timings = ?, updated_at = ? "
                    "WHERE job_id = ? AND state = 'running'",
                    (
                        measurement.measurement_id,
                        job_timings_text,
                        job_updated_at,
                        expected_job_id,
                    ),
                )
                if connection.execute("SELECT changes()").fetchone()[0] != 1:
                    raise JobNotPublishableError(
                        f"analysis job {expected_job_id!r} stopped before publication; "
                        "zero rows written"
                    )
            return measurement.measurement_id

        return self._in_write_transaction(body)

    # --- loads ------------------------------------------------------------------

    def _verify_and_reconstruct(
        self,
        stored_id: str,
        payload_text: str,
        reconstructor: Callable[[dict[str, Any]], T],
    ) -> T:
        if _sha256_hex(payload_text) != stored_id:
            raise PayloadIntegrityError(
                f"stored payload digest does not match stored id {stored_id!r}"
            )
        try:
            data = json.loads(payload_text)
        except ValueError as exc:
            raise PayloadIntegrityError(
                f"stored payload for {stored_id!r} is not valid JSON"
            ) from exc
        obj = reconstructor(data)
        # Byte-for-byte canonical verification: values passed through verbatim
        # must reserialize to the exact stored TEXT.
        if canonical_json(obj) != payload_text:
            raise PayloadIntegrityError(
                f"reconstructed payload for {stored_id!r} is not byte-canonical"
            )
        return obj

    def load_chain(self, series_uid: str) -> list[ContourRevision]:
        """Return the ordered revision chain (ORDER BY version) or [] if absent."""

        rows = (
            self._require_connection()
            .execute(
                "SELECT revision_id, payload FROM contour_revisions WHERE series_uid = ?"
                " ORDER BY version ASC",
                (series_uid,),
            )
            .fetchall()
        )
        chain: list[ContourRevision] = []
        for stored_id, payload_text in rows:
            reconstructed = self._verify_and_reconstruct(
                stored_id, payload_text, contour_revision_from_payload
            )
            chain.append(reconstructed)
        return chain

    def load_latest_measurement(self, series_uid: str) -> Measurement | None:
        """Latest by STORE INSERTION SEQUENCE (S2C-001), else ``None``.

        "Latest" is the highest store-assigned per-series sequence — actual
        insertion order, never a caller-controlled timestamp. Migrated v1/v2
        rows were backfilled from original ``rowid`` insertion order, so
        legacy histories keep their relative order. CFM-001: the additive
        ``range_table_version`` column is verified against the payload
        field before the record is returned.
        """

        rows = (
            self._require_connection()
            .execute(
                "SELECT measurement_id, payload, range_table_version FROM measurements"
                " WHERE series_uid = ?"
                " ORDER BY measurement_sequence DESC LIMIT 1",
                (series_uid,),
            )
            .fetchall()
        )
        if not rows:
            return None
        reconstructed = self._verify_and_reconstruct(
            rows[0][0], rows[0][1], measurement_from_payload
        )
        _assert_range_column_matches_payload(rows[0][2], reconstructed)
        return reconstructed

    def load_latest_selection(self, series_uid: str) -> EDESSelection | None:
        """Latest by STORE INSERTION SEQUENCE (S2C-001), else ``None``."""

        rows = (
            self._require_connection()
            .execute(
                "SELECT selection_id, payload FROM edes_selections WHERE series_uid = ?"
                " ORDER BY selection_sequence DESC LIMIT 1",
                (series_uid,),
            )
            .fetchall()
        )
        if not rows:
            return None
        return self._verify_and_reconstruct_selection(rows[0][0], rows[0][1])

    def load_measurement_for_state(
        self, series_uid: str, source_revision_id: str, selection_id: str
    ) -> Measurement | None:
        """The measurement binding exactly ``(revision, selection)``, else None.

        S2C-001 state lookup: reopen/refresh flows need THE coherent current
        measurement for a head revision + latest selection pair, not merely
        any newest row.
        """
        rows = (
            self._require_connection()
            .execute(
                "SELECT m.measurement_id, m.payload, m.range_table_version"
                " FROM measurements AS m"
                " WHERE m.series_uid = ? AND m.source_revision_id = ?"
                "   AND json_extract(m.payload, '$.selection_id') = ?"
                " ORDER BY m.measurement_sequence DESC LIMIT 1",
                (series_uid, source_revision_id, selection_id),
            )
            .fetchall()
        )
        if not rows:
            return None
        reconstructed = self._verify_and_reconstruct(
            rows[0][0], rows[0][1], measurement_from_payload
        )
        _assert_range_column_matches_payload(rows[0][2], reconstructed)
        return reconstructed

    def _verify_and_reconstruct_selection(self, stored_id: str, payload_text: str) -> EDESSelection:
        if _sha256_hex(payload_text) != stored_id:
            raise PayloadIntegrityError(
                f"stored selection payload digest does not match stored id {stored_id!r}"
            )
        try:
            data = json.loads(payload_text)
        except ValueError as exc:
            raise PayloadIntegrityError(
                f"stored selection payload {stored_id!r} is not valid JSON"
            ) from exc
        selection = edes_selection_from_payload(data)
        if edes_selection_payload_text(selection) != payload_text:
            raise PayloadIntegrityError(
                f"reconstructed selection payload {stored_id!r} is not byte-canonical"
            )
        # Cross-consistency against the imported Node 1 serializer (amendment A).
        if canonical_json(edes_selection_payload(selection)) != payload_text:
            raise PayloadIntegrityError(
                f"selection projection for {stored_id!r} diverges from canonical_json"
            )
        return selection

    def load_analysis_binding(self, analysis_id: str) -> SelectedPhaseAnalysisRecord | None:
        """Reconstruct ONE analysis binding by content id (digest-verified)."""
        row = (
            self._require_connection()
            .execute(
                "SELECT analysis_id, payload FROM selected_phase_analyses"
                " WHERE analysis_id = ?",
                (analysis_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return self._verify_and_reconstruct(
            str(row[0]), str(row[1]), analysis_record_from_payload
        )

    def load_analyses(self, series_uid: str) -> list[SelectedPhaseAnalysisRecord]:
        """All analysis bindings of a series in store-sequence order."""
        rows = (
            self._require_connection()
            .execute(
                "SELECT analysis_id, payload FROM selected_phase_analyses"
                " WHERE series_uid = ? ORDER BY analysis_sequence ASC",
                (series_uid,),
            )
            .fetchall()
        )
        return [
            self._verify_and_reconstruct(str(row[0]), str(row[1]), analysis_record_from_payload)
            for row in rows
        ]

    def _resolve_current_analysis(
        self,
        connection: sqlite3.Connection,
        *,
        series_uid: str,
        expected_n_slices: int | None,
    ) -> tuple[SelectedPhaseAnalysisRecord | None, Measurement | None, str | None]:
        """The ONE current coherent analysis for a series, inside a transaction.

        CMR_FUNCTION_V1_ANALYSIS_001: "current" means EVERY bound input still
        matches the persisted state, resolved in THIS transaction:

        * the analysis head is the CURRENT chain head (a contour/head change
          makes the previous analysis non-current);
        * its selection is the LATEST persisted selection (an ED/ES change
          makes the previous analysis non-current);
        * its inclusion set is the LATEST APPLICABLE set for that head and
          selection (an inclusion change makes the previous analysis
          non-current);
        * the measurement it names is bound to exactly that head/selection/
          inclusion and re-verifies digest-clean;
        * the persisted head's QC payload digest equals the analysis'
          ``qc_report_digest`` (a metric record can never pair with a QC report
          from a different state).

        Returns ``(record, measurement, None)`` for the newest matching
        analysis row, else ``(None, None, reason)``. Reasons are stable
        machine-readable tokens (``no_head``, ``no_selection``,
        ``head_qc_missing``, ``no_analysis``, ``not_current``).
        """
        head_row = self._current_head_row(connection, series_uid)
        if head_row is None:
            return None, None, "no_head"
        head_id = str(head_row[0])
        head_workflow = None if head_row[3] is None else str(head_row[3])
        selection_row = connection.execute(
            "SELECT selection_id, payload FROM edes_selections WHERE series_uid = ?"
            " ORDER BY selection_sequence DESC LIMIT 1",
            (series_uid,),
        ).fetchone()
        if selection_row is None:
            return None, None, "no_selection"
        selection_id = str(selection_row[0])
        selection = self._verify_and_reconstruct_selection(
            selection_id, str(selection_row[1])
        )
        resolved = self._resolve_latest_applicable_set(
            connection,
            series_uid=series_uid,
            head_revision_id=head_id,
            selection_id=selection_id,
            workflow_id=head_workflow,
            expected_n_slices=expected_n_slices,
        )
        applicable_id = None if resolved is None else resolved[0].inclusion_set_id
        rows = connection.execute(
            "SELECT analysis_id, payload FROM selected_phase_analyses"
            " WHERE series_uid = ? ORDER BY analysis_sequence DESC",
            (series_uid,),
        ).fetchall()
        if not rows:
            return None, None, "no_analysis"
        head_payload = self._verify_persisted_head_payload(head_row)
        head_qc = qc_report_from_payload(head_payload.get("qc_report"))
        if head_qc is None:
            return None, None, "head_qc_missing"
        head_qc_digest = canonical_digest(head_qc)
        for stored_id, payload_text in rows:
            record = self._verify_and_reconstruct(
                str(stored_id), str(payload_text), analysis_record_from_payload
            )
            if record.head_revision_id != head_id or record.selection_id != selection_id:
                continue
            if (record.inclusion_set_id or None) != (applicable_id or None):
                continue
            if (
                record.qc_report_digest != head_qc_digest
                or record.qc_verdict != head_qc.verdict
            ):
                raise PayloadIntegrityError(
                    f"analysis {record.analysis_id!r} binds QC digest "
                    f"{record.qc_report_digest!r} but the persisted head "
                    f"{head_id!r} carries {head_qc_digest!r}; the two authorities "
                    "disagree and neither can be returned"
                )
            measurement_row = connection.execute(
                "SELECT payload, range_table_version FROM measurements"
                " WHERE measurement_id = ?",
                (record.measurement_id,),
            ).fetchone()
            if measurement_row is None:
                raise PayloadIntegrityError(
                    f"analysis {record.analysis_id!r} names measurement "
                    f"{record.measurement_id!r} which is not persisted"
                )
            measurement = self._verify_and_reconstruct(
                record.measurement_id,
                str(measurement_row[0]),
                measurement_from_payload,
            )
            _assert_range_column_matches_payload(measurement_row[1], measurement)
            if (
                measurement.source_revision_id != head_id
                or (measurement.selection_id or None) != selection_id
                or (measurement.inclusion_set_id or None)
                != (record.inclusion_set_id or None)
                or measurement.measurement_id != record.measurement_id
                or selection.selection_id != selection_id
            ):
                raise PayloadIntegrityError(
                    f"analysis {record.analysis_id!r} does not match the "
                    "measurement state it names; refusing to return a mixed "
                    "metric/QC authority"
                )
            return record, measurement, None
        return None, None, "not_current"

    def load_current_analysis(
        self, series_uid: str, *, expected_n_slices: int | None = None
    ) -> tuple[SelectedPhaseAnalysisRecord | None, Measurement | None, str | None]:
        """Read-only resolution of the current coherent analysis (see internal)."""
        return self._resolve_current_analysis(
            self._require_connection(),
            series_uid=series_uid,
            expected_n_slices=expected_n_slices,
        )

    def load_reopen_snapshot(
        self, series_uid: str, *, expected_n_slices: int | None = None
    ) -> dict[str, Any]:
        """Atomically snapshot the full reopen projection (C1, R1 P1a).

        Inside ONE ``BEGIN IMMEDIATE`` read transaction on this connection,
        resolves: chain revision ids (ordered), latest selection, current
        coherent analysis + measurement + stale reason. Because SQLite
        defers a concurrent writer until this read transaction commits, no
        refresh commit can interleave BETWEEN the chain read and the
        analysis resolution — the projection can never mix a stale chain
        head with a newer analysis (or vice versa). Read-only: the
        transaction is rolled back before return; callers must treat the
        snapshot as point-in-time (re-verify at the seal boundary, which
        already re-resolves inside its write transaction).
        """
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            chain_rows = connection.execute(
                "SELECT revision_id FROM contour_revisions WHERE series_uid = ?"
                " ORDER BY version ASC",
                (series_uid,),
            ).fetchall()
            chain_ids = [str(row[0]) for row in chain_rows]
            analysis, measurement, reason = self._resolve_current_analysis(
                connection,
                series_uid=series_uid,
                expected_n_slices=expected_n_slices,
            )
            selection_row = connection.execute(
                "SELECT selection_id FROM edes_selections WHERE series_uid = ?"
                " ORDER BY selection_sequence DESC LIMIT 1",
                (series_uid,),
            ).fetchone()
            selection_id = None if selection_row is None else str(selection_row[0])
            # C2 (R2 P1): the applicable inclusion set is resolved INSIDE the
            # same read transaction, so the whole authority tuple (chain +
            # selection + inclusion + analysis) is one atomic snapshot.
            applicable_id: str | None = None
            if chain_ids and selection_id is not None:
                head_workflow = connection.execute(
                    "SELECT workflow_id FROM contour_revisions"
                    " WHERE series_uid = ? ORDER BY version DESC LIMIT 1",
                    (series_uid,),
                ).fetchone()
                resolved = self._resolve_latest_applicable_set(
                    connection,
                    series_uid=series_uid,
                    head_revision_id=chain_ids[-1],
                    selection_id=selection_id,
                    workflow_id=(
                        None if head_workflow[0] is None else str(head_workflow[0])
                    ),
                    expected_n_slices=expected_n_slices,
                )
                applicable_id = None if resolved is None else resolved[0].inclusion_set_id
            result: dict[str, Any] = {
                "chain_ids": chain_ids,
                "selection_id": selection_id,
                "applicable_inclusion_set_id": applicable_id,
                "analysis": analysis,
                "measurement": measurement,
                "analysis_stale_reason": reason,
            }
        finally:
            self._rollback_quietly(connection)
        return result

    def load_approvals(self, series_uid: str) -> list[ApprovalSnapshot]:
        """All sealed approvals for a series, deterministically ordered."""

        rows = (
            self._require_connection()
            .execute(
                "SELECT approval_id, payload FROM approval_snapshots WHERE series_uid = ?"
                " ORDER BY approved_at ASC, approval_id ASC",
                (series_uid,),
            )
            .fetchall()
        )
        approvals: list[ApprovalSnapshot] = []
        for stored_id, payload_text in rows:
            reconstructed = self._verify_and_reconstruct(
                stored_id, payload_text, approval_snapshot_from_payload
            )
            approvals.append(reconstructed)
        return approvals

    def load_series_workflow_ids(self, series_uid: str) -> list[str | None]:
        """Workflow column of EVERY persisted row of the series' tables.

        Correction round 2 R1: the controller's workflow coherence scan must
        cover ALL persisted selections and measurements (and revisions,
        approvals and decisions) — not merely the latest of each — so an
        A->B->A history or a bound row sitting beside legacy-unbound rows
        fails closed at reopen. ``None`` entries (legacy unbound rows) are
        preserved in the result; the CALLER decides the bound/unbound mixing
        verdict.
        """
        rows = (
            self._require_connection()
            .execute(
                "SELECT workflow_id FROM contour_revisions WHERE series_uid = ?"
                " UNION ALL"
                " SELECT workflow_id FROM edes_selections WHERE series_uid = ?"
                " UNION ALL"
                " SELECT workflow_id FROM measurements WHERE series_uid = ?"
                " UNION ALL"
                " SELECT workflow_id FROM approval_snapshots WHERE series_uid = ?"
                " UNION ALL"
                " SELECT workflow_id FROM qc_decision_records WHERE series_uid = ?"
                " UNION ALL"
                " SELECT workflow_id FROM selected_phase_analyses WHERE series_uid = ?",
                (series_uid,) * 6,
            )
            .fetchall()
        )
        return [None if row[0] is None else str(row[0]) for row in rows]

    def load_qc_decisions(self, series_uid: str) -> list[QCDecisionRecord]:
        """All persisted QC decision records for a series, ordered.

        Deterministic order: ``approved_at`` then ``decision_id`` (the same
        tie-broken ordering convention as :meth:`load_approvals`). Every row
        is digest-verified and byte-canonical re-verified before it is
        returned; any drift raises :class:`PayloadIntegrityError` instead of
        returning partial data.
        """

        rows = (
            self._require_connection()
            .execute(
                "SELECT decision_id, payload FROM qc_decision_records"
                " WHERE series_uid = ? ORDER BY approved_at ASC, decision_id ASC",
                (series_uid,),
            )
            .fetchall()
        )
        decisions: list[QCDecisionRecord] = []
        for stored_id, payload_text in rows:
            reconstructed = self._verify_and_reconstruct(
                stored_id, payload_text, qc_decision_from_payload
            )
            decisions.append(reconstructed)
        return decisions


def _assert_parent_links_share_series(
    connection: sqlite3.Connection, revision: ContourRevision
) -> None:
    """Reject cross-series parent links before the revision INSERT (DEFECT-1).

    Must run INSIDE the caller's ``BEGIN IMMEDIATE`` transaction and BEFORE
    any row is staged, so a rejection leaves zero staged rows. Every
    referenced parent present in the database is resolved to its stored
    series and compared against the new revision's ``series_uid``; a mismatch
    raises :class:`SeriesCoherenceError`. Parents that are absent are NOT
    resolved here — they fail closed on the foreign-key constraint at INSERT
    time, which preserves the established orphan-insert semantics.

    ``predictions`` carries no denormalized series column by contract, so its
    series is read from the stored canonical payload TEXT. A stored payload
    without a readable ``series_uid`` cannot be produced by this store (the
    digest identity is asserted on write), so it fails closed with
    :class:`PayloadIntegrityError`.

    S2C-001 correction F2: the workflow identities of every RESOLVED parent
    (prediction payload provenance and parent-revision workflow column) must
    agree with the child revision's provenance workflow under the
    bound/unbound coherence rule — mixing bound with unbound lineage members,
    or two different workflows, fails closed before any row is staged.
    """

    parent_prediction_workflow: str | None = None
    parent_prediction_present = False
    if revision.parent_prediction_id is not None:
        row = connection.execute(
            "SELECT payload FROM predictions WHERE prediction_id = ?",
            (revision.parent_prediction_id,),
        ).fetchone()
        if row is not None:
            parent_prediction_present = True
            try:
                parent_data = json.loads(row[0])
                parent_series = str(parent_data["series_uid"])
            except (ValueError, KeyError, TypeError) as exc:
                raise PayloadIntegrityError(
                    "stored prediction payload for "
                    f"{revision.parent_prediction_id!r} has no readable series_uid"
                ) from exc
            if parent_series != revision.series_uid:
                raise SeriesCoherenceError(
                    "revision series_uid "
                    f"{revision.series_uid!r} does not match parent prediction "
                    f"series_uid {parent_series!r}"
                )
            parent_prediction_workflow = _workflow_id_of_provenance(parent_data.get("provenance"))
    parent_revision_workflow: str | None = None
    parent_revision_present = False
    if revision.parent_revision_id is not None:
        row = connection.execute(
            "SELECT series_uid, workflow_id FROM contour_revisions WHERE revision_id = ?",
            (revision.parent_revision_id,),
        ).fetchone()
        if row is not None:
            parent_revision_present = True
            if str(row[0]) != revision.series_uid:
                raise SeriesCoherenceError(
                    "revision series_uid "
                    f"{revision.series_uid!r} does not match parent revision "
                    f"series_uid {str(row[0])!r}"
                )
            parent_revision_workflow = None if row[1] is None else str(row[1])
    # Correction F2: only lineage members that ACTUALLY exist participate in
    # the workflow agreement; an absent parent slot is not an "unbound"
    # participant.
    workflow_participants: dict[str, str | None] = {
        "child": revision.provenance.workflow_id,
    }
    if parent_prediction_present:
        workflow_participants["parent_prediction"] = parent_prediction_workflow
    if parent_revision_present:
        workflow_participants["parent_revision"] = parent_revision_workflow
    _assert_bound_workflow_agreement(**workflow_participants)


def _insert_revision_row(
    connection: sqlite3.Connection, revision: ContourRevision, payload_text: str
) -> None:
    connection.execute(
        "INSERT INTO contour_revisions (revision_id, series_uid, version,"
        " parent_prediction_id, parent_revision_id, payload, workflow_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            revision.revision_id,
            revision.series_uid,
            revision.version,
            revision.parent_prediction_id,
            revision.parent_revision_id,
            payload_text,
            revision.provenance.workflow_id,
        ),
    )


def _insert_slice_inclusion_set_row(
    connection: sqlite3.Connection,
    inclusion_set: SliceInclusionSet,
    payload_text: str,
    expected_n_slices: int | None = None,
) -> int:
    """Coherence-guard and INSERT one inclusion-set row inside the caller's
    transaction; returns the assigned ``inclusion_sequence``.

    Shared by :meth:`SqliteCmrStore.save_slice_inclusion_set` and the CFM-001
    compound checked analysis save. Guards: referenced selection exists in
    the set's series, anchored revision exists in the same series, FOUR-WAY
    workflow agreement (set column / set provenance / selection row /
    revision row), explicit per-entry ``frame_index`` matching the referenced
    selection's resolved frames, persisted manifest for bound workflows.
    """
    selection_row = connection.execute(
        "SELECT series_uid, workflow_id, payload FROM edes_selections WHERE selection_id = ?",
        (inclusion_set.selection_id,),
    ).fetchone()
    if selection_row is None:
        raise SeriesCoherenceError(
            f"referenced selection {inclusion_set.selection_id!r} is not present"
        )
    if str(selection_row[0]) != inclusion_set.series_uid:
        raise SeriesCoherenceError(
            "inclusion set series_uid "
            f"{inclusion_set.series_uid!r} does not match referenced "
            f"selection series {str(selection_row[0])!r}"
        )
    revision_row = connection.execute(
        "SELECT series_uid, workflow_id FROM contour_revisions WHERE revision_id = ?",
        (inclusion_set.source_revision_id,),
    ).fetchone()
    if revision_row is None:
        raise SeriesCoherenceError(
            f"source revision {inclusion_set.source_revision_id!r} is not present"
        )
    if str(revision_row[0]) != inclusion_set.series_uid:
        raise SeriesCoherenceError(
            "inclusion set series_uid "
            f"{inclusion_set.series_uid!r} does not match source revision "
            f"series {str(revision_row[0])!r}"
        )
    _assert_bound_workflow_agreement(
        set_column=inclusion_set.workflow_id,
        set_provenance=inclusion_set.provenance.workflow_id,
        selection=(None if selection_row[1] is None else str(selection_row[1])),
        revision=(None if revision_row[1] is None else str(revision_row[1])),
    )
    try:
        selection_payload = dict(json.loads(str(selection_row[2])))
        selection_ed_frame = int(selection_payload["ed_frame"])
        selection_es_frame = int(selection_payload["es_frame"])
    except (ValueError, KeyError, TypeError) as exc:
        raise PayloadIntegrityError(
            f"stored selection {inclusion_set.selection_id!r} payload has no readable ED/ES frames"
        ) from exc
    for entry in inclusion_set.entries:
        # Correction round 2 R3: NEW saves REQUIRE an explicit
        # frame_index on every entry (the None shape exists only for
        # spec-era fixtures; it is no longer acceptable at save time).
        if entry.frame_index is None:
            raise SeriesCoherenceError(
                "every inclusion entry must carry an explicit "
                "frame_index naming the ED/ES frame the decision "
                "speaks about; got None on "
                f"(slice {entry.slice_index}, role "
                f"{entry.frame_role.value}, structure "
                f"{entry.structure.value})"
            )
        expected_frame = (
            selection_ed_frame
            if entry.frame_role is SliceInclusionFrameRole.ED
            else selection_es_frame
        )
        if entry.frame_index != expected_frame:
            raise SeriesCoherenceError(
                f"inclusion entry frame_index {entry.frame_index} does "
                f"not match the referenced selection's "
                f"{entry.frame_role.value} frame {expected_frame}"
            )
    # CFM-001 correction F7: EVERY NEW write must carry the exact complete
    # 5xN grid over applicable groups only. Malformed PRE-v4 rows survive as
    # read-only history (they are never re-inserted), but this write path is
    # the boundary for all future records.
    try:
        parsed_payload = dict(json.loads(payload_text))
        covered_slices = validate_complete_grid_entries(parsed_payload["entries"])
    except ValueError as exc:
        raise SeriesCoherenceError(
            f"inclusion set {inclusion_set.inclusion_set_id!r} does not carry "
            f"a complete applicable entry grid: {exc}"
        ) from exc
    except (KeyError, TypeError) as exc:
        raise PayloadIntegrityError(
            f"inclusion set payload for {inclusion_set.inclusion_set_id!r} is unreadable"
        ) from exc
    if expected_n_slices is not None and covered_slices != expected_n_slices:
        raise SeriesCoherenceError(
            f"inclusion set {inclusion_set.inclusion_set_id!r} covers "
            f"{covered_slices} slices but the cine stack has "
            f"{expected_n_slices}; truncated-tail grids are refused at write"
        )
    if inclusion_set.workflow_id is not None:
        manifest_row = connection.execute(
            "SELECT 1 FROM dicom_source_manifests WHERE workflow_id = ?",
            (inclusion_set.workflow_id,),
        ).fetchone()
        if manifest_row is None:
            raise SeriesCoherenceError(
                f"workflow {inclusion_set.workflow_id!r} has no persisted source manifest"
            )
    sequence = _next_inclusion_sequence(
        connection, inclusion_set.series_uid, inclusion_set.workflow_id
    )
    connection.execute(
        "INSERT INTO slice_inclusion_sets (inclusion_set_id, workflow_id,"
        " series_uid, source_revision_id, selection_id, selected_at,"
        " inclusion_sequence, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            inclusion_set.inclusion_set_id,
            inclusion_set.workflow_id,
            inclusion_set.series_uid,
            inclusion_set.source_revision_id,
            inclusion_set.selection_id,
            inclusion_set.selected_at,
            sequence,
            payload_text,
        ),
    )
    return sequence


def _assert_model_run_columns_match_payload(
    stored: ModelRunRow, reconstructed: ModelRunRecord
) -> None:
    """Fail closed unless every denormalized ``model_runs`` column equals the
    canonical payload field (defense in depth against drifted rows)."""
    expected = (
        reconstructed.model_name,
        reconstructed.model_version,
        reconstructed.model_sha256,
        reconstructed.framework,
        reconstructed.study_uid,
        reconstructed.series_uid,
        reconstructed.input_content_hash,
        reconstructed.mask_artifact_sha256,
        reconstructed.uq_method,
        reconstructed.record_digest,
        reconstructed.created_at,
    )
    actual = (
        stored.model_name,
        stored.model_version,
        stored.model_sha256,
        stored.framework,
        stored.study_uid,
        stored.series_uid,
        stored.input_content_hash,
        stored.mask_artifact_sha256,
        stored.uq_method,
        stored.record_digest,
        stored.created_at,
    )
    if actual != expected:
        raise PayloadIntegrityError(
            f"model_runs denormalized columns {actual!r} diverge from payload "
            f"fields {expected!r} for model run {stored.model_run_id!r}"
        )


def _assert_whole_cine_prediction_columns_match_payload(
    stored: WholeCinePredictionRow, reconstructed: WholeCinePrediction
) -> None:
    """Fail closed unless denormalized prediction columns equal payload fields."""
    expected = (
        reconstructed.study_uid,
        reconstructed.series_uid,
        reconstructed.source,
        reconstructed.model_run_id,
        reconstructed.slice_count,
        reconstructed.frame_count,
        reconstructed.created_at,
    )
    actual = (
        stored.study_uid,
        stored.series_uid,
        stored.source,
        stored.model_run_id,
        stored.slice_count,
        stored.frame_count,
        stored.created_at,
    )
    if actual != expected:
        raise PayloadIntegrityError(
            f"whole_cine_predictions denormalized columns {actual!r} diverge "
            f"from payload fields {expected!r} for prediction "
            f"{stored.prediction_id!r}"
        )


def _assert_selected_phase_prediction_columns_match_payload(
    stored: SelectedPhasePredictionRow, reconstructed: SelectedPhasePrediction
) -> None:
    """Fail closed unless denormalized sparse-prediction columns equal payload.

    Includes the FNC-C identity-significant ``source_frame_indices``: the
    column TEXT must equal the canonical JSON projection of the payload's own
    tuple, never a parallel value. A tampered coverage column therefore fails
    closed instead of silently changing the artifact's declared phases.
    """
    expected = (
        reconstructed.study_uid,
        reconstructed.series_uid,
        reconstructed.source,
        reconstructed.model_run_id,
        reconstructed.slice_count,
        reconstructed.frame_count,
        selected_phase_frame_indices_json(reconstructed.source_frame_indices),
        reconstructed.created_at,
    )
    actual = (
        stored.study_uid,
        stored.series_uid,
        stored.source,
        stored.model_run_id,
        stored.slice_count,
        stored.frame_count,
        stored.source_frame_indices,
        stored.created_at,
    )
    if actual != expected:
        raise PayloadIntegrityError(
            f"selected_phase_predictions denormalized columns {actual!r} diverge "
            f"from payload fields {expected!r} for prediction "
            f"{stored.prediction_id!r}"
        )

def _assert_whole_cine_revision_columns_match_payload(
    stored: WholeCineRevisionRow, reconstructed: WholeCineRevision
) -> None:
    """Fail closed unless denormalized revision columns equal payload fields.

    Includes the CFM-001-style column/payload equality rule for the additive
    ``edited_keys_json`` store metadata: it must equal the canonical JSON array
    derived from the record's own ``edited_keys``, never a parallel value.
    """
    expected = (
        reconstructed.study_uid,
        reconstructed.series_uid,
        reconstructed.version,
        reconstructed.parent_prediction_id,
        reconstructed.parent_revision_id,
        reconstructed.editor,
        whole_cine_edited_keys_json(reconstructed),
        reconstructed.created_at,
    )
    actual = (
        stored.study_uid,
        stored.series_uid,
        stored.version,
        stored.parent_prediction_id,
        stored.parent_revision_id,
        stored.editor,
        stored.edited_keys_json,
        stored.created_at,
    )
    if actual != expected:
        raise PayloadIntegrityError(
            f"whole_cine_revisions denormalized columns {actual!r} diverge from "
            f"payload fields {expected!r} for revision {stored.revision_id!r}"
        )


__all__ = [
    "ANALYSIS_JOB_KINDS",
    "ANALYSIS_JOB_STATES",
    "AnalysisAuthorityError",
    "AnalysisJobRecord",
    "CINE_WORKFLOW_REGISTRATION_RECORD_SCHEMA",
    "CineWorkflowRegistrationV1",
    "SCHEMA_USER_VERSION",
    "SqliteCmrStore",
    "SeriesCoherenceError",
    "StaleHeadError",
    "JobNotPublishableError",
    "PayloadIntegrityError",
    "TABLES",
]
