"""cmr_persistence — Node 2 append-only embedded SQLite persistence.

Single-engine ``sqlite3`` store for the Node 1 clinical records plus payload
reconstruction helpers. Re-exports are resolved lazily (PEP 562) so importing
the package stays cheap and import-order safe.
"""

from __future__ import annotations

from typing import Any

_STORE_NAMES = (
    "AnalysisJobRecord",
    "CINE_WORKFLOW_REGISTRATION_RECORD_SCHEMA",
    "CineWorkflowRegistrationV1",
    "PayloadIntegrityError",
    "SCHEMA_USER_VERSION",
    "InferenceJobRecord",
    "StaleHeadError",
    "SeriesCoherenceError",
    "SqliteCmrStore",
    "TABLES",
)

_RECONSTRUCT_NAMES = (
    "approval_snapshot_from_payload",
    "contour_revision_from_payload",
    "edes_selection_from_payload",
    "edes_selection_payload",
    "edes_selection_payload_text",
    "measurement_from_payload",
    "model_run_from_payload",
    "prediction_from_payload",
    "selection_id_from_payload_text",
    "whole_cine_edited_keys_json",
    "whole_cine_prediction_from_payload",
    "whole_cine_revision_from_payload",
)


def __getattr__(name: str) -> Any:
    if name in _STORE_NAMES:
        from cmr_persistence import store

        return getattr(store, name)
    if name in _RECONSTRUCT_NAMES:
        from cmr_persistence import reconstruct

        return getattr(reconstruct, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_STORE_NAMES) | set(_RECONSTRUCT_NAMES))


__all__ = [*_STORE_NAMES, *_RECONSTRUCT_NAMES]
