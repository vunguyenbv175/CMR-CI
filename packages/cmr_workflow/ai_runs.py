"""Thin workflow-to-local-AI delegation seam for M5.

The workflow controller remains unchanged.  Callers provide the controller,
its explicit integration dependencies, and a bundle id; this module reopens
the accepted lineage and delegates to the provider integration service.  It
contains no clinical calculations and is synchronous library code.  A future
UI caller is responsible for invoking it away from the UI thread.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cmr_core.cine import CineVolume
from cmr_persistence.store import SqliteCmrStore
from inference.backends.budget_instrumentation import BudgetPolicy
from inference.backends.contract import CancelToken
from inference.providers.candidate.backend import NativeInferer
from inference.providers.integration import (
    CandidateBackendFactory,
    LocalInferenceOutcome,
    run_local_whole_cine_inference,
)
from inference.registry.service import RegistryService


@dataclass(frozen=True)
class AiRunDependencies:
    """Explicit non-clinical dependencies needed by the delegation seam."""

    store: SqliteCmrStore
    registry: RegistryService
    cine: CineVolume
    series_uid: str | None = None
    backend_factory: CandidateBackendFactory | None = None
    native_inferer: NativeInferer | None = None
    profile_hint: str = "CPU_BASELINE_V1"
    budget_policy: BudgetPolicy | None = None
    clock: Any | None = None
    uuid_factory: Any | None = None
    cancel_token: CancelToken | None = None


# Descriptive aliases make the small dependency object discoverable without
# introducing a second behavior or controller abstraction.
IntegrationDependencies = AiRunDependencies
LocalAiRunDependencies = AiRunDependencies


def _dependency(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _optional_dependency(value: object, name: str, default: object = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def run_ai_segmentation(
    controller: Any,
    integration_deps: AiRunDependencies | Mapping[str, object],
    bundle_id: str,
) -> LocalInferenceOutcome:
    """Reopen accepted workflow lineage and delegate one local AI run."""

    store = _dependency(integration_deps, "store")
    registry = _dependency(integration_deps, "registry")
    cine = _dependency(integration_deps, "cine")
    explicit_series_uid = _optional_dependency(integration_deps, "series_uid")
    series_uid = explicit_series_uid or cine.series_uid
    state = controller.reopen(store, series_uid, cine=cine)

    source_manifest = state.get("source_manifest")
    workflow_id = state.get("workflow_id")
    if workflow_id is None and source_manifest is not None:
        workflow_id = source_manifest.workflow_id
    if workflow_id is None:
        workflow_id = cine.workflow_id
    source_manifest_id = workflow_id
    study_uid = source_manifest.study_uid if source_manifest is not None else cine.study_uid
    if study_uid is None:
        study_uid = ""
    if series_uid is None:
        series_uid = ""
    if workflow_id is None:
        workflow_id = ""
    if source_manifest_id is None:
        source_manifest_id = ""

    return run_local_whole_cine_inference(
        store=store,
        registry=registry,
        bundle_id=bundle_id,
        cine=cine,
        study_uid=study_uid,
        series_uid=series_uid,
        workflow_id=workflow_id,
        source_manifest_id=source_manifest_id,
        backend_factory=_optional_dependency(integration_deps, "backend_factory"),
        native_inferer=_optional_dependency(integration_deps, "native_inferer"),
        profile_hint=_optional_dependency(
            integration_deps, "profile_hint", "CPU_BASELINE_V1"
        ),
        budget_policy=_optional_dependency(integration_deps, "budget_policy"),
        clock=_optional_dependency(integration_deps, "clock"),
        uuid_factory=_optional_dependency(integration_deps, "uuid_factory"),
        cancel_token=_optional_dependency(integration_deps, "cancel_token"),
    )


__all__ = [
    "AiRunDependencies",
    "IntegrationDependencies",
    "LocalAiRunDependencies",
    "run_ai_segmentation",
]
