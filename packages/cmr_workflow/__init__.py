"""cmr_workflow — Node 7 stateless workflow controller.

Pure composition layer over the accepted Node 2–6 primitives; no session
state, no inference, no GPU/cloud and no HTTP endpoint. Re-exports are
resolved lazily (PEP 562) so importing the package stays cheap and
import-order safe (the controller module pulls in the full primitive set,
including pydicom-backed ``cmr_dicom``).
"""

from __future__ import annotations

from typing import Any

_COVERAGE_NAMES = (
    "COVERAGE_INSUFFICIENT",
    "CoverageResolution",
    "FullCycleCapability",
    "ResolvedKeyCoverage",
    "SELECTED_PHASE_GROUPS",
    "can_satisfy_full_cycle",
    "resolve_full_cycle_coverage",
    "resolve_selected_phase_coverage",
    "selected_phase_required_keys",
)

_CONTROLLER_NAMES = (
    "OVERRIDE_MARKER",
    "PASS_VERDICT",
    "REVIEW_REQUIRED_VERDICT",
    "SENTINEL_ALGORITHM_VERSION",
    "SENTINEL_MEASUREMENT_ID",
    "SENTINEL_VERSION_SUFFIX",
    "WorkflowController",
    "check_pre_projection_topology",
    "embed_override_reason",
    "extract_override_reason",
    "materialize_legacy_chain_from_whole_cine",
    "project_persisted_whole_cine_to_legacy_draft",
)


def __getattr__(name: str) -> Any:
    if name in _CONTROLLER_NAMES:
        from cmr_workflow import controller

        return getattr(controller, name)
    if name in _COVERAGE_NAMES:
        from cmr_workflow import coverage

        return getattr(coverage, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CONTROLLER_NAMES) | set(_COVERAGE_NAMES))


__all__ = [*_CONTROLLER_NAMES, *_COVERAGE_NAMES]
