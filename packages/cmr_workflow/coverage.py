"""FNC-C selected-phase coverage resolver (read-only, shared).

PREIMPL review, "Sparse/all-25 compatibility and provenance": a shared
read-only coverage resolver must expose
``(source slice, source frame, structure) -> contour + artifact identity``
so materialization/quantification can consume an explicitly sparse
(selected-phase) prediction without ever pretending it is whole-cine, and
without silently falling back to inference.

This module is that resolver. It is READ-ONLY by construction: it loads
persisted immutable artifacts, resolves keys and reports what is missing. It
NEVER writes, NEVER runs a provider, NEVER rasterizes and NEVER computes a
clinical quantity. A cache miss is reported as a missing key — never as a
request to execute all-phase inference.

Binding resolution rules (summarized from the PREIMPL review — this is the
implementer's condensed phrasing, NOT a verbatim quotation; the authoritative
rule text is the "Sparse/all-25 compatibility and provenance" section of
``docs/reviews/PREIMPL_CMR_FUNCTION_ADVANCEMENT_REVIEW.md``):

1. exact same workflow/cine identity + exact source-frame key only. Every
   artifact consulted must agree on study, series, cine grid extents,
   model/preprocessing identity (name, version, checksum,
   preprocessing_version, label_map) and (when bound) the manifest workflow
   id; a key is served only at the EXACT source frame index, never at a
   "nearby" frame and never at a local slot.
2. physician revision wins over AI for display/metrics. If the store's
   latest physician revision covers a key, that key is exposed with
   physician authority; the AI artifact is never silently swapped in.
3. an all-phase prediction may satisfy selected ED/ES by explicit key
   projection: the contour is retrieved at the SAME source frame index.
4. a sparse prediction cannot satisfy full-cycle/curve tasks — see
   :func:`can_satisfy_full_cycle` and :func:`resolve_full_cycle_coverage`.
5. never combine two AI artifacts silently. Exactly ONE AI artifact is
   chosen per resolution (documented deterministic priority below) and the
   choice is recorded in the result; the other AI artifact never fills the
   chosen one's gaps.
6. legacy ``WholeCinePrediction`` canonical bytes/table/IDs are unchanged —
   this module only READS them.
7. no fallback from a sparse cache miss to all-phase EXECUTION. Rule 3 may
   PROJECT an EXISTING all-phase prediction; the resolver never triggers
   inference.

AI artifact priority (rule 5, deterministic and recorded; ACTUAL behavior):

* a persisted, CURRENT-ELIGIBLE (A1-bound)
  :class:`~cmr_core.domain_types.SelectedPhasePrediction` is ALWAYS
  preferred, UNCONDITIONALLY — even when its
  ``source_frame_indices`` do NOT contain both required selected frames. It
  remains the single chosen AI source; every key it cannot serve at the
  exact source frame is reported MISSING and is NEVER filled from the
  other artifact and never executed (rules 5 and 7). The choice and the
  declared-vs-required coverage mismatch are recorded in
  ``ai_artifact_choice``;
* the latest whole-cine (all-phase) prediction is used by explicit key
  projection (rule 3) ONLY when NO sparse artifact is persisted for the
  series;
* otherwise no AI artifact is chosen and every uncovered key is MISSING.

This unconditional sparse preference is deliberate: silently substituting an
all-phase artifact for a sparse one would mix two AI identities (rule 5) and
would let a stale all-phase artifact cover a newer selected-phase request.
The caller sees the missing keys and decides; the resolver never guesses.

Determinism: candidates are resolved by documented store order and the key
iteration order is canonical (sorted). No timestamps or dict-order
accidents participate.

Scope note (deliberate minimum): the physician authority surface of this
slice is the whole-cine physician revision head
(:class:`~cmr_core.domain_types.WholeCineRevision`), which shares the
``HoleAwareContour`` geometry type with both prediction families. The legacy
``Prediction``/``ContourRevision`` chain (``cmr_contours`` ``Contour``) keeps
its untouched owner (``materialize_legacy_chain_from_whole_cine``); FNC-C
does not re-model it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cmr_core.domain_types import (
    HoleAwareContour,
    SelectedPhasePrediction,
    WholeCinePrediction,
    WholeCineRevision,
)
from cmr_core.errors import INVALID_ARGUMENT, CMRError
from cmr_core.segmentation import Structure

__all__ = [
    "COVERAGE_INSUFFICIENT",
    "SELECTED_PHASE_GROUPS",
    "CoverageResolution",
    "FullCycleCapability",
    "ResolvedKeyCoverage",
    "can_satisfy_full_cycle",
    "resolve_full_cycle_coverage",
    "resolve_selected_phase_coverage",
    "selected_phase_required_keys",
]

#: Typed refusal code for "asked for coverage an artifact cannot honestly
#: provide" (a sparse artifact asked for full-cycle coverage). Local to this
#: module: no existing error code is reused and none is changed.
COVERAGE_INSUFFICIENT = "coverage.insufficient"

#: The exact five selected-phase geometry groups (FNC-B contract):
#: ED ``LV_CAVITY`` + ``LV_MYOCARDIUM`` + ``RV_CAVITY``, ES ``LV_CAVITY`` +
#: ``RV_CAVITY``. Mirrors
#: ``cmr_quantification.measurements._selected_phase_required_keys``; the
#: duplication is deliberate so the resolver imports no quantification
#: internals and FNC-B code stays untouched.
SELECTED_PHASE_GROUPS: tuple[tuple[str, Structure], ...] = (
    ("ed", Structure.LV_CAVITY),
    ("ed", Structure.LV_MYOCARDIUM),
    ("ed", Structure.RV_CAVITY),
    ("es", Structure.LV_CAVITY),
    ("es", Structure.RV_CAVITY),
)

WHOLE_CINE_PREDICTION_DOMAIN = "WholeCinePrediction"
SELECTED_PHASE_PREDICTION_DOMAIN = "SelectedPhasePrediction"
WHOLE_CINE_REVISION_DOMAIN = "WholeCineRevision"

PHYSICIAN_AUTHORITY = "physician"
AI_AUTHORITY = "ai"

_EXACT_SOURCE_FRAME = "exact-source-frame"
_ALL_PHASE_PROJECTION = "all-phase-projection"

ContourKey = tuple[int, int, Structure]


def selected_phase_required_keys(
    n_slices: int, ed_frame: int, es_frame: int
) -> tuple[ContourKey, ...]:
    """The canonical ordered five-group required key set for selected phases.

    One key per (slice, role, structure) group; order is deterministic
    (slice, then the frozen group order, so callers can compare tuples).
    """
    if isinstance(n_slices, bool) or not isinstance(n_slices, int) or n_slices < 1:
        raise CMRError(
            INVALID_ARGUMENT,
            f"n_slices must be an integer >= 1, got {n_slices!r}",
        )
    for name, frame in (("ed_frame", ed_frame), ("es_frame", es_frame)):
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise CMRError(
                INVALID_ARGUMENT,
                f"{name} must be a non-negative integer, got {frame!r}",
            )
    keys: list[ContourKey] = []
    for slice_index in range(n_slices):
        for role, structure in SELECTED_PHASE_GROUPS:
            frame_index = ed_frame if role == "ed" else es_frame
            keys.append((slice_index, frame_index, structure))
    return tuple(keys)


@dataclass(frozen=True)
class ResolvedKeyCoverage:
    """One resolved required key: geometry plus the artifact that supplied it."""

    key: ContourKey
    contours: tuple[HoleAwareContour, ...]
    #: Content-addressed identity of the supplying artifact.
    artifact_id: str
    #: Canonical domain of the supplying artifact.
    artifact_domain: str
    #: ``"physician"`` or ``"ai"`` (rule 2).
    authority: str
    #: How the key was matched: exact source frame (both families) or an
    #: explicit all-phase projection at the same source frame (rule 3).
    projection: str


@dataclass(frozen=True)
class CoverageResolution:
    """The complete, deterministic answer for one selected-phase resolution.

    ``resolved`` holds every required key an artifact could serve; ``missing``
    holds every required key nothing could serve (a sparse cache miss is
    reported here — it is NEVER an instruction to run inference, rule 7).

    ``unknown_grid`` is the documented P2-3 marker for "no persisted artifact
    (and no caller-supplied extent) declares a cine grid": the required key
    set cannot even be enumerated, so :attr:`complete` must be ``False``.
    When the caller supplies ``slice_count`` in that situation, ``missing``
    carries ALL required five-group keys instead.
    """

    series_uid: str
    ed_frame: int
    es_frame: int
    slice_count: int | None
    frame_count: int | None
    resolved: Mapping[ContourKey, ResolvedKeyCoverage]
    missing: tuple[ContourKey, ...]
    #: The single AI artifact chosen for this resolution (rule 5).
    ai_artifact_id: str | None
    ai_artifact_domain: str | None
    #: Why that artifact was chosen (recorded, never implicit).
    ai_artifact_choice: str | None
    #: The physician artifact that outranks it per key (rule 2), if any.
    physician_artifact_id: str | None
    notes: tuple[str, ...]
    #: P2-3: True when no cine grid extent could be established (no artifact
    #: grid, no caller-supplied ``slice_count``): completeness is never
    #: claimed on an unknown grid.
    unknown_grid: bool = False

    @property
    def complete(self) -> bool:
        """True only when the grid is known and every required key is resolved."""
        return not self.missing and not self.unknown_grid

    @property
    def physician_keys(self) -> tuple[ContourKey, ...]:
        return tuple(
            key for key, entry in self.resolved.items() if entry.authority == PHYSICIAN_AUTHORITY
        )

    @property
    def ai_keys(self) -> tuple[ContourKey, ...]:
        return tuple(
            key for key, entry in self.resolved.items() if entry.authority == AI_AUTHORITY
        )


@dataclass(frozen=True)
class FullCycleCapability:
    """Typed answer to "can this series' coverage satisfy a full-cycle task?"."""

    series_uid: str
    frame_count: int | None
    satisfied: bool
    reason: str
    artifact_id: str | None
    artifact_domain: str | None


def _artifact_grid(artifact: Any) -> tuple[int, int] | None:
    """The cine grid extents an artifact declares, or ``None``.

    ``WholeCineRevision`` carries the full post-edit contour state but no
    grid extents of its own; its grid is INHERITED from the prediction named
    by ``parent_prediction_id`` and is resolved separately by
    :func:`_load_physician_parent` (P1-2), never guessed here.
    """
    slice_count = getattr(artifact, "slice_count", None)
    frame_count = getattr(artifact, "frame_count", None)
    if slice_count is None or frame_count is None:
        return None
    return (int(slice_count), int(frame_count))


def _workflow_id_of(artifact: Any) -> str | None:
    workflow_id = getattr(artifact.provenance, "workflow_id", None)
    return None if workflow_id is None else str(workflow_id)


def _model_identity(artifact: Any) -> str | None:
    """Canonical model/preprocessing identity of an artifact, or ``None``.

    P1-1: artifacts that share study/workflow/grid but were produced by a
    different model (different name, version, checksum, preprocessing
    version or label map) must never be mixed into one resolution. The
    identity string is built from ``provenance.model`` (falling back to the
    prediction's own ``model`` field, which the dataclass pins to
    ``provenance.model`` for AI artifacts) and is deterministic: every
    component is a primitive and the label map is sorted.

    ``None`` means "this artifact declares no model identity" (a manual-seed
    prediction or a physician revision), which contributes no identity to
    compare — it is NOT a wildcard that matches any model.
    """
    provenance = getattr(artifact, "provenance", None)
    model = getattr(provenance, "model", None) if provenance is not None else None
    if model is None:
        model = getattr(artifact, "model", None)
    if model is None:
        return None
    label_map = getattr(model, "label_map", None) or {}
    label_text = ";".join(
        f"{str(key)}={str(value)}" for key, value in sorted(label_map.items())
    )
    checksum = "none" if model.checksum is None else str(model.checksum)
    return (
        f"{model.name}@{model.version}|preprocessing={model.preprocessing_version}"
        f"|checksum={checksum}|label_map=[{label_text}]"
    )


def _fail_coherence(message: str) -> CMRError:
    return CMRError(INVALID_ARGUMENT, message)


def _load_physician_parent(
    store: Any, series_uid: str, physician: WholeCineRevision
) -> WholeCinePrediction:
    """The prediction a physician revision INHERITS grid/model identity from.

    P1-2: a ``WholeCineRevision`` carries no grid extents of its own, so its
    inherited grid is the parent prediction's ``(slice_count, frame_count)``.
    An absent parent fails closed: without it the revision's grid and model
    identity cannot be validated, and a stale revision could otherwise
    override keys on a different cine grid (P1-1 also compares the parent's
    model identity).
    """
    parent = store.load_whole_cine_prediction(physician.parent_prediction_id)
    if parent is None:
        raise _fail_coherence(
            f"physician revision {physician.revision_id!r} references parent "
            f"prediction {physician.parent_prediction_id!r}, which is not "
            f"persisted for series {series_uid!r}; its inherited cine grid and "
            "model identity cannot be validated"
        )
    return parent


def _collect_candidates(
    store: Any, series_uid: str
) -> tuple[SelectedPhasePrediction | None, WholeCinePrediction | None, WholeCineRevision | None]:
    """Load the three read-only coverage candidates for one series.

    * current-ELIGIBLE sparse (selected-phase) prediction (A1: bound to the
      series' CURRENT head_revision_id + selection_id through a succeeded
      inference_jobs row; ineligible artifacts are audit-only and never
      affect current coverage — the unqualified latest-sparse read is for
      audit/history readers only);
    * latest whole-cine (all-phase) prediction;
    * the current whole-cine lineage head when it is a PHYSICIAN revision
      (``load_latest_whole_cine_head`` returns the newest prediction when no
      revision exists — that case leaves the physician slot ``None`` because
      a prediction is AI, rule 2).
    """
    sparse: SelectedPhasePrediction | None = None
    # A1: coverage is a current-state consumer, never an audit/history reader.
    # Resolve the nullable current head exactly as the eligibility predicate
    # expects.  A missing selection or an unreadable identity fails closed;
    # importantly, a chain-absent series passes ``head_revision_id=None`` to
    # the eligible accessor instead of falling back to an unbound artifact.
    _head_id: str | None = None
    _selection_id: str | None = None
    identity_read_ok = True
    try:
        chain = store.load_chain(series_uid)
        if chain:
            _head_id = chain[-1].revision_id
    except Exception:
        identity_read_ok = False
    try:
        latest_selection = store.load_latest_selection(series_uid)
        if latest_selection is not None:
            _selection_id = latest_selection.selection_id
    except Exception:
        identity_read_ok = False
    eligible_loader = getattr(store, "load_current_selected_phase_prediction", None)
    if eligible_loader is not None and identity_read_ok:
        try:
            sparse = eligible_loader(
                series_uid,
                head_revision_id=_head_id,
                selection_id=_selection_id,
            )
        except Exception:
            # Coverage fails closed on an unreadable current binding.  The
            # unqualified latest accessor is intentionally not used here; it
            # is reserved for explicit audit/history readers.
            sparse = None
    whole = store.load_latest_whole_cine_prediction(series_uid)
    head = store.load_latest_whole_cine_head(series_uid)
    physician = head if isinstance(head, WholeCineRevision) else None
    return sparse, whole, physician


def _assert_coherent_bindings(
    store: Any,
    series_uid: str,
    candidates: Sequence[Any],
) -> tuple[int | None, int | None]:
    """Rule 1: one study, cine grid, workflow and model identity per series.

    Fail closed on ANY disagreement (or on a series mismatch, which would
    mean the store returned a row for another series).

    P1-1: the comparison includes model/preprocessing identity for every
    consulted AI artifact, AND the physician revision's originating
    prediction (loaded by ``parent_prediction_id``), whose model identity it
    inherits.

    P1-2: a ``WholeCineRevision`` has no grid extents of its own; its
    inherited grid is the parent prediction's ``(slice_count, frame_count)``.
    A missing parent fails closed.
    """
    grids: set[tuple[int, int]] = set()
    studies: set[str] = set()
    workflows: set[str] = set()
    model_identities: set[str] = set()
    for artifact in candidates:
        if artifact is None:
            continue
        if isinstance(artifact, WholeCineRevision):
            # The revision's own grid/model identity is the parent's; the
            # revision itself keeps its own study/series/workflow checks.
            parent = _load_physician_parent(store, series_uid, artifact)
            artifacts: tuple[Any, ...] = (artifact, parent)
            # The revision inherits its model identity from the parent; if it
            # also declares one itself it must agree (it is compared too).
            identities: tuple[Any, ...] = (artifact, parent)
        else:
            artifacts = (artifact,)
            identities = (artifact,)
        for checked in artifacts:
            if str(checked.series_uid) != series_uid:
                raise _fail_coherence(
                    f"coverage artifact series {checked.series_uid!r} does not match "
                    f"requested series {series_uid!r}"
                )
            grid = _artifact_grid(checked)
            if grid is not None:
                grids.add(grid)
            studies.add(str(checked.study_uid))
            # Rule 1: one DICOM source workflow identity. ``None`` (legacy
            # unbound) is a VALUE here: mixing a bound artifact with an unbound
            # one would silently cover two different cine sources.
            workflows.add(str(_workflow_id_of(checked)))
        for checked in identities:
            identity = _model_identity(checked)
            if identity is not None:
                model_identities.add(identity)
    if len(studies) > 1:
        raise _fail_coherence(
            "conflicting study identities across coverage artifacts for series "
            f"{series_uid!r}: {sorted(studies)!r}"
        )
    if len(workflows) > 1:
        raise _fail_coherence(
            "conflicting DICOM workflow identities (including legacy-unbound None) "
            f"across coverage artifacts for series {series_uid!r}: "
            f"{sorted(str(value) for value in workflows)!r}"
        )
    if len(grids) > 1:
        raise _fail_coherence(
            "conflicting cine grid extents across coverage artifacts for series "
            f"{series_uid!r}: {sorted(grids)!r}"
        )
    if len(model_identities) > 1:
        raise _fail_coherence(
            "conflicting model/preprocessing identities across coverage artifacts "
            f"for series {series_uid!r}: {sorted(model_identities)!r}"
        )
    if not grids:
        return (None, None)
    grid = next(iter(grids))
    return (grid[0], grid[1])


def _choose_ai_artifact(
    sparse: SelectedPhasePrediction | None,
    whole: WholeCinePrediction | None,
    ed_frame: int,
    es_frame: int,
) -> tuple[Any | None, str | None, str]:
    """Rule 5: choose exactly ONE AI artifact and record why.

    Priority is UNCONDITIONAL: any persisted selected-phase artifact wins
    over the all-phase projection, even when it does not declare both
    required source frames (P3-7 documents the actual behavior). The chosen
    artifact is used for every AI-served key; whatever it does not carry at
    the exact source frame is reported MISSING — never filled from the other
    artifact (rule 5) and never executed (rule 7).
    """
    required = sorted({ed_frame, es_frame})
    if sparse is not None:
        declared = [int(index) for index in sparse.source_frame_indices]
        covered = set(declared)
        if all(frame in covered for frame in required):
            reason = (
                "sparse-selected-phase-artifact-chosen-as-the-single-ai-source:"
                "-declares-both-required-source-frames"
            )
        else:
            reason = (
                "sparse-selected-phase-artifact-chosen-as-the-single-ai-source-but"
                f"-does-not-declare-the-required-source-frames (declared {declared!r},"
                f" required {required!r}); uncovered keys are reported missing, never"
                " filled from another artifact and never executed"
            )
        return sparse, SELECTED_PHASE_PREDICTION_DOMAIN, reason
    if whole is not None:
        if ed_frame < whole.frame_count and es_frame < whole.frame_count:
            return (
                whole,
                WHOLE_CINE_PREDICTION_DOMAIN,
                "all-phase-prediction-projected-at-the-same-source-frame-indices",
            )
        return (
            None,
            None,
            "all-phase-prediction-grid-excludes-the-required-source-frames "
            f"(frame_count={whole.frame_count}, required {required!r})",
        )
    return (None, None, "no-prediction-artifact-persisted-for-this-series")


def resolve_selected_phase_coverage(
    store: Any,
    series_uid: str,
    ed_frame: int,
    es_frame: int,
    *,
    slice_count: int | None = None,
) -> CoverageResolution:
    """Resolve the five selected-phase groups for one series, read-only.

    Returns a :class:`CoverageResolution`: every required
    ``(slice, source frame, structure)`` key served by an exact source-frame
    match, its supplying artifact identity/domain, the single chosen AI
    artifact, and the keys nothing could serve.

    ``slice_count`` (the cine slice extent) may be supplied when the caller
    already knows the cine grid; otherwise it is derived from the persisted
    artifacts. When supplied it must agree with them (fail closed).

    Fail-closed cases: incoherent study/workflow/grid/model bindings, a
    supplied ``slice_count`` disagreeing with the artifacts, and selected
    frames outside the cine grid. No key is ever served at a frame other
    than the exact requested source frame.

    With NO persisted artifact (P2-3) completeness is never claimed: the
    result carries ``unknown_grid=True`` and, when the caller supplied
    ``slice_count``, lists ALL required five-group keys as ``missing``.
    """
    if not isinstance(series_uid, str) or not series_uid:
        raise _fail_coherence("series_uid is required and must be a non-empty string")
    for name, frame in (("ed_frame", ed_frame), ("es_frame", es_frame)):
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise _fail_coherence(
                f"{name} must be a non-negative integer, got {frame!r}"
            )
    if slice_count is not None and (
        isinstance(slice_count, bool) or not isinstance(slice_count, int) or slice_count < 1
    ):
        raise _fail_coherence(
            f"slice_count must be an integer >= 1 when supplied, got {slice_count!r}"
        )

    sparse, whole, physician = _collect_candidates(store, series_uid)
    derived_slices, derived_frames = _assert_coherent_bindings(
        store, series_uid, (sparse, whole, physician)
    )

    notes: list[str] = []
    if derived_slices is None or derived_frames is None:
        # P2-3: an absent prediction is INCOMPLETE coverage, never complete.
        # With a caller-supplied extent every required key is enumerable and
        # is reported missing; without one the grid is unknown and
        # completeness cannot be claimed at all.
        missing_unknown: tuple[ContourKey, ...] = ()
        if slice_count is not None:
            missing_unknown = selected_phase_required_keys(
                slice_count, ed_frame, es_frame
            )
            notes.append(
                "missing-keys-reported-not-executed: "
                f"{len(missing_unknown)} required key(s) have no persisted coverage"
            )
        notes.append(
            "unknown-grid: no persisted artifact declares a cine grid for series "
            f"{series_uid!r} (caller-supplied slice_count="
            f"{slice_count!r}); completeness is never claimed and no inference "
            "is triggered"
        )
        notes.append(
            "no-prediction-artifact-persisted-for-this-series: coverage is unknown, "
            "no key is resolved and no inference is triggered"
        )
        return CoverageResolution(
            series_uid=series_uid,
            ed_frame=ed_frame,
            es_frame=es_frame,
            slice_count=slice_count,
            frame_count=None,
            resolved=MappingProxyType({}),
            missing=missing_unknown,
            ai_artifact_id=None,
            ai_artifact_domain=None,
            ai_artifact_choice="no-prediction-artifact-persisted-for-this-series",
            physician_artifact_id=None,
            notes=tuple(notes),
            unknown_grid=True,
        )

    frame_count = derived_frames
    resolved_slices = derived_slices
    if slice_count is not None and slice_count != derived_slices:
        raise _fail_coherence(
            f"supplied slice_count {slice_count!r} does not match the persisted "
            f"coverage grid {derived_slices!r} for series {series_uid!r}"
        )
    if slice_count is not None:
        resolved_slices = slice_count
    out_of_range = [
        frame for frame in sorted({ed_frame, es_frame}) if not 0 <= frame < frame_count
    ]
    if out_of_range:
        raise _fail_coherence(
            "selected frame indices must fall within [0, frame_count); outside: "
            f"{out_of_range!r} (frame_count={frame_count})"
        )

    chosen, chosen_domain, choice_reason = _choose_ai_artifact(
        sparse, whole, ed_frame, es_frame
    )
    notes.append(f"ai-artifact-choice: {choice_reason}")

    resolved: dict[ContourKey, ResolvedKeyCoverage] = {}
    missing: list[ContourKey] = []
    for key in selected_phase_required_keys(resolved_slices, ed_frame, es_frame):
        # Rule 2: a physician revision covering the key wins, always. A
        # persisted EMPTY tuple is the frozen "no contour" spelling, NOT
        # coverage: it is reported missing exactly like an absent key
        # (F19/F20 — an unauthorised EMPTY must never read as sealable
        # completeness; only a physician applicability decision recorded
        # in the inclusion grid can waive a key, and that waiver lives
        # downstream of this resolver, never inside it).
        if physician is not None and key in physician.contours:
            if not physician.contours[key]:
                missing.append(key)
                continue
            resolved[key] = ResolvedKeyCoverage(
                key=key,
                contours=tuple(physician.contours[key]),
                artifact_id=physician.revision_id,
                artifact_domain=WHOLE_CINE_REVISION_DOMAIN,
                authority=PHYSICIAN_AUTHORITY,
                projection=_EXACT_SOURCE_FRAME,
            )
            continue
        # Rules 1/3/5: the single chosen AI artifact, at the exact source frame.
        if chosen is not None and key in chosen.contours:
            if not chosen.contours[key]:
                missing.append(key)
                continue
            resolved[key] = ResolvedKeyCoverage(
                key=key,
                contours=tuple(chosen.contours[key]),
                artifact_id=chosen.prediction_id,
                artifact_domain=str(chosen_domain),
                authority=AI_AUTHORITY,
                projection=(
                    _ALL_PHASE_PROJECTION
                    if chosen_domain == WHOLE_CINE_PREDICTION_DOMAIN
                    else _EXACT_SOURCE_FRAME
                ),
            )
            continue
        missing.append(key)

    if missing:
        notes.append(
            "missing-keys-reported-not-executed: "
            f"{len(missing)} required key(s) have no persisted coverage"
        )

    return CoverageResolution(
        series_uid=series_uid,
        ed_frame=ed_frame,
        es_frame=es_frame,
        slice_count=resolved_slices,
        frame_count=frame_count,
        resolved=MappingProxyType(resolved),
        missing=tuple(missing),
        ai_artifact_id=None if chosen is None else chosen.prediction_id,
        ai_artifact_domain=chosen_domain,
        ai_artifact_choice=choice_reason,
        physician_artifact_id=None if physician is None else physician.revision_id,
        notes=tuple(notes),
    )


def _full_cycle_required_keys(
    slice_count: int, frame_count: int
) -> tuple[ContourKey, ...]:
    """The legacy full-cycle key set: LV/RV cavities for every slice x frame.

    Mirrors ``cmr_quantification.measurements.compute_measurement``'s coverage
    requirement (``LV_CAVITY`` and ``RV_CAVITY`` at every slice x frame);
    duplicated deliberately so the resolver imports no quantification
    internals.
    """
    return tuple(
        (slice_index, frame_index, structure)
        for slice_index in range(slice_count)
        for frame_index in range(frame_count)
        for structure in (Structure.LV_CAVITY, Structure.RV_CAVITY)
    )


def _missing_full_cycle_keys(
    artifact: Any, slice_count: int, frame_count: int
) -> list[ContourKey]:
    """Required full-cycle keys the artifact's contours do not actually carry."""
    contours = getattr(artifact, "contours", {})
    return [
        key
        for key in _full_cycle_required_keys(slice_count, frame_count)
        if key not in contours
    ]


def _full_cycle_refusal_reason(
    missing: Sequence[ContourKey],
    *,
    prefix: str,
) -> str:
    """Typed reason naming what spatial coverage is missing (P2-4)."""
    preview = ", ".join(
        f"({key[0]},{key[1]},{key[2].value})" for key in missing[:6]
    )
    extra = len(missing) - 6
    suffix = f" (+{extra} more)" if extra > 0 else ""
    return (
        f"{prefix}: the required full-cycle key set (LV_CAVITY + RV_CAVITY on "
        f"every slice x frame) is not actually present; missing {len(missing)} "
        f"key(s): {preview}{suffix}"
    )


def can_satisfy_full_cycle(store: Any, series_uid: str) -> FullCycleCapability:
    """Can the persisted coverage of ``series_uid`` serve a full-cycle task?

    Rule 4: a sparse (selected-phase) artifact CANNOT satisfy full-cycle or
    volume-curve tasks unless it declares EVERY frame of the grid AND
    actually carries the full-cycle key set (``LV_CAVITY`` + ``RV_CAVITY``
    on every slice x frame — the legacy quantification requirement). An
    all-phase whole-cine prediction must carry that same spatial key set:
    a frame declaration alone is not spatial coverage (P2-4).

    The verdict is returned, never raised, so callers can render it; use
    :func:`resolve_full_cycle_coverage` for the typed refusal.
    """
    sparse, whole, _physician = _collect_candidates(store, series_uid)
    # Full-cycle coverage is satisfied ONLY by the sparse or the whole-cine
    # prediction; the physician revision does not serve full-cycle keys, so
    # it is not part of this coherence set (it remains fully checked by
    # :func:`resolve_selected_phase_coverage`, where it can override keys).
    derived_slices, derived_frames = _assert_coherent_bindings(
        store, series_uid, (sparse, whole)
    )
    if sparse is not None:
        declared = tuple(int(index) for index in sparse.source_frame_indices)
        if derived_frames is not None and declared == tuple(range(derived_frames)):
            missing = _missing_full_cycle_keys(
                sparse,
                sparse.slice_count if derived_slices is None else derived_slices,
                derived_frames,
            )
            if not missing:
                return FullCycleCapability(
                    series_uid=series_uid,
                    frame_count=derived_frames,
                    satisfied=True,
                    reason=(
                        "selected-phase artifact declares the full frame grid and "
                        "carries the full-cycle spatial key set"
                    ),
                    artifact_id=sparse.prediction_id,
                    artifact_domain=SELECTED_PHASE_PREDICTION_DOMAIN,
                )
            return FullCycleCapability(
                series_uid=series_uid,
                frame_count=derived_frames,
                satisfied=False,
                reason=_full_cycle_refusal_reason(
                    missing,
                    prefix=(
                        "sparse selected-phase artifact cannot satisfy "
                        "full-cycle/volume-curve coverage"
                    ),
                ),
                artifact_id=sparse.prediction_id,
                artifact_domain=SELECTED_PHASE_PREDICTION_DOMAIN,
            )
        return FullCycleCapability(
            series_uid=series_uid,
            frame_count=derived_frames,
            satisfied=False,
            reason=(
                "sparse selected-phase artifact cannot satisfy full-cycle/volume-curve "
                f"coverage: it declares {list(declared)!r} of "
                f"{derived_frames!r} source frames"
            ),
            artifact_id=sparse.prediction_id,
            artifact_domain=SELECTED_PHASE_PREDICTION_DOMAIN,
        )
    if whole is not None:
        slice_extent = whole.slice_count if derived_slices is None else derived_slices
        missing = _missing_full_cycle_keys(whole, slice_extent, whole.frame_count)
        if missing:
            return FullCycleCapability(
                series_uid=series_uid,
                frame_count=whole.frame_count,
                satisfied=False,
                reason=_full_cycle_refusal_reason(
                    missing,
                    prefix=(
                        "all-phase whole-cine prediction cannot satisfy "
                        "full-cycle/volume-curve coverage"
                    ),
                ),
                artifact_id=whole.prediction_id,
                artifact_domain=WHOLE_CINE_PREDICTION_DOMAIN,
            )
        return FullCycleCapability(
            series_uid=series_uid,
            frame_count=whole.frame_count,
            satisfied=True,
            reason=(
                "all-phase whole-cine prediction covers the full frame grid and "
                "carries the full-cycle spatial key set"
            ),
            artifact_id=whole.prediction_id,
            artifact_domain=WHOLE_CINE_PREDICTION_DOMAIN,
        )
    return FullCycleCapability(
        series_uid=series_uid,
        frame_count=None,
        satisfied=False,
        reason="no prediction artifact persisted for this series",
        artifact_id=None,
        artifact_domain=None,
    )


def resolve_full_cycle_coverage(
    store: Any, series_uid: str, *, frame_count: int
) -> tuple[Any, str]:
    """Return ``(artifact, domain)`` for a full-cycle task, or fail closed.

    Rule 4 as a typed refusal: asking a sparse artifact for non-selected
    (full-cycle/curve) coverage raises :class:`CMRError` with
    :data:`COVERAGE_INSUFFICIENT` naming the artifact and its declared
    coverage. Nothing is executed and nothing is projected.
    """
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise _fail_coherence(
            f"frame_count must be an integer >= 1, got {frame_count!r}"
        )
    capability = can_satisfy_full_cycle(store, series_uid)
    if not capability.satisfied:
        raise CMRError(
            COVERAGE_INSUFFICIENT,
            f"series {series_uid!r} cannot satisfy full-cycle coverage for "
            f"{frame_count} frames: {capability.reason}",
            source=capability.artifact_id,
        )
    if capability.frame_count is not None and capability.frame_count != frame_count:
        raise CMRError(
            COVERAGE_INSUFFICIENT,
            f"series {series_uid!r} full-cycle coverage declares "
            f"{capability.frame_count} frames, not the requested {frame_count}",
            source=capability.artifact_id,
        )
    sparse, whole, _physician = _collect_candidates(store, series_uid)
    if (
        sparse is not None
        and capability.artifact_id == sparse.prediction_id
    ):
        return sparse, capability.artifact_domain or SELECTED_PHASE_PREDICTION_DOMAIN
    assert whole is not None  # capability satisfied implies an artifact exists
    return whole, WHOLE_CINE_PREDICTION_DOMAIN
