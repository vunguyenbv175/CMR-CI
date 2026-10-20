"""Persisted selected-phase analysis + sparse-inference jobs for the workstation.

This module owns the one optional workstation worker thread.  The persistence,
quantification, and workflow packages remain thread-free; a worker only
orchestrates their existing, deterministic primitives and publishes through a
checked store transaction.  A timed-out stop retains the live thread handle and
refuses restart until that thread exits.  On startup, running jobs older than
``RUNNING_STALE_S`` are marked ``failed/interrupted``; they are
not recomputed because their worker ownership and in-flight lineage are stale.

Inference lifecycle (FNC-E2b-2 C1, AMENDMENT-001 A3): frozen transitions
``queued->running`` (claim), ``queued->cancelled``, ``running->succeeded|
failed|cancelled``. Cancellation is effective ONLY while queued, at the claim
boundary, and at the pre-glue-call check; once the model call starts it runs
to completion and persists its append-only rows (cancellation then only marks
the job row — it does NOT imply the model stopped). Terminal rows accept
audit bindings once via ``attach_inference_result`` without changing state;
``cancelled`` is NEVER resurrected into ``succeeded``. Single-worker
ownership is assumed (one live JobRunner per database).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cmr_core.domain_types import canonical_digest
from cmr_core.errors import IssueSet
from cmr_core.provenance import VERSION
from cmr_persistence.records import SliceInclusionSet
from cmr_persistence.store import (
    AnalysisAuthorityError,
    AnalysisJobRecord,
    InferenceJobRecord,
    JobNotPublishableError,
    SqliteCmrStore,
    StaleHeadError,
)
from cmr_qc.snapshot import run_selected_phase_qc
from cmr_quantification.measurements import (
    ALGORITHM_VERSION_IPP_SELECTED_PHASES,
    ALGORITHM_VERSION_SINGLE_SLICE_SELECTED_PHASES,
    applicability_waivers,
    build_cine_contour_index,
    compute_selected_phase_measurement,
)
from cmr_quantification.reference_ranges import REFERENCE_TABLE_VERSION
from cmr_workflow.controller import WorkflowController
from cmr_workflow.coverage import resolve_selected_phase_coverage
from cmr_workflow.coverage import selected_phase_required_keys as gate_required_keys

try:
    from inference.backends.contract import CancelToken
except ModuleNotFoundError:  # pragma: no cover - minimal workstation wheel fallback
    class CancelToken:
        """Dependency-neutral cancellation token for the minimal wheel."""

        def __init__(self) -> None:
            self._requested = False

        def request(self) -> None:
            self._requested = True

        @property
        def is_requested(self) -> bool:
            return self._requested


LOGGER = logging.getLogger(__name__)
RECOMPUTE_DEBOUNCE_S = 1.5
JOB_POLL_INTERVAL_S = 0.25
# Ten minutes bounds recovery of a worker that died after claiming a job while
# avoiding premature interruption of a slow deterministic computation.
RUNNING_STALE_S = 600.0
TERMINAL_WRITE_RETRIES = 3
TERMINAL_WRITE_BACKOFF_S = 0.01
#: How often the worker loop re-checks for orphaned running jobs. The
#: interval bounds the worst-case extra wait for a young orphan after a
#: restart (RUNNING_STALE_S + RECONCILE_INTERVAL_S), keeping active
#: computation (live tokens) protected.
RECONCILE_INTERVAL_S = 60.0

_TIMING_KEYS = (
    "enqueued_at",
    "debounce_wait_s",
    "claimed_at",
    "coverage_resolution_s",
    "qc_s",
    "quantification_s",
    "publish_s",
    "finished_at",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _seconds(start: float) -> float:
    return max(0.0, time.perf_counter() - start)


def _timings(value: str) -> dict[str, Any]:
    result = json.loads(value)
    for key in _TIMING_KEYS:
        result.setdefault(key, None)
    return result


def _inference_timings(value: str) -> dict[str, Any]:
    result = json.loads(value)
    for key in (
        "enqueued_at",
        "claimed_at",
        "runtime_resolution_s",
        "inference_s",
        "publish_s",
        "finished_at",
    ):
        result.setdefault(key, None)
    return result


def request_identity(
    *,
    series_uid: str,
    head_revision_id: str | None,
    selection_id: str | None,
    inclusion_set_id: str | None,
    coverage_identity: Mapping[str, Any],
    software_version: str,
    algorithm_version: str,
    range_table_version: str,
) -> dict[str, Any]:
    """Return the canonical, clinical-lineage-bound request identity."""
    return {
        "series_uid": series_uid,
        "kind": "selected_phase_recompute",
        "head_revision_id": head_revision_id,
        "selection_id": selection_id,
        "inclusion_set_id": inclusion_set_id,
        "coverage": dict(coverage_identity),
        "software_version": software_version,
        "algorithm_version": algorithm_version,
        "range_table_version": range_table_version,
    }


def enqueue_selected_phase_recompute(
    store: SqliteCmrStore,
    *,
    series_uid: str,
    head_revision_id: str | None,
    selection_id: str | None,
    inclusion_set_id: str | None,
    coverage_identity: Mapping[str, Any],
    software_version: str = VERSION,
    algorithm_version: str = "selected-phase-recompute-v1",
    range_table_version: str = REFERENCE_TABLE_VERSION,
    created_at: str,
) -> AnalysisJobRecord | None:
    """Best-effort enqueue helper used after successful authoritative writes.

    Missing/null lineage or an unresolvable coverage identity is deliberately
    a no-op: a job that cannot compute anything is not useful audit history.
    Callers should catch all store errors around this helper so enqueueing can
    never roll back the clinical mutation.
    """
    if not head_revision_id or not selection_id:
        return None
    if not coverage_identity or coverage_identity.get("unknown_grid"):
        return None
    if coverage_identity.get("missing"):
        return None
    identity = request_identity(
        series_uid=series_uid,
        head_revision_id=head_revision_id,
        selection_id=selection_id,
        inclusion_set_id=inclusion_set_id,
        coverage_identity=coverage_identity,
        software_version=software_version,
        algorithm_version=algorithm_version,
        range_table_version=range_table_version,
    )
    return store.enqueue_analysis_job(
        series_uid=series_uid,
        kind="selected_phase_recompute",
        request_payload=identity,
        dedup_identity=identity,
        created_at=created_at,
    )


def _lineage_payload(job: AnalysisJobRecord) -> tuple[str | None, str | None, str | None]:
    payload = json.loads(job.request_payload)
    return (
        payload.get("head_revision_id"),
        payload.get("selection_id"),
        payload.get("inclusion_set_id"),
    )


def _typed_diagnostic(exc: BaseException, code: str = "inference_failed") -> str:
    """Path-safe typed diagnostic (F5).

    Returns ``"<ExceptionType>:<code>"`` — the exception TYPE plus a stable
    code. Raw text (``str(exc)``) is NEVER interpolated: checkpoint spellings
    and other host paths must not reach logs or HTTP bodies.
    """
    return f"{type(exc).__name__}:{code}"


def _inference_diagnostic(exc: BaseException) -> str:
    """Path-safe typed diagnostic for the inference surface (F5)."""
    return _typed_diagnostic(exc, "inference_failed")


def _inference_terminal_diagnostic(exc: BaseException) -> str:
    """Path-safe terminal-write diagnostic (F5): type + stable code only."""
    return _typed_diagnostic(exc, "inference_terminal_write_failed")


class JobRunner:
    """One optional daemon worker for persisted workstation jobs."""

    def __init__(
        self,
        db_path: Path,
        *,
        controller: WorkflowController,
        cine_by_uid: Mapping[str, Any],
        clock: Callable[[], str] = _utc_now,
        compute: Callable[[SqliteCmrStore, AnalysisJobRecord, CancelToken], Any] | None = None,
        inference_compute: Callable[
            [SqliteCmrStore, InferenceJobRecord, CancelToken, dict[str, Any]], Any
        ] | None = None,
        x1_weights_root: Path | None = None,
        debounce_s: float = RECOMPUTE_DEBOUNCE_S,
        poll_interval_s: float = JOB_POLL_INTERVAL_S,
        reconcile_interval_s: float = RECONCILE_INTERVAL_S,
    ) -> None:
        self.db_path = Path(db_path)
        self.controller = controller
        self.cine_by_uid = cine_by_uid
        self.clock = clock
        self.compute = compute
        self.inference_compute = inference_compute
        self.x1_weights_root = None if x1_weights_root is None else Path(x1_weights_root)
        self.debounce_s = debounce_s
        self.poll_interval_s = poll_interval_s
        self.reconcile_interval_s = reconcile_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tokens: dict[str, CancelToken] = {}
        self._tokens_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        with SqliteCmrStore(self.db_path) as store:
            reconciled = store.reconcile_stale_analysis_jobs(
                now=self.clock(), stale_after_s=RUNNING_STALE_S
            )
            reconciled += store.reconcile_stale_inference_jobs(
                now=self.clock(), stale_after_s=RUNNING_STALE_S
            )
        if reconciled:
            LOGGER.warning("reconciled %d orphaned analysis jobs", reconciled)
        self._thread = threading.Thread(
            target=self._run,
            name="cmr-analysis-jobs",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        """Request shutdown and return whether the worker actually stopped.

        A timed-out join leaves ``_thread`` intact, so ``is_running`` remains
        honest and ``start`` cannot create a second worker. Active computations
        receive cancellation requests before the join. A glue call is not
        interruptible: a late stop marks the persisted job cancelled, then the
        provider is allowed to return and its artifact identities are retained
        as terminal audit fields.
        """
        self._stop.set()
        with self._tokens_lock:
            for token in self._tokens.values():
                token.request()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        if thread.is_alive():
            LOGGER.error("analysis job worker did not stop within %.3fs", timeout)
            return False
        self._thread = None
        return True

    def cancel(self, job_id: str, *, series_uid: str) -> AnalysisJobRecord | InferenceJobRecord:
        """Kind-routed cancel (A6) through the existing route.

        Analysis ids cancel via the analysis path (unchanged); inference
        ids fall back to the inference path. The store mutation is scoped
        by ``series_uid`` FIRST — a wrong-series id raises ``KeyError``
        before any token is requested (``request_cancel_token`` no-ops on
        unknown ids), so neither row nor token changes.
        """
        with SqliteCmrStore(self.db_path) as store:
            try:
                record: AnalysisJobRecord | InferenceJobRecord = store.cancel_analysis_job(
                    job_id, updated_at=self.clock(), series_uid=series_uid
                )
            except KeyError:
                record = store.cancel_inference_job(
                    job_id, updated_at=self.clock(), series_uid=series_uid
                )
        self.request_cancel_token(job_id)
        return record

    def request_cancel_token(self, job_id: str) -> None:
        """Request in-memory cancellation after the scoped store mutation."""
        with self._tokens_lock:
            token = self._tokens.get(job_id)
            if token is not None:
                token.request()

    def _run(self) -> None:
        last_reconcile = 0.0
        while not self._stop.is_set():
            did_work = False
            try:
                with SqliteCmrStore(self.db_path) as store:
                    job = store.claim_next_queued_analysis_job(
                        now=self.clock(), debounce_s=self.debounce_s
                    )
                if job is not None:
                    did_work = True
                    self._execute(job)
                with SqliteCmrStore(self.db_path) as store:
                    inference_job = store.claim_next_queued_inference_job(now=self.clock())
                if inference_job is not None:
                    did_work = True
                    self._execute_inference(inference_job)
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                # Path-safe (F5): type + stable code, never raw text/traceback
                # (a store exception can spell host paths).
                LOGGER.warning(
                    "analysis job worker iteration failed: %s",
                    _typed_diagnostic(exc, "worker_iteration_failed"),
                )
            # R2 review correction: reconcile orphaned running jobs
            # periodically, not only at startup — a job orphaned just before
            # a restart is young at startup and would otherwise stay running
            # forever. Jobs owned by THIS live worker (live tokens) are
            # excluded, so active computation is protected.
            now_mono = time.monotonic()
            if now_mono - last_reconcile >= self.reconcile_interval_s:
                last_reconcile = now_mono
                self._reconcile_orphans()
            if not did_work:
                self._stop.wait(self.poll_interval_s)

    def _reconcile_orphans(self) -> None:
        """Production reconcile pass (startup + periodic).

        Marks running rows older than ``RUNNING_STALE_S`` terminal
        (``failed/interrupted``), EXCLUDING job ids with a live in-memory
        token — an active computation (slow CPU inference can exceed
        600 s) must never be reconciled while its worker owns it.
        """
        try:
            with self._tokens_lock:
                live_ids = tuple(self._tokens.keys())
            with SqliteCmrStore(self.db_path) as store:
                store.reconcile_stale_analysis_jobs(
                    now=self.clock(),
                    stale_after_s=RUNNING_STALE_S,
                    exclude_job_ids=live_ids,
                )
                store.reconcile_stale_inference_jobs(
                    now=self.clock(),
                    stale_after_s=RUNNING_STALE_S,
                    exclude_job_ids=live_ids,
                )
        except Exception as exc:  # pragma: no cover - defensive
            # Path-safe (F5): type + stable code only.
            LOGGER.warning(
                "analysis job reconcile pass failed: %s",
                _typed_diagnostic(exc, "reconcile_failed"),
            )

    def _execute(self, job: AnalysisJobRecord) -> None:
        # The production loop passes a claimed running record.  Claim here too
        # when tests or a direct caller provide a queued record, so this method
        # cannot bypass the lifecycle contract.
        with SqliteCmrStore(self.db_path) as state_store:
            current = next(
                (
                    item
                    for item in state_store.load_analysis_jobs(job.series_uid)
                    if item.job_id == job.job_id
                ),
                job,
            )
            if current.state == "queued":
                claimed = state_store.claim_next_queued_analysis_job(
                    series_uid=job.series_uid, now=self.clock(), debounce_s=0.0
                )
                if claimed is None:
                    return
                job = claimed
        token = CancelToken()
        with self._tokens_lock:
            self._tokens[job.job_id] = token
        timings = _timings(job.timings)
        try:
            with SqliteCmrStore(self.db_path) as state_store:
                current = next(
                    (item for item in state_store.load_analysis_jobs(job.series_uid)
                     if item.job_id == job.job_id),
                    job,
                )
            if current.state == "cancelled":
                return
            if self.compute is not None:
                measurement = self.compute_with_cancel(job, token)
            else:
                measurement = self._compute_selected_phase(job, token, timings)
            if token.is_requested:
                self._finish(job, "cancelled", "cancelled", None, timings)
                return
            publish_start = time.perf_counter()
            head_id, selection_id, inclusion_id = _lineage_payload(job)
            timings["publish_s"] = _seconds(publish_start)
            timings["finished_at"] = self.clock()
            with SqliteCmrStore(self.db_path) as store:
                store.save_selected_phase_analysis_checked(
                    measurement,
                    expected_head_revision_id=head_id,
                    expected_selection_id=selection_id,
                    expected_inclusion_set_id=inclusion_id,
                    expected_n_slices=len(measurement.stack_gaps_mm),
                    expected_job_id=job.job_id,
                    job_timings=timings,
                    job_updated_at=self.clock(),
                )
        except JobNotPublishableError:
            # Cancellation won the publication transaction.  Do not attempt a
            # terminal write that could overwrite the cancelled outcome.
            LOGGER.info("analysis job %s was no longer publishable", job.job_id)
        except AnalysisAuthorityError:
            # F14: a job that cannot produce the coherent metrics+QC authority
            # fails visibly with a stable code and publishes NOTHING (the
            # store transaction rolled back; no measurement, no binding).
            # Checked BEFORE the parent StaleHeadError on purpose.
            self._finish(
                job, "failed", "analysis_authority_unavailable", None, timings
            )
        except StaleHeadError:
            self._finish(job, "failed", "stale_lineage", None, timings)
        except _Cancelled:
            self._finish(job, "cancelled", "cancelled", None, timings)
        except Exception:
            if token.is_requested:
                self._finish(job, "cancelled", "cancelled", None, timings)
            else:
                LOGGER.debug("analysis job %s failed", job.job_id, exc_info=True)
                self._finish(job, "failed", "store_error", None, timings)
        finally:
            with self._tokens_lock:
                self._tokens.pop(job.job_id, None)

    def _execute_inference(self, job: InferenceJobRecord) -> None:
        # Lifecycle parity with _execute: the production loop passes a
        # claimed running record, but a direct caller may hand a queued
        # record.  Claim here too, so THIS method cannot finish a queued row
        # directly (the F3 queued->succeeded violation).  A queued row that
        # cannot be claimed (lost race) returns without model work.
        with SqliteCmrStore(self.db_path) as state_store:
            current = next(
                (
                    item
                    for item in state_store.load_inference_jobs(job.series_uid)
                    if item.job_id == job.job_id
                ),
                job,
            )
            if current.state == "queued":
                claimed = state_store.claim_next_queued_inference_job(now=self.clock())
                if claimed is None or claimed.job_id != job.job_id:
                    return
                job = claimed
                current = claimed
        # Token registration FIRST (C3/F3): a stop()/cancel request that
        # targets this job must be observed by the claim-boundary check
        # below. The registered token is the claim-time token — a caller
        # pre-registered token that is already requested is preserved.
        with self._tokens_lock:
            token = self._tokens.get(job.job_id)
            if token is None:
                token = CancelToken()
                self._tokens[job.job_id] = token
        timings = _inference_timings(job.timings)
        try:
            with SqliteCmrStore(self.db_path) as store:
                current = next(
                    (
                        item
                        for item in store.load_inference_jobs(job.series_uid)
                        if item.job_id == job.job_id
                    ),
                    job,
                )
            # Claim boundary: a queued cancel (or a pre-registered token)
            # terminalizes BEFORE any model work. Late cancel after the glue
            # starts cannot stop the model — see _compute docstring.
            # C3/F3: prefer the atomic running->cancelled write (keeps the
            # first pre-call diagnostic instead of overwriting it with
            # cancelled_before_start) and fall back to the retrying writer.
            if current.state == "cancelled" or token.is_requested:
                try:
                    with SqliteCmrStore(self.db_path) as store:
                        store.finish_inference_job(
                            job.job_id, state="cancelled",
                            error_code="cancelled_before_start",
                            timings=timings, updated_at=self.clock(),
                        )
                except ValueError:
                    # Row already terminal (a cancel raced the claim): keep
                    # the existing terminal state, which already records the
                    # cancellation. There is no outcome to preserve here —
                    # no glue has run.
                    pass
                return
            # C3/F2 single prepared-gate call: preparation, torch-free
            # verification, the final coherent lineage/live-row/token gate
            # and compute invocation are ONE indivisible boundary — no store
            # may be opened between the gate and the glue call. Fake and
            # real paths go through the SAME method (only the final compute
            # callable differs), so a stale/cancelled/unverifiable job
            # terminalizes with NO glue call on both paths.
            prepared_store, outcome = self._run_inference_once(
                current, token, timings
            )
            # Post-start: the glue ran to completion and persisted its rows
            # into prepared_store (or the fake observed its call). Finalize
            # ATOMICALLY against the authoritative row (C3/F3): the success
            # decision still re-reads coherent lineage/state/token AFTER the
            # glue (a stale completion never publishes as succeeded), and
            # the single finalize transaction either finishes the running
            # row or attach-once-audits an already-terminal row — audit ids
            # are never dropped, cancellation is never resurrected.
            try:
                decision = self._post_glue_decision(current, token, timings)
            except _Cancelled:
                decision = "cancelled"
            except StaleHeadError:
                decision = "stale_lineage"
            with SqliteCmrStore(self.db_path) as store:
                store.finalize_inference_outcome(
                    job.job_id,
                    prediction_id=outcome.prediction_id,
                    model_run_id=outcome.model_run_id,
                    timings=timings,
                    updated_at=self.clock(),
                    decision=decision,
                )
            prepared_store.close()
            return
        except _Cancelled:
            self._finish_inference(
                job, "cancelled", "cancelled_before_start", None, None, timings
            )
        except StaleHeadError:
            self._finish_inference(job, "failed", "stale_lineage", None, None, timings)
        except Exception as exc:
            from cmr_core.errors import CMRError as _CMRError

            if isinstance(exc, _CMRError) and exc.code == "verification_failed":
                self._finish_inference(
                    job, "failed", "verification_failed", None, None, timings
                )
            elif token.is_requested:
                # Late failure after a cancel: the row may already be
                # terminal (cancel won the transaction).  Preserve the job
                # as cancelled and attach nothing new — never resurrect.
                with SqliteCmrStore(self.db_path) as store:
                    terminal = next(
                        (
                            item
                            for item in store.load_inference_jobs(job.series_uid)
                            if item.job_id == job.job_id
                        ),
                        None,
                    )
                if terminal is not None and terminal.state == "running":
                    self._finish_inference(
                        job, "cancelled", "cancelled", None, None, timings
                    )
            else:
                LOGGER.debug(
                    "inference job %s failed: %s",
                    job.job_id,
                    _inference_diagnostic(exc),
                )
                self._finish_inference(
                    job, "failed", "execution_failed", None, None, timings
                )
        finally:
            with self._tokens_lock:
                self._tokens.pop(job.job_id, None)

    def _run_inference_once(
        self, job: InferenceJobRecord, token: CancelToken, timings: dict[str, Any]
    ) -> tuple[Any, Any]:
        """Single prepared-gate inference boundary (C3/F2).

        Acquire the compute store ONCE, lazily bind the compute callable
        (real glue import + pinned checkpoint-path resolution, or the
        injected fake), torch-free-verify the pinned checkpoints, run the
        final coherent lineage/live-row/token gate — then invoke compute
        with the SAME store, with NO further store initialization between
        the gate and the call. Returns ``(store, outcome)``; the store is
        left OPEN for the caller (it holds the persisted glue rows and
        must be closed after finalization). Raises ``StaleHeadError``,
        ``_Cancelled`` or ``CMRError(verification_failed)`` with NO glue
        call on any stale/cancelled/unverifiable boundary.
        """
        from inference.backends.contract import CancelToken as _CancelToken

        _ = _CancelToken
        selected_frames = [int(frame) for frame in json.loads(job.selected_frames)]
        cine = self.cine_by_uid[job.series_uid]
        inference_store = SqliteCmrStore(self.db_path)
        try:
            if self.inference_compute is not None:
                def compute(store: Any, current: InferenceJobRecord) -> Any:
                    return self.inference_compute(store, current, token, timings)
            else:
                from inference.providers.baai.x1_glue import (
                    run_x1_precontour_selected as _glue,
                )

                checkpoint_root = self.x1_weights_root
                if checkpoint_root is not None:
                    from inference.providers.baai.artifacts import (
                        STAGE1_PATH,
                        STAGE2_PATH,
                    )

                    stage1_checkpoint = str(checkpoint_root / STAGE1_PATH)
                    stage2_checkpoint = str(checkpoint_root / STAGE2_PATH)
                else:
                    stage1_checkpoint = job.stage1_checkpoint
                    stage2_checkpoint = job.stage2_checkpoint

                def compute(store: Any, current: InferenceJobRecord) -> Any:
                    resolution_started = time.perf_counter()
                    timings["runtime_resolution_s"] = _seconds(resolution_started)
                    started = time.perf_counter()
                    outcome = _glue(
                        store,
                        cine,
                        stage1_checkpoint,
                        stage2_checkpoint,
                        selected_frames=selected_frames,
                        device=job.device,
                        clock=self.clock,
                        persist=True,
                        checkpoint_provenance={
                            "stage1_artifact_path": "cine_seg_first_SA/Cine_SAX_seg1.pth",
                            "stage2_artifact_path": "cine_seg_second_SA/Cine_SAX_seg2.pth",
                        },
                    )
                    timings["inference_s"] = _seconds(started)
                    timings["runtime_resolution_s"] = timings.get("runtime_resolution_s")
                    return outcome

            # Torch-free checkpoint verification FIRST (A5/C3), then ONE
            # coherent gate on the SAME store immediately before compute.
            self._verify_inference_checkpoints()
            self._final_lineage_gate(inference_store, job, token)
            outcome = compute(inference_store, job)
            return inference_store, outcome
        except BaseException:
            inference_store.close()
            raise

    def _final_lineage_gate(
        self, store: SqliteCmrStore, job: InferenceJobRecord, token: CancelToken
    ) -> None:
        """ONE coherent pre-call gate (C3/F2) on the caller's open store.

        Re-reads the CURRENT head/selection vs the job's bound payload, the
        live job row, and the token — after ALL preparation (verification,
        import, path resolution) and immediately before the compute call.
        ``StaleHeadError``, ``_Cancelled`` or nothing. The caller must not
        open another store between this gate and the invocation.
        """
        payload = json.loads(job.request_payload)
        cine = self.cine_by_uid[job.series_uid]
        state = self.controller.reopen(store, job.series_uid, cine=cine)
        selection = state.get("latest_selection")
        head = state.get("chain", [])
        current_head = None if not head else head[-1].revision_id
        if selection is None:
            raise StaleHeadError("selected-phase inference requires a current selection")
        if (
            payload.get("head_revision_id") != current_head
            or payload.get("selection_id") != selection.selection_id
            or payload.get("ed_frame") != selection.ed_frame
            or payload.get("es_frame") != selection.es_frame
        ):
            raise StaleHeadError("selected-phase inference request lineage is stale")
        live = next(
            (
                item
                for item in store.load_inference_jobs(job.series_uid)
                if item.job_id == job.job_id
            ),
            None,
        )
        if live is not None and live.state == "cancelled":
            raise _Cancelled
        if token.is_requested:
            raise _Cancelled

    def _post_glue_decision(
        self, job: InferenceJobRecord, token: CancelToken, timings: dict[str, Any]
    ) -> str | None:
        """Re-read coherent lineage/state/token AFTER the glue (C3/F3).

        Returns ``None`` for success, ``"cancelled"`` or raises
        ``StaleHeadError``. Uses a FRESH read connection; the decision is
        consumed by the single atomic ``finalize_inference_outcome``
        transaction, so a stale completion can never publish as succeeded.
        """
        if token.is_requested:
            return "cancelled"
        payload = json.loads(job.request_payload)
        cine = self.cine_by_uid[job.series_uid]
        with SqliteCmrStore(self.db_path) as store:
            state = self.controller.reopen(store, job.series_uid, cine=cine)
            selection = state.get("latest_selection")
            head = state.get("chain", [])
            current_head = None if not head else head[-1].revision_id
            if selection is None:
                raise StaleHeadError("selected-phase inference requires a current selection")
            if (
                payload.get("head_revision_id") != current_head
                or payload.get("selection_id") != selection.selection_id
                or payload.get("ed_frame") != selection.ed_frame
                or payload.get("es_frame") != selection.es_frame
            ):
                raise StaleHeadError("selected-phase inference request lineage is stale")
            live = next(
                (
                    item
                    for item in store.load_inference_jobs(job.series_uid)
                    if item.job_id == job.job_id
                ),
                None,
            )
            if live is not None and live.state == "cancelled":
                return "cancelled"
            if token.is_requested:
                return "cancelled"
        timings["publish_s"] = _seconds(time.perf_counter())
        return None

    def _inference_preflight(self, job: InferenceJobRecord, token: CancelToken) -> None:
        """Torch-free verification FIRST, then ONE atomic pre-call check (F2).

        Order is the correction: the two pinned checkpoints are read+hashed
        (``_verify_inference_checkpoints``) BEFORE any final boundary check,
        because verification is potentially slow and a cancel/lineage change
        may land while it runs. Immediately after verification ONE fresh,
        coherent check re-reads the current head/selection vs payload, the
        live job row and the token — this is the single atomic gate before
        glue invocation. Raises ``StaleHeadError``, ``_Cancelled`` or
        ``CMRError(verification_failed)`` identically for both the real glue
        path and the ``inference_compute`` fake seam, so
        ``_execute_inference`` terminalizes with NO glue call.
        """
        self._verify_inference_checkpoints()
        payload = json.loads(job.request_payload)
        cine = self.cine_by_uid[job.series_uid]
        with SqliteCmrStore(self.db_path) as store:
            state = self.controller.reopen(store, job.series_uid, cine=cine)
            selection = state.get("latest_selection")
            head = state.get("chain", [])
            current_head = None if not head else head[-1].revision_id
            if selection is None:
                raise StaleHeadError("selected-phase inference requires a current selection")
            if (
                payload.get("head_revision_id") != current_head
                or payload.get("selection_id") != selection.selection_id
                or payload.get("ed_frame") != selection.ed_frame
                or payload.get("es_frame") != selection.es_frame
            ):
                raise StaleHeadError("selected-phase inference request lineage is stale")
            live = next(
                (
                    item
                    for item in store.load_inference_jobs(job.series_uid)
                    if item.job_id == job.job_id
                ),
                None,
            )
            if live is not None and live.state == "cancelled":
                raise _Cancelled
            if token.is_requested:
                raise _Cancelled

    def _retired_finish_post_verification(
        self,
        job: InferenceJobRecord,
        token: CancelToken,
        outcome: Any,
        timings: dict[str, Any],
        started: float,
    ) -> None:
        """RETIRED in C3 (superseded by ``_post_glue_decision`` +
        ``finalize_inference_outcome``). Kept as a named tombstone so any
        lingering caller fails loudly instead of silently changing
        semantics. DO NOT CALL: the old split read-then-finish path left
        the database-first cancellation window open (R2 F3 residual).

        Verify-then-gate: fresh lineage/state/token check before success.

        The checkpoint files were read+hashed during the pre-flight (or the
        real-path compute) BEFORE this point.  Verification is potentially
        slow, so a cancel or a head/selection change may have landed while
        it ran.  This single atomic gate re-reads the coherent lineage, the
        live job row and the token AFTER verification and immediately
        before the success transition: a cancelled/stale job terminalizes
        with NO glue call (the recording fake observes zero calls) and NO
        success transition — ``cancelled/cancelled_before_start`` for a
        cancel, ``failed/stale_lineage`` for a lineage change.
        """
        raise NotImplementedError(
            "_finish_post_verification is retired; "
            "use _post_glue_decision + finalize_inference_outcome"
        )

    def _compute_selected_phase_inference(
        self, job: InferenceJobRecord, token: CancelToken, timings: dict[str, Any]
    ) -> Any:
        """RETIRED in C3 (superseded by ``_run_inference_once``). Kept as a
        named tombstone so any lingering caller fails loudly. DO NOT CALL:
        the old split pre-flight-then-separate-store path left the R2 F2
        residual window open (store opened between the gate and the call).
        """
        raise NotImplementedError(
            "_compute_selected_phase_inference is retired; use _run_inference_once"
        )

    def _verify_inference_checkpoints(self) -> None:
        """Torch-free re-verification of both pinned files (A5).

        Reads the two checkpoint files under the configured weights root and
        verifies exact size + sha256 via the artifacts manifest — BEFORE any
        glue call and WITHOUT importing torch. Raises ``CMRError`` on any
        failure; the caller maps it to terminal
        ``failed/verification_failed``.
        """
        from cmr_core.errors import CMRError

        root = self.x1_weights_root
        if root is None:
            raise CMRError("verification_failed", "pinned X1 weights root is not configured")
        try:
            from inference.providers.baai.artifacts import (
                STAGE1_PATH,
                STAGE2_PATH,
                verify_both,
            )
        except Exception as exc:
            raise CMRError("verification_failed", "pinned X1 weights failed verification") from exc
        stage1 = root / STAGE1_PATH
        stage2 = root / STAGE2_PATH
        try:
            if not stage1.is_file() or not stage2.is_file():
                raise ValueError("one or more pinned X1 checkpoints are missing")
            verify_both(
                {STAGE1_PATH: stage1.read_bytes(), STAGE2_PATH: stage2.read_bytes()}
            )
        except Exception as exc:
            if isinstance(exc, CMRError) and exc.code == "verification_failed":
                raise
            raise CMRError(
                "verification_failed", "pinned X1 weights failed verification"
            ) from exc

    def _finish_inference(
        self,
        job: InferenceJobRecord,
        state: str,
        error_code: str | None,
        prediction_id: str | None,
        model_run_id: str | None,
        timings: dict[str, Any],
    ) -> None:
        timings["finished_at"] = self.clock()
        last_error: Exception | None = None
        for attempt in range(TERMINAL_WRITE_RETRIES):
            try:
                with SqliteCmrStore(self.db_path) as store:
                    store.finish_inference_job(
                        job.job_id,
                        state=state,
                        error_code=error_code,
                        prediction_id=prediction_id,
                        model_run_id=model_run_id,
                        timings=timings,
                        updated_at=self.clock(),
                    )
                return
            except Exception as exc:
                last_error = exc
                if attempt + 1 < TERMINAL_WRITE_RETRIES:
                    time.sleep(TERMINAL_WRITE_BACKOFF_S)
        LOGGER.critical(
            "inference job %s terminal write failed after %d attempts: %s",
            job.job_id,
            TERMINAL_WRITE_RETRIES,
            _inference_terminal_diagnostic(last_error)
            if last_error is not None
            else "unknown:inference_terminal_write_failed",
        )

    def compute_with_cancel(self, job: AnalysisJobRecord, token: CancelToken) -> Any:
        assert self.compute is not None
        store = SqliteCmrStore(self.db_path)
        try:
            return self.compute(store, job, token)
        finally:
            store.close()

    def _compute_selected_phase(
        self, job: AnalysisJobRecord, token: CancelToken, timings: dict[str, Any]
    ) -> Any:
        payload = json.loads(job.request_payload)
        cine = self.cine_by_uid[job.series_uid]
        selection_frame = payload["coverage"]
        ed_frame = int(selection_frame["ed_frame"])
        es_frame = int(selection_frame["es_frame"])
        started = time.perf_counter()
        with SqliteCmrStore(self.db_path) as store:
            state = self.controller.reopen(store, job.series_uid, cine=cine)
            selection = state["latest_selection"]
            chain = list(state["chain"])
            inclusion_record: SliceInclusionSet | None = state.get("applicable_inclusion_set")
            if selection is None or not chain:
                raise ValueError("selected-phase recompute requires a chain and selection")
            if selection.ed_frame != ed_frame or selection.es_frame != es_frame:
                raise StaleHeadError("selected-phase request selection is no longer current")
            coverage = resolve_selected_phase_coverage(  # noqa: F841 - kept as a coherence read; the gate below resolves chain+waiver
                store, job.series_uid, ed_frame, es_frame, slice_count=cine.n_slices
            )
            # C1 (P1-1): the coverage gate is waiver-aware. The resolver only
            # reads AI/physician coverage artifacts — it never reads the
            # legacy contour-revision chain, so a physician-drawn chain
            # (the seeded-study and stack-applicability shapes) resolves as
            # unknown_grid + all-missing. What the gate must refuse is a key
            # that is missing from BOTH the chain geometry AND the waiver
            # set. A key absent from the chain but explicitly waived by the
            # CURRENT applicable inclusion record is non-applicable, not
            # missing. Keys with neither geometry NOR waiver stay missing
            # (fail-closed).
            waived_here: set[tuple[int, int, Any]] = set()
            policy_for_gate = None
            if inclusion_record is not None:
                policy_for_gate = self.controller.slice_inclusion_policy_from_record(
                    inclusion_record,
                    ed_frame=selection.ed_frame,
                    es_frame=selection.es_frame,
                    n_slices=cine.n_slices,
                )
                waived_keys = applicability_waivers(policy_for_gate)
                from cmr_core.segmentation import Structure as _Structure

                _role_frame = {"ED": ed_frame, "ES": es_frame}
                for slice_index, role, structure_value in waived_keys:
                    frame = _role_frame.get(role)
                    if frame is None:
                        continue
                    try:
                        waived_here.add(
                            (slice_index, frame, _Structure(structure_value))
                        )
                    except ValueError:
                        continue
            chain_index_keys: set[tuple[int, int, Any]] = set()
            try:
                chain_index_keys = set(
                    build_cine_contour_index(chain, job.series_uid).contours.keys()
                )
            except Exception:
                chain_index_keys = set()
            required_here = set(
                gate_required_keys(cine.n_slices, ed_frame, es_frame)
            )
            uncovered = sorted(
                (
                    key
                    for key in required_here
                    if key not in chain_index_keys and key not in waived_here
                ),
                key=lambda k: (k[0], k[1], k[2].value),
            )
            if uncovered:
                raise ValueError("selected-phase coverage is incomplete")
            index = build_cine_contour_index(chain, job.series_uid)
            timings["coverage_resolution_s"] = _seconds(started)
            if token.is_requested:
                raise _Cancelled
            qc_started = time.perf_counter()
            # F14 (CMR_FUNCTION_V1_ANALYSIS_001): the server-side recompute
            # must publish the SAME coherent analysis authority the explicit
            # inclusion refresh publishes: metrics AND the QC report derived
            # from the exact same chain head + selection. The QC is
            # RE-DERIVED here with the same frozen primitive the refresh path
            # uses and verified byte-for-byte against the persisted head's
            # report — the job never writes a second QC authority and never
            # publishes metrics whose QC cannot be recovered from the head.
            # C1 (R1 P1b): keep SELECTED-PHASE semantics for both metrics
            # and QC. The full-cycle volume curve + run_revision_qc demanded
            # LV cavity contours at EVERY frame (KeyError on intermediate
            # frames the sparse artifact never provides); run_selected_phase_qc
            # evaluates the same frozen primitives over the five required
            # groups at the physician-authored ED/ES frames only.
            # C1 (P1-1): the recompute consumes the SAME policy the refresh
            # bound — the QC re-derivation carries it, so a waived-absent
            # key verifies against the head report instead of flagging.
            qc = run_selected_phase_qc(
                index, cine, selection, inclusion_policy=policy_for_gate
            )
            timings["qc_s"] = _seconds(qc_started)
            if token.is_requested:
                raise _Cancelled
            head_qc = chain[-1].qc_report
            if head_qc is None or canonical_digest(head_qc) != canonical_digest(qc):
                raise AnalysisAuthorityError(
                    "the current head revision does not carry the QC report "
                    "derived from the exact same head/selection state as the "
                    "selected-phase metrics; apply the confirmed inclusion grid "
                    "to refresh the analysis before the server-side recompute "
                    "can publish"
                )
            policy = policy_for_gate
            quant_started = time.perf_counter()
            issues = IssueSet()
            measurement = compute_selected_phase_measurement(
                index,
                selection,
                cine,
                software_version=str(payload["software_version"]),
                algorithm_version=(
                    ALGORITHM_VERSION_SINGLE_SLICE_SELECTED_PHASES
                    if cine.n_slices == 1
                    else ALGORITHM_VERSION_IPP_SELECTED_PHASES
                ),
                created_at=self.clock(),
                issues=issues,
                inclusion_policy=policy,
                range_table_version=str(payload["range_table_version"]),
            )
            timings["quantification_s"] = _seconds(quant_started)
            # F14: the verified QC above is the head's own report, so the
            # measurement is published through the store's coherent analysis
            # binding (metrics + QC + exact inputs); no second QC authority is
            # written here.
            return replace(
                measurement,
                selection_id=selection.selection_id,
                inclusion_set_id=(
                    None if inclusion_record is None else inclusion_record.inclusion_set_id
                ),
            )

    def _finish(
        self,
        job: AnalysisJobRecord,
        state: str,
        error_code: str | None,
        artifact: str | None,
        timings: dict[str, Any],
    ) -> None:
        timings["finished_at"] = self.clock()
        last_error: Exception | None = None
        for attempt in range(TERMINAL_WRITE_RETRIES):
            try:
                with SqliteCmrStore(self.db_path) as store:
                    store.finish_analysis_job(
                        job.job_id,
                        state=state,
                        error_code=error_code,
                        artifact_measurement_id=artifact,
                        timings=timings,
                        updated_at=self.clock(),
                    )
                return
            except Exception as exc:  # bounded retry for ordinary store outages
                last_error = exc
                if attempt + 1 < TERMINAL_WRITE_RETRIES:
                    time.sleep(TERMINAL_WRITE_BACKOFF_S)
        LOGGER.critical(
            "analysis job %s terminal write failed after %d attempts: %s",
            job.job_id,
            TERMINAL_WRITE_RETRIES,
            last_error,
        )


class _Cancelled(Exception):
    """Internal cancellation sentinel."""


__all__ = [
    "JOB_POLL_INTERVAL_S",
    "JobRunner",
    "RECOMPUTE_DEBOUNCE_S",
    "RECONCILE_INTERVAL_S",
    "RUNNING_STALE_S",
    "enqueue_selected_phase_recompute",
    "request_identity",
]
