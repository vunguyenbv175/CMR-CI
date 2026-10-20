"""Node 8 — ViewerApp: thin, testable composition over accepted primitives.

``ViewerApp`` wires an immutable :class:`~cmr_viewer.state.ViewerState`, the
deterministic :class:`~cmr_viewer.renderer.ViewerRenderer`, the pure
:class:`~cmr_viewer.playback.PlayheadController` and INJECTED workflow
dependencies (controller / store / clock / physician identity) into the
physician-review workflows. It is a THIN composition layer:

- it holds NO clinical logic, computes NO metrics and makes NO approval
  decisions — every mutation delegates to ``WorkflowController`` /
  ``EditCommandStack`` / store primitives;
- all dependencies are injected (AMENDMENT-001 H): tests supply fixed-clock
  controllers and stores; nothing here touches wall-clock time;
- dependencies are typed STRUCTURALLY (:class:`WorkflowControllerLike`,
  :class:`CmrStoreLike`) so this module never imports ``cmr_workflow`` at
  package-import time — the negative import-boundary gate stays honest even
  on machines where ``urllib`` is not already resident (see the one
  function-level exception below);
- failures NEVER escape to the console: every flow catches
  :class:`~cmr_core.errors.CMRError` (and store-side ``ValueError``
  subclasses) and surfaces ``"ERROR: <message>"`` in :attr:`status` while
  leaving the viewed state UNCHANGED.

Current-head-only policy (AMENDMENT-001 H): ``save_flow`` /
``approve_flow`` re-open the store BEFORE acting and verify that the
persisted chain head still equals the head this view was built from. A
mismatch is STALE STATE: visible error label, unchanged state, no write.

QC-report policy (AMENDMENT-001 B, documented choice): the state's
``qc_report`` is CARRIED EXPLICITLY from the chain head's own N6 gateway
field (``ContourRevision.qc_report``, snapshotted by the QC-only child
revision flow). The viewer NEVER recomputes QC silently, so what is
displayed is exactly what was sealed onto the chain — never silently stale.

Sentinel path (N7.1 B-LITE): approving a REVIEW REQUIRED head whose series
has no Measurement seals a durable decision record through the controller's
own policy. The sentinel measurement-id constant is imported lazily INSIDE
that branch (single source of truth stays ``cmr_workflow.controller``);
package import itself never pulls it in.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, Protocol

from cmr_contours.editing import EditCommand, EditCommandStack, InsertVertex
from cmr_contours.model import Contour
from cmr_core.cine import CineVolume
from cmr_core.domain_types import (
    ApprovalSnapshot,
    ContourRevision,
    EDESSelection,
    Measurement,
    Prediction,
)
from cmr_core.errors import INVALID_ARGUMENT, CMRError
from cmr_core.provenance import VERSION, Provenance
from cmr_core.segmentation import Structure
from cmr_qc.report import QCReport
from cmr_quantification.measurements import (
    PostInclusionEsvExceedsEdvError,
)

from cmr_viewer.playback import PlayheadController
from cmr_viewer.state import (
    ANALYSIS_UNAVAILABLE_ESV_EXCEEDS_EDV,
    DEFAULT_FPS,
    INCLUSION_GRID_GROUPS,
    InclusionToggle,
    ViewerState,
    all_included_toggles,
    toggles_of_inclusion_set,
)

__all__ = [
    "CmrStoreLike",
    "ViewerApp",
    "WorkflowControllerLike",
    "reopen_viewer_state",
]

#: Zero-argument factory producing deterministic ISO-8601 UTC stamps.
Clock = Callable[[], str]

#: Zero-argument factory producing the provenance of a manual ED/ES pick.
ProvenanceFactory = Callable[[], Provenance]

EditKey = tuple[int, int, Structure]


class WorkflowControllerLike(Protocol):
    """Structural surface of ``WorkflowController`` used by the viewer."""

    def select_ed_es(
        self,
        cine: CineVolume,
        ed_frame: int,
        es_frame: int,
        selected_by: str,
        created_at: str,
        provenance_factory: ProvenanceFactory,
    ) -> EDESSelection: ...

    def record_ed_es(
        self,
        store: Any,
        cine: CineVolume,
        ed_frame: int,
        es_frame: int,
        selected_by: str,
        created_at: str,
        provenance_factory: ProvenanceFactory,
    ) -> EDESSelection:
        """S2C-001 §8.3: selections persist ONLY through the controller."""
        ...

    def edit_contour(
        self, stack: EditCommandStack, command: EditCommand
    ) -> Contour: ...

    def undo(self, stack: EditCommandStack) -> Contour: ...

    def redo(self, stack: EditCommandStack) -> Contour: ...

    def save(
        self,
        store: Any,
        parent: Prediction | ContourRevision,
        edits: Mapping[Any, Contour],
        editor: str,
        created_at: str,
        *,
        software_version: str = VERSION,
        qc_report: QCReport | None = None,
        measurement: Measurement | None = None,
    ) -> ContourRevision: ...

    def reopen(
        self, store: Any, series_uid: str, *, cine: CineVolume | None = None
    ) -> dict[str, Any]: ...

    def refresh_head_analysis(
        self,
        store: Any,
        cine: CineVolume,
        series_uid: str,
        *,
        software_version: str = VERSION,
        created_at: str,
    ) -> dict[str, Any]: ...

    def approve(
        self,
        store: Any,
        revision_id: str,
        measurement_id: str,
        qc_report: QCReport,
        approved_by: str,
        approved_at: str,
        *,
        measurement: Measurement | None = None,
        selection: EDESSelection | None = None,
        override: bool = False,
        override_reason: str | None = None,
    ) -> ApprovalSnapshot: ...


class CmrStoreLike(Protocol):
    """Structural store surface used directly by the viewer.

    S2C-001 §8.3: the viewer NEVER calls ``store.save_edes_selection`` —
    selection persistence is routed through
    ``WorkflowController.record_ed_es`` — so this protocol carries only what
    the app itself touches.
    """


def reopen_viewer_state(
    controller: WorkflowControllerLike,
    store: CmrStoreLike,
    series_uid: str,
    cine: CineVolume,
    *,
    fps: int = DEFAULT_FPS,
    current_slice_index: int = 0,
    current_frame_index: int = 0,
    playing: bool = False,
    draft_contours: tuple[Contour, ...] = (),
) -> ViewerState:
    """Rebuild a :class:`ViewerState` from ``controller.reopen()`` plus cine.

    Persistence stores NO pixels by design (AMENDMENT-001 B), so the
    RELOADED :class:`~cmr_core.cine.CineVolume` is an explicit argument.
    Correction F1: the cine travels into EVERY controller reopen so
    manifest-bound history can verify its source identity. The QC report is
    carried explicitly from the chain head's N6 gateway field (never
    recomputed, never silently stale — see module docstring).
    """
    data = controller.reopen(store, series_uid, cine=cine)
    chain = tuple(data["chain"])
    # CFM-001 §1.8: the draft initializes from the APPLICABLE persisted set
    # when one exists; otherwise every toggle starts CHECKED under the
    # disclosed legacy all-slices default (never a silent default decision).
    applicable_set = data.get("applicable_inclusion_set")
    if applicable_set is not None:
        inclusion_draft = toggles_of_inclusion_set(applicable_set)
    else:
        inclusion_draft = all_included_toggles(cine.n_slices)
    return ViewerState(
        series_uid=series_uid,
        cine=cine,
        current_slice_index=current_slice_index,
        current_frame_index=current_frame_index,
        selection=data["latest_selection"],
        latest_measurement=data["latest_measurement"],
        qc_report=chain[-1].qc_report if chain else None,
        approvals=tuple(data["approvals"]),
        qc_decisions=tuple(data["qc_decisions"]),
        revision_chain=chain,
        draft_contours=draft_contours,
        playing=playing,
        fps=fps,
        inclusion_draft=inclusion_draft,
        persisted_inclusion_set=applicable_set,
    )


class ViewerApp:
    """Application core behind both the interactive shell and the tests."""

    def __init__(
        self,
        *,
        controller: WorkflowControllerLike,
        store: CmrStoreLike,
        clock: Clock,
        physician: str,
        state: ViewerState,
    ) -> None:
        if not isinstance(physician, str) or not physician.strip():
            raise CMRError(
                INVALID_ARGUMENT,
                "physician identity must be a non-empty string",
            )
        self._controller = controller
        self._store = store
        self._clock = clock
        self._physician = physician
        self._state = state
        self._playhead = PlayheadController(fps=state.fps)
        self._playhead.bind(state.cine.n_frames, frame_index=state.current_frame_index)
        self._stacks: dict[EditKey, EditCommandStack] = {}
        self._active_key: EditKey | None = None
        self._status = "ready"
        self._status_is_error = False

    # --- construction ---------------------------------------------------------

    @classmethod
    def open(
        cls,
        *,
        controller: WorkflowControllerLike,
        store: CmrStoreLike,
        cine: CineVolume,
        series_uid: str,
        clock: Clock,
        physician: str,
        fps: int = DEFAULT_FPS,
    ) -> ViewerApp:
        """Open a fresh view over the persisted state of one series."""
        state = reopen_viewer_state(
            controller, store, series_uid, cine, fps=fps
        )
        return cls(
            controller=controller,
            store=store,
            clock=clock,
            physician=physician,
            state=state,
        )

    # --- introspection ----------------------------------------------------------

    @property
    def state(self) -> ViewerState:
        """Current immutable viewer state."""
        return self._state

    @property
    def status(self) -> str:
        """Last status/error message (errors prefixed ``ERROR: ``)."""
        return self._status

    @property
    def status_is_error(self) -> bool:
        """Whether :attr:`status` currently carries an error."""
        return self._status_is_error

    @property
    def physician(self) -> str:
        """Injected physician identity used for every persisted action."""
        return self._physician

    @property
    def playhead(self) -> PlayheadController:
        """The pure playhead session (shell reads ``interval_ms`` from it)."""
        return self._playhead

    @property
    def active_edit_key(self) -> EditKey | None:
        """Key of the most recently edited (unsaved) contour, else ``None``."""
        return self._active_key

    @property
    def active_stack(self) -> EditCommandStack | None:
        """Edit-command stack of :attr:`active_edit_key`, else ``None``."""
        return None if self._active_key is None else self._stacks.get(self._active_key)

    # --- internals ---------------------------------------------------------------

    def _set_status(self, message: str, *, is_error: bool = False) -> None:
        self._status = message
        self._status_is_error = is_error

    def _error(self, message: str) -> None:
        self._set_status(f"ERROR: {message}", is_error=True)

    def surface_exception(self, err: BaseException) -> None:
        """Surface an unexpected exception as a visible, non-committal error.

        Correction round 2 R2 final boundary: this message can NEVER claim a
        rollback or a completed write — an unexpected failure may occur before
        OR after a successful commit, so the physician is told to inspect the
        visible state and Reopen instead of trusting either conclusion.
        """
        self._error(
            f"unexpected {type(err).__name__}; no clinical conclusion can be "
            "drawn from this message — if a save/seal/refresh was in flight it "
            "may ALREADY be committed; check the visible state and press "
            "Reopen before retrying"
        )

    @staticmethod
    def _store_error_message(err: sqlite3.Error) -> str:
        """Retry-safe wording for raw store failures (correction F6)."""
        return (
            f"store error ({type(err).__name__}); the transaction rolled back "
            "and NOTHING was written — the store may be busy or locked; retry"
        )

    def _provenance_factory(self) -> Provenance:
        head = self._state.head_revision
        # Correction F1: selections carry the manifest-bound workflow of the
        # viewed lineage so downstream coherence checks stay one-workflow.
        return Provenance(
            study_uid=head.provenance.study_uid if head else None,
            series_uid=self._state.series_uid,
            software_version=VERSION,
            model=None,
            created_at=self._clock(),
            revision=head.provenance.revision if head else 0,
            workflow_id=head.provenance.workflow_id if head else None,
        )

    def _fresh_chain_head(self) -> ContourRevision | None:
        """Reopen the store; enforce the current-head-only view contract.

        Returns the fresh head when coherent. On any mismatch (missing
        history, stale view) sets an error status and returns ``None``
        WITHOUT touching the viewed state.
        """
        try:
            # Correction F1: the RELOADED cine travels with every reopen so
            # manifest-bound history can verify its source identity.
            data = self._controller.reopen(
                self._store, self._state.series_uid, cine=self._state.cine
            )
        except (CMRError, ValueError) as err:
            self._error(f"reopen failed: {_message_of(err)}")
            return None
        except sqlite3.Error as err:
            self._error(f"reopen failed: {self._store_error_message(err)}")
            return None
        chain = tuple(data["chain"])
        if not chain:
            self._error("no persisted revision chain for this series; nothing to act on")
            return None
        fresh_head_id = chain[-1].revision_id
        viewed_head_id = self._state.head_revision_id
        if viewed_head_id is not None and fresh_head_id != viewed_head_id:
            self._error(
                "stale view: persisted chain head advanced beyond the viewed "
                f"head ({viewed_head_id[:12]}.. -> {fresh_head_id[:12]}..); "
                "reload via Reopen before saving or approving"
            )
            return None
        return chain[-1]

    def _refresh_state(self) -> None:
        """Rebuild state from the store, preserving view position and drafts."""
        self._state = reopen_viewer_state(
            self._controller,
            self._store,
            self._state.series_uid,
            self._state.cine,
            fps=self._state.fps,
            current_slice_index=self._state.current_slice_index,
            current_frame_index=self._state.current_frame_index,
            playing=self._state.playing,
            draft_contours=self._state.draft_contours,
        )
        self._playhead.bind(
            self._state.cine.n_frames,
            frame_index=self._state.current_frame_index,
        )

    # --- keyboard/playback actions ------------------------------------------------

    def step_forward(self) -> None:
        """Advance one frame (modulo ``n_frames``) and keep the playhead in sync."""
        n_frames = self._state.cine.n_frames
        next_index = PlayheadController.advance(
            self._state.current_frame_index, n_frames
        )
        self._playhead.sync(next_index)
        self._state = replace(self._state, current_frame_index=next_index)
        self._set_status(self._position_label())

    def step_backward(self) -> None:
        """Step back one frame (wrapping to the last frame at index 0)."""
        previous = (
            self._state.current_frame_index - 1
        ) % self._state.cine.n_frames
        self._playhead.sync(previous)
        self._state = replace(self._state, current_frame_index=previous)
        self._set_status(self._position_label())

    def toggle_play(self) -> None:
        """Toggle playback; ticks advance ONLY while playing (pure playhead)."""
        playing = not self._state.playing
        if playing:
            self._playhead.bind(
                self._state.cine.n_frames,
                frame_index=self._state.current_frame_index,
            )
            self._playhead.start()
        else:
            self._playhead.stop()
        self._state = replace(self._state, playing=playing)
        self._set_status("playing" if playing else "paused")

    def next_slice(self) -> None:
        """Move to the next slice (modulo ``n_slices``); frame index kept."""
        n_slices = self._state.cine.n_slices
        next_slice = (self._state.current_slice_index + 1) % n_slices
        self._state = replace(self._state, current_slice_index=next_slice)
        self._set_status(self._position_label())

    def prev_slice(self) -> None:
        """Move to the previous slice (wrapping at index 0); frame kept."""
        n_slices = self._state.cine.n_slices
        previous = (self._state.current_slice_index - 1) % n_slices
        self._state = replace(self._state, current_slice_index=previous)
        self._set_status(self._position_label())

    def set_fps(self, fps: int) -> int:
        """Clamped fps change shared by the playhead session and the state."""
        effective = self._playhead.set_fps(fps)
        self._state = replace(self._state, fps=effective)
        self._set_status(f"fps set to {effective}")
        return effective

    def consume_playhead_tick(self) -> None:
        """Advance the VIEW from one playhead tick while playing (shell timer).

        The shell schedules this through Tk ``after()`` at
        :attr:`~cmr_viewer.playback.PlayheadController.interval_ms`; paused
        ticks leave everything unchanged.
        """
        if not self._state.playing:
            return
        next_index = self._playhead.tick()
        self._state = replace(self._state, current_frame_index=next_index)

    def _position_label(self) -> str:
        return (
            f"slice {self._state.current_slice_index + 1}/"
            f"{self._state.cine.n_slices}   frame "
            f"{self._state.current_frame_index + 1}/{self._state.cine.n_frames}"
        )

    # --- manual authoritative ED/ES selection ---------------------------------

    def set_ed(self) -> bool:
        """Mark the CURRENT frame as end-diastole via the controller."""
        return self._apply_selection(which="ed")

    def set_es(self) -> bool:
        """Mark the CURRENT frame as end-systole via the controller."""
        return self._apply_selection(which="es")

    def _apply_selection(self, which: str) -> bool:
        state = self._state
        current = state.current_frame_index
        other = (
            (state.selection.es_frame if which == "ed" else state.selection.ed_frame)
            if state.selection is not None
            else current
        )
        ed, es = (current, other) if which == "ed" else (other, current)
        try:
            # S2C-001 §8.3 routing: validation AND persistence in one
            # controller step — the viewer never touches the store primitive.
            selection = self._controller.record_ed_es(
                self._store,
                state.cine,
                ed,
                es,
                self._physician,
                self._clock(),
                self._provenance_factory,
            )
        except (CMRError, ValueError) as err:
            self._error(f"set {which.upper()} failed: {_message_of(err)}")
            return False
        except sqlite3.Error as err:
            self._error(
                f"set {which.upper()} failed: {self._store_error_message(err)}"
            )
            return False
        self._state = replace(state, selection=selection)
        # CFM-001 §1.8: an ED/ES change must not silently carry a prior set
        # forward — grid resets to the visible all-included default.
        self._reset_inclusion_draft_to_default()
        self._set_status(
            f"ED={selection.ed_frame} ES={selection.es_frame} recorded; "
            "analysis is STALE until Refresh Analysis"
        )
        return True

    # --- minimal contour-edit gesture -------------------------------------------

    def add_vertex_at(
        self,
        slice_index: int,
        frame_index: int,
        structure: Structure,
        position: tuple[float, float],
    ) -> bool:
        """Add one vertex at pixel ``position``; UNSAVED until ``save_flow``.

        Delegates to the Node 4 :class:`~cmr_contours.editing.EditCommandStack`
        through the controller. Failures (no base contour, invalid position)
        surface in the status label and leave state unchanged.
        """
        state = self._state
        if not 0 <= slice_index < state.cine.n_slices or not 0 <= frame_index < state.cine.n_frames:
            self._error("add vertex failed: slice/frame outside the cine grid")
            return False
        key: EditKey = (slice_index, frame_index, structure)
        stack = self._stacks.get(key)
        if stack is None:
            base = self._base_contour_for(slice_index, frame_index, structure)
            if base is None:
                self._error(
                    f"add vertex failed: no {structure.value} contour at slice "
                    f"{slice_index} frame {frame_index} on the chain head"
                )
                return False
            stack = EditCommandStack(base)
        try:
            updated = self._controller.edit_contour(stack, InsertVertex(position))
        except (CMRError, ValueError) as err:
            self._error(f"add vertex failed: {_message_of(err)}")
            return False
        self._stacks[key] = stack
        self._active_key = key
        drafts = tuple(
            contour
            for contour in state.draft_contours
            if (contour.slice_index, contour.frame_index, contour.structure) != key
        ) + (updated,)
        self._state = replace(state, draft_contours=drafts)
        self._set_status(
            f"vertex added at ({position[0]:g}, {position[1]:g}) px on "
            f"{structure.value} slice {slice_index} frame {frame_index} (unsaved)"
        )
        return True

    def _base_contour_for(
        self, slice_index: int, frame_index: int, structure: Structure
    ) -> Contour | None:
        # Correction round 1 F1 consistency: edits target what the physician
        # SEES — the chain-merged latest contour at this (slice, frame),
        # never a head-revision-only lookup.
        for contour in self._state.merged_contours_at(slice_index, frame_index):
            if contour.structure is structure:
                return contour
        return None

    # --- undo/redo ---------------------------------------------------------------

    def request_undo(self, stack: EditCommandStack) -> Contour | None:
        """Undo through the controller; failure -> status label, no change."""
        try:
            restored = self._controller.undo(stack)
        except (CMRError, ValueError) as err:
            self._error(f"undo failed: {_message_of(err)}")
            return None
        self._sync_stack_result(stack, restored, action="undo")
        return restored

    def request_redo(self, stack: EditCommandStack) -> Contour | None:
        """Redo through the controller; failure -> status label, no change."""
        try:
            reapplied = self._controller.redo(stack)
        except (CMRError, ValueError) as err:
            self._error(f"redo failed: {_message_of(err)}")
            return None
        self._sync_stack_result(stack, reapplied, action="redo")
        return reapplied

    def _sync_stack_result(
        self, stack: EditCommandStack, result: Contour, *, action: str
    ) -> None:
        key = next((k for k, v in self._stacks.items() if v is stack), None)
        if key is not None:
            self._active_key = key
            drafts = tuple(
                contour
                for contour in self._state.draft_contours
                if (contour.slice_index, contour.frame_index, contour.structure) != key
            ) + (result,)
            self._state = replace(self._state, draft_contours=drafts)
        self._set_status(f"{action} applied (unsaved)")

    def undo_active(self) -> bool:
        """Convenience: undo on the active stack (shell Undo button/binding)."""
        stack = self.active_stack
        if stack is None:
            self._error("undo failed: no contour edit in progress")
            return False
        return self.request_undo(stack) is not None

    def redo_active(self) -> bool:
        """Convenience: redo on the active stack (shell Redo button/binding)."""
        stack = self.active_stack
        if stack is None:
            self._error("redo failed: no contour edit in progress")
            return False
        return self.request_redo(stack) is not None

    # --- CFM-001 basal-inclusion grid flows ------------------------------------

    def _reset_inclusion_draft_to_default(
        self, *, keep_persisted_set: bool = False
    ) -> None:
        """Reset the draft to the visible all-included default (unsaved).

        CFM-001 §1.8: changing ED/ES or the contour head must never silently
        carry a prior set forward — the grid resets to the all-included
        default and the old set remains available only as read-only history.

        Correction round 2 (finding 1): ``reset_all_included`` KEEPS the
        persisted applicable set so an all-included draft over a set that
        contains exclusions stays visibly divergent (never masquerading as
        approved).
        """
        self._state = replace(
            self._state,
            inclusion_draft=all_included_toggles(self._state.cine.n_slices),
            persisted_inclusion_set=(
                self._state.persisted_inclusion_set
                if keep_persisted_set
                else None
            ),
        )

    def toggle_inclusion(
        self, slice_index: int, role: str, structure: str, included: bool
    ) -> bool:
        """Set one draft checkbox; invalidates metrics immediately.

        Values must be ACTUAL Booleans (integers/floats fail closed). The
        change stays UNSAVED until ``apply_inclusion_and_refresh_flow``; it
        makes the displayed analysis STALE right away through the state's
        inclusion-diff trigger.
        """
        if not isinstance(included, bool):
            self._error("toggle failed: included must be an actual Boolean")
            return False
        if (role, structure) not in INCLUSION_GRID_GROUPS:
            self._error(
                f"toggle failed: unknown group (role={role!r}, "
                f"structure={structure!r}); LV mass applies to ED only"
            )
            return False
        if not 0 <= slice_index < self._state.cine.n_slices:
            self._error("toggle failed: slice outside the cine stack")
            return False
        draft = tuple(
            (
                toggle
                if toggle.key != (slice_index, role, structure)
                else InclusionToggle(slice_index, role, structure, included)
            )
            for toggle in self._state.inclusion_draft
        )
        self._state = replace(self._state, inclusion_draft=draft)
        self._set_status(
            f"inclusion draft updated: slice {slice_index} {role} "
            f"{structure} {'INCLUDED' if included else 'EXCLUDED'} (unsaved); "
            "metrics are stale until Apply Inclusion & Refresh Analysis"
        )
        return True

    def reset_all_included(self) -> bool:
        """'Reset all included': changes ONLY the unsaved draft.

        Correction round 2 (finding 1): the persisted applicable set is
        PRESERVED, so resetting over a set that contains exclusions keeps the
        divergence visible (stale/unapproved) instead of hiding it.
        """
        self._reset_inclusion_draft_to_default(keep_persisted_set=True)
        self._set_status("inclusion draft reset: every toggle INCLUDED (unsaved)")
        return True

    def navigate_to_inclusion_row(self, slice_index: int, role: str) -> bool:
        """Clicking a row navigates to that slice and its resolved frame."""
        if not 0 <= slice_index < self._state.cine.n_slices:
            self._error("navigation failed: slice outside the cine stack")
            return False
        selection = self._state.selection
        if selection is None:
            self._error("navigation failed: no ED/ES selection recorded yet")
            return False
        frame = selection.ed_frame if role == "ED" else selection.es_frame
        if role not in ("ED", "ES"):
            self._error(f"navigation failed: unknown frame role {role!r}")
            return False
        self._state = replace(
            self._state,
            current_slice_index=slice_index,
            current_frame_index=frame,
        )
        self._playhead.bind(self._state.cine.n_frames, frame_index=frame)
        self._set_status(f"slice {slice_index + 1}, {role} frame {frame}")
        return True

    def apply_inclusion_and_refresh_flow(self) -> bool:
        """Validate + atomically persist the applicable set AND the analysis.

        Delegates to ``WorkflowController.apply_inclusion_and_refresh_analysis``
        with the CURRENT unsaved draft. Failure leaves the previous durable
        state intact and the draft visible; current metrics are CLEARED from
        the display. On success the state is refreshed from the store.
        """
        entries = [
            {
                "slice_index": toggle.slice_index,
                "frame_role": toggle.role,
                "structure": toggle.structure,
                "included": toggle.included,
            }
            for toggle in self._state.inclusion_draft
        ]
        try:
            receipt = self._controller_apply_inclusion(entries)
        except CMRError as err:
            # Correction round 2 (finding 6): ONLY the dedicated
            # post-inclusion ESV>EDV reason maps to the frozen viewer text;
            # unrelated qc.volume_curve errors keep their own message.
            if _is_post_inclusion_esv_error(err):
                self.mark_analysis_unavailable_esv_exceeds_edv()
                return False
            self._error(f"apply inclusion failed: {_message_of(err)}")
            return False
        except (ValueError) as err:
            self._error(f"apply inclusion failed: {_message_of(err)}")
            return False
        except sqlite3.Error as err:
            self._error(
                "apply inclusion failed: "
                f"{self._store_error_message(err)}"
            )
            return False
        try:
            self._refresh_state()
        except sqlite3.Error as err:
            self._set_status(
                "inclusion applied & COMMITTED; automatic reload failed "
                f"({type(err).__name__}) — press Reopen before retrying",
                is_error=True,
            )
            return True
        if receipt.get("status") == "unchanged":
            self._set_status("inclusion already coherent for the current head")
        else:
            self._set_status(
                "inclusion applied & analysis refreshed: verdict "
                f"{receipt.get('qc_verdict')}; visible state: "
                f"{self._state.approval_label}"
            )
        return True

    def _controller_apply_inclusion(self, entries: list[dict[str, Any]]) -> Any:
        """Single call site for the controller compound flow (test seam)."""
        apply_fn = getattr(self._controller, "apply_inclusion_and_refresh_analysis")
        return apply_fn(
            self._store,
            self._state.cine,
            self._state.series_uid,
            entries=entries,
            selected_by=self._physician,
            software_version=self._provenance_factory().software_version,
            created_at=self._clock(),
        )

    def mark_analysis_unavailable_esv_exceeds_edv(self) -> None:
        """Post-inclusion ESV>EDV viewer contract (CFM-001 §1.6).

        The inclusion draft is RETAINED, current metrics are CLEARED from
        the display and the exact literal is surfaced as the status text.
        Nothing durable changes: no Measurement and no ApprovalSnapshot may
        be written for a rejected curve.
        """
        self._state = replace(self._state, latest_measurement=None)
        self._error(ANALYSIS_UNAVAILABLE_ESV_EXCEEDS_EDV)

    # --- persistence flows (current head only) ------------------------------------

    def save_flow(self) -> bool:
        """Persist the ACTIVE edited contour as a child of the CURRENT head.

        Stale views (persisted head advanced beyond the viewed head) are
        rejected with a visible error and zero writes. On success the state
        is refreshed from the store; the saved draft is consumed while other
        unsaved drafts survive.
        """
        active_key = self._active_key
        stack = self.active_stack
        if active_key is None or stack is None:
            self._error("save failed: no contour edit in progress")
            return False
        head = self._fresh_chain_head()
        if head is None:
            return False
        _, _, structure = active_key
        try:
            self._controller.save(
                self._store,
                head,
                {structure: stack.current},
                self._physician,
                self._clock(),
            )
        except (CMRError, ValueError) as err:
            self._error(f"save failed: {_message_of(err)}")
            return False
        except sqlite3.Error as err:
            self._error(f"save failed: {self._store_error_message(err)}")
            return False
        saved_key = active_key
        self._stacks.pop(saved_key, None)
        self._active_key = None
        remaining_drafts = tuple(
            contour
            for contour in self._state.draft_contours
            if (contour.slice_index, contour.frame_index, contour.structure)
            != saved_key
        )
        self._state = replace(self._state, draft_contours=remaining_drafts)
        # Correction round 2 R2: the commit SUCCEEDED above. A post-commit
        # reload failure must never be reported as a failed save — say
        # exactly that the edit is durable and a manual reopen is needed.
        try:
            self._refresh_state()
        except sqlite3.Error as err:
            self._set_status(
                "saved & COMMITTED; automatic reload failed "
                f"({type(err).__name__}) — press Reopen before retrying",
                is_error=True,
            )
            return True
        head_after = self._state.head_revision
        saved_version = head_after.version if head_after is not None else "-"
        # CFM-001 §1.8: a contour-head change must not silently carry a
        # prior set forward — grid resets to the visible all-included
        # default; the old set stays readable only as history.
        self._reset_inclusion_draft_to_default()
        self._set_status(f"saved revision version {saved_version}")
        return True

    def approve_flow(
        self,
        *,
        override: bool = False,
        override_reason: str | None = None,
    ) -> bool:
        """Approve the CURRENT head through the controller approval policy.

        The QC report comes from the chain head's own gateway field; the
        measurement/selection come from the freshly reopened store;
        ``override``/``override_reason`` pass through UNCHANGED to
        ``WorkflowController.approve`` (B-LITE sealing included). Stale
        views fail closed before any write.

        S2C-001 clusters C+D: the controller seals through
        ``store.save_approval_checked`` / ``save_qc_decision_checked`` —
        head freshness, persisted-QC digest equality and binding coherence
        are verified INSIDE one BEGIN IMMEDIATE transaction, so the former
        post-seal TOCTOU probe is gone by construction: a head that advanced
        before the transaction writes ZERO rows and surfaces here as a
        visible error with unchanged state.
        """
        head = self._fresh_chain_head()
        if head is None:
            return False
        qc_report = head.qc_report
        if qc_report is None:
            self._error(
                "approve failed: the chain head carries no QC report; refresh "
                "the analysis (or snapshot a QC outcome) first"
            )
            return False
        try:
            # Correction F1: reopen carries the reloaded cine.
            data = self._controller.reopen(
                self._store, self._state.series_uid, cine=self._state.cine
            )
        except (CMRError, ValueError) as err:
            self._error(f"reopen failed: {_message_of(err)}")
            return False
        except sqlite3.Error as err:
            self._error(f"approve failed: {self._store_error_message(err)}")
            return False
        measurement = data["latest_measurement"]
        selection = data["latest_selection"]
        if measurement is None:
            # Lazy import keeps package import free of cmr_workflow (import
            # boundary gate); see module docstring "Sentinel path".
            from cmr_workflow.controller import SENTINEL_MEASUREMENT_ID

            measurement_id = SENTINEL_MEASUREMENT_ID
        else:
            measurement_id = measurement.measurement_id
        try:
            snapshot = self._controller.approve(
                self._store,
                head.revision_id,
                measurement_id,
                qc_report,
                self._physician,
                self._clock(),
                measurement=measurement,
                selection=selection,
                override=override,
                override_reason=override_reason,
            )
        except (CMRError, ValueError) as err:
            self._error(f"approve failed: {_message_of(err)}")
            return False
        except sqlite3.Error as err:
            self._error(f"approve failed: {self._store_error_message(err)}")
            return False

        # Correction round 2 R2: the seal SUCCEEDED. A post-commit reload
        # failure must never imply the decision was lost — it is durable.
        try:
            self._refresh_state()
        except sqlite3.Error as err:
            self._set_status(
                "decision sealed & COMMITTED; automatic reload failed "
                f"({type(err).__name__}) — press Reopen before retrying",
                is_error=True,
            )
            return True
        self._set_status(
            f"decision sealed: verdict {snapshot.qc_verdict}; "
            f"visible state: {self._state.approval_label}"
        )
        return True

    def refresh_head_analysis(self) -> bool:
        """Recompute QC + metrics on the CURRENT head (S2C-001 §8.2).

        Delegates to ``WorkflowController.refresh_head_analysis`` with the
        viewed cine and series; failures surface in the status label and
        leave the viewed state untouched. On success the state is refreshed
        from the store so head, QC, measurement and selection are coherent.
        """
        try:
            receipt = self._controller.refresh_head_analysis(
                self._store,
                self._state.cine,
                self._state.series_uid,
                software_version=self._provenance_factory().software_version,
                created_at=self._clock(),
            )
        except (CMRError, ValueError) as err:
            self._error(f"refresh analysis failed: {_message_of(err)}")
            return False
        except sqlite3.Error as err:
            self._error(f"refresh analysis failed: {self._store_error_message(err)}")
            return False
        # Correction round 2 R2: the refresh transaction SUCCEEDED (or was an
        # idempotent no-op). A post-commit reload failure must never imply a
        # rollback — say the new head is durable and a manual reopen is due.
        try:
            self._refresh_state()
        except sqlite3.Error as err:
            self._set_status(
                "analysis refreshed & COMMITTED; automatic reload failed "
                f"({type(err).__name__}) — press Reopen before retrying",
                is_error=True,
            )
            return True
        status = receipt.get("status")
        if status == "unchanged":
            self._set_status("analysis already coherent for the current head")
        else:
            self._set_status(
                f"analysis refreshed: verdict {receipt.get('qc_verdict')}; "
                f"visible state: {self._state.approval_label}"
            )
        return True

    # --- reopen -------------------------------------------------------------------

    def reopen_view(self, store: CmrStoreLike, series_uid: str, cine: CineVolume) -> ViewerState:
        """Rebuild the viewed state per AMENDMENT-001 B and switch to it.

        ``cine`` is the RELOADED volume (persistence stores no pixels).
        Unsaved drafts and edit stacks belong to the discarded view and are
        dropped; view position resets to the first slice/frame.

        Correction round 1 F2 (AMENDMENT-001 H): a failing reopen dependency
        (store unavailable, payload integrity drift, ...) is caught HERE,
        surfaces as ``ERROR: reopen failed: <message>`` in the status label,
        and leaves the CURRENT state object untouched — the old view keeps
        rendering instead of crashing or half-switching.
        """
        try:
            state = reopen_viewer_state(
                self._controller, store, series_uid, cine, fps=self._state.fps
            )
        except (CMRError, ValueError) as err:
            self._error(f"reopen failed: {_message_of(err)}")
            return self._state
        except sqlite3.Error as err:
            self._error(f"reopen failed: {self._store_error_message(err)}")
            return self._state
        self._store = store
        self._state = state
        self._stacks.clear()
        self._active_key = None
        self._playhead.bind(cine.n_frames, frame_index=state.current_frame_index)
        self._set_status(
            f"reopened {series_uid}: visible state {state.approval_label}"
        )
        return state


def _is_post_inclusion_esv_error(err: BaseException) -> bool:
    """Correction round 3 (R7): dispatch on the TYPED reason subtype."""
    return isinstance(err, PostInclusionEsvExceedsEdvError)


def _message_of(err: CMRError | ValueError) -> str:
    """Extract the human-readable message from the caught error types."""
    message = getattr(err, "message", None)
    if isinstance(message, str) and message:
        return message
    return str(err)
