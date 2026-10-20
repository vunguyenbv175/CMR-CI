"""Node 8 — interactive Tk/TkAgg shell over the same ViewerApp core.

AMENDMENT-001 A: the deliverable is the deterministic Agg renderer PLUS this
real interactive adapter (Tk 8.6 ships with the interpreter; stdlib only, no
new dependency). :class:`ViewerShell` wires the rendered frames into a real
display surface with REAL controls:

- buttons: Play/Pause, Step +, Step -, Slice +, Slice -, Set ED, Set ES,
  Undo, Redo, Save, Refresh Analysis, Approve, Approve+Reason, Reopen,
  Add Vertex;
- keyboard bindings mapped to the SAME ``ViewerApp`` action methods:
  Right/Left step frames, Up/Down change slices, Space plays/pauses,
  Ctrl+Z/Ctrl+Y undo/redo, Ctrl+S save, Ctrl+R refresh analysis,
  Ctrl+D/Ctrl+E set ED/ES, Ctrl+P approve, F5 reopen;
- a visible status/error label mirroring :attr:`ViewerApp.status` (errors
  shown in red with their ``ERROR:`` prefix — failures surface in the UI,
  never on the console);
- playback ticks are scheduled cooperatively through Tk ``after()`` at
  :attr:`~cmr_viewer.playback.PlayheadController.interval_ms`; there is NO
  thread anywhere.

S2C-001 §8.1 exact click mapping: an armed left click converts the Tk
top-left event origin to Matplotlib's bottom-left display origin, resolves
the OUTER imshow pixel-centre coordinates through the live ``transData``
inverse (handles window resizing, letterboxing, aspect), applies the
+0.5 pixel-centre-to-edge rule and delegates to the pure renderer helper
:func:`~cmr_viewer.renderer.pixel_from_outer_canvas_point` carrying THIS
frame's millimetre->RGB affine. Out-of-axes / off-grid / non-finite results
are REJECTED (never clipped) and never reach ``add_vertex_at``.

Import safety: importing THIS module never requires a display and never
switches the matplotlib backend — ``matplotlib.use('TkAgg')`` happens ONLY
inside :func:`main` (AMENDMENT-001 A), after which a root window may be
instantiated. Tests import the module and assert its structure without ever
constructing Tk objects; :func:`build_synthetic_demo_session` (S2C-001 §8.4)
is fully headless so the demo seeding can run without starting Tk.

:func:`main` launches against SYNTHETIC data only (gradient cine, rectangle
contours, throwaway temp-dir store) so the owner can run the manual QA
checklist (:file:`docs/manual/NODE8_VIEWER_QA_CHECKLIST.md`) without any
patient file, GPU, cloud or network.
"""

from __future__ import annotations

import tkinter as tk
from functools import partial
from pathlib import Path
from typing import Any

from cmr_core.segmentation import Structure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from cmr_viewer.app import CmrStoreLike, ViewerApp
from cmr_viewer.renderer import (
    DPI,
    FIGURE_SIZE_IN,
    RenderedFrame,
    ViewerRenderer,
    pixel_from_outer_canvas_point,
)

__all__ = [
    "VIEWER_WINDOW_TITLE",
    "BUTTON_LABELS",
    "INCLUSION_BUTTON_LABELS",
    "ViewerShell",
    "build_synthetic_demo_session",
    "main",
]

#: Fixed window title (documented in the manual QA checklist).
VIEWER_WINDOW_TITLE = "CMR Studio - Node 8 cine SAX review viewer"

#: Button labels in fixed layout order (documented in the QA checklist).
BUTTON_LABELS: tuple[str, ...] = (
    "Play/Pause",
    "Step +",
    "Step -",
    "Slice +",
    "Slice -",
    "Set ED",
    "Set ES",
    "Undo",
    "Redo",
    "Save",
    "Refresh Analysis",
    "Approve",
    "Approve+Reason",
    "Reopen",
    "Add Vertex",
)

#: CFM-001 inclusion-panel buttons (separate frame; deliberately NOT part of
#: the frozen main-toolbar BUTTON_LABELS tuple).
INCLUSION_BUTTON_LABELS: tuple[str, ...] = (
    "Reset All Included",
    "Apply Inclusion & Refresh Analysis",
)

#: Human-readable structure labels of the inclusion grid (spec §1.8).
INCLUSION_STRUCTURE_LABELS: dict[str, str] = {
    "lv_cavity": "LV cavity",
    "rv_cavity": "RV cavity",
    "lv_myocardium": "LV mass",
}


def inclusion_row_label(
    toggle: Any,
    position_mm: float,
    resolved_frame: int | None,
) -> str:
    """Pure row label used by the interactive grid (correction F12/F5).

    Carries the stable ``slice_index``, geometric POSITION (mm), structure,
    frame ROLE and RESOLVED frame number; rebuilt whenever the selection
    changes so resolved frames never go stale.
    """
    structure_label = INCLUSION_STRUCTURE_LABELS.get(
        toggle.structure, toggle.structure
    )
    resolved = "" if resolved_frame is None else f" | frame {resolved_frame}"
    return (
        f"slice {toggle.slice_index} @ {position_mm:.1f} mm | "
        f"{structure_label} | Frame role: {toggle.role}{resolved}"
    )


def selection_signature(selection: Any) -> tuple[int, int] | None:
    """Resolved (ED, ES) pair used to detect selection-driven rebuilds."""
    if selection is None:
        return None
    return (selection.ed_frame, selection.es_frame)


class ViewerShell(tk.Tk):
    """Tk window binding every control to the shared ``ViewerApp`` methods."""

    def __init__(
        self,
        app: ViewerApp,
        *,
        renderer: ViewerRenderer | None = None,
        reopen_store: CmrStoreLike | None = None,
        reopen_series_uid: str | None = None,
        reopen_cine: Any = None,
    ) -> None:
        super().__init__()
        self.title(VIEWER_WINDOW_TITLE)
        self._app = app
        self._renderer = renderer if renderer is not None else ViewerRenderer()
        self._reopen_store = reopen_store
        self._reopen_series_uid = reopen_series_uid
        self._reopen_cine = reopen_cine

        self._toolbar = tk.Frame(self, bd=1, relief="raised")
        self._toolbar.pack(side="top", fill="x")
        self._buttons: dict[str, tk.Button] = {}
        for label in BUTTON_LABELS:
            button = tk.Button(
                self._toolbar,
                text=label,
                width=max(8, len(label) + 1),
                command=partial(self._on_button, label),
            )
            button.pack(side="left", padx=2, pady=2)
            self._buttons[label] = button

        from matplotlib.figure import Figure

        self._figure = Figure(figsize=FIGURE_SIZE_IN, dpi=DPI)
        self._axes = self._figure.add_axes((0.0, 0.0, 1.0, 1.0))
        self._axes.set_axis_off()
        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        widget = self._canvas.get_tk_widget()
        widget.pack(side="top", fill="both", expand=True)
        widget.bind("<Button-1>", self._on_canvas_click)

        self._status_var = tk.StringVar(value=app.status)
        self._status_label = tk.Label(
            self,
            textvariable=self._status_var,
            anchor="w",
            bd=1,
            relief="sunken",
            padx=6,
            pady=3,
        )
        self._status_label.pack(side="bottom", fill="x")
        # S2C-001 §8.1: the RenderedFrame of the LAST refresh carries the
        # millimetre->RGB affine used for exact click mapping.
        self._last_rendered_frame: RenderedFrame | None = None

        # Edit-gesture affordances: structure picker + arm toggle. While
        # armed, a LEFT CLICK on the image places one vertex on the chosen
        # structure of the CURRENT slice/frame via ViewerApp.add_vertex_at.
        self._vertex_armed = tk.BooleanVar(value=False)
        self._edit_structure = tk.StringVar(value=Structure.LV_CAVITY.value)
        self._structure_menu = tk.OptionMenu(
            self._toolbar, self._edit_structure, Structure.LV_CAVITY.value
        )
        self._structure_menu.pack(side="right", padx=2, pady=2)

        # --- CFM-001 basal-inclusion grid panel ---------------------------------
        # Always-visible SCMR uncertainty notice beside the grid (never a
        # tooltip), one Checkbutton per (slice, frame role, structure) row,
        # LV mass rows forced to ED (LV mass + ES disabled as not
        # applicable), Reset/Apply buttons touching only the draft until
        # Apply validates and atomically persists set + analysis.
        self._inclusion_panel = tk.Frame(self, bd=1, relief="ridge")
        self._inclusion_panel.pack(side="right", fill="y")
        from cmr_viewer.state import SCMR_BASAL_UNCERTAINTY_NOTICE

        notice_label = tk.Label(
            self._inclusion_panel,
            text=SCMR_BASAL_UNCERTAINTY_NOTICE,
            wraplength=260,
            justify="left",
            anchor="w",
        )
        notice_label.pack(side="top", fill="x", padx=4, pady=4)
        self._inclusion_status_var = tk.StringVar(value="")
        tk.Label(
            self._inclusion_panel,
            textvariable=self._inclusion_status_var,
            wraplength=260,
            justify="left",
            anchor="w",
            fg="#202020",
        ).pack(side="top", fill="x", padx=4)
        grid_holder = tk.Frame(self._inclusion_panel)
        grid_holder.pack(side="top", fill="both", expand=True)
        canvas = tk.Canvas(grid_holder)
        scrollbar = tk.Scrollbar(
            grid_holder, orient="vertical", command=canvas.yview
        )
        self._inclusion_rows_frame = tk.Frame(canvas)
        self._inclusion_rows_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(
                scrollregion=canvas.bbox("all")
            ),
        )
        canvas.create_window((0, 0), window=self._inclusion_rows_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for label in INCLUSION_BUTTON_LABELS:
            if label.startswith("Reset"):
                command: Any = self._reset_all_included
            else:
                command = self._apply_inclusion
            button = tk.Button(self._inclusion_panel, text=label)

            def invoke(fn: Any = command) -> None:
                self._action(fn)

            button.configure(command=invoke)
            button.pack(side="bottom", fill="x", padx=2, pady=2)
        self._inclusion_vars: dict[tuple[int, str, str], tk.BooleanVar] = {}
        self._inclusion_rows: dict[tuple[int, str, str], tk.Frame] = {}
        # Correction round 2 (finding 5): frame-role / structure selectors.
        # Selecting 'LV mass' FORCES the role back to ED (ES is not
        # applicable); selector changes never invalidate analysis - they only
        # highlight/navigate to the first matching row.
        self._selector_role_var = tk.StringVar(value="ED")
        self._selector_structure_var = tk.StringVar(value="LV cavity")

        def on_role_selector(choice: str) -> None:
            if (
                self._selector_structure_var.get() == "LV mass"
                and choice == "ES"
            ):
                # LV mass forces ED.
                self._selector_role_var.set("ED")
            self._highlight_first_matching_row()

        def on_structure_selector(choice: str) -> None:
            if choice == "LV mass":
                self._selector_role_var.set("ED")
            self._highlight_first_matching_row()

        role_menu = tk.OptionMenu(
            self._inclusion_panel,
            self._selector_role_var,
            "ED",
            "ES",
            command=on_role_selector,  # type: ignore[arg-type,unused-ignore]
        )
        role_menu.configure(width=6)
        tk.Label(self._inclusion_panel, text="Frame role:").pack(
            side="top", anchor="w", padx=4
        )
        role_menu.pack(side="top", anchor="w", padx=4)

        structure_menu = tk.OptionMenu(
            self._inclusion_panel,
            self._selector_structure_var,
            *INCLUSION_STRUCTURE_LABELS.values(),
            command=on_structure_selector,  # type: ignore[arg-type]
        )
        structure_menu.configure(width=10)
        tk.Label(self._inclusion_panel, text="Structure:").pack(
            side="top", anchor="w", padx=4
        )
        structure_menu.pack(side="top", anchor="w", padx=4)
        self._grid_selection_signature: tuple[int, int] | None = None

        self._bind_keys()
        self._schedule_tick()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.refresh()

    # --- wiring ---------------------------------------------------------------

    def _bind_keys(self) -> None:
        bindings = {
            "<Right>": lambda event: self._action(self._app.step_forward),
            "<Left>": lambda event: self._action(self._app.step_backward),
            "<space>": lambda event: self._action(self._app.toggle_play),
            "<Up>": lambda event: self._action(self._app.next_slice),
            "<Down>": lambda event: self._action(self._app.prev_slice),
            "<Control-z>": lambda event: self._action(self._app.undo_active),
            "<Control-y>": lambda event: self._action(self._app.redo_active),
            "<Control-s>": lambda event: self._action(self._app.save_flow),
            "<Control-r>": lambda event: self._action(self._app.refresh_head_analysis),
            "<Control-d>": lambda event: self._action(self._app.set_ed),
            "<Control-e>": lambda event: self._action(self._app.set_es),
            "<Control-p>": lambda event: self._action(self._app.approve_flow),
            "<F5>": lambda event: self._action(self._reopen),
            "<v>": lambda event: self._toggle_vertex_mode,
        }
        for sequence, handler in bindings.items():
            self.bind(sequence, handler)

    def _on_button(self, label: str) -> None:
        actions = {
            "Play/Pause": self._app.toggle_play,
            "Step +": self._app.step_forward,
            "Step -": self._app.step_backward,
            "Slice +": self._app.next_slice,
            "Slice -": self._app.prev_slice,
            "Set ED": self._app.set_ed,
            "Set ES": self._app.set_es,
            "Undo": self._app.undo_active,
            "Redo": self._app.redo_active,
            "Save": self._app.save_flow,
            "Refresh Analysis": self._app.refresh_head_analysis,
            "Approve": self._app.approve_flow,
            "Approve+Reason": self._approve_with_reason,
            "Reopen": self._reopen,
            "Add Vertex": self._toggle_vertex_mode,
        }
        action = actions.get(label)
        if action is not None:
            self._action(action)

    def _action(self, action: Any) -> None:
        # Correction F6 final exception boundary: NO action can crash the Tk
        # event loop; unexpected failures surface as a retry-safe visible
        # error via ViewerApp and the view keeps rendering.
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - deliberate UI boundary
            self._app.surface_exception(exc)
        self.refresh()

    def _approve_with_reason(self) -> None:
        from tkinter.simpledialog import askstring

        reason = askstring(
            "Physician override",
            "Documented override reason (required by the approval policy):",
            parent=self,
        )
        if reason is None or not reason.strip():
            return  # cancelled: no decision, state untouched
        self._action(lambda: self._app.approve_flow(override=True, override_reason=reason))

    # --- minimal edit gesture: armed click -> add_vertex_at -------------------

    def _toggle_vertex_mode(self) -> None:
        """Arm/disarm click-to-add-vertex on the image."""
        self._vertex_armed.set(not self._vertex_armed.get())
        self._buttons["Add Vertex"].config(
            relief="sunken" if self._vertex_armed.get() else "raised",
            bg="#dfe8d8" if self._vertex_armed.get() else "SystemButtonFace",
        )

    def _set_edit_structure(self, value: str) -> None:
        self._edit_structure.set(value)

    def _refresh_structure_menu(self) -> None:
        # Correction round 1 F1: offer exactly what the renderer draws —
        # chain-MERGED contours at the current slice/frame (latest per key
        # across the whole revision_chain), never head-only lookups.
        state = self._app.state
        available = [
            contour.structure.value
            for contour in state.merged_contours_at(
                state.current_slice_index, state.current_frame_index
            )
        ]
        menu = self._structure_menu["menu"]
        menu.delete(0, "end")
        for value in available:
            menu.add_command(label=value, command=partial(self._set_edit_structure, value))
        current = self._edit_structure.get()
        if available and current not in available:
            self._edit_structure.set(available[0])

    def _on_canvas_click(self, event: Any) -> None:
        if not self._vertex_armed.get():
            return  # disarmed clicks never edit
        state = self._app.state
        geometry = state.cine.geometries[state.current_slice_index]
        frame = self._last_rendered_frame
        if frame is None or frame.mm_from_rgb_affine == ():
            return  # no rendered affine yet: never guess a mapping
        # S2C-001 §8.1 steps 1-3: Tk top-left origin -> Matplotlib bottom-left
        # display origin -> OUTER imshow pixel-centre coordinates through the
        # live axes inverse (window resizing / letterboxing / aspect aware).
        widget_height = float(event.widget.winfo_height())
        x_display = float(event.x)
        y_display = widget_height - float(event.y)
        x_outer, y_outer = self._axes.transData.inverted().transform(
            (x_display, y_display)
        )
        # Steps 4-8 (+0.5 pixel-centre rule, affine into millimetres,
        # anisotropic spacing division, strict rejection) live in the pure
        # headless renderer helper.
        point = pixel_from_outer_canvas_point(
            geometry, frame.mm_from_rgb_affine, x_outer, y_outer
        )
        if point is None:
            return  # outside the image grid: reject WITHOUT clipping

        def place() -> None:
            self._app.add_vertex_at(
                state.current_slice_index,
                state.current_frame_index,
                Structure(self._edit_structure.get()),
                point,
            )

        self._action(place)

    def _reopen(self) -> None:
        if (
            self._reopen_store is None
            or self._reopen_series_uid is None
            or self._reopen_cine is None
        ):
            return  # shell was constructed without a reopen context
        self._app.reopen_view(
            self._reopen_store, self._reopen_series_uid, self._reopen_cine
        )
        self.refresh()

    # --- CFM-001 inclusion panel wiring ---------------------------------------

    def _reset_all_included(self) -> None:
        self._app.reset_all_included()

    def _apply_inclusion(self) -> None:
        self._app.apply_inclusion_and_refresh_flow()

    def _rebuild_inclusion_grid(self) -> None:
        """Rebuild the per-slice checkbox rows from the current state.

        CFM-001 correction F12: each stable row displays its ``slice_index``,
        the geometric slice POSITION (mm) and the RESOLVED frame number for
        its role, and clicking the row navigates to that slice/frame.
        """
        from cmr_geometry.transforms import slice_position_mm

        state = self._app.state
        selection = state.selection
        for child in self._inclusion_rows_frame.winfo_children():
            child.destroy()
        self._inclusion_vars = {}
        self._inclusion_rows = {}
        self._grid_selection_signature = selection_signature(selection)
        for toggle in state.inclusion_draft:
            position_mm = slice_position_mm(
                state.cine.geometries[toggle.slice_index]
            )
            resolved_frame = None
            if selection is not None:
                resolved_frame = (
                    selection.ed_frame if toggle.role == "ED"
                    else selection.es_frame
                )
            row_label = inclusion_row_label(
                toggle, position_mm, resolved_frame
            )
            row = tk.Frame(self._inclusion_rows_frame)
            row.pack(side="top", fill="x")
            var = tk.BooleanVar(value=toggle.included)
            check = tk.Checkbutton(row, variable=var)
            check.pack(side="left")
            label = tk.Label(row, text=row_label, anchor="w", justify="left")
            label.pack(side="left", fill="x", expand=True)
            if toggle.structure == "lv_myocardium" and toggle.role == "ES":
                # LV mass + ES is disabled as not applicable (spec §1.8).
                check.configure(state="disabled")
            self._inclusion_vars[toggle.key] = var
            self._inclusion_rows[toggle.key] = row

            def on_toggle(var_var: tk.BooleanVar = var, key=toggle.key) -> None:
                self._action(
                    lambda: self._app.toggle_inclusion(
                        key[0], key[1], key[2], bool(var_var.get())
                    )
                )

            check.configure(command=on_toggle)

            def on_row_click(key=toggle.key) -> None:
                self._action(lambda: self._app.navigate_to_inclusion_row(
                    key[0], key[1]
                ))

            def on_click(event: Any, fn: Any = on_row_click) -> None:
                fn()

            for widget in (row, label):
                widget.bind("<Button-1>", on_click)

        status = state.inclusion_status_label or ""
        self._inclusion_status_var.set(status)

    def _sync_inclusion_vars(self) -> None:
        """Push draft values into the existing checkbox variables."""
        for toggle in self._app.state.inclusion_draft:
            var = self._inclusion_vars.get(toggle.key)
            if var is not None and var.get() != toggle.included:
                var.set(toggle.included)
        status = self._app.state.inclusion_status_label or ""
        if self._inclusion_status_var.get() != status:
            self._inclusion_status_var.set(status)

    # --- cooperative playback timer (NO threads) --------------------------------

    def _schedule_tick(self) -> None:
        interval = max(1, self._app.playhead.interval_ms)
        self.after(interval, self._on_tick)

    def _on_tick(self) -> None:
        if self._app.state.playing:
            self._app.consume_playhead_tick()
            self.refresh()
        self._schedule_tick()

    # --- display -----------------------------------------------------------------

    def refresh(self) -> None:
        """Redraw the canvas and the status/error label from current state."""
        frame = self._renderer.render(self._app.state)
        self._last_rendered_frame = frame
        self._axes.clear()
        self._axes.set_axis_off()
        self._axes.imshow(frame.canvas, interpolation="nearest")
        self._canvas.draw_idle()
        self._status_var.set(self._app.status)
        self._status_label.config(
            fg="#b22222" if self._app.status_is_error else "#202020"
        )
        playing = self._app.state.playing
        self._buttons["Play/Pause"].config(text="Pause" if playing else "Play")
        self._refresh_structure_menu()
        signature = selection_signature(self._app.state.selection)
        if not self._inclusion_vars or (
            signature != self._grid_selection_signature
        ):
            # Correction round 2 (finding 5): an ED/ES change REBUILDS the
            # rows so resolved-frame labels can never go stale.
            self._rebuild_inclusion_grid()
        else:
            self._sync_inclusion_vars()

    def _highlight_first_matching_row(self) -> None:
        """Navigate to the first row matching the selector combination."""
        role = self._selector_role_var.get()
        structure_by_label = {v: k for k, v in INCLUSION_STRUCTURE_LABELS.items()}
        structure = structure_by_label.get(self._selector_structure_var.get())
        if structure is None:
            return
        for toggle in self._app.state.inclusion_draft:
            if toggle.role == role and toggle.structure == structure:
                self._action(
                    lambda: self._app.navigate_to_inclusion_row(
                        toggle.slice_index, role
                    )
                )
                return


def build_synthetic_demo_session(
    store_path: Path,
) -> dict[str, Any]:
    """Build the SYNTHETIC demo workflow state, fully headless (S2C-001 §8.4).

    Corrected sequence — one coherent chain, no divergent measurement/QC
    fork:

        persist prediction
        -> persist contour chain including final contour head
        -> persist ED/ES selection (through ``record_ed_es``)
        -> ``refresh_head_analysis()`` creates the QC-bearing child revision
           AND the measurement bound to that child
        -> caller opens the shell; Approve seals the SAME current head

    Fixed timestamps and fixed gradient/rectangle content keep every derived
    identity deterministic. No Tk window, no display, no patient data.
    """
    import numpy as np
    from cmr_contours.editing import save_to_revision
    from cmr_contours.model import Contour
    from cmr_core.cine import CardiacFrame, CineVolume
    from cmr_core.domain_types import Prediction
    from cmr_core.geometry import ImageGeometry
    from cmr_core.provenance import Provenance
    from cmr_persistence.store import SqliteCmrStore
    from cmr_workflow.controller import WorkflowController

    stamp = "2026-08-23T00:00:00+00:00"
    series_uid = "synthetic-node8-demo-series"
    study_uid = "synthetic-node8-demo-study"
    software = "cmr-studio-node8-demo"

    rows, cols, n_frames = 48, 64, 8
    # Anisotropic spacing ON PURPOSE: the mm-extent renderer must show an
    # undistorted physical image (QA checklist item R1).
    geometry = ImageGeometry(
        rows=rows,
        cols=cols,
        pixel_spacing=(0.9, 1.1),
        slice_thickness=8.0,
        image_position_patient=(0.0, 0.0, 0.0),
    )

    def rect(x0: float, y0: float, w: float, h: float) -> list[tuple[float, float]]:
        return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]

    frames = [
        [
            CardiacFrame(
                0,
                f,
                (
                    np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
                    % 97.0
                )
                / 97.0
                + 0.05 * f,
                f"SYNTH-NODE8.{f}",
            )
            for f in range(n_frames)
        ]
    ]
    cine = CineVolume([geometry], frames)

    lv = Structure.LV_CAVITY
    rv = Structure.RV_CAVITY
    myo = Structure.LV_MYOCARDIUM

    # Frame 0 seed (ED): large cavities + epicardial myocardium ring.
    seed = {
        lv: Contour(0, 0, lv, rect(18, 14, 28, 22), source="manual"),
        rv: Contour(0, 0, rv, rect(50, 30, 10, 10), source="manual"),
        myo: Contour(0, 0, myo, rect(16, 12, 32, 26), source="manual"),
    }

    def frame_edits(frame_index: int) -> dict[Structure, Contour]:
        # Cavities shrink towards mid-systole (frame 4) so ED > ES honestly.
        shrink = 2.0 * min(frame_index, n_frames - 1 - frame_index)
        return {
            lv: Contour(
                0,
                frame_index,
                lv,
                rect(19 + shrink / 2, 15 + shrink / 2, 26 - shrink, 20 - shrink),
                source="manual",
            ),
            rv: Contour(
                0,
                frame_index,
                rv,
                rect(
                    51 + shrink / 4,
                    31 + shrink / 4,
                    8 - shrink / 2,
                    8 - shrink / 2,
                ),
                source="manual",
            ),
        }

    provenance = Provenance(study_uid, series_uid, software, None, stamp, 0)
    prediction = Prediction(
        series_uid, series_uid, "manual-seed", seed, None, provenance, stamp
    )

    controller = WorkflowController()
    store = SqliteCmrStore(store_path)
    try:
        # (1) prediction row first (FK root).
        store.save_prediction(prediction)
        # (2) contour chain INCLUDING the final contour head: v1 carries the
        # frame-0 seed verbatim (one contour per structure per revision is a
        # frozen constraint); later children add frames 1..n_frames-1.
        parent: Any = prediction
        seed_child = save_to_revision(
            parent,
            series_uid=series_uid,
            edits={},
            editor="demo-seed",
            created_at=stamp,
        )
        store.save_revision(seed_child)
        parent = seed_child
        for frame in range(1, n_frames):
            revision = save_to_revision(
                parent,
                series_uid=series_uid,
                edits=frame_edits(frame),
                editor="demo-seed",
                created_at=stamp,
            )
            store.save_revision(revision)
            parent = revision
        # (3) ED/ES selection through the S2C routing method.
        selection = controller.record_ed_es(
            store,
            cine,
            0,
            4,
            "demo-physician",
            stamp,
            lambda: Provenance(study_uid, series_uid, software, None, stamp, 0),
        )
        # (4) CFM-001: analysis refresh THROUGH the physician-authored
        # inclusion set — one atomic set+QC-child+measurement transaction.
        # The demo physician authors the complete all-included grid.
        entries = [
            {
                "slice_index": slice_index,
                "frame_role": role,
                "structure": structure,
                "included": True,
            }
            for slice_index in range(cine.n_slices)
            for role, structure in (
                ("ED", "lv_cavity"),
                ("ES", "lv_cavity"),
                ("ED", "rv_cavity"),
                ("ES", "rv_cavity"),
                ("ED", "lv_myocardium"),
            )
        ]
        receipt = controller.apply_inclusion_and_refresh_analysis(
            store,
            cine,
            series_uid,
            entries=entries,
            selected_by="demo-physician",
            software_version=software,
            created_at=stamp,
        )
        if receipt["status"] != "refreshed":
            raise RuntimeError("demo seeding expected a refreshed analysis")
        return {
            "controller": controller,
            "store": store,
            "cine": cine,
            "series_uid": series_uid,
            "study_uid": study_uid,
            "physician": "demo-physician",
            "selection": selection,
            "head_revision_id": receipt["head_revision_id"],
            "measurement": receipt["measurement"],
            "qc_verdict": receipt["qc_verdict"],
            "stamp": stamp,
        }
    except BaseException:
        store.close()
        raise


def main(argv: list[str] | None = None) -> int:
    """Launch the interactive viewer on a SYNTHETIC cine session.

    ``matplotlib.use('TkAgg')`` runs here ONLY — never at package import.
    The session is built by :func:`build_synthetic_demo_session` (fixed
    gradient frames, rectangle contours, throwaway SQLite store in a temp
    directory); see that function and the manual QA checklist. Returns the
    Tk exit code.
    """
    import tempfile

    import matplotlib

    matplotlib.use("TkAgg")

    store_dir = Path(tempfile.mkdtemp(prefix="cmr_node8_demo_"))
    session = build_synthetic_demo_session(store_dir / "demo.db")
    store: Any = session["store"]
    controller = session["controller"]
    cine = session["cine"]
    series_uid = session["series_uid"]

    def clock() -> str:
        return session["stamp"]  # fixed demo clock: deterministic identities

    app = ViewerApp.open(
        controller=controller,
        store=store,
        cine=cine,
        series_uid=series_uid,
        clock=clock,
        physician=session["physician"],
    )
    shell = ViewerShell(app, reopen_store=store, reopen_series_uid=series_uid, reopen_cine=cine)
    shell.mainloop()
    if store is not None:
        store.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
