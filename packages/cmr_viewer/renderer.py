"""Node 8 — deterministic Agg renderer: viewer state -> RGB frame + overlays.

``ViewerRenderer`` turns a frozen :class:`~cmr_viewer.state.ViewerState` into
a :class:`RenderedFrame`: a contiguous ``uint8`` RGB canvas (HxWx3) plus
structured overlay metadata (tuple of plain dicts with ``kind`` keys
``'contour' | 'marker' | 'hud'``). Rendering is PURE and DETERMINISTIC:

- ``matplotlib.figure.Figure`` is used DIRECTLY (never pyplot), one fresh
  figure per call, no global matplotlib state is read or written;
- figure size (:data:`FIGURE_SIZE_IN` inches), dpi (:data:`DPI`), font size
  (:data:`FONT_SIZE`), colormaps, interpolation, axis limits, line styles,
  colors and draw order are pinned constants below;
- ``vmin``/``vmax`` come ONLY from the displayed frame's own data range
  (degenerate single-value frames widen the range by exactly 1.0);
- NO PNG encoding participates anywhere: digests hash contiguous RGB pixels
  via :meth:`~numpy.ndarray.tobytes` plus canonical sorted-key JSON of the
  overlay metadata (see :func:`frame_digest`).

Physical display convention (AMENDMENT-001 D): the image is drawn with a
MILLIMETRE extent ``[0, cols*col_spacing_mm, rows*row_spacing_mm, 0]`` and
equal aspect, so anatomy is never distorted under anisotropic spacing.
Contour points are transformed ``(x*col_spacing_mm, y*row_spacing_mm)`` —
equivalent to :meth:`~cmr_contours.model.Contour.as_mm` over
:class:`~cmr_core.geometry.ImageGeometry.pixel_spacing`. Overlay metadata
carries BOTH the canonical pixel points and the derived millimetre points.

Pixel grid convention: pixel ``(x, y)`` covers ``[x, x+1] x [y, y+1]``
(matching :mod:`cmr_contours.conversion`); with ``origin='upper'`` and the
inverted y-axis above, pixel coordinates map onto millimetre axes exactly.

Draw order (pinned): (1) grayscale image, (2) chain-merged contours — latest
contour per ``(slice, frame, structure)`` across the WHOLE ``revision_chain``
(``build_cine_contour_index`` semantics, see
:meth:`ViewerState.merged_contours_at`) — in ``Structure`` definition order,
(3) unsaved draft contours (dashed), (4) ED marker band, (5) ES marker band,
(6) status/metrics/QC panel text. ED and ES markers are drawn INDEPENDENTLY at
fixed distinct vertical bands, so an ``ed_frame == es_frame`` selection
renders both labels legibly overlapping the same frame.

Quantification summary (AMENDMENT-001 G): every ``Quantity`` field actually
present on :class:`~cmr_core.domain_types.Measurement` is rendered with its
value and EXACT ``Unit.value`` string. No indexed quantity is ever
synthesized. When Node 5 legitimately produced no Measurement because QC
failed the input (N7.1 B-LITE sentinel path, visible through a decision
record anchored to the head), the literal sentinel text below replaces every
number. The upstream volume method is per-slice disc/slab summation — a
geometric area-times-gap method, not Simpson's-rule integration; that is the
single sanctioned mention in this package and it never appears in any
user-visible string.

S3-008B — display-only controls (spec v0.3.2-FROZEN §2.2/§2.3/§2.4):

- TRUE WINDOW/LEVEL: an explicit :class:`~cmr_viewer.state
  .DisplayWindowLevel` replaces the per-frame auto range with the exact DICOM
  linear boundaries ``lower = center - 0.5 - (width - 1.0)/2`` and
  ``upper = center - 0.5 + (width - 1.0)/2`` passed as ``vmin``/``vmax`` to
  the SAME grayscale ``imshow``. Source arrays are NEVER rescaled. This v1
  consumes RAW decoded pixel values — no modality/VOI LUT support exists or
  is implied.
- VISIBILITY: chain-merged AND unsaved draft contours are filtered by the
  effective visible set BEFORE drawing and BEFORE overlay metadata. Image,
  ED/ES bands, HUD, metrics, QC and inclusion display always remain visible.
  No contour object is mutated and editing authority is unchanged.
- DEFAULT IDENTITY: neither control emits any status line or display metadata
  while BOTH are ``None``, so the legacy render bytes are EXACTLY unchanged
  (golden-default identity, §5.2). The two conditional ``[STATUS]`` lines and
  their ``:g`` formatting are test-frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from cmr_contours.model import Contour
from cmr_core.geometry import ImageGeometry
from cmr_core.segmentation import Structure
from cmr_quantification.measurements import PAPILLARY_TRABECULATION_CONVENTION_V1
from cmr_quantification.reference_ranges import (
    REFERENCE_TABLE_VERSION,
    ReferenceQuantity,
    format_reference_line,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from cmr_viewer.state import (
    REFERENCE_UNAVAILABLE_STATUS,
    SCMR_BASAL_UNCERTAINTY_NOTICE,
    ViewerState,
)

__all__ = [
    "DPI",
    "ED_MARKER_COLOR",
    "ES_MARKER_COLOR",
    "FIGURE_SIZE_IN",
    "FONT_SIZE",
    "RenderedFrame",
    "SENTINEL_METRICS_TEXT",
    "STALE_ANALYSIS_NOTICE",
    "ViewerRenderer",
    "frame_digest",
    "pixel_from_outer_canvas_point",
]

#: Pinned figure size in inches (determinism contract F).
FIGURE_SIZE_IN = (8.0, 5.0)

#: Pinned render resolution (determinism contract F).
DPI = 100

#: Pinned font size in points for every drawn text element.
FONT_SIZE = 8

#: Pinned axes rectangles in figure fractions: image left, panel right.
IMAGE_AXES_RECT = (0.03, 0.05, 0.52, 0.90)
PANEL_AXES_RECT = (0.585, 0.04, 0.395, 0.92)

#: Fixed structure colors (documented, deterministic).
CONTOUR_COLORS: dict[Structure, str] = {
    Structure.LV_CAVITY: "#d62728",
    Structure.LV_MYOCARDIUM: "#ff7f0e",
    Structure.RV_CAVITY: "#1f77b4",
}

#: Fixed marker colors, distinct from each other and from all structures.
ED_MARKER_COLOR = "#2ca02c"
ES_MARKER_COLOR = "#9467bd"

#: Vertical band positions (fraction of image height) for the two marker
#: bars: ED occupies the topmost band, ES the band right below it, so an
#: ED==ES selection shows BOTH bars overlapping the same frame independently.
ED_MARKER_BAND = (0.000, 0.030)
ES_MARKER_BAND = (0.035, 0.065)

#: Literal replacing numbers when no Measurement exists because QC failed
#: the input (N7.1 B-LITE sentinel path).
SENTINEL_METRICS_TEXT = "metrics unavailable due to QC"

#: Literal for a series where metrics simply have not been computed yet.
NO_METRICS_TEXT = "no metrics recorded"

#: S2C-001 §8.3: visible stale notice shown beside a displayed measurement
#: whose values no longer describe the current head + latest selection.
STALE_ANALYSIS_NOTICE = "values NOT current (STALE) — press Refresh Analysis"

#: Every Quantity field of Measurement in fixed display order. Labels are
#: presentation-only; units come verbatim from ``Unit.value``.
QUANTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("LV EDV", "edv_ml"),
    ("LV ESV", "esv_ml"),
    ("LV SV", "sv_ml"),
    ("LV EF", "ef_pct"),
    ("LV mass", "lv_mass_g"),
    ("RV EDV", "rv_edv_ml"),
    ("RV ESV", "rv_esv_ml"),
    ("RV SV", "rv_sv_ml"),
    ("RV EF", "rv_ef_pct"),
)


def pixel_from_outer_canvas_point(
    geometry: ImageGeometry,
    mm_from_rgb: Sequence[Sequence[float]],
    x_outer: float,
    y_outer: float,
) -> tuple[float, float] | None:
    """Pure inverse click mapping: outer canvas pixel -> source pixel.

    S2C-001 §8.1 headless helper (the Tk shell only converts its top-left
    event origin to Matplotlib's bottom-left display origin and delegates):

    1. ``(u, v) = (x_outer + 0.5, y_outer + 0.5)`` converts OUTER ``imshow``
       pixel-centre coordinates to RGB EDGE coordinates (omitting the +0.5
       recreates the off-by-half-pixel error);
    2. ``(x_mm, y_mm, 1) = A_mm_from_rgb @ (u, v, 1)`` maps into clinical
       millimetres through the renderer affine captured at draw time;
    3. division by anisotropic spacings yields continuous source pixels.

    Returns ``None`` (never clips) when the inverse is non-finite or the
    result falls outside ``0..cols-1`` / ``0..rows-1`` (a 1e-6 float-noise
    tolerance keeps legitimate boundary pixels inside the honest domain).
    """
    matrix = np.asarray(mm_from_rgb, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"A_mm_from_rgb must be 3x3, got shape {matrix.shape}")
    u = float(x_outer) + 0.5
    v = float(y_outer) + 0.5
    x_mm, y_mm, _w = matrix @ np.array([u, v, 1.0])
    if _w == 0 or not (math.isfinite(x_mm) and math.isfinite(y_mm)):
        return None
    x_px = float(x_mm) / geometry.col_spacing_mm
    y_px = float(y_mm) / geometry.row_spacing_mm
    if not (math.isfinite(x_px) and math.isfinite(y_px)):
        return None
    tolerance = 1e-6
    if not (
        -tolerance <= x_px <= (geometry.cols - 1) + tolerance
        and -tolerance <= y_px <= (geometry.rows - 1) + tolerance
    ):
        return None
    return (
        min(max(x_px, 0.0), geometry.cols - 1.0),
        min(max(y_px, 0.0), geometry.rows - 1.0),
    )


@dataclass(frozen=True)
class RenderedFrame:
    """One deterministic render result.

    ``canvas`` is a contiguous ``uint8`` array of shape ``(H, W, 3)`` (RGB).
    ``overlay_metadata`` is a tuple of plain dicts keyed by ``kind``
    (``'contour'``, ``'marker'`` or ``'hud'``); contour entries carry both
    ``points_pixel`` and ``points_mm`` tuples.

    S2C-001 §8.1: ``mm_from_rgb_affine`` carries the 3x3 homogeneous affine
    from clinical millimetres to rendered RGB CANVAS EDGE coordinates,
    captured from the actual image-axes transform after drawing (display
    flip folded in), so exact click mapping round-trips through THIS frame.
    """

    canvas: np.ndarray
    overlay_metadata: tuple[dict[str, Any], ...]
    mm_from_rgb_affine: tuple[tuple[float, float, float], ...] = field(default=())


def frame_digest(frame: RenderedFrame) -> str:
    """SHA-256 over contiguous RGB pixels + canonical metadata JSON.

    NEVER hashes encoded PNG bytes (determinism contract F). Metadata is
    serialized with sorted keys and compact separators so equal logical
    content yields equal bytes. The click-mapping affine participates so two
    frames with identical pixels but different geometry transforms digest
    differently (S2C-001 §8.1).
    """
    metadata_json = json.dumps(
        list(frame.overlay_metadata),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    hasher = hashlib.sha256()
    hasher.update(frame.canvas.tobytes())
    hasher.update(b"\x00")
    hasher.update(metadata_json.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(
        json.dumps(
            [list(row) for row in frame.mm_from_rgb_affine],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    return hasher.hexdigest()


def _display_status_lines(state: ViewerState) -> tuple[str, ...]:
    """The two CONDITIONAL S3-008B ``[STATUS]`` lines (§2.4, test-frozen).

    Emitted ONLY when at least one display control is explicit — with BOTH
    controls ``None`` this returns the empty tuple, which is exactly what
    keeps the legacy default render bytes unchanged (§5.2).

    Exact literals and ``:g`` number formatting, in this order:

    - ``window: C=<center:g> W=<width:g>`` when W/L is explicit, else
      ``window: auto`` (W/L ``None`` while visibility IS explicit);
    - ``structures: <comma-space joined Structure.value>`` for a nonempty
      tuple, ``structures: none`` for the empty tuple, ``structures: all``
      when visibility is ``None`` while W/L IS explicit.
    """
    window_level = state.window_level
    visible = state.visible_structures
    if window_level is None and visible is None:
        return ()
    lines = [
        (
            f"window: C={window_level.center:g} W={window_level.width:g}"
            if window_level is not None
            else "window: auto"
        )
    ]
    if visible is None:
        lines.append("structures: all")
    elif not visible:
        lines.append("structures: none")
    else:
        lines.append(
            "structures: " + ", ".join(structure.value for structure in visible)
        )
    return tuple(lines)


def _contour_entries(
    state: ViewerState,
    slice_index: int,
    frame_index: int,
    *,
    draft: bool,
) -> tuple[dict[str, Any], ...]:
    """Overlay metadata entries for chain-merged or draft contours on view.

    S3-008B §2.3: candidates are filtered by the effective visible set BEFORE
    any metadata is produced, so a hidden contour is absent from BOTH the
    drawn pixels and the overlay metadata. Nothing is mutated and no editing
    authority changes.
    """
    geometry = state.cine.geometries[slice_index]
    row_spacing = geometry.row_spacing_mm
    col_spacing = geometry.col_spacing_mm
    entries: list[dict[str, Any]] = []
    if draft:
        candidates: tuple[Contour, ...] | list[Contour] = [
            contour
            for contour in state.draft_contours
            if contour.slice_index == slice_index
            and contour.frame_index == frame_index
        ]
    else:
        # Correction round 1 F1: the SAME chain-merge semantics as
        # build_cine_contour_index — latest contour per (slice, frame,
        # structure) across the WHOLE revision_chain, never head-only.
        candidates = state.merged_contours_at(slice_index, frame_index)
    # §2.3 visibility filter — applied to BOTH chain-merged and draft sets.
    candidates = [
        contour for contour in candidates if state.is_structure_visible(contour.structure)
    ]
    for contour in candidates:
        points_pixel = tuple((float(x), float(y)) for x, y in contour.points)
        points_mm = tuple(
            (x * col_spacing, y * row_spacing) for x, y in points_pixel
        )
        entries.append(
            {
                "kind": "contour",
                "structure": contour.structure.value,
                "slice_index": slice_index,
                "frame_index": frame_index,
                "draft": draft,
                "color": CONTOUR_COLORS[contour.structure],
                "source": contour.source,
                "edited": bool(contour.edited),
                "points_pixel": points_pixel,
                "points_mm": points_mm,
            }
        )
    return tuple(entries)


def _status_lines(state: ViewerState) -> tuple[str, ...]:
    geometry = state.cine.geometries[state.current_slice_index]
    height_mm = geometry.rows * geometry.row_spacing_mm
    width_mm = geometry.cols * geometry.col_spacing_mm
    playback = f"{'playing' if state.playing else 'paused'} @ {state.fps} fps"
    selection = (
        f"selection: ED={state.selection.ed_frame} ES={state.selection.es_frame}"
        if state.selection is not None
        else "selection: none"
    )
    inclusion_label = state.inclusion_status_label
    return (
        f"status: {state.approval_label}",
        f"series: {state.series_uid}",
        f"slice {state.current_slice_index + 1}/{state.cine.n_slices}   "
        f"frame {state.current_frame_index + 1}/{state.cine.n_frames}   {playback}",
        f"extent: {height_mm:.2f} x {width_mm:.2f} mm",
        f"spacing: row {geometry.row_spacing_mm:g} mm / "
        f"col {geometry.col_spacing_mm:g} mm",
        selection,
        # S2C-001 cluster F disclosure surfaced in the HUD verbatim (never
        # appended to any clinical algorithm version).
        f"convention: {PAPILLARY_TRABECULATION_CONVENTION_V1}",
        f"chain head version: "
        f"{state.head_revision.version if state.head_revision else '-'}",
        # CFM-001 §1.8: the exact no-set label / applied-set status, plus the
        # ALWAYS-VISIBLE SCMR uncertainty notice beside the grid (never
        # hidden in a tooltip).
        *(("" , inclusion_label) if inclusion_label else ()),
        SCMR_BASAL_UNCERTAINTY_NOTICE,
        # S3-008B §2.4: the two conditional display lines come LAST, after
        # every accepted line, and are absent entirely when both display
        # controls are None (default render bytes unchanged).
        *_display_status_lines(state),
    )


def _metrics_lines(state: ViewerState) -> tuple[str, ...]:
    measurement = state.latest_measurement
    if measurement is None:
        # Sentinel honesty: the literal replaces numbers ONLY when QC failed
        # the measurement itself (decision record anchored to the current
        # head); a pristine series simply reports that nothing was computed.
        anchored_decision = any(
            record.source_revision_id == state.head_revision_id
            for record in state.qc_decisions
        )
        return (SENTINEL_METRICS_TEXT,) if anchored_decision else (NO_METRICS_TEXT,)
    lines = [
        f"{label}: {getattr(measurement, field).value:.3f} "
        f"{getattr(measurement, field).unit.value}"
        for label, field in QUANTITY_FIELDS
    ]
    lines.append(f"algorithm: {measurement.algorithm_version}")
    lines.append(f"measurement id: {measurement.measurement_id[:16]}..")
    # S2C-001 §8.3 honesty rule: the metrics panel shows either the coherent
    # current measurement or a VISIBLE stale notice — old values are never
    # presented as current.
    if state.analysis_is_stale:
        lines.append(STALE_ANALYSIS_NOTICE)
    return tuple(lines)


def _reference_lines(state: ViewerState) -> tuple[str, ...]:
    """Neutral contextual reference-range lines (CFM-001 §2.7).

    Ranges are CONTEXT, never QC: no z-scores, percentiles, traffic lights,
    "normal/abnormal" wording or range-derived flags exist here, and a
    displayed range never alters approval status or any QCReport.

    Suppression rules (fail-honest):

    - legacy null-bound measurements carry NO table provenance → no ranges
      and no invented context;
    - an UNKNOWN bound ``range_table_version`` suppresses ranges rather
      than falling forward to the latest table;
    - without an explicitly supplied supported sex the LVEF range is
      suppressed; indexed quantities (LVEDVi/LVESVi/LVMi) additionally
      require ALREADY-indexed values which this repository cannot derive
      (no BSA provenance exists) — raw EDV/ESV/mass are NEVER converted —
      so a single neutral "reference unavailable" status is shown instead.
    """
    measurement = state.latest_measurement
    if measurement is None:
        return ()
    version = getattr(measurement, "range_table_version", None)
    if version is None:
        return ()
    if version != REFERENCE_TABLE_VERSION:
        return (REFERENCE_UNAVAILABLE_STATUS,)
    sex = state.reference_sex
    if sex is None:
        return (REFERENCE_UNAVAILABLE_STATUS,)
    ef_line = format_reference_line(
        measurement.ef_pct, sex, ReferenceQuantity.LVEF
    )
    if ef_line is None:
        return (REFERENCE_UNAVAILABLE_STATUS,)
    # format_reference_line itself carries the table/source provenance and
    # the exact general disclaimer (spec §2.7), plus the papillary warning
    # for LVMi rows; split multi-line output into panel lines.
    lines: list[str] = []
    for chunk in ef_line.split("\n"):
        lines.append(chunk)
    return tuple(lines)


def _qc_lines(state: ViewerState) -> tuple[str, ...]:
    report = state.qc_report
    if report is None:
        return ("verdict: -", "no QC report carried by the chain head")
    lines = [f"verdict: {report.verdict}", f"flags ({len(report.flags)}):"]
    lines.extend(
        f"{flag.severity.value} | {flag.code} | "
        f"source={flag.source if flag.source is not None else '-'} | "
        f"{flag.message}"
        for flag in report.flags
    )
    return tuple(lines)


class ViewerRenderer:
    """Pure deterministic renderer; holds no mutable session state."""

    def render(self, state: ViewerState) -> RenderedFrame:
        """Render ``state`` into an RGB canvas plus overlay metadata.

        See the module docstring for the pinned determinism contract and the
        documented draw order. This method performs no I/O, spawns no
        threads and touches no pyplot globals.
        """
        geometry = state.cine.geometries[state.current_slice_index]
        rows = geometry.rows
        cols = geometry.cols
        row_spacing = geometry.row_spacing_mm
        col_spacing = geometry.col_spacing_mm
        height_mm = rows * row_spacing
        width_mm = cols * col_spacing

        # The source array is read for display ONLY: it is never rescaled,
        # rewritten or persisted (S3-008B §5.5 display-only invariance).
        frame_data = np.asarray(
            state.cine.frame(
                state.current_slice_index, state.current_frame_index
            ).pixel_data,
            dtype=np.float64,
        )
        window_level = state.window_level
        if window_level is None:
            # Existing per-frame auto range EXACTLY as accepted: degenerate
            # single-value frames widen the range by exactly 1.0.
            value_min = float(frame_data.min())
            value_max = float(frame_data.max())
            if value_max <= value_min:
                value_max = value_min + 1.0
        else:
            # §2.2 exact DICOM linear display-window boundaries. Values below
            # /above are clipped by the colormap itself — no source rescale.
            value_min = window_level.lower
            value_max = window_level.upper

        figure = Figure(figsize=FIGURE_SIZE_IN, dpi=DPI)
        agg_canvas = FigureCanvasAgg(figure)
        ax_image = figure.add_axes(IMAGE_AXES_RECT)
        ax_panel = figure.add_axes(PANEL_AXES_RECT)

        # (1) grayscale image, millimetre extent, equal physical aspect.
        ax_image.imshow(
            frame_data,
            cmap="gray",
            vmin=value_min,
            vmax=value_max,
            interpolation="nearest",
            origin="upper",
            extent=(0.0, width_mm, height_mm, 0.0),
            aspect="equal",
        )
        ax_image.set_xlim(0.0, width_mm)
        ax_image.set_ylim(height_mm, 0.0)
        ax_image.tick_params(labelsize=FONT_SIZE)

        metadata: list[dict[str, Any]] = []

        # (2) chain-head contours (pixel -> mm transform), definition order.
        for entry in _contour_entries(
            state, state.current_slice_index, state.current_frame_index, draft=False
        ):
            points_mm = entry["points_mm"]
            xs_mm = [point[0] for point in points_mm] + [points_mm[0][0]]
            ys_mm = [point[1] for point in points_mm] + [points_mm[0][1]]
            ax_image.plot(
                xs_mm,
                ys_mm,
                color=str(entry["color"]),
                linewidth=1.5,
                linestyle="-",
            )
            metadata.append(entry)

        # (3) unsaved draft contours (the app's edit gesture), dashed.
        for entry in _contour_entries(
            state, state.current_slice_index, state.current_frame_index, draft=True
        ):
            points_mm = entry["points_mm"]
            xs_mm = [point[0] for point in points_mm] + [points_mm[0][0]]
            ys_mm = [point[1] for point in points_mm] + [points_mm[0][1]]
            ax_image.plot(
                xs_mm,
                ys_mm,
                color=str(entry["color"]),
                linewidth=1.2,
                linestyle="--",
            )
            metadata.append(entry)

        # (4)+(5) ED / ES marker bands, drawn independently (overlap legal).
        if state.selection is not None:
            for marker_label, marker_frame, color, band in (
                ("ED", state.selection.ed_frame, ED_MARKER_COLOR, ED_MARKER_BAND),
                ("ES", state.selection.es_frame, ES_MARKER_COLOR, ES_MARKER_BAND),
            ):
                if state.current_frame_index != marker_frame:
                    continue
                band_top = band[0] * height_mm
                band_bottom = band[1] * height_mm
                ax_image.add_patch(
                    Rectangle(
                        (0.0, band_top),
                        width_mm,
                        band_bottom - band_top,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.25,
                        linewidth=1.2,
                    )
                )
                ax_image.text(
                    0.01 * width_mm,
                    0.5 * (band_top + band_bottom),
                    marker_label,
                    color=color,
                    fontsize=FONT_SIZE,
                    va="center",
                    ha="left",
                    fontweight="bold",
                )
                metadata.append(
                    {
                        "kind": "marker",
                        "marker": marker_label,
                        "frame_index": marker_frame,
                        "current_frame_index": state.current_frame_index,
                        "color": color,
                    }
                )

        # (6) status / metrics / reference / QC panel.
        sections = (
            ("status", _status_lines(state)),
            ("metrics", _metrics_lines(state)),
            ("reference", _reference_lines(state)),
            ("qc", _qc_lines(state)),
        )
        panel_lines: list[str] = []
        for section_name, lines in sections:
            panel_lines.append(f"[{section_name.upper()}]")
            panel_lines.extend(lines)
            panel_lines.append("")
        ax_panel.axis("off")
        ax_panel.text(
            0.0,
            1.0,
            "\n".join(panel_lines),
            family="monospace",
            fontsize=FONT_SIZE,
            va="top",
            ha="left",
            transform=ax_panel.transAxes,
        )
        for section_name, lines in sections:
            metadata.append(
                {
                    "kind": "hud",
                    "section": section_name,
                    "lines": tuple(lines),
                    # CFM-001 §2.7: ranges render with ONE neutral style
                    # regardless of where the value lies relative to the
                    # range — identical style metadata for in-range and
                    # out-of-range values, by construction.
                    "style": "neutral",
                }
            )

        agg_canvas.draw()
        rgba = np.asarray(agg_canvas.buffer_rgba(), dtype=np.uint8)
        canvas_rgb = np.ascontiguousarray(rgba[:, :, :3])

        # S2C-001 §8.1: capture the EXACT image-axes affine from clinical
        # millimetres to display coordinates AFTER drawing (so window/aspect
        # resolution is final), then fold in the RGB row flip
        # F = [[1,0,0],[0,-1,H],[0,0,1]] to reach RGB canvas EDGE
        # coordinates. The inverse is applied by ``pixel_from_outer_canvas_point``.
        H = int(canvas_rgb.shape[0])
        a_display_from_mm = np.asarray(
            ax_image.transData.get_affine().get_matrix(), dtype=np.float64
        )
        flip_to_rgb = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(H)], [0.0, 0.0, 1.0]])
        rgb_from_mm = flip_to_rgb @ a_display_from_mm
        mm_from_rgb = np.linalg.inv(rgb_from_mm)
        affine = tuple(
            tuple(float(value) for value in row) for row in mm_from_rgb
        )
        typed_affine: tuple[tuple[float, float, float], ...] = (
            (affine[0][0], affine[0][1], affine[0][2]),
            (affine[1][0], affine[1][1], affine[1][2]),
            (affine[2][0], affine[2][1], affine[2][2]),
        )
        return RenderedFrame(
            canvas=canvas_rgb,
            overlay_metadata=tuple(metadata),
            mm_from_rgb_affine=typed_affine,
        )
