"""Node 8 — immutable viewer state over accepted primitives only.

``ViewerState`` is a FROZEN dataclass composing exclusively accepted
primitives (:class:`~cmr_core.cine.CineVolume`,
:class:`~cmr_core.domain_types.ContourRevision` chain,
:class:`~cmr_core.domain_types.EDESSelection`,
:class:`~cmr_core.domain_types.Measurement`,
:class:`~cmr_core.domain_types.ApprovalSnapshot`,
:class:`~cmr_qc.report.QCReport`, N7.1
:class:`~cmr_persistence.records.QCDecisionRecord`). It holds NO clinical
logic, computes NO metrics and makes NO approval decisions — it carries
exactly what ``WorkflowController.reopen()`` plus the reloaded cine provide,
so the renderer and status surfaces can display it deterministically.

Hardening (AMENDMENT-001 B/C):

- collection-valued fields are TUPLES (approvals, qc_decisions,
  revision_chain, draft_contours); nothing list-shaped crosses a boundary;
- ``revision_chain`` is carried explicitly because persistence stores no
  pixels AND the renderer must draw chain-head geometry;
- ``__post_init__`` coherence checks fail closed: series identity against
  every chain revision (head provenance series included), current slice/
  frame indices within the cine grid, selection frames within cine bounds
  when present, fps bounded to :data:`MIN_FPS`..:data:`MAX_FPS`;
- ``draft_contours`` are UNSAVED working-copy overlays produced by the app's
  edit gesture (never persisted, never part of the chain identity);
- collection fields REJECT non-tuple inputs fail-closed and the carried
  ``qc_report`` must match the chain head's gateway report (correction
  round 1 F3), so stale QC can never be displayed;
- :meth:`merged_contours_at` exposes the chain-merge view (latest contour per
  ``(slice, frame, structure)`` across the WHOLE chain,
  ``build_cine_contour_index`` semantics) that rendering and editing consume
  (correction round 1 F1).

Three distinct approval states (AMENDMENT-001 C) with precedence
``APPROVED > REVIEW DECISION RECORDED > UNAPPROVED`` are exposed through
:data:`base_approval_label <ViewerState.base_approval_label>` using EXACTLY
these label strings:

- :data:`APPROVED_LABEL` = ``"APPROVED"``;
- :data:`DECISION_RECORDED_LABEL` =
  ``"REVIEW DECISION RECORDED - NOT APPROVED"``;
- :data:`UNAPPROVED_LABEL` = ``"UNAPPROVED"``.

Both non-default states are anchored to the CURRENT head revision
(current-head-only policy, mirroring save/approve): a decision or approval
recorded on an older revision honestly describes that older state, not the
current head.

S2C-001 cluster E — STALE modifier: the three base states are PRESERVED
verbatim; staleness is a SEPARATE modifier exposed through
:data:`STALE_SUFFIX` appended by :data:`approval_label
<ViewerState.approval_label>` ONLY when the base is ``UNAPPROVED`` and a
displayed artifact (historical measurement, approval, decision, or an
unsaved draft) does not match the CURRENT head plus the latest selection.
A newer selection wins by STORE SEQUENCE, never by ``selected_at``.

S3-008B (display controls, spec v0.3.2-FROZEN §2.1) adds two EPHEMERAL
DISPLAY-ONLY trailing fields — :attr:`ViewerState.window_level` and
:attr:`ViewerState.visible_structures` — together with the standalone
:class:`DisplayWindowLevel` domain type. They are request-scoped view
context: they participate in NO approval, staleness, measurement, inclusion
or provenance decision anywhere in this module, are NEVER persisted and are
NEVER clinical truth. Both default to ``None``, which reproduces the
pre-S3-008B rendering byte-for-byte (the golden-default identity, spec §5.2).

Binding-aware strictness (S2C-001 clusters B/C): when the displayed records
carry S2C binding metadata (``selection_id``, ``qc_report_digest``,
manifest ``workflow_id``), ``APPROVED`` additionally requires no drafts,
current-head/measurement/selection/approval coherence, approval QC-digest
equality against the current head's gateway report, and workflow equality.
Purely legacy (unbound) records keep the original two-key rule — binding
fields are read defensively so structural display stubs stay legal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from cmr_contours.model import Contour
from cmr_core.cine import CineVolume
from cmr_core.domain_types import (
    ApprovalSnapshot,
    ContourRevision,
    EDESSelection,
    Measurement,
    canonical_digest,
    head_tombstoned_keys,
)
from cmr_core.errors import INVALID_ARGUMENT, CMRError
from cmr_core.segmentation import Structure
from cmr_persistence.records import (
    QCDecisionRecord,
    SliceInclusionSet,
    TopologyDispositionRecord,
)
from cmr_qc.report import QCReport
from cmr_quantification.reference_ranges import ReferenceSex

__all__ = [
    "ANALYSIS_UNAVAILABLE_ESV_EXCEEDS_EDV",
    "APPROVED_LABEL",
    "BASAL_INCLUSION_UNCONFIRMED_LABEL",
    "DECISION_RECORDED_LABEL",
    "DEFAULT_FPS",
    "DisplayWindowLevel",
    "InclusionToggle",
    "MAX_FPS",
    "MIN_FPS",
    "NUMBER_LEX_RE",
    "REFERENCE_UNAVAILABLE_STATUS",
    "SCMR_BASAL_UNCERTAINTY_NOTICE",
    "STALE_SUFFIX",
    "STRUCTURE_ORDER",
    "UNAPPROVED_LABEL",
    "ViewerState",
    "canonical_number_lex",
    "canonical_visible_structures_json",
    "default_visible_structures",
    "window_level_json",
]

#: Exact visible label for a verified approval bound to head + measurement.
APPROVED_LABEL = "APPROVED"

#: Exact visible label for the N7.1 sentinel decision-record path.
DECISION_RECORDED_LABEL = "REVIEW DECISION RECORDED - NOT APPROVED"

#: Exact visible label for everything else.
UNAPPROVED_LABEL = "UNAPPROVED"

#: S2C-001 stale modifier appended AFTER a base label (never replaces it).
STALE_SUFFIX = " — STALE"

#: CFM-001 §1.3 exact label when no applicable inclusion set exists.
BASAL_INCLUSION_UNCONFIRMED_LABEL = (
    "BASAL INCLUSION UNCONFIRMED — LEGACY ALL-SLICES DEFAULT"
)

#: CFM-001 §1.8 always-visible SCMR uncertainty notice (verbatim).
SCMR_BASAL_UNCERTAINTY_NOTICE = (
    "BASAL-SLICE UNCERTAINTY — Physician judgment required. SCMR 2013 "
    "combines myocardial enclosure with systolic expansion; SCMR 2020 lists "
    "an “at least 50% surrounded” alternative and states there is no expert "
    "consensus. CMR Studio applies no automatic threshold. Each control "
    "includes or excludes the entire slice slab."
)

#: CFM-001 §1.6 exact viewer text after post-inclusion ESV>EDV rejection.
ANALYSIS_UNAVAILABLE_ESV_EXCEEDS_EDV = (
    "ANALYSIS UNAVAILABLE — ESV EXCEEDS EDV AFTER SLICE INCLUSION"
)

#: CFM-001 §2.7 neutral status when a range cannot be shown honestly.
REFERENCE_UNAVAILABLE_STATUS = "reference unavailable"

#: The five mandatory inclusion groups in canonical order
#: ``(role, structure)``; ED < ES and lv_cavity < rv_cavity < lv_myocardium.
# Correction round 2 (finding 5): role-primary canonical order, matching the
# frozen spec order and measurements._MANDATORY_GROUPS.
INCLUSION_GRID_GROUPS: tuple[tuple[str, str], ...] = (
    ("ED", "lv_cavity"),
    ("ED", "rv_cavity"),
    ("ED", "lv_myocardium"),
    ("ES", "lv_cavity"),
    ("ES", "rv_cavity"),
)


@dataclass(frozen=True)
class InclusionToggle:
    """One immutable draft checkbox of the basal-inclusion grid."""

    slice_index: int
    role: str
    structure: str
    included: bool

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.slice_index, self.role, self.structure)


def all_included_toggles(n_slices: int) -> tuple[InclusionToggle, ...]:
    """The visible all-included default grid: every toggle checked.

    Canonical row order is ``(slice_index, role_order, structure_order)``
    with ED < ES and lv_cavity < rv_cavity < lv_myocardium.
    """
    toggles: list[InclusionToggle] = []
    for slice_index in range(n_slices):
        for role, structure in INCLUSION_GRID_GROUPS:
            toggles.append(
                InclusionToggle(slice_index, role, structure, True)
            )
    return tuple(toggles)


def toggles_of_inclusion_set(record: SliceInclusionSet) -> tuple[InclusionToggle, ...]:
    """Draft grid initialized FROM a persisted applicable set."""
    mapping = {
        (entry.slice_index, entry.frame_role.value, entry.structure.value): (
            entry.included
        )
        for entry in record.entries
    }
    return tuple(
        InclusionToggle(slice_index, role, structure, mapping[(slice_index, role,
                                                                 structure)])
        for slice_index in range(max(key[0] for key in mapping) + 1)
        for role, structure in INCLUSION_GRID_GROUPS
    )

#: Inclusive lower playback-rate bound (frames per second).
MIN_FPS = 1

#: Inclusive upper playback-rate bound (frames per second).
MAX_FPS = 60

#: Default playback rate used by fresh sessions and the interactive shell.
DEFAULT_FPS = 10


def _field_or_none(record: object, name: str) -> object:
    """Defensive optional-field reader for structural display records.

    Legacy v2 payloads reconstruct into records WITHOUT the S2C additive
    fields populated; purely structural display stubs may omit them
    entirely. Missing/unset binding metadata reads as ``None`` so strict
    coherence checks engage ONLY for genuinely bound records.
    """
    value = getattr(record, name, None)
    return value


def _workflow_of(record: object) -> str | None:
    provenance = _field_or_none(record, "provenance")
    if provenance is None:
        return None
    workflow = getattr(provenance, "workflow_id", None)
    return None if workflow is None else str(workflow)


#: The frozen ``Structure`` definition order (S3-008B §2.3/§3.3): visibility
#: checkboxes, overlay filtering and serialization all use EXACTLY this order.
STRUCTURE_ORDER: tuple[Structure, ...] = tuple(Structure)


def canonical_number_lex(value: float) -> str:
    """Render one domain-normalized number with the SHARED LEXICAL GRAMMAR.

    S3-008B §3.2 (frozen, cross-runtime): shortest-roundtrip decimal, then
    drop a trailing ``".0"``. The Python recipe is
    ``repr(float(value))`` plus the ``".0"`` strip; the JS recipe is
    ``String(Number(value))``. Both produce the SAME token stream over the
    whole accepted domain, which the §5.7 cross-runtime test pins.

    Exponent spellings (``1e+16``) are outside the frozen grammar: §2.1
    already rejects non-finite values, and the caller must reject any value
    whose lexical form leaves the grammar rather than emit it.
    """
    token = repr(float(value))
    if token.endswith(".0"):
        token = token[:-2]
    return token


def _number_lex_or_fail(value: float, name: str) -> str:
    token = canonical_number_lex(value)
    if not NUMBER_LEX_RE.fullmatch(token):
        raise CMRError(
            INVALID_ARGUMENT,
            f"{name} {value!r} is outside the frozen display number grammar "
            f"(no exponent forms, no leading zeros, no trailing '.')",
        )
    return token


#: The frozen number-lex grammar (§3.2): decimal only, no exponent forms, no
#: leading zeros on the integer part, at least one digit after '.'. Exported
#: so the wire layer can validate the LEXICAL form of an incoming number
#: (Python's json accepts several spellings this grammar excludes).
NUMBER_LEX_RE = re.compile(r"^(0|[1-9][0-9]*(\.[0-9]+)?)$")


def window_level_json(window_level: DisplayWindowLevel) -> str:
    """Canonical compact sorted-key JSON for one W/L (S3-008B §3.2).

    Returns e.g. ``{"center":40,"width":400}`` — sorted keys, compact
    separators, numbers rendered by the shared lexical grammar. The bytes
    are the cache-key/query input and are test-frozen.
    """
    return (
        '{"center":'
        + _number_lex_or_fail(window_level.center, "window_level.center")
        + ',"width":'
        + _number_lex_or_fail(window_level.width, "window_level.width")
        + "}"
    )


def canonical_visible_structures_json(visible: tuple[Structure, ...]) -> str:
    """Canonical compact JSON array of ``Structure.value`` strings (§3.2).

    Order is the frozen enum order (already enforced at construction).
    """
    return "[" + ",".join(f'"{structure.value}"' for structure in visible) + "]"


def default_visible_structures() -> tuple[Structure, ...]:
    """The ALL-visible default: every ``Structure`` in frozen enum order."""
    return STRUCTURE_ORDER


def _normalize_number(value: object, name: str) -> float:
    """Accept an actual ``int``/``float`` and return a normalized ``float``.

    S3-008B §2.1: Booleans are REJECTED (a bool is not a number here), as are
    non-numeric types, NaN and infinities. Every accepted value is normalized
    with ``float(value)`` BEFORE any serialization, digest, cache-key or
    equality use, so ``DisplayWindowLevel(1, 512)`` and
    ``DisplayWindowLevel(1.0, 512.0)`` share one canonical representation.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CMRError(
            INVALID_ARGUMENT,
            f"{name} must be an actual int or float number "
            f"(Booleans are rejected), got {type(value).__name__}",
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CMRError(
            INVALID_ARGUMENT,
            f"{name} must be a finite number, got {normalized!r}",
        )
    return normalized


@dataclass(frozen=True)
class DisplayWindowLevel:
    """One explicit display-only grayscale window (S3-008B §2.1).

    ``center``/``width`` are RAW decoded pixel values (this v1 consumes raw
    values; DICOM rescale/VOI LUT is out of scope and is never implied).

    Validation is STANDALONE: ``__post_init__`` raises
    :class:`~cmr_core.errors.CMRError` with ``INVALID_ARGUMENT`` for Booleans,
    non-numeric types, non-finite values and any ``width <= 1.0``, so every
    surviving instance is valid before any ``ViewerState`` or renderer sees
    it. The constructor also NORMALIZES both numbers to Python ``float``
    (finding 2), so ``repr``/hash/equality never carry an ``int`` spelling.
    """

    center: float
    width: float

    def __post_init__(self) -> None:
        # Normalize FIRST (int -> float), then validate the normalized value.
        # object.__setattr__ is required by frozen dataclass semantics; this
        # is the single normalization point mandated by §2.1.
        center = _normalize_number(self.center, "window_level.center")
        width = _normalize_number(self.width, "window_level.width")
        if not width > 1.0:
            raise CMRError(
                INVALID_ARGUMENT,
                f"window_level.width must be greater than 1.0, got {width!r}",
            )
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "width", width)

    @property
    def lower(self) -> float:
        """Inclusive lower display boundary (§2.2 exact formula)."""
        return self.center - 0.5 - (self.width - 1.0) / 2.0

    @property
    def upper(self) -> float:
        """Inclusive upper display boundary (§2.2 exact formula)."""
        return self.center - 0.5 + (self.width - 1.0) / 2.0

    def canonical_json(self) -> str:
        """The frozen canonical JSON spelling (§3.2)."""
        return window_level_json(self)


def _validate_visible_structures(
    value: object,
) -> tuple[Structure, ...] | None:
    """Validate/normalize the ``visible_structures`` display field (§2.1).

    ``None`` means ALL structures. An empty tuple means NO contours. A
    nonempty tuple must contain unique actual :class:`Structure` members in
    EXACT enum-definition order; tuples only (never list/set/dict), and
    duplicates or out-of-order members fail closed.
    """
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise CMRError(
            INVALID_ARGUMENT,
            "visible_structures must be a tuple of Structure members or None "
            f"(frozen state carries no mutable collections), got "
            f"{type(value).__name__}",
        )
    for member in value:
        if not isinstance(member, Structure):
            raise CMRError(
                INVALID_ARGUMENT,
                "visible_structures entries must be Structure enum members, got "
                f"{type(member).__name__}",
            )
    if len(set(value)) != len(value):
        raise CMRError(
            INVALID_ARGUMENT,
            "visible_structures must not contain duplicate structures",
        )
    expected = tuple(
        structure for structure in STRUCTURE_ORDER if structure in set(value)
    )
    if expected != value:
        raise CMRError(
            INVALID_ARGUMENT,
            "visible_structures must be sorted in Structure enum definition "
            "order, got "
            f"{[member.value for member in value]!r}",
        )
    return value


def validate_fps(fps: int) -> int:
    """Return ``fps`` unchanged when it is an integer within 1..60.

    Fails closed on bools, non-integers and out-of-range values (the frozen
    state validates; interactive clamping lives in the playhead controller).
    """
    if isinstance(fps, bool) or not isinstance(fps, int):
        raise CMRError(
            INVALID_ARGUMENT,
            f"fps must be an integer number of frames per second, got {fps!r}",
        )
    if not MIN_FPS <= fps <= MAX_FPS:
        raise CMRError(
            INVALID_ARGUMENT,
            f"fps {fps} is outside the supported range {MIN_FPS}..{MAX_FPS}",
        )
    return fps


def _qc_report_matches_head(
    carried: QCReport | None, head_report: QCReport | None
) -> bool:
    """Whether the carried QC report IS the chain head's gateway report.

    Content comparison over the flag lists (frozen clones store tuples while
    freshly built reports store lists, so identity/dataclass equality would
    produce false mismatches). Both-absent agrees; one-sided disagrees —
    displaying no QC while the head carries one (or vice versa) would be
    exactly the stale-display hazard correction round 1 F3 closes.
    """
    if carried is None or head_report is None:
        return carried is None and head_report is None
    return list(carried.flags) == list(head_report.flags)


@dataclass(frozen=True)
class ViewerState:
    """Immutable snapshot of everything the viewer displays for one series.

    See the module docstring for the hardening rules and the three-state
    approval semantics. Construction fails closed on any incoherence instead
    of displaying a misleading clinical surface.
    """

    series_uid: str
    cine: CineVolume
    current_slice_index: int
    current_frame_index: int
    selection: EDESSelection | None
    latest_measurement: Measurement | None
    qc_report: QCReport | None
    approvals: tuple[ApprovalSnapshot, ...] = ()
    qc_decisions: tuple[QCDecisionRecord, ...] = ()
    revision_chain: tuple[ContourRevision, ...] = ()
    draft_contours: tuple[Contour, ...] = ()
    playing: bool = False
    fps: int = DEFAULT_FPS
    # --- CFM-001 basal-inclusion display state ---------------------------------
    #: UNSAVED checkbox draft over the complete 5xN grid. Checkbox changes
    #: create the draft and invalidate metrics immediately (stale).
    inclusion_draft: tuple[InclusionToggle, ...] = ()
    #: The APPLICABLE persisted set for the CURRENT head + latest selection,
    #: or None when no applicable set exists (legacy all-slices default).
    persisted_inclusion_set: SliceInclusionSet | None = None
    #: Explicitly supplied reference sex for HUD ranges (never inferred,
    #: never persisted; CFM-001 introduces no demographics storage).
    reference_sex: ReferenceSex | None = None
    # --- S3-008B display-only view context (spec v0.3.2-FROZEN §2.1) -------
    #: EPHEMERAL DISPLAY-ONLY explicit grayscale window, or ``None`` for the
    #: existing per-frame auto range. Never persisted, never clinical truth,
    #: never an action field. MUST stay LAST-but-one so appending it cannot
    #: reorder any accepted field (``visible_structures`` is appended last).
    window_level: DisplayWindowLevel | None = None
    #: EPHEMERAL DISPLAY-ONLY contour visibility filter: ``None`` = all
    #: structures, empty tuple = no contours, else unique ``Structure``
    #: members in exact enum order. Filters drawing + overlay metadata only;
    #: it never deletes/modifies contour objects or editing authority.
    visible_structures: tuple[Structure, ...] | None = None
    # --- S3-009 D7 whole-cine lineage identity (prediction or revision CAS head) ---
    whole_cine_head_id: str | None = None
    # --- Astra F1-F5: topology disposition display state (append-only) -----
    #: Every durable physician disposition recorded for this series, in the
    #: store's deterministic order. Display/audit only: a disposition never
    #: supplies geometry and never enters quantification.
    topology_dispositions: tuple[TopologyDispositionRecord, ...] = ()
    # --- Viewer disposition controls v4 (additive, trailing) -----------------
    #: Precomputed topology resolutions for the CURRENT head + selection,
    #: populated by ``services.workstation.actions.build_viewer_state`` via
    #: ``WorkflowController.resolve_topology``. Each entry is a
    #: ``TopologyResolution`` (kept as ``Any`` here so this domain layer does
    #: not depend on the workflow resolver). Empty means no known blocks or a
    #: legacy construction without resolution inputs; the wire projection
    #: renders an empty key list in that case. Display only.
    topology_resolutions: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.series_uid, str) or not self.series_uid.strip():
            raise CMRError(
                INVALID_ARGUMENT,
                f"series_uid must be a non-empty string, got {self.series_uid!r}",
            )
        if not isinstance(self.playing, bool):
            raise CMRError(
                INVALID_ARGUMENT,
                f"playing must be a bool, got {type(self.playing).__name__}",
            )
        validate_fps(self.fps)
        if self.whole_cine_head_id is not None and (
            not isinstance(self.whole_cine_head_id, str) or not self.whole_cine_head_id.strip()
        ):
            raise CMRError(
                INVALID_ARGUMENT,
                "whole_cine_head_id must be a non-empty string or None, "
                f"got {self.whole_cine_head_id!r}",
            )

        # Correction round 1 F3: collection-valued fields MUST be tuples.
        # A list or dict would keep the "frozen" state shallowly mutable
        # (Sol critique point 3); reject anything else fail-closed.
        for field_name in (
            "approvals",
            "qc_decisions",
            "revision_chain",
            "draft_contours",
            "inclusion_draft",
            "topology_resolutions",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"{field_name} must be a tuple (frozen state carries no "
                    f"mutable collections), got {type(value).__name__}",
                )
        for toggle in self.inclusion_draft:
            if not isinstance(toggle, InclusionToggle):
                raise CMRError(
                    INVALID_ARGUMENT,
                    "inclusion_draft entries must be InclusionToggle instances, "
                    f"got {type(toggle).__name__}",
                )
            if not isinstance(toggle.included, bool):
                raise CMRError(
                    INVALID_ARGUMENT,
                    "inclusion toggle values must be actual Booleans, got "
                    f"{toggle.included!r}",
                )
        if self.reference_sex is not None and not isinstance(
            self.reference_sex, ReferenceSex
        ):
            raise CMRError(
                INVALID_ARGUMENT,
                "reference_sex must be a ReferenceSex member or None; sex is "
                "never inferred from names, contours or DICOM fields",
            )
        # --- S3-008B §2.1 display-context validation ----------------------
        # Appended AFTER every accepted check so no pre-existing validation is
        # reordered. A DisplayWindowLevel is self-validating by construction;
        # this rejects any non-None value that is not that exact type.
        if self.window_level is not None and not isinstance(
            self.window_level, DisplayWindowLevel
        ):
            raise CMRError(
                INVALID_ARGUMENT,
                "window_level must be a DisplayWindowLevel instance or None, got "
                f"{type(self.window_level).__name__}",
            )
        # The field is validated/canonicalized here; a frozen dataclass needs
        # object.__setattr__ to store the normalized tuple.
        object.__setattr__(
            self,
            "visible_structures",
            _validate_visible_structures(self.visible_structures),
        )
        if self.persisted_inclusion_set is not None and (
            self.persisted_inclusion_set.series_uid != self.series_uid
        ):
            raise CMRError(
                INVALID_ARGUMENT,
                "persisted inclusion set carries series "
                f"{self.persisted_inclusion_set.series_uid!r}, state carries "
                f"{self.series_uid!r}",
            )

        n_slices = self.cine.n_slices
        n_frames = self.cine.n_frames
        if not 0 <= self.current_slice_index < n_slices:
            raise CMRError(
                INVALID_ARGUMENT,
                f"current_slice_index {self.current_slice_index} is outside the "
                f"cine slice range 0..{n_slices - 1}",
            )
        if not 0 <= self.current_frame_index < n_frames:
            raise CMRError(
                INVALID_ARGUMENT,
                f"current_frame_index {self.current_frame_index} is outside the "
                f"cine frame range 0..{n_frames - 1}",
            )

        for revision in self.revision_chain:
            if revision.series_uid != self.series_uid:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"revision {revision.revision_id!r} carries series "
                    f"{revision.series_uid!r}, state carries {self.series_uid!r}; "
                    "the chain belongs to exactly one series",
                )
        head = self.head_revision
        if head is not None and head.provenance.series_uid != self.series_uid:
            raise CMRError(
                INVALID_ARGUMENT,
                f"chain head provenance carries series "
                f"{head.provenance.series_uid!r}, state carries "
                f"{self.series_uid!r}",
            )

        # Correction round 1 F3: the displayed QC outcome must BE the chain
        # head's gateway report (or both absent) — never a stale/mismatched
        # report from an earlier revision or an unsealed recomputation.
        if not _qc_report_matches_head(self.qc_report, None if head is None else head.qc_report):
            raise CMRError(
                INVALID_ARGUMENT,
                "qc_report does not match the revision chain head's gateway "
                "QCReport; stale QC output cannot be displayed (carry the head "
                "report verbatim via reopen_viewer_state)",
            )

        if self.selection is not None:
            if self.selection.series_uid != self.series_uid:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"selection series {self.selection.series_uid!r} does not match "
                    f"state series {self.series_uid!r}",
                )
            if not (
                0 <= self.selection.ed_frame < n_frames
                and 0 <= self.selection.es_frame < n_frames
            ):
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"selection ED/ES frames ({self.selection.ed_frame}, "
                    f"{self.selection.es_frame}) are outside the cine frame range "
                    f"0..{n_frames - 1}",
                )
            if (
                self.selection.n_frames is not None
                and self.selection.n_frames != n_frames
            ):
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"selection n_frames {self.selection.n_frames} disagrees with "
                    f"cine n_frames {n_frames}",
                )

        for contour in self.draft_contours:
            if not 0 <= contour.slice_index < n_slices:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"draft contour slice_index {contour.slice_index} is outside "
                    f"the cine slice range 0..{n_slices - 1}",
                )
            if not 0 <= contour.frame_index < n_frames:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"draft contour frame_index {contour.frame_index} is outside "
                    f"the cine frame range 0..{n_frames - 1}",
                )
            if not isinstance(contour.structure, Structure):
                raise CMRError(
                    INVALID_ARGUMENT,
                    "draft contour structure must be a Structure enum member",
                )

    # --- derived views ------------------------------------------------------

    @property
    def head_revision(self) -> ContourRevision | None:
        """Highest-version revision of the carried chain, else ``None``."""
        return self.revision_chain[-1] if self.revision_chain else None

    @property
    def effective_visible_structures(self) -> tuple[Structure, ...]:
        """The visible set actually applied by the renderer (S3-008B §2.3).

        ``visible_structures=None`` resolves to ALL structures in frozen enum
        order; an explicit tuple is returned verbatim (already validated at
        construction). The empty tuple means no contours are drawn.
        """
        if self.visible_structures is None:
            return STRUCTURE_ORDER
        return self.visible_structures

    def is_structure_visible(self, structure: Structure) -> bool:
        """Whether ``structure``'s contours are drawn for this state (§2.3)."""
        return structure in self.effective_visible_structures

    @property
    def head_revision_id(self) -> str | None:
        """Content-addressed id of :attr:`head_revision`, else ``None``."""
        head = self.head_revision
        return None if head is None else head.revision_id

    def merged_contours_at(
        self, slice_index: int, frame_index: int
    ) -> tuple[Contour, ...]:
        """Latest contour per structure at ``(slice, frame)`` across the chain.

        Correction round 1 F1: this mirrors the merge semantics of
        ``cmr_quantification.measurements.build_cine_contour_index`` — walking
        ``revision_chain`` oldest-first, each ``(slice_index, frame_index,
        structure)`` key resolves to the LATEST (highest-version) revision
        supplying it, so a later physician re-edit of an earlier frame wins
        over that frame's original seed. A single revision carries at most one
        contour per structure (frozen domain constraint), which is why full
        cine coverage spreads across several revisions.

        Returns contours in ``Structure`` definition order (deterministic draw
        order). The renderer and the interactive shell's edit picker MUST use
        this view — never head-revision-only lookups.

        Tombstone rule (CMR_CREATE_A_001, R2 C1): keys whose newest chain
        event is a ``remove`` tombstone read ABSENT here. Ordinary ancestor
        merging or fallback must not silently resurrect removed geometry.
        """
        suppressed = head_tombstoned_keys(self.revision_chain)
        merged: dict[Structure, Contour] = {}
        for revision in self.revision_chain:
            for structure in Structure:
                candidate = revision.contours.get(structure)
                if candidate is None:
                    continue
                if (
                    candidate.slice_index == slice_index
                    and candidate.frame_index == frame_index
                ):
                    merged[structure] = candidate  # later revisions overwrite
        for structure in tuple(merged):
            if (slice_index, frame_index, structure) in suppressed:
                del merged[structure]
        return tuple(merged[structure] for structure in Structure if structure in merged)

    @property
    def base_approval_label(self) -> str:
        """One of the three exact base labels (precedence C > A > U).

        ``APPROVED`` requires a persisted :class:`~cmr_core.domain_types
        .ApprovalSnapshot` matching the current head revision id and the
        latest measurement id, plus (S2C-001, ONLY when binding metadata is
        present on the displayed records) draft-freedom, head/measurement/
        selection/approval coherence, approval QC-digest equality against the
        current head's gateway report and workflow equality. Otherwise a
        durable N7.1 :class:`~cmr_persistence.records.QCDecisionRecord`
        anchored to the current head yields the decision-recorded label.
        Anything else is unapproved. This method decides NOTHING clinically —
        it reads what persistence already sealed.
        """
        # CFM-001 correction F2: ANY §1.8 inclusion trigger forces the BASE
        # state to UNAPPROVED — an inclusion change or a newer set can never
        # remain visibly APPROVED / decision-recorded.
        if self._inclusion_trigger_active():
            return UNAPPROVED_LABEL
        head_id = self.head_revision_id
        if head_id is not None and self.latest_measurement is not None:
            measurement = self.latest_measurement
            measurement_id = measurement.measurement_id
            for snapshot in self.approvals:
                if (
                    snapshot.source_revision_id == head_id
                    and snapshot.measurement_id == measurement_id
                    and self._approval_binding_coherent(snapshot)
                ):
                    return APPROVED_LABEL
        if head_id is not None:
            for record in self.qc_decisions:
                if record.source_revision_id == head_id:
                    return DECISION_RECORDED_LABEL
        return UNAPPROVED_LABEL

    def _inclusion_trigger_active(self) -> bool:
        """Any §1.8 inclusion staleness trigger (correction F2).

        Used BOTH by :attr:`analysis_is_stale` and by
        :attr:`base_approval_label` — a triggered inclusion state can never
        display ``APPROVED`` or ``REVIEW DECISION RECORDED``; it forces the
        base label down to ``UNAPPROVED``.
        """
        measurement = self.latest_measurement
        # (1) unsaved draft differs from the persisted applicable set;
        if self._inclusion_grid_differs_from_persisted():
            return True
        if measurement is None:
            return False
        bound = _field_or_none(measurement, "inclusion_set_id")
        if self.persisted_inclusion_set is not None:
            # (2) newer/different set than the measurement-bound one.
            return bound != self.persisted_inclusion_set.inclusion_set_id
        # Correction round 2 (finding 1): with NO applicable current set, ANY
        # displayed measurement is unconfirmed/stale - whether its bound id
        # is null or historical.
        return True

    def _approval_binding_coherent(self, snapshot: ApprovalSnapshot) -> bool:
        """S2C-001 strict coherence checks, engaged ONLY for bound records.

        Legacy (unbound) approvals/measurements carry none of ``selection_id``
        /``qc_report_digest``/manifest ``workflow_id``, so every strict branch
        below reads as vacuous and the historical two-key rule alone decides.

        Correction F5: the workflow check now rejects BOTH mixed bound/unbound
        participants AND two different bound workflows — a single non-null
        value can no longer bypass coherence when other participants are
        unbound members of the same displayed lineage.
        """
        if self.draft_contours:
            return False  # unsaved edits never describe an approved state
        # CFM-001 corrections F2 + R1: the full inclusion trigger set (draft
        # divergence, newer/different set, no-applicable-set with displayed
        # metrics) can never describe an approved state.
        if self._inclusion_trigger_active():
            return False
        current_set_id = (
            None if self.persisted_inclusion_set is None
            else self.persisted_inclusion_set.inclusion_set_id
        )
        measurement = self.latest_measurement
        selection = self.selection
        head = self.head_revision

        # Workflow equality across ALL displayed lineage members: when ANY
        # participant carries a manifest binding, EVERY participant must carry
        # the SAME binding (bound/unbound mixing fails just like two
        # conflicting ids do).
        #
        # Presence and binding must be counted separately. A single tuple of
        # workflow values cannot express both, because an ABSENT participant and
        # a PRESENT-but-unbound participant both yield None. The old guard scored
        # absent participants as disagreements AND compared a count of DISTINCT
        # ids as if it were a count of members, so it rejected every lineage that
        # shared one id -- including the ordinary two-member approved case -- and
        # reported it UNAPPROVED.
        lineage_members = (snapshot, measurement, selection, head)
        present_workflows = [
            _workflow_of(member) for member in lineage_members if member is not None
        ]
        bound_workflows = {value for value in present_workflows if value is not None}
        if len(bound_workflows) > 1:
            return False
        if bound_workflows and any(value is None for value in present_workflows):
            return False  # mixed bound/unbound lineage members

        # Current-head measurement coherence when the field exists.
        source_revision = _field_or_none(measurement, "source_revision_id")
        if (
            source_revision is not None
            and self.head_revision_id is not None
            and source_revision != self.head_revision_id
        ):
            return False

        # Explicit selection binding across measurement/approval/selection.
        binding_ids = {
            value
            for value in (
                _field_or_none(measurement, "selection_id"),
                _field_or_none(snapshot, "selection_id"),
            )
            if isinstance(value, str)
        }
        if len(binding_ids) > 1:
            return False
        if binding_ids:
            required_selection_id = next(iter(binding_ids))
            if (
                selection is None
                or getattr(selection, "selection_id", None)
                != required_selection_id
            ):
                return False

        # Inclusion identity agreement: snapshot, measurement and the
        # CURRENT applicable set must carry ONE set id.
        inclusion_ids = {
            value
            for value in (
                _field_or_none(snapshot, "inclusion_set_id"),
                _field_or_none(measurement, "inclusion_set_id")
                if measurement is not None
                else None,
                current_set_id,
            )
            if isinstance(value, str)
        }
        if len(inclusion_ids) > 1:
            return False
        if current_set_id is not None:
            required = current_set_id
        elif inclusion_ids:
            required = next(iter(inclusion_ids))
        else:
            required = None
        if required is not None:
            for participant in (
                _field_or_none(snapshot, "inclusion_set_id"),
                _field_or_none(measurement, "inclusion_set_id")
                if measurement is not None
                else None,
            ):
                if participant != required:
                    return False

        # Approval QC digest must equal the CURRENT head's gateway report.
        approval_digest = _field_or_none(snapshot, "qc_report_digest")
        if (
            isinstance(approval_digest, str)
            and head is not None
            and head.qc_report is not None
            and canonical_digest(head.qc_report) != approval_digest
        ):
            return False
        return True

    @property
    def basal_inclusion_unconfirmed(self) -> bool:
        """CFM-001 §1.3: no applicable set exists for the current state."""
        return self.persisted_inclusion_set is None

    @property
    def inclusion_status_label(self) -> str | None:
        """The exact no-set label, or the applied-set status line."""
        if self.persisted_inclusion_set is None:
            return BASAL_INCLUSION_UNCONFIRMED_LABEL
        return (
            "basal inclusion: APPLIED — set "
            f"{self.persisted_inclusion_set.inclusion_set_id[:12]}.."
        )

    def _inclusion_grid_differs_from_persisted(self) -> bool:
        """Any draft checkbox differing from the persisted applicable set.

        An EMPTY draft represents "no physician interaction yet" (pure
        display state): it can never diverge on its own.
        """
        if not self.inclusion_draft:
            return False
        if self.persisted_inclusion_set is None:
            # With no applicable set, the all-included default IS the
            # disclosed legacy behavior: unchecking anything creates the
            # unsaved difference.
            return any(not toggle.included for toggle in self.inclusion_draft)
        persisted = {
            (
                entry.slice_index,
                entry.frame_role.value,
                entry.structure.value,
                entry.included,
            )
            for entry in self.persisted_inclusion_set.entries
        }
        draft = {toggle.key + (toggle.included,) for toggle in self.inclusion_draft}
        return draft != persisted

    @property
    def analysis_is_stale(self) -> bool:
        """Whether a DISPLAYED artifact no longer matches head + selection.

        S2C-001 §8.3: true when an unsaved draft exists, or a displayed
        historical measurement/approval/decision disagrees with the CURRENT
        head revision plus the latest (store-sequence) selection. Binding
        fields are read defensively so legacy states behave exactly as before.
        """
        if self.draft_contours:
            return True
        head_id = self.head_revision_id
        if head_id is None:
            return False  # nothing persisted to be stale against
        measurement = self.latest_measurement
        selection = self.selection

        # --- CFM-001 §1.8 inclusion-derived staleness triggers -------------
        # Correction round 3 (R1): the FULL trigger set - including a
        # NON-NULL historical set binding with no applicable current set -
        # participates in staleness. Reuse _inclusion_trigger_active()
        # verbatim so the STALE suffix can never fall through on a non-null
        # historical id.
        if self._inclusion_trigger_active():
            return True

        # A measurement anchored to another revision describes the past.
        source_revision = _field_or_none(measurement, "source_revision_id")
        if measurement is not None and source_revision is not None and (
            source_revision != head_id
        ):
            return True

        # Correction F5: an APPROVAL or DECISION anchored to an OLDER head is
        # stale EVEN when its QC digest would still match — it describes a
        # superseded state, not the current head plus latest selection.
        for snapshot in self.approvals:
            if snapshot.source_revision_id != head_id:
                return True
        for record in self.qc_decisions:
            if record.source_revision_id != head_id:
                return True

        # A bound measurement must match the LATEST selection id.
        measurement_selection = _field_or_none(measurement, "selection_id")
        if measurement_selection is not None:
            if selection is None or getattr(selection, "selection_id", None) != (
                measurement_selection
            ):
                return True

        # An APPROVAL whose verified QC digest no longer matches the current
        # head report means the approved state was superseded.
        head_report = None if self.head_revision is None else self.head_revision.qc_report
        if head_report is not None:
            for snapshot in self.approvals:
                approval_digest = _field_or_none(snapshot, "qc_report_digest")
                if isinstance(approval_digest, str) and (
                    canonical_digest(head_report) != approval_digest
                ):
                    return True

        # A DECISION bound to a different (older) selection than the latest.
        latest_selection_id = (
            None if selection is None else getattr(selection, "selection_id", None)
        )
        for record in self.qc_decisions:
            record_selection = _field_or_none(record, "selection_id")
            if (
                isinstance(record_selection, str)
                and latest_selection_id is not None
                and record_selection != latest_selection_id
                and record.source_revision_id == head_id
            ):
                return True
        return False

    @property
    def approval_label(self) -> str:
        """Base label with the S2C-001 STALE modifier appended when stale.

        The three base strings are preserved verbatim; ``" — STALE"`` is
        appended ONLY to ``UNAPPROVED`` states whose displayed artifacts do
        not match the current head plus latest selection. Changing ED or ES
        therefore immediately yields ``UNAPPROVED — STALE`` until refresh and
        reapproval.
        """
        base = self.base_approval_label
        if base == UNAPPROVED_LABEL and self.analysis_is_stale:
            return UNAPPROVED_LABEL + STALE_SUFFIX
        return base
