"""Node 7 — stateless workflow controller composing accepted primitives.

``WorkflowController`` is a PURE COMPOSITION LAYER over the accepted Node
2–6 primitives (``cmr_persistence.SqliteCmrStore``, ``cmr_dicom.pipeline
.load_cine_volume``, ``cmr_contours.editing``, ``cmr_quantification
.measurements``, ``cmr_qc.snapshot``). It holds NO mutable session state:
the class declares ``__slots__ = ()`` so instances are structurally unable
to carry attributes, and EVERY method takes explicit inputs and returns
explicit outputs. There is no inference, no GPU, no cloud and no HTTP
endpoint of any kind here; raw DICOM stays the imaging source of truth and
the controller never touches pixels.

Method map (contract scope item 1):

- :meth:`WorkflowController.select_series` — delegates verbatim to
  ``cmr_dicom.pipeline.load_cine_volume`` with an AUTHORITATIVE explicit
  ``series_uid``; classification remains ANNOTATE_ONLY (it annotates the
  returned :class:`~cmr_core.study.CMRSeries` and never reorders or
  auto-selects candidates). The leading ``store`` argument is part of the
  contract signature (interface uniformity across controller methods) but
  selection performs NO persistence: the argument is deliberately unused
  (recorded interpretation, see the report).
- :meth:`WorkflowController.select_ed_es` — validates ``0 <= frame <
  cine.n_frames`` BEFORE anything else and builds an immutable
  :class:`~cmr_core.domain_types.EDESSelection` through a caller-supplied
  zero-argument ``provenance_factory`` (which carries the series identity,
  because :class:`~cmr_core.cine.CineVolume` itself carries none). A flat
  selection (``ed_frame == es_frame``) is ALLOWED per the N5 policy;
  ``detect_ed_es`` is NEVER imported nor called anywhere in this module —
  the manual selection is authoritative.
- :meth:`WorkflowController.edit_contour` / ``undo`` / ``redo`` — thin
  passthrough over ``EditCommandStack.execute/undo/redo`` semantics
  (snapshot-based history lives in Node 4; the controller adds nothing).
- :meth:`WorkflowController.save` — composes
  ``cmr_contours.editing.save_to_revision(...)`` then persists in the
  documented foreign-key order. PRECONDITION (mirroring every accepted
  node flow): ancestor rows referenced by ``parent`` (its Prediction and
  prior revisions) must already exist in ``store``; otherwise the store's
  transactional guards roll back with zero partial rows. When a
  ``Measurement`` is supplied it MUST bind to the revision composed by
  this call and the pair is persisted atomically via
  ``save_revision_with_measurement`` (revision row inserted BEFORE the
  measurement row inside one BEGIN IMMEDIATE transaction); a QCReport
  supplied through ``qc_report`` is attached to the composed revision at
  composition time (the N6 gateway field
  :attr:`~cmr_core.domain_types.ContourRevision.qc_report`).
- :meth:`WorkflowController.compute_metrics` — delegates to
  ``cmr_quantification.measurements.compute_measurement``; the manual
  selection stays authoritative and the REAL canonical algorithm version
  is resolved inside Node 5.
- :meth:`WorkflowController.run_qc` — delegates to
  ``cmr_qc.snapshot.run_revision_qc`` (full frozen rule bundle over the
  composed index, ESV>EDV supplement included).
- :meth:`WorkflowController.reopen` — loads chain +
  latest_measurement + latest_selection + approvals + qc_decisions from a
  :class:`~cmr_persistence.store.SqliteCmrStore` as a plain dict.
- :meth:`WorkflowController.approve` — accepts the actual QCReport object,
  derives the verdict internally, enforces the binding approval policy
  (below) and stores one immutable
  :class:`~cmr_core.domain_types.ApprovalSnapshot` via
  ``store.save_approval`` with the REAL ``algorithm_version`` propagated
  from the Measurement (never the placeholder default). The owner-decided
  B-LITE no-measurement mode seals a sentinel decision record in hand AND
  persists it durably when Node 5 legitimately produced no Measurement
  (see below; milestone N7.1).

Approval policy (binding, contract scope item 2; correction round 1 F0)
-----------------------------------------------------------------------

``approve()`` accepts the actual :class:`~cmr_qc.report.QCReport` OBJECT
and derives the verdict internally via ``report.verdict`` — the caller can
never relabel a flagged report as ``"PASS"`` through a string argument
(review P0 fix). The derived verdict must be one of ``"PASS"`` /
``"REVIEW REQUIRED"`` (validated fail-closed). The verdict then REQUIRES
``"PASS"``, unless ``override=True`` is passed explicitly together with a
non-empty ``override_reason``. The rule is UNIFORM for every
``"REVIEW REQUIRED"`` verdict regardless of which flag class produced it;
the distinction called out by the contract is documented here rather than
encoded as special cases:

- **ESV>EDV class** (flag code ``qc.volume_curve``): the dependency-graph
  acceptance wording — QC yields REVIEW REQUIRED while the state remains
  approvable WITH override. Two honest realizations exist and both are
  pinned end-to-end by tests:
  1. *Measured state*: the legal flat-curve path of Node 5 (SV = 0 ml /
     EF = 0 % plus the ``qc.volume_curve`` warning) yields a REAL
     Measurement whose single-flag report is REVIEW REQUIRED solely
     through ``qc.volume_curve``; override seals a full approval snapshot
     bound to that Measurement exactly like any other override.
  2. *Unmeasurable state* (owner decision B-LITE, correction round 1):
     when Node 5 fail-closes the Measurement itself (strictly rising curve
     at the selected frames ⇒ ``quantification.invalid_input``, so NO
     Measurement exists), an explicit no-measurement mode still lets a
     documented physician decision be SEALED as a clinical-decision record
     without any numeric results — see "B-LITE no-measurement sealing"
     below. Without override the state stays blocked in BOTH cases.
- **Any other failure class** (geometry flags such as missing structures,
  self-intersections, containment escapes): identically subject to the
  uniform override rule — override applies to ANY REVIEW REQUIRED verdict.

Override-reason embedding mechanism (ONE deterministic choice)
--------------------------------------------------------------

The reason string is embedded into ``ApprovalSnapshot.provenance`` using
the **software_version suffix convention**: the snapshot's provenance
carries ``software_version = "<base>" + "+override=" +
urllib.parse.quote(reason, safe="")`` where ``<base>`` is the clean
``Measurement.software_version`` (or :data:`VERSION` in B-LITE mode,
which has no Measurement). In B-LITE mode the deterministic tail
``+metrics_unavailable`` (:data:`SENTINEL_VERSION_SUFFIX`) is appended
AFTER the encoded reason so the decision record is self-describing. The
encoding is fully deterministic (percent-encoding of the UTF-8 reason
text), reversible through :func:`extract_override_reason`, and
digest-relevant (different reasons yield different ``approval_id``
values). The snapshot's TOP-LEVEL ``software_version`` field keeps the
clean base so machine-readable version identity stays unpolluted. This
mechanism was chosen because the alternative ("append field") would
require changing the frozen Node 1 domain types, which is outside this
node's write set. A base version that already contains the
``"+override="`` marker is rejected fail-closed.

B-LITE no-measurement sealing (owner decision, correction round 1)
------------------------------------------------------------------

When the derived verdict is ``"REVIEW REQUIRED"``, ``override=True`` and
a non-empty reason are supplied, and ``measurement=None``, ``approve()``
seals an immutable :class:`~cmr_core.domain_types.ApprovalSnapshot`
WITHOUT numeric results:

- ``measurement_id`` is forced to the deterministic sentinel string
  ``"METRICS_UNAVAILABLE_DUE_TO_QC"``
  (:data:`SENTINEL_MEASUREMENT_ID`); the ``measurement_id`` argument must
  equal it explicitly so call sites stay honest;
- ``algorithm_version`` is forced to ``"metrics-unavailable-due-to-qc"``
  (:data:`SENTINEL_ALGORITHM_VERSION`) — never a real Node 5 algorithm
  identity;
- series/frames come from the explicitly supplied ``selection``
  (fail closed when absent); ``source_revision_id`` is the caller-supplied
  revision id and is verified to exist on the persisted chain of that
  series before anything is sealed;
- the override reason rides in the provenance per the embedding mechanism
  above, plus :data:`SENTINEL_VERSION_SUFFIX`.

The sealed snapshot is returned to the caller. Milestone N7.1 (owner-delegated
autonomous design decision, binding): the SAME decision is ALSO persisted
durably as a separate append-only
:class:`~cmr_persistence.records.QCDecisionRecord` through
``store.save_qc_decision``, into the dedicated ``qc_decision_records``
table added by the documented schema migration hatch
(``PRAGMA user_version`` 1 -> 2). The immutable clinical
``approval_snapshots`` table and its foreign key to ``measurements`` are
NOT weakened: the decision record anchors to the persisted
``contour_revisions`` chain instead (verified above before anything is
sealed), carries the physician's ``override_reason`` as a FIRST-CLASS
field, and shares one provenance with the sealed snapshot so a single
decision correlates across both tables byte-for-byte.
``store.save_approval(sealed)`` still verifiably refuses sentinel snapshots
(``SeriesCoherenceError`` + column-level foreign key), so the durable store
keeps refusing unverifiable APPROVALS while persisting verified DECISIONS —
defense in depth intact on both sides.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import numpy as np
from cmr_contours.editing import (
    EditCommand,
    EditCommandStack,
    compose_create_revision,
    compose_remove_revision,
    copy_contour_across_frames,
    copy_contour_across_slices,
    delete_vertex,
    insert_vertex,
    save_to_revision,
    translate_contour,
)
from cmr_contours.model import Contour
from cmr_core.cine import CineVolume
from cmr_core.domain_types import (
    EDIT_OP_CREATE,
    EDIT_OP_REMOVE,
    ApprovalSnapshot,
    ContourRevision,
    EDESSelection,
    HoleAwareContour,
    Measurement,
    PolygonRing,
    Prediction,
    SelectedPhasePrediction,
    TopologyBlock,
    WholeCinePrediction,
    WholeCineRevision,
    canonical_digest,
    effective_chain_contour,
    head_tombstoned_keys,
    revision_operation,
)
from cmr_core.errors import CONTOUR_INVALID, INVALID_ARGUMENT, CMRError, IssueSet
from cmr_core.geometry import _AREA_EPSILON, _signed_area, _validate_no_self_intersection
from cmr_core.provenance import VERSION, Provenance
from cmr_core.segmentation import Structure
from cmr_core.source_manifest import DicomSourceManifest
from cmr_core.study import CMRSeries
from cmr_dicom.pipeline import load_cine_volume
from cmr_geometry.polygon import point_on_segment, polygon_contains, segments_intersect
from cmr_persistence.records import (
    QC_DECISION_ALGORITHM_VERSION,
    QCDecisionRecord,
    SliceInclusionEntry,
    SliceInclusionFrameRole,
    SliceInclusionSet,
    SliceInclusionStructure,
    applicable_topology_dispositions,
    chain_key_authoring_times,
    topology_disposition_set_digest,
)
from cmr_persistence.store import SqliteCmrStore, StaleHeadError
from cmr_qc.report import QCReport
from cmr_qc.snapshot import run_revision_qc, run_selected_phase_qc, snapshot_qc
from cmr_quantification.measurements import (
    INCLUSION_ROLE_ED,
    INCLUSION_ROLE_ES,
    CineContourIndex,
    SliceInclusionPolicy,
    applicability_waivers,
    build_cine_contour_index,
    build_slice_inclusion_policy,
    compute_measurement,
    compute_selected_phase_measurement,
)
from cmr_quantification.reference_ranges import REFERENCE_TABLE_VERSION

from cmr_workflow.coverage import SELECTED_PHASE_GROUPS, selected_phase_required_keys
from cmr_workflow.topology_disposition import (
    TopologyResolution,
    assert_topology_resolved_for_approval,
    resolve_topology_dispositions,
)

__all__ = [
    "ADOPT_BINDING_MISMATCH",
    "ADOPT_CHAIN_REQUIRED",
    "ADOPT_INCOMPLETE",
    "ADOPT_INELIGIBLE",
    "ADOPT_NOTHING_TO_ADOPT",
    "ADOPT_SELECTION_REQUIRED",
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
    "project_persisted_whole_cine_to_legacy_draft",
]

#: Marker separating the clean base software version from the embedded,
#: percent-encoded override reason (see module docstring).
OVERRIDE_MARKER = "+override="

#: Deterministic tail appended AFTER the encoded override reason on B-LITE
#: no-measurement decision records (self-describing, still reversible).
SENTINEL_VERSION_SUFFIX = "+metrics_unavailable"

#: Forced measurement identity on B-LITE decision records: no Measurement
#: exists because Node 5 fail-closed; nothing numeric is derivable.
SENTINEL_MEASUREMENT_ID = "METRICS_UNAVAILABLE_DUE_TO_QC"

#: Forced algorithm identity on B-LITE decision records (never a real
#: Node 5 algorithm version). Aliased from the persistence-local record type,
#: which is the single source of truth for the durable ``qc_decision_records``
#: artifact (milestone N7.1).
SENTINEL_ALGORITHM_VERSION = QC_DECISION_ALGORITHM_VERSION

# --- CMR_FUNCV1_ADOPT_001: typed refusals for the explicit sparse adoption ---
#
# Local typed codes (same precedent as ``cmr_workflow.coverage
# .COVERAGE_INSUFFICIENT``): no existing error code is reused and
# ``cmr_core.errors`` stays byte-untouched. Stale head/selection are typed as
# :class:`~cmr_persistence.store.StaleHeadError` (the store CAS shape the
# legacy chain already raises); the remaining refusals carry these codes.
ADOPT_CHAIN_REQUIRED = "adopt.chain_required"
ADOPT_SELECTION_REQUIRED = "adopt.selection_required"
ADOPT_INELIGIBLE = "adopt.selected_phase_prediction_ineligible"
ADOPT_INCOMPLETE = "adopt.selected_phase_coverage_incomplete"
ADOPT_BINDING_MISMATCH = "adopt.binding_mismatch"
ADOPT_NOTHING_TO_ADOPT = "adopt.nothing_to_adopt"

#: The only verdict approvable without an explicit override.
PASS_VERDICT = "PASS"

#: Verdict produced by :class:`~cmr_qc.report.QCReport` when flags fired.
REVIEW_REQUIRED_VERDICT = "REVIEW REQUIRED"

#: Zero-argument factory producing the provenance for a manual ED/ES
#: selection. The factory carries the series identity because
#: :class:`~cmr_core.cine.CineVolume` has none of its own.
ProvenanceFactory = Callable[[], Provenance]


def _cine_contour_key(
    slice_index: Any, frame_index: Any, structure: Any
) -> tuple[int, int, Structure]:
    """Normalize one edit target to the frozen whole-cine contour key type.

    The D7 web action's wire grammar carries ``structure`` as its
    :class:`str` enum VALUE (``"lv_cavity"``), while the persisted
    :class:`~cmr_core.domain_types.WholeCinePrediction` /
    ``WholeCineRevision`` contour maps are keyed by the
    :class:`~cmr_core.segmentation.Structure` ENUM member. A raw string key
    would silently miss every lookup (and would be rejected by
    ``_s3_validate_key`` on write), so this helper is the SINGLE boundary
    where the two spellings meet.

    Accepted inputs: a ``Structure`` member (used verbatim — already
    normalized in-process callers) or its exact ``Structure.value`` string
    (the web wire). Bool is never an int here or anywhere in this module.
    """
    for name, value in (("slice_index", slice_index), ("frame_index", frame_index)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CMRError(
                INVALID_ARGUMENT,
                f"contour edit {name} must be an integer, got {value!r}",
            )
    if isinstance(structure, Structure):
        resolved = structure
    elif isinstance(structure, str):
        try:
            resolved = Structure(structure)
        except ValueError as exc:
            known = ", ".join(item.value for item in Structure)
            raise CMRError(
                INVALID_ARGUMENT,
                f"contour edit structure {structure!r} is unknown; known: {known}",
            ) from exc
    else:
        raise CMRError(
            INVALID_ARGUMENT,
            f"contour edit structure must be a Structure or its value string,"
            f" got {structure!r}",
        )
    return (int(slice_index), int(frame_index), resolved)


def _shoelace_area(points: Any) -> float:
    """Absolute shoelace area of one exterior polygon (pixel^2)."""
    return abs(
        0.5
        * sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    )


def _largest_component(components: tuple[Any, ...]) -> Any:
    """The largest-area exterior component (the frozen primary choice).

    Mirrors the QC routing's primary-component rule: absolute exterior
    area first, then the deterministic canonical-JSON tie-break.
    """
    from cmr_core.domain_types import canonical_json

    return max(
        components,
        key=lambda contour: (
            _shoelace_area(contour.exterior.points),
            canonical_json(contour),
        ),
    )

_ROLE_TO_FRAME = {
    INCLUSION_ROLE_ED: SliceInclusionFrameRole.ED,
    INCLUSION_ROLE_ES: SliceInclusionFrameRole.ES,
}

_STRUCTURE_BY_VALUE = {
    "lv_cavity": SliceInclusionStructure.LV_CAVITY,
    "rv_cavity": SliceInclusionStructure.RV_CAVITY,
    "lv_myocardium": SliceInclusionStructure.LV_MYOCARDIUM,
}


def _persistence_entry(
    entry: Mapping[str, Any], selection: EDESSelection
) -> SliceInclusionEntry:
    """Build one persistence entry, resolving its role to the exact frame.

    CFM-001 §1.2: role resolution is exact — ``ED -> selection.ed_frame`` and
    ``ES -> selection.es_frame`` — and every persisted entry names that frame
    explicitly (verified again at save time).
    """
    role = entry["frame_role"]
    if role == INCLUSION_ROLE_ED:
        frame_index = selection.ed_frame
    elif role == INCLUSION_ROLE_ES:
        frame_index = selection.es_frame
    else:
        raise CMRError(
            INVALID_ARGUMENT,
            f"frame_role must be 'ED' or 'ES', got {role!r}",
        )
    return SliceInclusionEntry(
        slice_index=entry["slice_index"],
        frame_role=_ROLE_TO_FRAME[role],
        structure=_STRUCTURE_BY_VALUE[entry["structure"]],
        included=entry["included"],
        frame_index=frame_index,
    )


def embed_override_reason(base_software_version: str, reason: str) -> str:
    """Return ``base`` suffixed with the deterministically encoded reason.

    The result is ``"<base>+override=<percent-encoded reason>"``; see the
    module docstring "Override-reason embedding mechanism". Fails closed on
    an empty base or reason and on a base that already carries the marker.
    """
    if not isinstance(base_software_version, str) or not base_software_version.strip():
        raise CMRError(
            INVALID_ARGUMENT,
            f"base software_version must be a non-empty string, got "
            f"{base_software_version!r}",
        )
    if OVERRIDE_MARKER in base_software_version:
        raise CMRError(
            INVALID_ARGUMENT,
            f"base software_version {base_software_version!r} already carries the "
            f"{OVERRIDE_MARKER!r} marker; refusing to nest override embeddings",
        )
    if not isinstance(reason, str) or not reason.strip():
        raise CMRError(INVALID_ARGUMENT, "override_reason must be a non-empty string")
    return f"{base_software_version}{OVERRIDE_MARKER}{quote(reason, safe='')}"


def extract_override_reason(version_string: str) -> tuple[str, str | None]:
    """Split a version string back into ``(base, reason_or_None)``.

    Exact inverse of :func:`embed_override_reason` for strings produced by
    it (including B-LITE strings carrying the trailing
    :data:`SENTINEL_VERSION_SUFFIX`, which is stripped before decoding);
    strings without the marker return ``(value, None)`` unchanged.
    """
    core = version_string
    if core.endswith(SENTINEL_VERSION_SUFFIX):
        core = core[: -len(SENTINEL_VERSION_SUFFIX)]
    if OVERRIDE_MARKER not in core:
        return (core, None)
    base, _, encoded = core.partition(OVERRIDE_MARKER)
    return (base, unquote(encoded))


def check_pre_projection_topology(
    key: tuple[int, int, Structure], source: tuple[HoleAwareContour, ...]
) -> None:
    """Validate that hole-aware source contours satisfy legacy projection invariants."""
    if len(source) == 0:
        raise CMRError(CONTOUR_INVALID, f"key {key!r} carries empty contour")
    if len(source) > 1:
        raise CMRError(
            CONTOUR_INVALID,
            f"key {key!r} carries {len(source)} disconnected components; "
            "legacy Contour cannot encode multi-component topologies",
        )
    h = source[0]
    if len(h.holes) > 0:
        raise CMRError(
            CONTOUR_INVALID,
            f"key {key!r} carries {len(h.holes)} internal holes; "
            "legacy Contour cannot encode annular/holed topologies",
        )
    exterior = h.exterior
    if len(exterior.points) < 3:
        raise CMRError(
            CONTOUR_INVALID,
            f"key {key!r} exterior ring is invalid or self-intersecting",
        )
    if abs(_signed_area(exterior.points)) < _AREA_EPSILON:
        raise CMRError(
            CONTOUR_INVALID,
            f"key {key!r} exterior ring is invalid or self-intersecting",
        )
    try:
        _validate_no_self_intersection(exterior.points)
    except Exception as exc:
        raise CMRError(
            CONTOUR_INVALID,
            f"key {key!r} exterior ring is invalid or self-intersecting",
        ) from exc
    if h.structure != key[2]:
        raise CMRError(
            CONTOUR_INVALID,
            f"key structure {key[2].value!r} disagrees with contour structure "
            f"{h.structure.value!r}",
        )


def _project_whole_cine_prediction_to_legacy_draft(
    store: SqliteCmrStore,
    series_uid: str,
    head: WholeCinePrediction,
    slice_index: int,
    frame_index: int,
    structure: Structure,
) -> Contour:
    """Private helper projecting a verified-persisted whole-cine prediction plane."""
    key = (slice_index, frame_index, structure)
    source = head.contours.get(key, ())
    try:
        check_pre_projection_topology(key, source)
    except CMRError as e:
        reason = (
            f"BLOCKED_NON_PROJECTABLE_TOPOLOGY: key (s={slice_index}, f={frame_index}, "
            f"struct={structure.value}) non-projectable ({e.message})"
        )
        now_str = datetime.now(timezone.utc).isoformat()
        store.record_topology_block(
            series_uid=series_uid,
            slice_index=slice_index,
            frame_index=frame_index,
            structure=structure.value,
            reason=reason,
            created_at=now_str,
        )
        raise CMRError(CONTOUR_INVALID, reason) from e
    h = source[0]
    pts = list(h.exterior.points)
    return Contour(
        slice_index=slice_index,
        frame_index=frame_index,
        structure=structure,
        points=pts,
        source="ai",
        edited=False,
        approved=False,
        revision=0,
        model=head.model,
        provenance=head.provenance,
    )


def project_persisted_whole_cine_to_legacy_draft(
    store: SqliteCmrStore,
    series_uid: str,
    slice_index: int,
    frame_index: int,
    structure: Structure,
) -> Contour:
    """Project one persisted whole-cine prediction plane to an editable legacy Contour.

    Normative U12 specification:
    The study's AI evidence MUST already be persisted in whole_cine_predictions.
    Public U12 ALWAYS loads via store.load_latest_whole_cine_prediction(series_uid).
    If no record is found, raises CMRError(INVALID_ARGUMENT, "no whole-cine prediction found").
    """
    head = store.load_latest_whole_cine_prediction(series_uid)
    if head is None:
        raise CMRError(INVALID_ARGUMENT, "no whole-cine prediction found")
    return _project_whole_cine_prediction_to_legacy_draft(
        store, series_uid, head, slice_index, frame_index, structure
    )


# --- create/remove myocardial containment gate (CMR_CREATE_A_001, R2 C4/C5)
#
# Boolean strict-containment + no-boundary-contact gate over the complete
# effective post-command state of every affected (slice, frame) plane. Uses
# the observed ``cmr_geometry.polygon`` primitives with explicit
# endpoint/on-segment checks in both directions; the shared-endpoint
# semantics of :func:`segments_intersect` are NOT modified. There is no
# minimum thickness, no radial sampling and no clinical arithmetic here —
# only the boolean gate (R2 C5).
#
# Acceleration (CMR_MYO_PERF_001, pure broad-phase, zero semantic change;
# CMR_MYO_PERF_C1_001 pruning-only correction: NO replacement numeric path):
# ``_check_myo_ring_pair`` below keeps the EXACT refusal vocabulary and
# check order of the original implementation — degenerate checks first,
# then the SHARED ``polygon_contains`` ("endo escapes epi"), then the
# edge-pair loop ("rings cross" / "boundary contact", endo-outer / epi-inner
# first-hit order). Three sound pruning filters only skip work the exact
# primitives must clear:
# 1. per-ring bbox sweep (``_myo_ring_bboxes_prove_simple``): two closed
#    segments that intersect (including touching) always share a bbox
#    point, so pairwise-disjoint non-adjacent edge bboxes prove the exact
#    zero-epsilon ``_validate_no_self_intersection`` check must pass.
#    Pure min/max/comparison arithmetic — no rounding, no tolerance.
# 2. bbox escape prefilter: a vertex outside the CLOSED epi bbox cannot be
#    inside the polygon, so the gate refuses early with the identical
#    "endo escapes epi" detail; otherwise the SHARED ``polygon_contains``
#    primitive decides the verdict (never a replacement predicate).
# 3. edge-pair pruning (``_myo_contact_candidates``): a pair whose bboxes
#    are separated in x or y by more than
#    ``M = max(eps, 2*eps/|A|, 2*eps/|B|)`` (plus a relative pad) cannot
#    trigger ``segments_intersect``/``point_on_segment`` under the shared
#    ``eps`` — every TRUE outcome (proper crossing, T-junction, shared
#    endpoint, on-segment contact) forces the pair separation within that
#    margin. ``eps`` mirrors the primitive default and is never retuned.
# Any ring carrying a non-finite coordinate takes the verbatim legacy path
# (no pruning at all), so the soundness arguments above — which hold over
# finite IEEE-754 arithmetic — cover every input. ``False``/fallback from
# any filter always means "run the exact check unchanged".


#: Broad-phase epsilon mirroring the shared ``cmr_geometry.polygon``
#: primitive default (``segments_intersect`` / ``point_on_segment``
#: ``eps=1e-9``). Never retuned: the differential regression guards it
#: against drift.
_MYO_PRUNE_EPS = 1e-9

#: Relative pad on the pair-prune margin so 1-ulp platform variance in the
#: edge-length computation can never flip a skip decision (the pad exceeds
#: any IEEE-754 rounding error by ~7 orders of magnitude).
_MYO_PRUNE_PAD = 1e-9


def _myo_all_finite(ring: list[tuple[float, float]]) -> bool:
    """True when every ring coordinate is finite (pruning precondition)."""
    return bool(np.all(np.isfinite(np.asarray(ring, dtype=np.float64))))


def _myo_ring_edge_boxes(
    ring: list[tuple[float, float]],
) -> tuple[Any, Any, Any]:
    """Per-edge bboxes and prune coefficients for one converted float ring.

    Returns ``(mins, maxs, coefs)``: ``(n, 2)`` float64 arrays of per-edge
    minima/maxima and an ``(n,)`` array with ``2 * eps / edge_length``
    (``inf`` for a zero-length edge, which is then never pruned).
    """
    pts = np.asarray(ring, dtype=np.float64)
    nxt = np.roll(pts, -1, axis=0)
    mins = np.minimum(pts, nxt)
    maxs = np.maximum(pts, nxt)
    vec = nxt - pts
    lens = np.hypot(vec[:, 0], vec[:, 1])
    coefs = np.divide(
        2.0 * _MYO_PRUNE_EPS,
        lens,
        out=np.full(lens.shape, np.inf),
        where=(lens != 0.0),
    )
    return mins, maxs, coefs


def _myo_ring_bboxes_prove_simple(ring: list[tuple[float, float]]) -> bool:
    """True only when per-edge bboxes PROVE the ring has no self-intersection.

    ``False`` means "unknown" — the caller runs the exact check unchanged,
    so the verdict is identical on every input. Callers must additionally
    require :func:`_myo_all_finite` first (non-finite coordinates always
    take the legacy path).
    """
    n = len(ring)
    if n < 4:
        return True
    mins, maxs, _ = _myo_ring_edge_boxes(ring)
    mnx = mins[:, 0]
    mxx = maxs[:, 0]
    mny = mins[:, 1]
    mxy = maxs[:, 1]
    overlap = (
        (mnx[:, None] <= mxx[None, :])
        & (mnx[None, :] <= mxx[:, None])
        & (mny[:, None] <= mxy[None, :])
        & (mny[None, :] <= mxy[:, None])
    )
    span = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    non_adjacent = (span > 1) & (span < n - 1)
    return bool(not np.any(overlap & non_adjacent))


def _myo_contact_candidates(
    endo: list[tuple[float, float]], epi: list[tuple[float, float]]
) -> list[tuple[int, int]]:
    """Edge-pair indices that can possibly trigger the exact contact verdicts.

    Returned in row-major (endo-outer, epi-inner) order so the caller's
    first-hit refusal matches the unpruned loop exactly. Callers must only
    use this on all-finite rings (see :func:`_myo_all_finite`).
    """
    e_mins, e_maxs, e_coef = _myo_ring_edge_boxes(endo)
    p_mins, p_maxs, p_coef = _myo_ring_edge_boxes(epi)
    sep_x = np.maximum(
        e_mins[:, 0, None] - p_maxs[None, :, 0],
        p_mins[None, :, 0] - e_maxs[:, 0, None],
    )
    sep_y = np.maximum(
        e_mins[:, 1, None] - p_maxs[None, :, 1],
        p_mins[None, :, 1] - e_maxs[:, 1, None],
    )
    margin = np.maximum(e_coef[:, None], p_coef[None, :])
    margin = np.maximum(margin, _MYO_PRUNE_EPS) * (1.0 + _MYO_PRUNE_PAD)
    evaluable = ~((sep_x > margin) | (sep_y > margin))
    return [(int(i), int(j)) for i, j in np.argwhere(evaluable)]


def _refuse_myo_containment(
    slice_index: int, frame_index: int, detail: str
) -> CMRError:
    return CMRError(
        CONTOUR_INVALID,
        f"BLOCKED_MYO_CONTAINMENT: key (s={slice_index}, f={frame_index}) "
        f"endo not strictly inside epi ({detail}); "
        f"next: on plane (s={slice_index}, f={frame_index}) redraw the LV "
        "cavity/myo pair nested with endo strictly inside epi, then retry",
    )


def _check_myo_ring_pair(
    slice_index: int,
    frame_index: int,
    endo_points: Any,
    epi_points: Any,
) -> None:
    """Refuse unless endo is strictly inside epi with no boundary contact.

    Check order and refusal vocabulary are frozen (CMR_CREATE_A_001 R2
    C4/C5): per-ring degenerate checks, then the SHARED ``polygon_contains``
    ("endo escapes epi"), then the edge-pair loop ("rings cross" /
    "boundary contact"). CMR_MYO_PERF_C1_001 is pruning-only: a bbox escape
    early-out plus edge-pair pruning skip work the exact primitives must
    clear, every filter fallback runs the exact legacy check, and
    non-finite rings take the verbatim legacy path, so verdicts are
    identical on every input.
    """
    endo = [(float(x), float(y)) for x, y in endo_points]
    epi = [(float(x), float(y)) for x, y in epi_points]
    for name, ring in (("endo", endo), ("epi", epi)):
        if len(ring) < 3:
            raise _refuse_myo_containment(
                slice_index, frame_index, f"{name} ring is degenerate"
            )
        if abs(_signed_area(ring)) < _AREA_EPSILON:
            raise _refuse_myo_containment(
                slice_index, frame_index, f"{name} ring is degenerate"
            )
        if _myo_all_finite(ring) and _myo_ring_bboxes_prove_simple(ring):
            pass
        else:
            try:
                _validate_no_self_intersection(ring)
            except Exception as exc:
                raise _refuse_myo_containment(
                    slice_index, frame_index, f"{name} ring is not simple"
                ) from exc
    pruned = _myo_all_finite(endo) and _myo_all_finite(epi)
    if pruned:
        epi_xs = [p[0] for p in epi]
        epi_ys = [p[1] for p in epi]
        lo_x, hi_x = min(epi_xs), max(epi_xs)
        lo_y, hi_y = min(epi_ys), max(epi_ys)
        for x, y in endo:
            if x < lo_x or x > hi_x or y < lo_y or y > hi_y:
                raise _refuse_myo_containment(
                    slice_index, frame_index, "endo escapes epi"
                )
    if not polygon_contains(epi, endo):
        raise _refuse_myo_containment(
            slice_index, frame_index, "endo escapes epi"
        )
    endo_edges = [(endo[i], endo[(i + 1) % len(endo)]) for i in range(len(endo))]
    epi_edges = [(epi[i], epi[(i + 1) % len(epi)]) for i in range(len(epi))]
    if pruned:
        pairs = _myo_contact_candidates(endo, epi)
    else:
        pairs = [
            (i, j) for i in range(len(endo_edges)) for j in range(len(epi_edges))
        ]
    for i, j in pairs:
        a1, a2 = endo_edges[i]
        b1, b2 = epi_edges[j]
        if segments_intersect(a1, a2, b1, b2):
            raise _refuse_myo_containment(
                slice_index, frame_index, "rings cross"
            )
        if (
            point_on_segment(a1, b1, b2)
            or point_on_segment(a2, b1, b2)
            or point_on_segment(b1, a1, a2)
            or point_on_segment(b2, a1, a2)
        ):
            raise _refuse_myo_containment(
                slice_index, frame_index, "boundary contact"
            )


def _myo_gate_live_contour(
    chain: Sequence[Any],
    revision: ContourRevision,
    key: tuple[int, int, Structure],
) -> Contour | None:
    """Effective post-command contour for ``key`` (newest assertion wins).

    The composed revision is consulted first (a ``remove`` tombstone for the
    key reads absent even though the pre-save chain head still carries the
    created contour); older chain state resolves through the tombstone-aware
    effective lookup.
    """
    edit_op, edit_key, _ = revision_operation(revision)
    if edit_op == EDIT_OP_REMOVE and edit_key == key:
        return None
    candidate = revision.contours.get(key[2])
    if (
        candidate is not None
        and candidate.slice_index == key[0]
        and candidate.frame_index == key[1]
    ):
        return candidate
    return effective_chain_contour(chain, key)


def _enforce_myo_containment(
    chain: Sequence[Any],
    revision: ContourRevision,
    *,
    affected_planes: set[tuple[int, int]] | None = None,
) -> None:
    """Run the R2 C4 containment gate over the operation's affected planes.

    The affected planes are derived from the command identity alone: for an
    operation-marked (create/remove) revision that is the composed revision's
    single operation key; for a general legacy save (translate/insert/delete/
    replace, one or more commands) the caller passes the exact planes carried
    by the command's edited contours. Planes merely carried by the revision
    are never examined, so invalid LV geometry on an unrelated carried plane
    cannot block a valid save. On each affected plane BOTH LV counterparts
    are resolved (including unchanged counterparts from the chain), even for
    an RV command. A plane where one or both LV rings are genuinely absent
    is skipped — an epi-only apex stays valid partial presence — while a
    present-but-malformed ring refuses (malformed geometry is not "absent"
    and must not bypass topology validation). A revision without an
    operation key and without caller-supplied planes has no defined affected
    plane and refuses fail-closed rather than scanning unrelated planes.
    """
    if affected_planes is None:
        _, edit_key, _ = revision_operation(revision)
        if edit_key is None:
            raise CMRError(
                CONTOUR_INVALID,
                "BLOCKED_MYO_CONTAINMENT: containment gate requires an "
                "operation-marked revision carrying its affected-plane key; "
                "next: save via an operation-marked revision carrying "
                "its affected-plane key, then retry",
            )
        planes = {(edit_key[0], edit_key[1])}
    else:
        planes = set(affected_planes)
        if not planes:
            return
    for slice_index, frame_index in sorted(planes):
        endo = _myo_gate_live_contour(
            chain, revision, (slice_index, frame_index, Structure.LV_CAVITY)
        )
        epi = _myo_gate_live_contour(
            chain, revision, (slice_index, frame_index, Structure.LV_MYOCARDIUM)
        )
        if endo is None or epi is None:
            continue
        _check_myo_ring_pair(slice_index, frame_index, endo.points, epi.points)


def _whole_cine_lv_ring_points(
    contours: Mapping[Any, Any],
    key: tuple[int, int, Structure],
) -> list[tuple[float, float]] | None:
    """Exterior points for one whole-cine LV key, or ``None`` when absent.

    Genuine absence (no key, a ``None`` entry or an empty component tuple)
    reads as missing and skips the gate — a cavity-only seed or an epi-only
    apex stays valid partial presence. A present-but-malformed counterpart
    (multi-component, annular/with-holes, or an unresolved holder) is NOT
    absence and refuses ``BLOCKED_MYO_CONTAINMENT`` instead of bypassing
    topology validation.
    """
    slice_index, frame_index, structure = key
    components = contours.get(key, ())
    if components is None:
        return None
    try:
        items = tuple(components)
    except TypeError as exc:
        raise _refuse_myo_containment(
            slice_index, frame_index, f"{structure.value} counterpart is unresolved"
        ) from exc
    if len(items) == 0:
        return None
    if len(items) != 1:
        raise _refuse_myo_containment(
            slice_index,
            frame_index,
            f"{structure.value} counterpart is multi-component "
            f"({len(items)} components)",
        )
    (component,) = items
    holes = getattr(component, "holes", ())
    if holes:
        raise _refuse_myo_containment(
            slice_index,
            frame_index,
            f"{structure.value} counterpart is annular ({len(holes)} holes)",
        )
    exterior = getattr(component, "exterior", None)
    points = getattr(exterior, "points", None)
    if not points:
        raise _refuse_myo_containment(
            slice_index, frame_index, f"{structure.value} counterpart is unresolved"
        )
    return [(float(x), float(y)) for x, y in points]


def _enforce_whole_cine_myo_containment(
    contours: Mapping[Any, Any],
    affected_planes: set[tuple[int, int]],
) -> None:
    """Run the R2 C4 containment gate over whole-cine affected planes.

    ``contours`` is the complete effective post-command state (unedited keys
    carried through verbatim, edited keys replaced); only ``affected_planes``
    — the ``(slice, frame)`` projection of the command's edited keys — are
    examined, so invalid LV geometry on an unrelated plane never blocks a
    valid save. On each affected plane BOTH LV counterparts resolve from the
    composed state (including unchanged counterparts), even for an RV-only
    edit. Genuine absence skips the plane; malformed counterparts refuse.
    Runs AFTER per-target topology validation (topology-first refusals keep
    ``BLOCKED_NON_PROJECTABLE_TOPOLOGY``) and BEFORE the checked append (a
    refusal writes zero rows).
    """
    for slice_index, frame_index in sorted(affected_planes):
        endo = _whole_cine_lv_ring_points(
            contours, (slice_index, frame_index, Structure.LV_CAVITY)
        )
        epi = _whole_cine_lv_ring_points(
            contours, (slice_index, frame_index, Structure.LV_MYOCARDIUM)
        )
        if endo is None or epi is None:
            continue
        _check_myo_ring_pair(slice_index, frame_index, endo, epi)


def _prevalidate_copy_containment(
    chain: Sequence[Any],
    copies: Sequence[Contour],
) -> None:
    """COPY-R1-001 atomicity gate: myo containment over the FULLY COMPOSED post-copy state.

    The legacy contour map holds ONE contour per structure, so sibling copies
    sharing the source's structure cannot be composed into one map. Instead
    each destination plane is evaluated with its post-copy LV pair resolved
    as: the copy itself when it carries that counterpart, otherwise the
    tombstone-aware effective chain contour. LV-cavity copies re-check the
    destination plane's epicardium; LV-myocardium copies re-check the
    destination plane's cavity; other structures evaluate the plane's
    unchanged LV pair (a copy landing on a plane whose LV pair is already
    invalid still refuses — the copy would seal the invalidity into a new
    revision). Genuine absence skips the plane (same partial-presence rule
    as :func:`_enforce_myo_containment`). Runs BEFORE any row is written, so
    any refusal writes zero rows.
    """
    by_plane: dict[tuple[int, int], list[Contour]] = {}
    for sibling in copies:
        by_plane.setdefault((sibling.slice_index, sibling.frame_index), []).append(
            sibling
        )
    for (slice_index, frame_index), siblings in sorted(by_plane.items()):
        endo_points: list[tuple[float, float]] | None = None
        epi_points: list[tuple[float, float]] | None = None
        for structure, slot in (
            (Structure.LV_CAVITY, "endo"),
            (Structure.LV_MYOCARDIUM, "epi"),
        ):
            carried = next(
                (c for c in siblings if c.structure == structure), None
            )
            if carried is not None:
                points = [(float(x), float(y)) for x, y in carried.points]
            else:
                live = effective_chain_contour(
                    chain, (slice_index, frame_index, structure)
                )
                if live is None:
                    continue
                points = [(float(x), float(y)) for x, y in live.points]
            if slot == "endo":
                endo_points = points
            else:
                epi_points = points
        if endo_points is None or epi_points is None:
            continue
        _check_myo_ring_pair(slice_index, frame_index, endo_points, epi_points)


def _is_physician_owned_contour(contour: Contour) -> bool:
    """Whether an EXISTING chain contour carries physician authority.

    CMR_FUNCV1_ADOPT_001 physician-wins classification (exact, auditable):

    * ``source == "manual"`` — a human-drawn contour;
    * ``edited is True`` — a physician correction of prior AI geometry
      (Node 4 preserves the original ``source`` and sets ``edited``).

    An unedited AI-origin contour (``source == "ai"``, ``edited is False``)
    is provenance from a PRIOR AI artifact, not physician authority: it may
    be superseded by the CURRENT-eligible sparse artifact. The classification
    reads only the persisted flags — never geometry, timestamps or names.
    """
    return contour.source == "manual" or bool(contour.edited)


def _project_selected_phase_to_legacy_draft(
    store: SqliteCmrStore,
    series_uid: str,
    sparse: SelectedPhasePrediction,
    key: tuple[int, int, Structure],
    *,
    created_at: str,
) -> Contour:
    """Project one verified sparse selected-phase key to a legacy draft Contour.

    The SAME pre-projection gate the whole-cine materialization uses
    (:func:`check_pre_projection_topology`) runs BEFORE any object is built:
    a non-projectable key (multi-component / annular / self-intersecting)
    records its durable per-key topology block and raises
    ``CMRError(CONTOUR_INVALID, "BLOCKED_NON_PROJECTABLE_TOPOLOGY: ...")`` —
    fail-visible, never silently flattened to the largest component or a
    hole-free ring.

    Only the FIRST (and, after the gate, only) component's exterior enters the
    legacy ``Contour``. ``source="ai"`` + the artifact's ``model`` and
    ``provenance`` preserve the AI origin so a later physician edit is
    auditable as AI-origin/physician-edited.
    """
    source = sparse.contours[key]
    try:
        check_pre_projection_topology(key, source)
    except CMRError as exc:
        reason = (
            f"BLOCKED_NON_PROJECTABLE_TOPOLOGY: key (s={key[0]}, f={key[1]}, "
            f"struct={key[2].value}) non-projectable ({exc.message})"
        )
        store.record_topology_block(
            series_uid=series_uid,
            slice_index=key[0],
            frame_index=key[1],
            structure=key[2].value,
            reason=reason,
            created_at=created_at,
        )
        raise CMRError(CONTOUR_INVALID, reason) from exc
    component = source[0]
    return Contour(
        slice_index=key[0],
        frame_index=key[1],
        structure=key[2],
        points=list(component.exterior.points),
        source="ai",
        edited=False,
        approved=False,
        revision=0,
        model=sparse.model,
        provenance=sparse.provenance,
    )


def _resolve_save_lineage(
    store: SqliteCmrStore,
    parent: Prediction | ContourRevision,
) -> tuple[Any, ...]:
    """Full pre-save lineage for the myo gate when ``chain`` is omitted.

    CMR_GATE_BYPASS_001: the old ``(parent,)`` default missed counterparts
    living in older revisions (each revision carries at most one contour per
    structure), so the gate skipped via the genuine-absence path and admitted
    escaping geometry on viewer-invoked saves (``cmr_viewer/app.py``
    ``save_flow`` calls ``save`` without ``chain``). Loading the persisted
    chain closes the bypass. A ``Prediction`` parent keeps the legacy
    ``(parent,)`` lineage (it is the seed root, not a truncated revision
    chain). A persisted revision parent resolves to its persisted prefix
    (the full chain when it is the head); an unpersisted in-memory head
    is refused fail-closed (BLOCKED_SAVE_LINEAGE_UNRESOLVED) since its
    ancestry cannot be validated. Store read failures propagate fail-closed (the gate runs
    before any row is written), never as a silent skip.

    CMR_GATE_BYPASS_C1_001 (R1 N3): the two residual fallbacks below refused
    fail-closed instead of silently skipping. Neither establishes that the
    available history is complete for the gate, so both raise
    ``CMRError(CONTOUR_INVALID, "BLOCKED_SAVE_LINEAGE_UNRESOLVED: ...")``
    BEFORE any row is written (zero rows; the in-memory composition is pure
    and persists nothing):

    * empty persisted history with a non-``Prediction`` parent: the parent
      claims revision ancestry that is absent from the store, so a
      ``(parent,)`` lookup cannot prove completeness;
    * unfound revision parent: the parent is absent from the persisted
      chain, so appending it cannot prove the ancestry between the
      persisted prefix and the parent is complete.

    Callers holding a legitimate unpersisted head must persist it first (or
    pass the explicit ``chain`` they composed from) and then retry; the
    guidance in each message says exactly that.
    """
    if isinstance(parent, Prediction):
        return (parent,)
    persisted = tuple(store.load_chain(parent.series_uid))
    if not persisted:
        raise CMRError(
            CONTOUR_INVALID,
            "BLOCKED_SAVE_LINEAGE_UNRESOLVED: no persisted revision chain "
            f"found for series {parent.series_uid!r} but the save parent is "
            "a revision (not the Prediction seed root), so the available "
            "history cannot be proven complete for the containment gate; "
            "zero rows written; "
            "next: persist the parent revision chain first (or retry the "
            "seed save from the Prediction root), or pass the explicit "
            "chain composed from, then retry",
        )
    parent_id = getattr(parent, "revision_id", None)
    for idx, rev in enumerate(persisted):
        if getattr(rev, "revision_id", None) == parent_id:
            return persisted[: idx + 1]
    raise CMRError(
        CONTOUR_INVALID,
        "BLOCKED_SAVE_LINEAGE_UNRESOLVED: the save parent revision "
        f"{parent_id!r} was not found on the persisted chain for series "
        f"{parent.series_uid!r}, so the ancestry between the persisted "
        "prefix and the parent cannot be proven complete for the "
        "containment gate; zero rows written; "
        "next: reload the persisted chain head and retry from it (or pass "
        "the explicit chain the parent was composed from), then retry",
    )


class WorkflowController:
    """Stateless composition layer over the accepted Node 2–6 primitives.

    See the module docstring for the full method map, the binding approval
    policy and the override-embedding mechanism. ``__slots__ = ()`` makes
    session-state drift structurally impossible: instances carry no
    attributes at all, so every workflow fact must travel through method
    arguments and return values.
    """

    __slots__ = ()

    def select_series(
        self,
        store: SqliteCmrStore,
        directory: Path,
        series_uid: str,
        issues: IssueSet | None = None,
    ) -> tuple[CineVolume | None, CMRSeries | None]:
        """Delegate to ``load_cine_volume`` and bind the source workflow.

        S2C-001 cluster A: an EXPLICIT ``series_uid`` is mandatory and
        authoritative — a blank/whitespace-only value fails closed (never a
        silent default fallback). After a successful reconstruction the
        returned cine and series UIDs MUST equal the requested UID, their
        study/workflow/cine identities MUST agree, and the DICOM source
        manifest is PERSISTED before returning, so one exact raw-source
        workflow identity travels end-to-end. Returns ``(None, None)`` plus
        descriptive warnings when the series cannot be reconstructed.
        """
        if not isinstance(series_uid, str) or not series_uid.strip():
            raise CMRError(
                INVALID_ARGUMENT,
                f"explicit series_uid must be a non-blank string, got "
                f"{series_uid!r}; pass a real SeriesInstanceUID",
            )
        volume, series = load_cine_volume(directory, series_uid=series_uid, issues=issues)
        if volume is None or series is None:
            return None, None
        manifest = volume.source_manifest
        if (
            volume.series_uid != series_uid
            or series.series_uid != series_uid
            or manifest is None
        ):
            raise CMRError(
                INVALID_ARGUMENT,
                "reconstructed identities disagree with the explicit request or "
                "carry no source manifest; refusing to return an unverified "
                "selection",
            )
        if (
            series.study_uid != manifest.study_uid
            or series.workflow_id != manifest.workflow_id
            or series.cine_volume_digest != manifest.cine_volume_digest
            or volume.workflow_id != manifest.workflow_id
            or volume.study_uid != manifest.study_uid
            or volume.cine_volume_digest != manifest.cine_volume_digest
        ):
            raise CMRError(
                INVALID_ARGUMENT,
                "cine/series source identities disagree with the persisted "
                "manifest; refusing to return an incoherent selection",
            )
        store.save_source_manifest(manifest)
        return volume, series

    def select_ed_es(
        self,
        cine: CineVolume,
        ed_frame: int,
        es_frame: int,
        selected_by: str,
        created_at: str,
        provenance_factory: ProvenanceFactory,
    ) -> EDESSelection:
        """Validate frames against the cine grid and build the selection.

        Frame validation happens FIRST and fails closed with
        ``app.invalid_argument`` before any object is constructed or any
        provenance is requested, so out-of-range input can never leave a
        partial write behind. ``ed_frame == es_frame`` is legal (flat
        selection per the N5 policy). ``detect_ed_es`` is never called:
        the physician's manual choice is authoritative. The series identity
        comes from the zero-argument ``provenance_factory`` result (fail
        closed when it carries no series_uid).
        """
        n_frames = cine.n_frames
        for label, value in (("ed_frame", ed_frame), ("es_frame", es_frame)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"{label} must be an integer frame index, got {value!r}",
                )
            if not 0 <= value < n_frames:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"{label} {value} is outside the cine frame range 0..{n_frames - 1}; "
                    "refusing to build an out-of-range EDESSelection",
                )
        if not isinstance(selected_by, str) or not selected_by.strip():
            raise CMRError(INVALID_ARGUMENT, "selected_by must be a non-empty string")
        provenance = provenance_factory()
        series_uid = provenance.series_uid
        if not isinstance(series_uid, str) or not series_uid.strip():
            raise CMRError(
                INVALID_ARGUMENT,
                "provenance_factory must return provenance carrying a non-empty "
                "series_uid; CineVolume itself carries none",
            )
        return EDESSelection(
            series_uid=series_uid,
            ed_frame=ed_frame,
            es_frame=es_frame,
            selected_by=selected_by,
            selected_at=created_at,
            provenance=provenance,
            n_frames=n_frames,
        )

    def edit_contour(self, stack: EditCommandStack, command: EditCommand) -> Contour:
        """Apply one pure-data command through the Node 4 stack (passthrough)."""
        return stack.execute(command)

    def undo(self, stack: EditCommandStack) -> Contour:
        """Undo the most recent command via the Node 4 stack (passthrough)."""
        return stack.undo()

    def redo(self, stack: EditCommandStack) -> Contour:
        """Re-apply the most recently undone command (passthrough)."""
        return stack.redo()

    def save(
        self,
        store: SqliteCmrStore,
        parent: Prediction | ContourRevision,
        edits: Mapping[Any, Contour],
        editor: str,
        created_at: str,
        *,
        cine: CineVolume | None = None,
        expected_head_revision_id: str | None = None,
        software_version: str = VERSION,
        qc_report: QCReport | None = None,
        measurement: Measurement | None = None,
        chain: Sequence[Any] | None = None,
        expected_selection_id: str | None = None,
        expected_inclusion_set_id: str | None = None,
        expected_n_slices: int | None = None,
    ) -> ContourRevision:
        """Compose a child revision via Node 4 and persist it in FK order.

        Composition goes through the unchanged ``save_to_revision``
        (mapping-key coherence, unapproved-fork policy and parent-link
        monotonicity are inherited). Persistence order follows the store's
        documented contract: when ``measurement`` is omitted the revision
        alone is inserted; when supplied, the measurement MUST reference
        exactly the revision composed by THIS call (checked here, then
        re-enforced by the store guard) and both rows are written in ONE
        transaction with the revision row first. Ancestor rows referenced
        by ``parent`` must already exist (precondition shared by every
        accepted node flow). A supplied ``qc_report`` rides along through
        composition and lands in the revision's N6 ``qc_report`` gateway
        field. The ``parent`` object is never mutated.

        When ``cine`` is supplied, Tier-2 grid bounds are checked:
        every contour must satisfy ``0 <= slice_index < cine.n_slices``
        and ``0 <= frame_index < cine.n_frames``.

        Create/remove boundary (CMR_CREATE_A_001, R2 C4): when the composed
        revision carries a create/remove operation identity, the myocardial
        containment gate runs AFTER composition and BEFORE any persistence
        row is written (a refusal writes zero rows). The optional ``chain``
        supplies the pre-save lineage for resolving unchanged counterpart
        geometry; when omitted the FULL persisted lineage is loaded via
        ``_resolve_save_lineage`` (CMR_GATE_BYPASS_001: never a parent-only
        silent skip). F4 follow-up
        (CMR_CREATE_A_F4_SCOPE_001, R2 C4): the SAME gate covers every
        legacy save kind supported on this path (translate/insert/delete/
        replace/create/remove) — the affected planes are exactly the planes
        carried by the command's edited contours (an empty edit set gates
        nothing), and the complete effective post-command state (including
        unchanged counterparts from the chain, even for RV-only edits) is
        resolved per affected plane. Unrelated planes are never scanned.
        """
        if cine is not None:
            for contour in edits.values():
                if not (
                    0 <= contour.slice_index < cine.n_slices
                    and 0 <= contour.frame_index < cine.n_frames
                ):
                    raise CMRError(
                        INVALID_ARGUMENT,
                        f"contour ({contour.slice_index}, {contour.frame_index}) "
                        f"out of cine grid bounds "
                        f"(n_slices={cine.n_slices}, n_frames={cine.n_frames})",
                    )
        revision = save_to_revision(
            parent,
            series_uid=parent.series_uid,
            edits=edits,
            editor=editor,
            created_at=created_at,
            software_version=software_version,
            qc_report=qc_report,
        )
        if revision.edit_op in (EDIT_OP_CREATE, EDIT_OP_REMOVE):
            gate_chain = chain if chain is not None else _resolve_save_lineage(store, parent)
            _enforce_myo_containment(gate_chain, revision)
        else:
            # F4 (CMR_CREATE_A_F4_SCOPE_001): gate translate/insert/delete/
            # replace saves on exactly the command-affected planes. Per-ring
            # validation already ran inside the editing operations during
            # composition (topology order preserved); the gate resolves the
            # complete effective post-command state per affected plane and
            # refuses BEFORE any row is written.
            affected_planes = {
                (contour.slice_index, contour.frame_index)
                for contour in edits.values()
            }
            if affected_planes:
                gate_chain = chain if chain is not None else _resolve_save_lineage(store, parent)
                _enforce_myo_containment(
                    gate_chain,
                    revision,
                    affected_planes=affected_planes,
                )
        if measurement is None:
            if expected_head_revision_id is not None:
                # PENDING_EDIT_BIND_001 (R1 P1-005): re-verify the CAPTURED
                # (authored) selection/inclusion authority pre-persist. The
                # ordinary path does NOT opt into the source/topology CAS
                # block (expected_n_slices=None keeps the historical head-only
                # CAS shape — source CAS is create/remove-scoped per C1 and
                # ordinary fixtures legitimately persist source predictions).
                # A stale selection/inclusion therefore refuses HERE (before
                # any row) with StaleHeadError + zero rows instead of
                # committing under a stale verdict.
                # PENDING_EDIT_BIND_C2_001 (R2 P1-005 OPEN): fail closed on
                # UNKNOWN (null) captured authority. A null expectation is
                # only legitimate when no live authority exists either
                # (head-only CAS history). When live selection/inclusion IS
                # present, an unknown-authority edit must refuse with
                # StaleHeadError + zero rows instead of committing under a
                # known authority it never proved.
                live_selection = store.load_latest_selection(
                    revision.series_uid
                )
                live_selection_id = (
                    None if live_selection is None else live_selection.selection_id
                )
                live_inclusion_id: str | None = None
                if live_selection_id is not None and cine is not None:
                    live_workflow_id = parent.provenance.workflow_id
                    if live_workflow_id is None:
                        live_workflow_id = cine.workflow_id
                    live_applicable = store.load_latest_applicable_inclusion_set(
                        revision.series_uid,
                        head_revision_id=expected_head_revision_id,
                        selection_id=live_selection_id,
                        workflow_id=live_workflow_id,
                        expected_n_slices=cine.n_slices,
                    )
                    if live_applicable is not None:
                        live_inclusion_id = live_applicable[0].inclusion_set_id
                if expected_selection_id is None and expected_inclusion_set_id is None:
                    if live_selection_id is not None or live_inclusion_id is not None:
                        raise StaleHeadError(
                            "unknown authoring authority "
                            f"(expected selection None, inclusion None) "
                            f"vs current selection {live_selection_id!r}, "
                            f"inclusion {live_inclusion_id!r}; zero rows written"
                        )
                else:
                    if expected_selection_id != live_selection_id:
                        raise StaleHeadError(
                            f"selection mismatch: expected {expected_selection_id!r}, "
                            f"current {live_selection_id!r}; zero rows written"
                        )
                    if expected_inclusion_set_id != live_inclusion_id:
                        raise StaleHeadError(
                            f"inclusion mismatch: expected {expected_inclusion_set_id!r}, "
                            f"current {live_inclusion_id!r}; zero rows written"
                        )
                store.save_revision_checked(
                    expected_head_revision_id,
                    revision,
                )
            else:
                store.save_revision(revision)
        else:
            if measurement.source_revision_id != revision.revision_id:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"measurement source_revision_id "
                    f"{measurement.source_revision_id!r} does not match the revision "
                    f"composed by this save call {revision.revision_id!r}; the pair-save "
                    "requires an exact association",
                )
            store.save_revision_with_measurement(revision, measurement)
        return revision

    def create_contour_checked(
        self,
        store: SqliteCmrStore,
        series_uid: str,
        *,
        slice_index: int,
        frame_index: int,
        structure: Structure,
        points: Any,
        editor: str,
        created_at: str,
        cine: CineVolume,
        expected_head_revision_id: str,
        expected_selection_id: str | None = None,
        expected_inclusion_set_id: str | None = None,
        software_version: str = VERSION,
    ) -> ContourRevision:
        """Authoritatively create one missing contour (controller-owned op).

        Legacy-clinical-path only (R2 C2). The controller owns every
        eligibility decision in the mandatory R2 C4 order: exact lineage/head
        identity binding (null never matches persisted non-null; null
        inclusion only for the genuinely-resolved default), persisted-set
        inclusion (never the view draft) with ED/ES grid mapping,
        effective-target absence, topology-block safety, source-availability,
        per-ring validation, in-memory composition, the affected-plane
        myocardial containment gate, then ONE checked append that commits the
        exact preflight eligibility snapshot atomically in-transaction (the
        store re-verifies selection/inclusion/source/topology bindings inside
        the same BEGIN IMMEDIATE as the insert). Any refusal writes zero rows.

        Stable refusal vocabulary (R2 C3): ``BLOCKED_CREATE_SELECTION``,
        ``BLOCKED_CREATE_INCLUSION``, ``BLOCKED_CREATE_TARGET_EXISTS``,
        ``BLOCKED_CREATE_SOURCE_AVAILABLE``,
        ``BLOCKED_CREATE_SOURCE_UNRESOLVED`` (all ``CONTOUR_INVALID``).
        Lineage/eligibility races refuse as ``StaleHeadError``.
        """
        key = self._checked_create_remove_key(
            "create", slice_index, frame_index, structure, cine=cine
        )
        if not isinstance(series_uid, str) or not series_uid.strip():
            raise CMRError(INVALID_ARGUMENT, "series_uid must be a non-empty string")
        if not isinstance(editor, str) or not editor.strip():
            raise CMRError(
                INVALID_ARGUMENT,
                "editor must be a non-empty physician identity",
            )
        chain = list(store.load_chain(series_uid))
        if not chain:
            raise CMRError(
                INVALID_ARGUMENT,
                f"no legacy revision chain found for series {series_uid!r}; "
                "bootstrap required before creating contours",
            )
        head = chain[-1]
        if head.revision_id != expected_head_revision_id:
            raise StaleHeadError(
                f"stale head: expected {expected_head_revision_id!r} but the "
                f"persisted chain head is {head.revision_id!r}; zero rows written"
            )
        selection = store.load_latest_selection(series_uid)
        if selection is None:
            raise CMRError(
                CONTOUR_INVALID,
                "BLOCKED_CREATE_SELECTION: a physician-confirmed ED/ES "
                "selection is required before creating contours; none is "
                f"persisted for series {series_uid!r}; "
                "next: confirm the physician ED/ES selection for this "
                "series, then retry the create",
            )
        # Exact selection identity (R2 C3 / CREATE-A C1 F2): a null expectation
        # never matches a persisted non-null selection. This point is only
        # reached when a selection IS persisted (absence refuses above), so an
        # expected null always refuses here as stale lineage.
        if expected_selection_id != selection.selection_id:
            raise StaleHeadError(
                f"selection mismatch: expected {expected_selection_id!r}, "
                f"current {selection.selection_id!r}; zero rows written"
            )
        roles = self._create_key_roles(key, selection, cine)
        applicable_id = self._create_inclusion_gate(
            store, series_uid, head, selection, cine, key, roles
        )
        # Exact inclusion identity including None (R2 C3 / CREATE-A C1 F2):
        # null matches only the genuinely-resolved default (no applicable
        # persisted set); a null expectation beside a persisted applicable set
        # — or a stale non-null expectation — refuses as stale lineage.
        if expected_inclusion_set_id != applicable_id:
            raise StaleHeadError(
                f"inclusion mismatch: expected {expected_inclusion_set_id!r}, "
                f"current {applicable_id!r}; zero rows written"
            )
        if effective_chain_contour(chain, key) is not None:
            raise CMRError(
                CONTOUR_INVALID,
                f"BLOCKED_CREATE_TARGET_EXISTS: effective physician-chain "
                f"geometry already exists for key (s={key[0]}, f={key[1]}, "
                f"struct={key[2].value}); create cannot duplicate it; "
                "next: edit or remove the existing contour instead of "
                "creating a duplicate",
            )
        topology_block_ids, topology_disposition_ids = self._create_topology_gate(
            store, series_uid, head, selection, key
        )
        sparse_id, whole_id = self._create_source_gate(store, series_uid, head, selection, key)
        revision = compose_create_revision(
            head,
            slice_index=key[0],
            frame_index=key[1],
            structure=key[2],
            points=points,
            editor=editor,
            created_at=created_at,
            series_uid=series_uid,
            software_version=software_version,
            cine=cine,
        )
        _enforce_myo_containment(chain, revision)
        # CREATE-A C1 F1: the EXACT eligibility snapshot consumed by the
        # gates above commits atomically with the append — the store
        # re-verifies every binding inside the same BEGIN IMMEDIATE
        # transaction as the insert, so a selection/inclusion/source/
        # topology change landing after preflight refuses with zero rows
        # instead of committing under a stale verdict.
        store.save_revision_checked(
            expected_head_revision_id,
            revision,
            expected_selection_id=selection.selection_id,
            expected_inclusion_set_id=applicable_id,
            expected_sparse_prediction_id=sparse_id,
            expected_whole_prediction_id=whole_id,
            expected_topology_block_ids=topology_block_ids,
            expected_topology_disposition_ids=topology_disposition_ids,
            expected_n_slices=cine.n_slices,
        )
        return revision

    def remove_contour_checked(
        self,
        store: SqliteCmrStore,
        series_uid: str,
        *,
        slice_index: int,
        frame_index: int,
        structure: Structure,
        editor: str,
        created_at: str,
        cine: CineVolume,
        expected_head_revision_id: str,
        software_version: str = VERSION,
    ) -> ContourRevision:
        """Authoritatively remove one created contour as its create-inverse.

        Bounded immediate-head-only authority (R2 C1): the persisted head
        must be a durably identified single-key ``create`` revision for this
        exact key, authored by the current physician, still carrying the
        created contour. A successfully redone create qualifies (it is a
        fresh create revision). Anything else refuses
        ``BLOCKED_REMOVE_NOT_CREATE_INVERSE`` with zero rows written.
        History rows stay intact and AI source records stay immutable; the
        removal appends a parent-linked tombstone revision instead.

        REMOVE-ATOMIC-001: the append commits the same transactional
        eligibility snapshot the create path commits (selection /
        inclusion / source / topology identities plus ``n_slices``),
        re-verified inside the store's single ``BEGIN IMMEDIATE``
        transaction. A binding committed after this preflight refuses
        with zero rows instead of committing under a stale verdict. No
        create-eligibility refusals are enforced here: removal authority
        stays immediate-head create-inverse only; the snapshot is purely
        the atomicity binding.
        """
        key = self._checked_create_remove_key(
            "remove", slice_index, frame_index, structure, cine=cine
        )
        if not isinstance(series_uid, str) or not series_uid.strip():
            raise CMRError(INVALID_ARGUMENT, "series_uid must be a non-empty string")
        if not isinstance(editor, str) or not editor.strip():
            raise CMRError(
                INVALID_ARGUMENT,
                "editor must be a non-empty physician identity",
            )
        chain = list(store.load_chain(series_uid))
        if not chain:
            raise CMRError(
                INVALID_ARGUMENT,
                f"no legacy revision chain found for series {series_uid!r}; "
                "nothing to remove",
            )
        head = chain[-1]
        if head.revision_id != expected_head_revision_id:
            raise StaleHeadError(
                f"stale head: expected {expected_head_revision_id!r} but the "
                f"persisted chain head is {head.revision_id!r}; zero rows written"
            )
        if (
            head.edit_op != EDIT_OP_CREATE
            or head.edit_key != key
            or head.editor != editor
        ):
            raise CMRError(
                CONTOUR_INVALID,
                "BLOCKED_REMOVE_NOT_CREATE_INVERSE: removal is admitted only "
                f"as the immediate-head single-key create-inverse for key "
                f"(s={key[0]}, f={key[1]}, struct={key[2].value}) by the same "
                "physician; the current head is not that create revision; "
                "next: refresh to the immediate-head single-key create by "
                "the same physician, then retry the removal",
            )
        carried = head.contours.get(key[2])
        if (
            carried is None
            or carried.slice_index != key[0]
            or carried.frame_index != key[1]
        ):
            raise CMRError(
                CONTOUR_INVALID,
                "BLOCKED_REMOVE_NOT_CREATE_INVERSE: the created contour for "
                f"key (s={key[0]}, f={key[1]}, struct={key[2].value}) is no "
                "longer present as that created contour; "
                "next: reload the head revision — the created contour is "
                "gone, so there is nothing left to remove",
            )
        revision = compose_remove_revision(
            head,
            slice_index=key[0],
            frame_index=key[1],
            structure=key[2],
            inverse_of=head.revision_id,
            editor=editor,
            created_at=created_at,
            series_uid=series_uid,
            software_version=software_version,
        )
        _enforce_myo_containment(chain, revision)
        reloaded = store.load_chain(series_uid)
        if not reloaded or reloaded[-1].revision_id != head.revision_id:
            raise StaleHeadError(
                "persisted chain head changed during remove; zero rows written"
            )
        # REMOVE-ATOMIC-001: resolve the exact eligibility snapshot the
        # create path commits (same shape: selection / applicable-inclusion
        # / sparse / whole-cine / topology identities + n_slices) and commit
        # it atomically with the append. Read-only here; the store
        # re-verifies every binding inside the same BEGIN IMMEDIATE
        # transaction as the insert. No create-gate refusals are enforced:
        # an excluded key, an available source, or a blocked topology never
        # blocks a removal; the snapshot only binds atomicity.
        _remove_selection = store.load_latest_selection(series_uid)
        _remove_selection_id = (
            None if _remove_selection is None else _remove_selection.selection_id
        )
        _remove_workflow_id = head.provenance.workflow_id
        if _remove_workflow_id is None:
            _remove_workflow_id = cine.workflow_id
        _remove_applicable = store.load_latest_applicable_inclusion_set(
            series_uid,
            head_revision_id=head.revision_id,
            selection_id=_remove_selection_id,
            workflow_id=_remove_workflow_id,
            expected_n_slices=cine.n_slices,
        )
        _remove_applicable_id = (
            None if _remove_applicable is None else _remove_applicable[0].inclusion_set_id
        )
        _remove_sparse = store.load_current_selected_phase_prediction(
            series_uid,
            head_revision_id=head.revision_id,
            selection_id=_remove_selection_id,
        )
        _remove_sparse_id = (
            None if _remove_sparse is None else _remove_sparse.prediction_id
        )
        _remove_whole = store.load_latest_whole_cine_prediction(series_uid)
        _remove_whole_id = (
            None if _remove_whole is None else _remove_whole.prediction_id
        )
        _remove_blocks = store.load_topology_blocks(series_uid)
        _remove_block_ids = tuple(
            sorted(str(block.block_id) for block in _remove_blocks)
        )
        _remove_dispositions = store.load_topology_dispositions(series_uid)
        _remove_disposition_ids = tuple(
            sorted(str(record.disposition_id) for record in _remove_dispositions)
        )
        store.save_revision_checked(
            expected_head_revision_id,
            revision,
            expected_selection_id=_remove_selection_id,
            expected_inclusion_set_id=_remove_applicable_id,
            expected_sparse_prediction_id=_remove_sparse_id,
            expected_whole_prediction_id=_remove_whole_id,
            expected_topology_block_ids=_remove_block_ids,
            expected_topology_disposition_ids=_remove_disposition_ids,
            expected_n_slices=cine.n_slices,
        )
        return revision

    @staticmethod
    def _checked_create_remove_key(
        op: str,
        slice_index: int,
        frame_index: int,
        structure: Structure,
        *,
        cine: CineVolume | None,
    ) -> tuple[int, int, Structure]:
        """Validate one create/remove target key against the cine grid."""
        for name, value in (("slice_index", slice_index), ("frame_index", frame_index)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise CMRError(
                    INVALID_ARGUMENT, f"{op} {name} must be an integer, got {value!r}"
                )
        if not isinstance(structure, Structure):
            raise CMRError(
                INVALID_ARGUMENT,
                f"{op} structure must be a Structure member, got {structure!r}",
            )
        if cine is None:
            raise CMRError(
                INVALID_ARGUMENT,
                f"{op} requires the cine grid to validate the target key",
            )
        if not (0 <= slice_index < cine.n_slices and 0 <= frame_index < cine.n_frames):
            raise CMRError(
                INVALID_ARGUMENT,
                f"{op} target ({slice_index}, {frame_index}) out of cine grid "
                f"bounds (n_slices={cine.n_slices}, n_frames={cine.n_frames})",
            )
        return (slice_index, frame_index, structure)

    @staticmethod
    def _create_key_roles(
        key: tuple[int, int, Structure],
        selection: EDESSelection,
        cine: CineVolume,
    ) -> list[str]:
        """ED/ES roles the requested key represents (R2 C3 grid mapping).

        Only exact keys represented by the five selected-phase groups are
        permitted: no nonselected-frame create, no ES-only myocardium
        create. Refuses ``BLOCKED_CREATE_SELECTION`` otherwise, including
        when the persisted selection frames fall outside the cine grid (the
        mapping cannot then be established).
        """
        slice_index, frame_index, structure = key
        for name, frame in (
            ("ed_frame", selection.ed_frame),
            ("es_frame", selection.es_frame),
        ):
            if isinstance(frame, bool) or not isinstance(frame, int):
                raise CMRError(
                    CONTOUR_INVALID,
                    "BLOCKED_CREATE_SOURCE_UNRESOLVED: persisted selection "
                    f"{name} {frame!r} is not an integer; source/binding "
                    "eligibility cannot be established; "
                    "next: repair the persisted ED/ES selection frames to "
                    "valid integers inside the cine grid, then retry",
                )
            if not 0 <= frame < cine.n_frames:
                raise CMRError(
                    CONTOUR_INVALID,
                    "BLOCKED_CREATE_SOURCE_UNRESOLVED: persisted selection "
                    f"{name} {frame!r} is outside the cine frame grid "
                    f"[0, {cine.n_frames}); source/binding eligibility "
                    "cannot be established; "
                    "next: repair the persisted ED/ES selection frames to "
                    "lie inside the cine grid, then retry",
                )
        roles = [
            role
            for role, group_structure in SELECTED_PHASE_GROUPS
            if group_structure == structure
            and (selection.ed_frame if role == "ed" else selection.es_frame)
            == frame_index
        ]
        if not roles:
            raise CMRError(
                CONTOUR_INVALID,
                "BLOCKED_CREATE_SELECTION: key "
                f"(s={slice_index}, f={frame_index}, "
                f"struct={structure.value}) is outside the selected-phase "
                "groups (no nonselected-frame create; no ES-only "
                "myocardium create); "
                "next: create only on a selected-phase ED/ES key, then retry",
            )
        return roles

    @staticmethod
    def _create_inclusion_gate(
        store: SqliteCmrStore,
        series_uid: str,
        head: ContourRevision,
        selection: EDESSelection,
        cine: CineVolume,
        key: tuple[int, int, Structure],
        roles: list[str],
    ) -> str | None:
        """Persisted-set inclusion verdict for one create key (R2 C3).

        Resolves inclusion from persisted server state WITHOUT applying any
        view draft: the current applicable persisted set through the
        existing policy conversion, else the disclosed all-included default
        (bound to the null applicable-set identity). The default is never
        substituted for an explicit exclusion, a malformed record or an
        unreadable authority. On a flat shared-frame selection every
        represented role for the physical key must be included; conflicting
        flags refuse. Returns the applicable inclusion-set id (or ``None``
        for the default case).
        """
        try:
            workflow_id = head.provenance.workflow_id
            if workflow_id is None:
                workflow_id = cine.workflow_id
            applicable = store.load_latest_applicable_inclusion_set(
                series_uid,
                head_revision_id=head.revision_id,
                selection_id=selection.selection_id,
                workflow_id=workflow_id,
                expected_n_slices=cine.n_slices,
            )
        except CMRError:
            raise
        except Exception as exc:
            raise CMRError(
                CONTOUR_INVALID,
                "BLOCKED_CREATE_INCLUSION: persisted inclusion authority is "
                f"unreadable ({exc}); refusing rather than defaulting; "
                "next: reload the persisted inclusion authority, then "
                "retry the create",
            ) from exc
        if applicable is None:
            return None
        inclusion_set = applicable[0]
        decisions = {
            (
                entry.slice_index,
                entry.frame_role.value,
                entry.structure.value,
            ): entry
            for entry in inclusion_set.entries
        }
        for role in roles:
            drawn = selection.ed_frame if role == "ed" else selection.es_frame
            entry = decisions.get((key[0], role.upper(), key[2].value))
            if entry is None:
                raise CMRError(
                    CONTOUR_INVALID,
                    "BLOCKED_CREATE_INCLUSION: no persisted inclusion "
                    f"decision for key (s={key[0]}, role={role.upper()}, "
                    f"struct={key[2].value}); unknown or malformed "
                    "applicability refuses; "
                    "next: persist an inclusion decision for this key, "
                    "then retry the create",
                )
            if entry.frame_index is not None and entry.frame_index != drawn:
                raise CMRError(
                    CONTOUR_INVALID,
                    "BLOCKED_CREATE_INCLUSION: persisted inclusion entry "
                    f"for key (s={key[0]}, role={role.upper()}, "
                    f"struct={key[2].value}) speaks about frame "
                    f"{entry.frame_index}, not the selected {drawn}; "
                    "malformed applicability refuses; "
                    "next: correct the persisted inclusion entry to the "
                    "selected frame, then retry the create",
                )
            if not entry.included:
                raise CMRError(
                    CONTOUR_INVALID,
                    "BLOCKED_CREATE_INCLUSION: key "
                    f"(s={key[0]}, f={key[1]}, struct={key[2].value}) is "
                    "excluded by the persisted inclusion set (an unsaved "
                    "checkbox change is not create permission; conflicting "
                    "flat-selection flags refuse); "
                    "next: include this key in the persisted inclusion set "
                    "and save it, then retry the create",
                )
        return inclusion_set.inclusion_set_id

    @staticmethod
    def _create_topology_gate(
        store: SqliteCmrStore,
        series_uid: str,
        head: ContourRevision,
        selection: EDESSelection,
        key: tuple[int, int, Structure],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Refuse creating over unsafely-blocked topology (R2 C3).

        A durable topology block for the exact key keeps the
        ``BLOCKED_NON_PROJECTABLE_TOPOLOGY`` verdict unless an applicable
        disposition bound to the current head + selection resolves it; a
        blocked source is never permission to create over it. Read failures
        fail closed as ``BLOCKED_CREATE_SOURCE_UNRESOLVED``.

        Returns the full-series sorted ``(block_ids, disposition_ids)``
        identity snapshot consumed here, so the store can re-verify it
        atomically with the append (CREATE-A C1 F1): any concurrent topology
        or disposition persist aborts the create fail-closed, even for an
        unrelated key.
        """
        try:
            blocks = store.load_topology_blocks(series_uid)
        except CMRError:
            raise
        except Exception as exc:
            raise CMRError(
                CONTOUR_INVALID,
                "BLOCKED_CREATE_SOURCE_UNRESOLVED: durable topology state "
                f"is unreadable ({exc}); source/binding eligibility cannot "
                "be established; "
                "next: reload durable topology state, then retry the create",
            ) from exc
        try:
            all_dispositions = store.load_topology_dispositions(series_uid)
        except CMRError:
            raise
        except Exception as exc:
            raise CMRError(
                CONTOUR_INVALID,
                "BLOCKED_CREATE_SOURCE_UNRESOLVED: durable disposition "
                f"state is unreadable ({exc}); source/binding eligibility "
                "cannot be established; "
                "next: reload durable disposition state, then retry "
                "the create",
            ) from exc
        relevant = [
            block
            for block in blocks
            if block.slice_index == key[0]
            and block.frame_index == key[1]
            and block.structure == key[2].value
        ]
        if relevant:
            try:
                dispositions = applicable_topology_dispositions(
                    all_dispositions,
                    head_revision_id=head.revision_id,
                    selection_id=selection.selection_id,
                )
            except CMRError:
                raise
            except Exception as exc:
                raise CMRError(
                    CONTOUR_INVALID,
                    "BLOCKED_CREATE_SOURCE_UNRESOLVED: durable disposition "
                    f"state is unreadable ({exc}); source/binding eligibility "
                    "cannot be established; "
                    "next: reload durable disposition state, then retry "
                    "the create",
                ) from exc
            resolved = {
                (record.slice_index, record.frame_index, record.structure)
                for record in dispositions
            }
            if (key[0], key[1], key[2].value) not in resolved:
                raise CMRError(
                    CONTOUR_INVALID,
                    "BLOCKED_NON_PROJECTABLE_TOPOLOGY: key "
                    f"(s={key[0]}, f={key[1]}, struct={key[2].value}) carries an "
                    "unresolved durable topology block; it is not permission "
                    "to create over the source",
                )
        return (
            tuple(sorted(str(block.block_id) for block in blocks)),
            tuple(sorted(str(record.disposition_id) for record in all_dispositions)),
        )

    @staticmethod
    def _create_source_gate(
        store: SqliteCmrStore,
        series_uid: str,
        head: ContourRevision,
        selection: EDESSelection,
        key: tuple[int, int, Structure],
    ) -> tuple[str | None, str | None]:
        """Refuse creating where eligible projectable source geometry exists (R2 C3).

        Read-only preflight (no adoption, no bootstrap, no inference, no
        mutation of any kind): when the current-eligible sparse artifact or
        the latest whole-cine prediction already serves the key with
        non-empty PROJECTABLE geometry, creation refuses
        ``BLOCKED_CREATE_SOURCE_AVAILABLE`` (the physician must adopt or
        edit the usable source instead of creating over it).

        A non-empty source that is NON-PROJECTABLE for the exact key
        (multi-component or holed per :func:`check_pre_projection_topology`,
        i.e. geometry the physician can never adopt or project onto the
        legacy single-ring path) does NOT refuse: the hand-drawn create
        proceeds so the physician can replace the unusable AI source with
        authored geometry. The durable topology block for the key still
        stands until a REPLACED/EXCLUDED disposition resolves it
        (``_create_topology_gate`` runs first and is untouched; no
        disposition semantics change here). A persisted empty tuple is the
        frozen "no contour" spelling and reads as missing. Returns the
        ``(sparse_id, whole_id)`` binding snapshot for pre-persist
        revalidation.
        """
        try:
            sparse = store.load_current_selected_phase_prediction(
                series_uid,
                head_revision_id=head.revision_id,
                selection_id=selection.selection_id,
            )
        except CMRError:
            raise
        except Exception as exc:
            raise CMRError(
                CONTOUR_INVALID,
                "BLOCKED_CREATE_SOURCE_UNRESOLVED: current sparse source "
                f"binding is unreadable ({exc}); source/binding eligibility "
                "cannot be established; "
                "next: reload the current sparse source binding, then "
                "retry the create",
            ) from exc
        try:
            whole = store.load_latest_whole_cine_prediction(series_uid)
        except CMRError:
            raise
        except Exception as exc:
            raise CMRError(
                CONTOUR_INVALID,
                "BLOCKED_CREATE_SOURCE_UNRESOLVED: whole-cine source state "
                f"is unreadable ({exc}); source/binding eligibility cannot "
                "be established; "
                "next: reload whole-cine source state, then retry the create",
            ) from exc
        for domain, artifact in (("sparse", sparse), ("whole-cine", whole)):
            if artifact is None:
                continue
            try:
                components = artifact.contours.get(key, ())
            except Exception as exc:
                raise CMRError(
                    CONTOUR_INVALID,
                    "BLOCKED_CREATE_SOURCE_UNRESOLVED: source geometry for "
                    f"key (s={key[0]}, f={key[1]}, struct={key[2].value}) "
                    f"cannot be established ({exc}); "
                    "next: repair the source artifact geometry for this "
                    "key, then retry the create",
                ) from exc
            if components:
                try:
                    check_pre_projection_topology(key, components)
                except CMRError:
                    # Non-projectable AI source (holed / multi-component) is
                    # unusable on the legacy path the physician can never
                    # adopt or project: it is not "eligible source geometry"
                    # and must not block the hand-drawn create that replaces
                    # it. The topology gate already ran (untouched) and the
                    # store re-verifies bindings atomically, so this skips
                    # only the source-availability refusal for this key.
                    # Malformed contours inside components raise ValueError
                    # from frozen-dataclass construction, never CMRError, so
                    # only the projection predicate lands here.
                    continue
                raise CMRError(
                    CONTOUR_INVALID,
                    "BLOCKED_CREATE_SOURCE_AVAILABLE: current eligible "
                    f"{domain} source geometry already exists for key "
                    f"(s={key[0]}, f={key[1]}, struct={key[2].value}); "
                    "create cannot duplicate it; "
                    "next: adopt or edit the eligible source geometry "
                    "instead of creating over it",
                )
        sparse_id = None if sparse is None else sparse.prediction_id
        whole_id = None if whole is None else whole.prediction_id
        return (sparse_id, whole_id)

    def compute_metrics(
        self,
        revision_chain_index: CineContourIndex,
        selection: EDESSelection,
        cine: CineVolume,
        *,
        software_version: str,
        algorithm_version: str | None = None,
        created_at: str,
        issues: IssueSet | None = None,
        inclusion_policy: SliceInclusionPolicy | None = None,
        range_table_version: str | None = None,
    ) -> Measurement:
        """Delegate to Node 5 ``compute_measurement`` (manual selection wins).

        ``revision_chain_index`` is the assembled cine-wide index (see
        ``build_cine_contour_index`` over the ordered reopened chain).
        Deterministic metrics only: no AI, no LLM, no rounding. CFM-001:
        an explicit ``inclusion_policy`` switches Node 5 onto the binary
        whole-slab arithmetic and the ``disc-cylinder-v2/...`` identity;
        omitting it keeps the legacy null-set bytes exactly.
        """
        return compute_measurement(
            revision_chain_index,
            selection,
            cine,
            software_version=software_version,
            algorithm_version=algorithm_version,
            created_at=created_at,
            issues=issues,
            inclusion_policy=inclusion_policy,
            range_table_version=range_table_version,
        )

    def run_qc(
        self,
        index: CineContourIndex,
        cine: CineVolume,
        selection: EDESSelection,
        volumes_by_frame: list[float] | None = None,
    ) -> QCReport:
        """Delegate to Node 6 ``run_revision_qc`` over the composed index."""
        return run_revision_qc(index, cine, selection, volumes_by_frame=volumes_by_frame)

    def compute_selected_phase_metrics(
        self,
        revision_chain_index: CineContourIndex,
        selection: EDESSelection,
        cine: CineVolume,
        *,
        software_version: str,
        algorithm_version: str | None = None,
        created_at: str,
        issues: IssueSet | None = None,
        inclusion_policy: SliceInclusionPolicy | None = None,
        range_table_version: str | None = None,
    ) -> Measurement:
        """Delegate to Node 5 ``compute_selected_phase_measurement``.

        The selected-phase clinical path: consumes exactly the five required
        geometry groups at the physician-authored selection frames, with the
        SAME physician-authoritative applicability record
        (``inclusion_policy``) the QC entry consumes. A required key absent
        from the index but explicitly waived contributes zero with no
        fabricated contour; an unwaived-absent key fails closed. The
        canonical algorithm identity carries
        ``coverage=selected-phases-v1`` (never a whole-cine identity).
        """
        return compute_selected_phase_measurement(
            revision_chain_index,
            selection,
            cine,
            software_version=software_version,
            algorithm_version=algorithm_version,
            created_at=created_at,
            issues=issues,
            inclusion_policy=inclusion_policy,
            range_table_version=range_table_version,
        )

    def run_selected_phase_qc(
        self,
        index: CineContourIndex,
        cine: CineVolume,
        selection: EDESSelection,
        *,
        inclusion_policy: SliceInclusionPolicy | None = None,
    ) -> QCReport:
        """Delegate to Node 6 ``run_selected_phase_qc`` with the SAME policy.

        The selected-phase QC consumes the SAME applicability authority the
        selected-phase measurement consumes; the policy identity is verified
        (series/head-ancestor/selection/workflow/frames/slices) before any
        waiver is applied. ``None`` preserves the no-authority behavior.
        """
        return run_selected_phase_qc(
            index, cine, selection, inclusion_policy=inclusion_policy
        )

    def refresh_head_analysis(
        self,
        store: SqliteCmrStore,
        cine: CineVolume,
        series_uid: str,
        *,
        software_version: str = VERSION,
        created_at: str,
        editor: str = "analysis-refresh",
        selected_by: str = "physician",
    ) -> dict[str, Any]:
        """CFM-001 correction F5: post-CFM refresh is NOT independent.

        Refresh routes EXCLUSIVELY through
        :meth:`apply_inclusion_and_refresh_analysis`, reusing the CURRENT
        applicable inclusion grid as the physician confirmation. When no
        applicable set exists this fails closed demanding explicit
        confirmation (the viewer's BASAL INCLUSION UNCONFIRMED state).
        """
        del cine, software_version, created_at, editor, selected_by
        raise CMRError(
            INVALID_ARGUMENT,
            "post-CFM Refresh Analysis requires the confirmed inclusion "
            "grid; use apply_inclusion_and_refresh_analysis (viewer button "
            "'Apply Inclusion & Refresh Analysis') instead of the retired "
            "legacy null-set refresh",
        )

    def record_ed_es(
        self,
        store: SqliteCmrStore,
        cine: CineVolume,
        ed_frame: int,
        es_frame: int,
        selected_by: str,
        created_at: str,
        provenance_factory: ProvenanceFactory,
    ) -> EDESSelection:
        """Validate, build and PERSIST one authoritative ED/ES selection.

        S2C-001 cluster E routing: the viewer must not touch
        ``store.save_edes_selection`` directly — every selection goes through
        this controller method so validation and persistence stay one
        coherent step.
        """
        selection = self.select_ed_es(
            cine, ed_frame, es_frame, selected_by, created_at, provenance_factory
        )
        store.save_edes_selection(selection)
        return selection

    def record_ed_es_checked(
        self,
        store: SqliteCmrStore,
        cine: CineVolume,
        ed_frame: int,
        es_frame: int,
        selected_by: str,
        created_at: str,
        provenance_factory: ProvenanceFactory,
        *,
        expected_head_revision_id: str | None,
        expected_selection_id: str | None,
        expected_inclusion_set_id: str | None,
    ) -> EDESSelection:
        """Build and persist ED/ES through the nullable checked CAS surface.

        This is the workstation-only composition entry point.  The legacy
        ``record_ed_es`` method remains available to existing callers, while
        this method always carries the client's complete nullable lineage
        expectations into one checked store transaction.
        """
        selection = self.select_ed_es(
            cine, ed_frame, es_frame, selected_by, created_at, provenance_factory
        )
        store.save_edes_selection_checked(
            selection,
            expected_head_revision_id=expected_head_revision_id,
            expected_selection_id=expected_selection_id,
            expected_inclusion_set_id=expected_inclusion_set_id,
            expected_n_slices=cine.n_slices,
        )
        return selection

    # --- CFM-001 basal-slice inclusion flows ----------------------------------

    @staticmethod
    def slice_inclusion_policy_from_record(
        record: SliceInclusionSet,
        *,
        ed_frame: int,
        es_frame: int,
        n_slices: int,
    ) -> SliceInclusionPolicy:
        """Convert a persisted inclusion-set record into a validated policy.

        CFM-001 §1.7 dependency rule: the persistence record is converted
        HERE (controller layer) only — quantification never imports
        persistence. The conversion runs the complete-grid validator via
        :func:`build_slice_inclusion_policy`, so a malformed persisted grid
        fails closed before any arithmetic or sealing can use it.
        """
        entries = [
            {
                "slice_index": entry.slice_index,
                "frame_role": (
                    INCLUSION_ROLE_ED
                    if entry.frame_role is SliceInclusionFrameRole.ED
                    else INCLUSION_ROLE_ES
                ),
                "structure": entry.structure.value,
                "included": entry.included,
            }
            for entry in record.entries
        ]
        return build_slice_inclusion_policy(
            entries,
            series_uid=record.series_uid,
            source_revision_id=record.source_revision_id,
            selection_id=record.selection_id,
            inclusion_set_id=record.inclusion_set_id,
            ed_frame=ed_frame,
            es_frame=es_frame,
            n_slices=n_slices,
            workflow_id=record.workflow_id,
        )

    def apply_inclusion_and_refresh_analysis(
        self,
        store: SqliteCmrStore,
        cine: CineVolume,
        series_uid: str,
        *,
        entries: Any,
        selected_by: str,
        software_version: str = VERSION,
        created_at: str,
        editor: str = "analysis-refresh",
    ) -> dict[str, Any]:
        """Validate + atomically persist the applicable set AND the analysis.

        CFM-001 §1.8 "Apply Inclusion & Refresh Analysis": the physician's
        complete binary grid (plain ``{slice_index, frame_role, structure,
        included}`` mappings; strict Booleans; exactly the five mandatory
        groups over every cine slice) is turned into ONE content-addressed
        :class:`~cmr_persistence.records.SliceInclusionSet` anchored to the
        CURRENT head plus latest selection, and the refreshed QC-bearing
        child revision plus its measurement are sealed in THE SAME
        ``BEGIN IMMEDIATE`` transaction via
        ``store.save_inclusion_set_with_analysis_checked``. Any failure
        leaves the previous durable state intact with zero partial rows.

        Idempotent: when the current head already carries exactly this QC
        report AND the latest applicable set equals this grid AND the
        coherent measurement binds it, nothing is written and
        ``status == "unchanged"``.
        """
        state = self.reopen(store, series_uid, cine=cine)
        chain = list(state["chain"])
        if not chain:
            raise CMRError(
                INVALID_ARGUMENT,
                f"no persisted revision chain for series {series_uid!r}; "
                "nothing to refresh",
            )
        head = chain[-1]
        selection = state["latest_selection"]
        if selection is None:
            raise CMRError(
                INVALID_ARGUMENT,
                "analysis refresh requires a persisted ED/ES selection; record "
                "one first",
            )
        # Grid validation happens INSIDE build_slice_inclusion_policy
        # (complete 5xN grid, strict booleans, five non-empty groups).
        build_slice_inclusion_policy(
            entries,
            series_uid=series_uid,
            source_revision_id=head.revision_id,
            selection_id=selection.selection_id,
            inclusion_set_id="probe-not-persisted",
            ed_frame=selection.ed_frame,
            es_frame=selection.es_frame,
            n_slices=cine.n_slices,
            workflow_id=head.provenance.workflow_id,
        )
        # The PERSISTED record resolves each decision to the exact frame its
        # role speaks about (save-time verification re-checks this).
        resolved_entries = [
            {
                "slice_index": int(entry["slice_index"]),
                "frame_role": str(entry["frame_role"]),
                "structure": str(entry["structure"]),
                "included": bool(entry["included"]),
            }
            for entry in entries
        ]
        record = SliceInclusionSet(
            series_uid=series_uid,
            source_revision_id=head.revision_id,
            selection_id=selection.selection_id,
            entries=tuple(
                _persistence_entry(resolved_entry, selection)
                for resolved_entry in resolved_entries
            ),
            selected_by=selected_by,
            selected_at=created_at,
            provenance=Provenance(
                study_uid=head.provenance.study_uid,
                series_uid=series_uid,
                software_version=software_version,
                model=None,
                created_at=created_at,
                revision=head.provenance.revision,
                workflow_id=head.provenance.workflow_id,
            ),
            workflow_id=head.provenance.workflow_id,
        )
        # The probe's placeholder identity is replaced by the real
        # content-addressed one of the persisted record.
        policy = self.slice_inclusion_policy_from_record(
            record,
            ed_frame=selection.ed_frame,
            es_frame=selection.es_frame,
            n_slices=cine.n_slices,
        )

        # C1 (P1-1): the refresh composes the SELECTED-PHASE pair with the
        # SAME applicability policy the measurement binds. The legacy
        # full-cycle curve + whole-cine measurement demanded cavity geometry
        # on EVERY frame, so an absent-but-waived key died with a bare
        # KeyError here; the selected-phase entries consume the waived key
        # as zero with no fabricated contour (fail-closed UNKNOWN unchanged).
        # Selected-phase identity note: the sealed measurement carries
        # ``coverage=selected-phases-v1`` (never the whole-cine
        # ``inclusion=whole-slice-v1`` identity) — the refresh path is the
        # selected-phase clinical path, same as the jobs recompute (F14
        # P1b) and the sealproof recompute.
        index = build_cine_contour_index(chain, series_uid)
        report = self.run_selected_phase_qc(
            index, cine, selection, inclusion_policy=policy
        )

        latest_measurement = state["latest_measurement"]
        applicable = store.load_latest_applicable_inclusion_set(
            series_uid,
            head_revision_id=head.revision_id,
            selection_id=selection.selection_id,
            workflow_id=head.provenance.workflow_id,
            expected_n_slices=cine.n_slices,
        )
        # Idempotency compares the physician-meaningful GRID CONTENT (the
        # resolved binary decisions), never authorship metadata: rebuilding
        # the same grid on a refreshed head must stay "unchanged".
        def _grid_signature(resolved: list[dict[str, Any]]) -> str:
            return canonical_digest(sorted(
                (
                    int(e["slice_index"]),
                    str(e["frame_role"]),
                    str(e["structure"]),
                    bool(e["included"]),
                )
                for e in resolved
            ))

        current_signature = _grid_signature(resolved_entries)
        if (
            head.qc_report is not None
            and canonical_digest(head.qc_report) == canonical_digest(report)
            and latest_measurement is not None
            and applicable is not None
            and _grid_signature([
                {
                    "slice_index": entry.slice_index,
                    "frame_role": entry.frame_role.value,
                    "structure": entry.structure.value,
                    "included": entry.included,
                }
                for entry in applicable[0].entries
            ]) == current_signature
            and latest_measurement.source_revision_id == head.revision_id
            and latest_measurement.selection_id == selection.selection_id
            and latest_measurement.inclusion_set_id
            == applicable[0].inclusion_set_id
        ):
            return {
                "status": "unchanged",
                "series_uid": series_uid,
                "head_revision_id": head.revision_id,
                "selection_id": selection.selection_id,
                "inclusion_set_id": applicable[0].inclusion_set_id,
                "measurement": latest_measurement,
                "child_revision": None,
                "qc_verdict": report.verdict,
                "analysis_id": (
                    None if state["analysis"] is None else state["analysis"].analysis_id
                ),
            }

        child = snapshot_qc(
            head,
            report,
            series_uid=series_uid,
            editor=editor,
            created_at=created_at,
            software_version=software_version,
        )
        child_index = build_cine_contour_index([*chain, child], series_uid)
        recheck_report = self.run_selected_phase_qc(
            child_index, cine, selection, inclusion_policy=policy
        )
        if canonical_digest(recheck_report) != canonical_digest(report):
            raise CMRError(
                INVALID_ARGUMENT,
                "QC digest drifted between the parent and QC-bearing child "
                "index; refusing to persist an unstable analysis",
            )
        # Correction F11: the NEW clinical workflow stamps the frozen table
        # version explicitly (the low-level compatibility path stays None).
        # C1 (P1-1): the measurement is the SELECTED-PHASE entry with the
        # same policy the QC above consumed (same authority, same identity).
        measurement = self.compute_selected_phase_metrics(
            child_index,
            selection,
            cine,
            software_version=software_version,
            created_at=created_at,
            inclusion_policy=policy,
            range_table_version=REFERENCE_TABLE_VERSION,
        )
        # R4: the compound checked path binds coverage to len(gaps); assert
        # coherence with the cine here so drift fails loudly at the boundary.
        assert len(measurement.stack_gaps_mm) == cine.n_slices
        store.save_inclusion_set_with_analysis_checked(
            record,
            head.revision_id,
            selection.selection_id,
            child,
            measurement,
        )
        analysis, _analysis_measurement, _reason = store.load_current_analysis(
            series_uid, expected_n_slices=cine.n_slices
        )
        return {
            "status": "refreshed",
            "series_uid": series_uid,
            "parent_head_revision_id": head.revision_id,
            "head_revision_id": child.revision_id,
            "selection_id": selection.selection_id,
            "inclusion_set_id": measurement.inclusion_set_id,
            "measurement": measurement,
            "child_revision": child,
            "qc_verdict": report.verdict,
            "analysis_id": None if analysis is None else analysis.analysis_id,
        }

    def record_applicability_decision_checked(
        self,
        store: SqliteCmrStore,
        cine: CineVolume,
        series_uid: str,
        *,
        entries: Any,
        selected_by: str,
        software_version: str = VERSION,
        created_at: str,
        expected_head_revision_id: str | None,
        expected_selection_id: str | None,
        expected_inclusion_set_id: str | None,
    ) -> dict[str, Any]:
        """Checked persist of one physician applicability grid (P1-1, C1).

        FUNCV1 physician applicability loop C1 correction for R1 P1-1: the
        decision-persist path is INDEPENDENT of recompute success. The grid
        (complete 5xN, strict Booleans, five non-empty groups) is validated
        via the SAME :func:`build_slice_inclusion_policy` probe, bound to
        the CURRENT head + latest selection (role-exact frame resolution via
        :func:`_persistence_entry`), and persisted through the SAME checked
        store surface family (three-way nullable lineage CAS, content-
        addressed :class:`SliceInclusionSet`, zero partial rows). The
        persisted set becomes the LATEST APPLICABLE set, so the previous
        coherent analysis/approval can no longer resolve as current — the
        old result is invalidated WITHOUT any new measurement row (reopen
        yields ``analysis=None`` / ``latest_measurement=None`` /
        ``has_stale_analysis=True``; the seal refuses
        ``AnalysisAuthorityError``/``StaleHeadError``). Missing geometry
        keeps refusing recompute/seal fail-closed: this method computes
        NOTHING and fabricates no contour. No new table, model, or state
        machine: the row lands in the EXISTING ``slice_inclusion_sets``
        table. Idempotent: an identical grid returns ``status="unchanged"``
        with the current set id (no duplicate row).
        """
        state = self.reopen(store, series_uid, cine=cine)
        chain = list(state["chain"])
        if not chain:
            raise CMRError(
                INVALID_ARGUMENT,
                f"no persisted revision chain for series {series_uid!r}; "
                "nothing to record",
            )
        head = chain[-1]
        selection = state["latest_selection"]
        if selection is None:
            raise CMRError(
                INVALID_ARGUMENT,
                "applicability persist requires a persisted ED/ES selection; "
                "record one first",
            )
        build_slice_inclusion_policy(
            entries,
            series_uid=series_uid,
            source_revision_id=head.revision_id,
            selection_id=selection.selection_id,
            inclusion_set_id="probe-not-persisted",
            ed_frame=selection.ed_frame,
            es_frame=selection.es_frame,
            n_slices=cine.n_slices,
            workflow_id=head.provenance.workflow_id,
        )
        resolved_entries = [
            {
                "slice_index": int(entry["slice_index"]),
                "frame_role": str(entry["frame_role"]),
                "structure": str(entry["structure"]),
                "included": entry["included"],
            }
            for entry in entries
        ]
        record = SliceInclusionSet(
            series_uid=series_uid,
            source_revision_id=head.revision_id,
            selection_id=selection.selection_id,
            entries=tuple(
                _persistence_entry(resolved_entry, selection)
                for resolved_entry in resolved_entries
            ),
            selected_by=selected_by,
            selected_at=created_at,
            provenance=Provenance(
                study_uid=head.provenance.study_uid,
                series_uid=series_uid,
                software_version=software_version,
                model=None,
                created_at=created_at,
                revision=head.provenance.revision,
                workflow_id=head.provenance.workflow_id,
            ),
            workflow_id=head.provenance.workflow_id,
        )
        saved_set_id = store.save_slice_inclusion_set_checked(
            record,
            expected_head_revision_id=expected_head_revision_id,
            expected_selection_id=expected_selection_id,
            expected_inclusion_set_id=expected_inclusion_set_id,
            expected_n_slices=cine.n_slices,
        )
        reopened = self.reopen(store, series_uid, cine=cine)
        current = reopened["applicable_inclusion_set"]
        if current is None or current.inclusion_set_id != saved_set_id:
            raise CMRError(
                INVALID_ARGUMENT,
                "checked applicability persist committed but the set does "
                "not resolve as applicable; refusing to report success",
            )
        return {
            "status": (
                "unchanged"
                if saved_set_id == expected_inclusion_set_id
                and state.get("applicable_inclusion_set") is not None
                and state["applicable_inclusion_set"].inclusion_set_id
                == saved_set_id
                else "recorded"
            ),
            "series_uid": series_uid,
            "head_revision_id": head.revision_id,
            "selection_id": selection.selection_id,
            "inclusion_set_id": saved_set_id,
            "analysis": reopened["analysis"],
            "analysis_stale_reason": reopened["analysis_stale_reason"],
            "has_stale_analysis": reopened["has_stale_analysis"],
            "latest_measurement": reopened["latest_measurement"],
        }

    def apply_inclusion_and_refresh_analysis_checked(
        self,
        store: SqliteCmrStore,
        cine: CineVolume,
        series_uid: str,
        *,
        entries: Any,
        selected_by: str,
        software_version: str = VERSION,
        created_at: str,
        editor: str = "analysis-refresh",
        expected_head_revision_id: str | None,
        expected_selection_id: str | None,
        expected_inclusion_set_id: str | None,
    ) -> dict[str, Any]:
        """Checked workstation inclusion refresh with a three-way nullable CAS.

        Candidate QC, policy and measurement objects may be prepared from a
        read snapshot.  The final checked persistence call is responsible for
        resolving current head/selection/inclusion identities and performs
        its comparison before both its idempotency decision and all inserts.
        Thus a competing commit in the preparation-to-write window becomes a
        deterministic stale action rather than a re-bound refresh.
        """
        state = self.reopen(store, series_uid, cine=cine)
        chain = list(state["chain"])
        if not chain:
            raise CMRError(
                INVALID_ARGUMENT,
                f"no persisted revision chain for series {series_uid!r}; "
                "nothing to refresh",
            )
        head = chain[-1]
        selection = state["latest_selection"]
        if selection is None:
            raise CMRError(
                INVALID_ARGUMENT,
                "analysis refresh requires a persisted ED/ES selection; record "
                "one first",
            )

        # Validation is intentionally performed before candidate construction,
        # but not used as a substitute for the checked transaction's CAS.
        build_slice_inclusion_policy(
            entries,
            series_uid=series_uid,
            source_revision_id=head.revision_id,
            selection_id=selection.selection_id,
            inclusion_set_id="probe-not-persisted",
            ed_frame=selection.ed_frame,
            es_frame=selection.es_frame,
            n_slices=cine.n_slices,
            workflow_id=head.provenance.workflow_id,
        )
        resolved_entries = [
            {
                "slice_index": int(entry["slice_index"]),
                "frame_role": str(entry["frame_role"]),
                "structure": str(entry["structure"]),
                "included": entry["included"],
            }
            for entry in entries
        ]
        record = SliceInclusionSet(
            series_uid=series_uid,
            source_revision_id=head.revision_id,
            selection_id=selection.selection_id,
            entries=tuple(
                _persistence_entry(resolved_entry, selection)
                for resolved_entry in resolved_entries
            ),
            selected_by=selected_by,
            selected_at=created_at,
            provenance=Provenance(
                study_uid=head.provenance.study_uid,
                series_uid=series_uid,
                software_version=software_version,
                model=None,
                created_at=created_at,
                revision=head.provenance.revision,
                workflow_id=head.provenance.workflow_id,
            ),
            workflow_id=head.provenance.workflow_id,
        )
        policy = self.slice_inclusion_policy_from_record(
            record,
            ed_frame=selection.ed_frame,
            es_frame=selection.es_frame,
            n_slices=cine.n_slices,
        )
        # C1 (P1-1): same selected-phase composition as the unchecked path —
        # the SAME policy feeds the selected-phase QC and the selected-phase
        # measurement, so an absent-but-waived key computes (zero, no
        # contour) instead of raising a bare KeyError from the legacy
        # full-cycle curve.
        index = build_cine_contour_index(chain, series_uid)
        report = self.run_selected_phase_qc(
            index, cine, selection, inclusion_policy=policy
        )
        child = snapshot_qc(
            head,
            report,
            series_uid=series_uid,
            editor=editor,
            created_at=created_at,
            software_version=software_version,
        )
        child_index = build_cine_contour_index([*chain, child], series_uid)
        recheck_report = self.run_selected_phase_qc(
            child_index, cine, selection, inclusion_policy=policy
        )
        if canonical_digest(recheck_report) != canonical_digest(report):
            raise CMRError(
                INVALID_ARGUMENT,
                "QC digest drifted between the parent and QC-bearing child "
                "index; refusing to persist an unstable analysis",
            )
        measurement = self.compute_selected_phase_metrics(
            child_index,
            selection,
            cine,
            software_version=software_version,
            created_at=created_at,
            inclusion_policy=policy,
            range_table_version=REFERENCE_TABLE_VERSION,
        )
        if len(measurement.stack_gaps_mm) != cine.n_slices:
            raise CMRError(
                INVALID_ARGUMENT,
                "measurement stack gaps do not cover the complete cine stack",
            )

        saved_set_id, saved_child_id = (
            store.save_inclusion_set_with_analysis_expected_checked(
                record,
                expected_head_revision_id,
                expected_selection_id,
                expected_inclusion_set_id,
                cine.n_slices,
                child,
                measurement,
            )
        )
        analysis, _analysis_measurement, _reason = store.load_current_analysis(
            series_uid, expected_n_slices=cine.n_slices
        )
        if saved_child_id == head.revision_id:
            current_measurement = state["latest_measurement"]
            if current_measurement is None:
                raise CMRError(
                    INVALID_ARGUMENT,
                    "checked store reported unchanged analysis without a "
                    "coherent current measurement",
                )
            return {
                "status": "unchanged",
                "series_uid": series_uid,
                "head_revision_id": head.revision_id,
                "selection_id": selection.selection_id,
                "inclusion_set_id": saved_set_id,
                "measurement": current_measurement,
                "child_revision": None,
                "qc_verdict": report.verdict,
                "analysis_id": None if analysis is None else analysis.analysis_id,
            }
        return {
            "status": "refreshed",
            "series_uid": series_uid,
            "parent_head_revision_id": head.revision_id,
            "head_revision_id": child.revision_id,
            "selection_id": selection.selection_id,
            "inclusion_set_id": saved_set_id,
            "measurement": measurement,
            "child_revision": child,
            "qc_verdict": report.verdict,
            "analysis_id": None if analysis is None else analysis.analysis_id,
        }

    # --- S3-009 D7: checked whole-cine contour save (web action backing) ------

    def save_contour_edits_checked(
        self,
        store: SqliteCmrStore,
        series_uid: str,
        *,
        edits: list[dict[str, Any]],
        editor: str,
        created_at: str,
        software_version: str = VERSION,
        expected_whole_cine_head_id: str | None,
    ) -> dict[str, Any]:
        """Apply physician contour edits and append ONE whole-cine revision.

        S3-009 D7 (thin composition; the edit semantics live in Node 4
        ``cmr_contours.editing`` and persistence in the checked store):

        1. resolves the current whole-cine lineage head
           (:meth:`SqliteCmrStore.load_latest_whole_cine_head`);
        2. requires the caller's expected head identity to match (stale
           fail-closed before ANY edit is applied);
        3. TYPE-REJECTS non-projectable edit targets (topology containment,
           2026-09-09): each target must satisfy
           :func:`check_pre_projection_topology` — exactly ONE component,
           zero holes and a valid exterior ring — because the edited result
           re-enters storage as a single hole-free exterior. A
           multi-component or annular target raises
           ``CMRError(CONTOUR_INVALID, "BLOCKED_NON_PROJECTABLE_TOPOLOGY:
            ...")`` BEFORE the offending target's draft is built and before
            any batch revision is persisted (zero rows written) — per-target
            in-loop rejection, not batch-wide prevalidation —
            instead of silently dropping components / stripping holes;
        4. applies each edit command to the head's contour for its
           ``(slice_index, frame_index, structure)`` key through the frozen
           non-mutating Node 4 operations
           (:func:`~cmr_contours.editing.translate_contour` /
           ``insert_vertex`` / ``delete_vertex`` — every result is a NEW
           unapproved contour);
        5. composes the FULL post-edit contour state (unedited keys carried
           through verbatim) and appends ONE
           :class:`~cmr_core.domain_types.WholeCineRevision` with the edited
           keys + explicit supersedes lineage through
           :meth:`SqliteCmrStore.save_whole_cine_revision_checked` (the
           CAS re-verifies the head inside the SAME transaction).
        6. F4 (CMR_CREATE_A_F4_SCOPE_001, R2 C4): between composition and
           the checked append, the myocardial containment gate resolves the
           complete effective post-command LV pair on every command-affected
           plane (unchanged counterparts included, RV-only edits included;
           unrelated planes never scanned) and refuses
           ``CMRError(CONTOUR_INVALID, "BLOCKED_MYO_CONTAINMENT: ...")``
           with zero rows written. Genuine absence (cavity-only seeds,
           epi-only apex) skips the plane; malformed counterparts refuse.
           Topology-first order is preserved: a non-projectable edit
           target still refuses ``BLOCKED_NON_PROJECTABLE_TOPOLOGY``.

        The original AI prediction row is never modified (invariant 10).
        """
        head = store.load_latest_whole_cine_head(series_uid)
        if head is None:
            raise CMRError(
                INVALID_ARGUMENT,
                f"no whole-cine prediction lineage exists for series "
                f"{series_uid!r}; a contour save requires an AI or seed "
                "prediction first",
            )
        if expected_whole_cine_head_id is None:
            raise CMRError(
                INVALID_ARGUMENT,
                "save_contour_edits requires the client's expected whole-cine "
                "head identity; a null lineage cannot be superseded",
            )
        head_id = (
            head.prediction_id
            if isinstance(head, WholeCinePrediction)
            else head.revision_id
        )
        if head_id != expected_whole_cine_head_id:
            raise CMRError(
                INVALID_ARGUMENT,
                f"stale whole-cine head: expected {expected_whole_cine_head_id!r} "
                f"but the current head is {head_id!r}; reload and re-apply",
            )

        version = 1 if isinstance(head, WholeCinePrediction) else head.version + 1
        contours: dict[tuple[int, int, Any], Any] = dict(head.contours)
        edited_keys: list[tuple[int, int, Any]] = []
        for position, edit in enumerate(edits):
            # The wire/API spelling of the target is normalized ONCE here:
            # ``structure`` arrives as its string value from the web action
            # but the contour map is keyed by the Structure enum member.
            key = _cine_contour_key(
                edit["slice_index"], edit["frame_index"], edit["structure"]
            )
            kind = str(edit["kind"])
            if kind not in ("translate", "insert", "delete"):
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"contour edit {position} kind must be one of translate,"
                    f" insert, delete; got {kind!r}",
                )
            components = contours.get(key, ())
            if not components:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"contour edit {position} targets "
                    f"{key[0]}:{key[1]}:{key[2].value} but the lineage head has "
                    "no contour for that (slice, frame, structure)",
                )
            # TOPOLOGY CONTAINMENT (S3-009 D7 amendment, 2026-09-09): the
            # whole-cine edit path can only express ONE simple polygon per
            # key — the edited result re-enters storage as a single exterior
            # with NO holes (see below), so editing a multi-component or
            # annular key would silently DESTROY topology (dropped
            # components / stripped holes) while still reporting SUCCESS.
            # Fail closed BEFORE any draft or revision is built, using the
            # SAME invariants the legacy projection uses. No durable
            # topology block is recorded here (this is a refusal, not a
            # projection outcome) and approval semantics are untouched.
            try:
                check_pre_projection_topology(key, tuple(components))
            except CMRError as exc:
                raise CMRError(
                    CONTOUR_INVALID,
                    f"BLOCKED_NON_PROJECTABLE_TOPOLOGY: key "
                    f"(s={key[0]}, f={key[1]}, struct={key[2].value}) "
                    f"non-projectable (editing annular/multi-component "
                    f"whole-cine contours is unsupported: "
                    f"{len(components)} components, "
                    f"{sum(len(component.holes) for component in components)} "
                    f"holes; {exc.message})",
                ) from exc
            primary = _largest_component(components)
            draft = Contour(
                slice_index=key[0],
                frame_index=key[1],
                structure=key[2],
                points=[
                    (float(x), float(y)) for x, y in primary.exterior.points
                ],
                source="ai" if getattr(head, "source", "ai") == "ai" else "manual",
            )
            if kind == "translate":
                edited = translate_contour(
                    draft, float(edit["dx"]), float(edit["dy"])
                )
            elif kind == "insert":
                position = edit["index_or_position"]
                edited = insert_vertex(
                    draft,
                    position
                    if isinstance(position, int)
                    else [float(component) for component in position],
                )
            else:
                edited = delete_vertex(draft, int(edit["vertex_index"]))
            # The EDITED result re-enters the hole-aware representation as a
            # single exterior with NO holes: Node 4's point-level operations
            # own the polygon, and a physician edit cannot re-derive the AI
            # mask's internal topology (S2C-001 cluster-F disclosure stands).
            contours[key] = (
                HoleAwareContour(
                    exterior=PolygonRing(
                        points=tuple((x, y) for x, y in edited.points)
                    ),
                    structure=key[2],
                ),
            )
            edited_keys.append(key)

        # F4 myocardial containment gate (CMR_CREATE_A_F4_SCOPE_001, R2 C4):
        # in-memory effective post-command composition is complete here, so
        # resolve BOTH LV counterparts per command-affected plane (including
        # unchanged counterparts carried verbatim, even for RV-only edits)
        # and refuse BEFORE the checked append (zero rows on refusal). Runs
        # AFTER the per-target topology checks above, so annular/
        # multi-component edit targets keep BLOCKED_NON_PROJECTABLE_TOPOLOGY.
        # The command vocabulary is UNCHANGED (translate/insert/delete only).
        affected_planes = {(key[0], key[1]) for key in edited_keys}
        if affected_planes:
            _enforce_whole_cine_myo_containment(contours, affected_planes)

        provenance = Provenance(
            study_uid=head.study_uid,
            series_uid=series_uid,
            software_version=software_version,
            model=None,
            created_at=created_at,
            revision=version - 1,
            workflow_id=head.provenance.workflow_id,
        )
        revision = WholeCineRevision(
            study_uid=head.study_uid,
            series_uid=series_uid,
            version=version,
            parent_prediction_id=(
                head.prediction_id
                if isinstance(head, WholeCinePrediction)
                else head.parent_prediction_id
            ),
            parent_revision_id=(
                None
                if isinstance(head, WholeCinePrediction)
                else head.revision_id
            ),
            editor=editor,
            edited_keys=tuple(edited_keys),
            contours=contours,
            provenance=provenance,
            created_at=created_at,
        )
        revision_id = store.save_whole_cine_revision_checked(
            revision,
            expected_parent_head_id=expected_whole_cine_head_id,
        )
        return {
            "status": "saved",
            "series_uid": series_uid,
            "whole_cine_head_id": head_id,
            "revision_id": revision_id,
            "version": version,
            "edited_keys": [
                f"{key[0]}:{key[1]}:{key[2].value}" for key in revision.edited_keys
            ],
            "parent_head_id": expected_whole_cine_head_id,
        }

    def copy_contours_checked(
        self,
        store: SqliteCmrStore,
        series_uid: str,
        *,
        src_slice_index: int,
        src_frame_index: int,
        dst_frame_indices: Sequence[Any],
        structure: Any,
        editor: str,
        created_at: str,
        cine: CineVolume,
        software_version: str = VERSION,
        expected_head_revision_id: str | None,
        expected_selection_id: str | None = None,
        expected_inclusion_set_id: str | None = None,
    ) -> ContourRevision:
        """Duplicate one verified contour onto empty neighbor frames (copy, Phase 6).

        LEGACY-clinical-path operation (same lineage discipline as
        :meth:`create_contour_checked` / :meth:`remove_contour_checked`): the
        SOURCE key ``(src_slice_index, src_frame_index, structure)`` is
        resolved through the effective chain (tombstone-aware); its geometry
        is duplicated verbatim onto each destination frame of the SAME slice
        + structure via
        :func:`~cmr_contours.editing.copy_contour_across_frames` — no
        interpolation, no propagation, no smoothing. ONE legacy revision per
        destination frame is appended through the existing checked path
        (same-shape as multi-key saves: each copy is its own revision so the
        durable ``(edit_op, edit_key)`` identity stays single-key).

        Fail-closed BEFORE any row is written (zero rows on any refusal):

        1. chain must exist; ``expected_head_revision_id`` must match the
           head (stale raises ``StaleHeadError``, same vocabulary as create);
        2. destination list validated by ``copy_contour_across_frames``
           (non-empty, integer, ``>= 0``, no dups, never the source frame);
        3. source and destination keys pass the cine grid check;
        4. the source key must be EFFECTIVELY PRESENT on the chain
           (tombstoned/absent refuses);
        5. every destination key must be EFFECTIVELY ABSENT on the chain
           (exists refuses — copy never overwrites; edit instead);
        6. the source ring must satisfy the replacement-ring validator
           (simple non-degenerate ring, via ``copy_contour_across_frames``
           composition);
        7. the myocardial containment gate runs on every destination plane
           AFTER composition and BEFORE any persistence row is written.

        The original AI prediction row is never modified (invariant 10).
        Returns the LAST appended revision (callers needing every id read
        the chain).
        """
        chain = list(store.load_chain(series_uid))
        if not chain:
            raise CMRError(
                INVALID_ARGUMENT,
                f"no legacy revision chain found for series {series_uid!r}; "
                "bootstrap required before copying contours",
            )
        head = chain[-1]
        if head.revision_id != expected_head_revision_id:
            raise StaleHeadError(
                f"stale head: expected {expected_head_revision_id!r} but the "
                f"persisted chain head is {head.revision_id!r}; zero rows written"
            )
        try:
            structure_enum = (
                structure if isinstance(structure, Structure) else Structure(structure)
            )
        except ValueError as exc:
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy structure {structure!r} is unknown",
            ) from exc
        if not isinstance(src_slice_index, int) or isinstance(src_slice_index, bool):
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy src_slice_index must be an integer, got {src_slice_index!r}",
            )
        if not isinstance(src_frame_index, int) or isinstance(src_frame_index, bool):
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy src_frame_index must be an integer, got {src_frame_index!r}",
            )
        if cine is None:
            raise CMRError(INVALID_ARGUMENT, "copy requires the cine grid")
        if not (0 <= src_slice_index < cine.n_slices):
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy source slice {src_slice_index} out of cine slice range "
                f"(0..{cine.n_slices - 1})",
            )
        if not (0 <= src_frame_index < cine.n_frames):
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy source frame {src_frame_index} out of cine frame range "
                f"(0..{cine.n_frames - 1})",
            )
        try:
            draft_targets = list(dst_frame_indices)
        except TypeError as exc:
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy dst_frame_indices must be a list, got {dst_frame_indices!r}",
            ) from exc
        for frame in draft_targets:
            if not (isinstance(frame, int) and not isinstance(frame, bool)):
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"copy destination frame must be an integer, got {frame!r}",
                )
            if not (0 <= frame < cine.n_frames):
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"copy destination frame {frame} out of cine frame range "
                    f"(0..{cine.n_frames - 1})",
                )
        src_key = (src_slice_index, src_frame_index, structure_enum)
        source = effective_chain_contour(chain, src_key)
        if source is None:
            raise CMRError(
                CONTOUR_INVALID,
                f"copy source {src_key[0]}:{src_key[1]}:{src_key[2].value} "
                "is absent on the chain (missing or removed); a copy requires "
                "an existing source contour",
            )
        try:
            copies = copy_contour_across_frames(source, draft_targets)
        except CMRError as exc:
            # COPY-R4-001: preserve the source-gate vocabulary — a degenerate
            # source refuses with the ring validator's own code
            # (contour.invalid), not a re-wrapped invalid-argument.
            raise CMRError(exc.code, f"copy source invalid: {exc.message}") from exc
        for sibling in copies:
            dst_key = (sibling.slice_index, sibling.frame_index, sibling.structure)
            if effective_chain_contour(chain, dst_key) is not None:
                raise CMRError(
                    CONTOUR_INVALID,
                    f"copy destination {dst_key[0]}:{dst_key[1]}:{dst_key[2].value} "
                    "already carries a contour on the chain; copy never "
                    "overwrites (edit the existing contour instead)",
                )
        # COPY-R1-001 atomicity: prevalidate EVERY destination's myo
        # containment against the FULLY COMPOSED post-copy state BEFORE any
        # row is written. A per-copy sequential save would persist an early
        # copy and then refuse a later one (partial success); instead the
        # gate below resolves each destination plane against the chain PLUS
        # all sibling copies, so any refusal writes zero rows.
        _prevalidate_copy_containment(chain, copies)
        saved: ContourRevision | None = None
        current: ContourRevision | Prediction = head
        for sibling in copies:
            dst_key = (sibling.slice_index, sibling.frame_index, sibling.structure)
            try:
                # Atomicity (COPY-R1-001): every destination was prevalidated
                # above (occupancy + full-state containment), so per-copy
                # saves below cannot refuse on those grounds; any residual
                # refusal (authority race landing between prevalidation and
                # append) still writes zero FURTHER rows.
                current = self.save(
                    store,
                    current,
                    {sibling.structure: sibling},
                    editor,
                    created_at,
                    cine=cine,
                    expected_head_revision_id=current.revision_id,
                    expected_selection_id=expected_selection_id,
                    expected_inclusion_set_id=expected_inclusion_set_id,
                )
            except CMRError as exc:
                raise CMRError(
                    exc.code,
                    f"copy destination {dst_key[0]}:{dst_key[1]}:{dst_key[2].value} "
                    f"refused ({exc.message}); zero further rows written",
                ) from exc
            assert isinstance(current, ContourRevision)
            saved = current
            chain = list(store.load_chain(series_uid))
        assert saved is not None
        return saved

    def copy_contours_across_slices_checked(
        self,
        store: SqliteCmrStore,
        series_uid: str,
        *,
        src_slice_index: int,
        src_frame_index: int,
        dst_slice_indices: Sequence[Any],
        structure: Any,
        editor: str,
        created_at: str,
        cine: CineVolume,
        software_version: str = VERSION,
        expected_head_revision_id: str | None,
        expected_selection_id: str | None = None,
        expected_inclusion_set_id: str | None = None,
    ) -> ContourRevision:
        """Duplicate one verified contour onto empty neighbor slices (copy, Phase 6).

        S16-transpose of :meth:`copy_contours_checked`: the SOURCE key
        ``(src_slice_index, src_frame_index, structure)`` is resolved through
        the effective chain (tombstone-aware); its geometry is duplicated
        verbatim onto each destination slice at the SAME frame + structure
        via :func:`~cmr_contours.editing.copy_contour_across_slices` — no
        interpolation, no propagation, no smoothing. Every guard is the
        slice-transpose of the frame-copy guards: chain exists,
        ``StaleHeadError`` on stale head, cine grid checks (source slice +
        frame, destination slices), source effectively present, destinations
        effectively absent (copy never overwrites), source-ring validator
        vocabulary preserved (``copy source invalid:`` keeps
        ``contour.invalid``), ``_prevalidate_copy_containment`` over the
        composed post-copy planes BEFORE any row (plane-generic — keyed by
        ``(slice, frame)`` pairs, so slice-varying destinations are covered),
        per-copy ``self.save()`` backstop with captured authority. Zero rows
        before the first write on any refusal; zero further rows after.
        Returns the LAST appended revision.
        """
        chain = list(store.load_chain(series_uid))
        if not chain:
            raise CMRError(
                INVALID_ARGUMENT,
                f"no legacy revision chain found for series {series_uid!r}; "
                "bootstrap required before copying contours",
            )
        head = chain[-1]
        if head.revision_id != expected_head_revision_id:
            raise StaleHeadError(
                f"stale head: expected {expected_head_revision_id!r} but the "
                f"persisted chain head is {head.revision_id!r}; zero rows written"
            )
        try:
            structure_enum = (
                structure if isinstance(structure, Structure) else Structure(structure)
            )
        except ValueError as exc:
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy structure {structure!r} is unknown",
            ) from exc
        if not isinstance(src_slice_index, int) or isinstance(src_slice_index, bool):
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy src_slice_index must be an integer, got {src_slice_index!r}",
            )
        if not isinstance(src_frame_index, int) or isinstance(src_frame_index, bool):
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy src_frame_index must be an integer, got {src_frame_index!r}",
            )
        if cine is None:
            raise CMRError(INVALID_ARGUMENT, "copy requires the cine grid")
        if not (0 <= src_slice_index < cine.n_slices):
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy source slice {src_slice_index} out of cine slice range "
                f"(0..{cine.n_slices - 1})",
            )
        if not (0 <= src_frame_index < cine.n_frames):
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy source frame {src_frame_index} out of cine frame range "
                f"(0..{cine.n_frames - 1})",
            )
        try:
            draft_targets = list(dst_slice_indices)
        except TypeError as exc:
            raise CMRError(
                INVALID_ARGUMENT,
                f"copy dst_slice_indices must be a list, got {dst_slice_indices!r}",
            ) from exc
        for sl in draft_targets:
            if not (isinstance(sl, int) and not isinstance(sl, bool)):
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"copy destination slice must be an integer, got {sl!r}",
                )
            if not (0 <= sl < cine.n_slices):
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"copy destination slice {sl} out of cine slice range "
                    f"(0..{cine.n_slices - 1})",
                )
        src_key = (src_slice_index, src_frame_index, structure_enum)
        source = effective_chain_contour(chain, src_key)
        if source is None:
            raise CMRError(
                CONTOUR_INVALID,
                f"copy source {src_key[0]}:{src_key[1]}:{src_key[2].value} "
                "is absent on the chain (missing or removed); a copy requires "
                "an existing source contour",
            )
        try:
            copies = copy_contour_across_slices(source, draft_targets)
        except CMRError as exc:
            raise CMRError(exc.code, f"copy source invalid: {exc.message}") from exc
        for sibling in copies:
            dst_key = (sibling.slice_index, sibling.frame_index, sibling.structure)
            if effective_chain_contour(chain, dst_key) is not None:
                raise CMRError(
                    CONTOUR_INVALID,
                    f"copy destination {dst_key[0]}:{dst_key[1]}:{dst_key[2].value} "
                    "already carries a contour on the chain; copy never "
                    "overwrites (edit the existing contour instead)",
                )
        _prevalidate_copy_containment(chain, copies)
        saved: ContourRevision | None = None
        current: ContourRevision | Prediction = head
        for sibling in copies:
            dst_key = (sibling.slice_index, sibling.frame_index, sibling.structure)
            try:
                current = self.save(
                    store,
                    current,
                    {sibling.structure: sibling},
                    editor,
                    created_at,
                    cine=cine,
                    expected_head_revision_id=current.revision_id,
                    expected_selection_id=expected_selection_id,
                    expected_inclusion_set_id=expected_inclusion_set_id,
                )
            except CMRError as exc:
                raise CMRError(
                    exc.code,
                    f"copy destination {dst_key[0]}:{dst_key[1]}:{dst_key[2].value} "
                    f"refused ({exc.message}); zero further rows written",
                ) from exc
            assert isinstance(current, ContourRevision)
            saved = current
            chain = list(store.load_chain(series_uid))
        assert saved is not None
        return saved

    def adopt_selected_phase_ai_checked(
        self,
        store: SqliteCmrStore,
        cine: CineVolume,
        series_uid: str,
        *,
        editor: str,
        created_at: str,
        expected_head_revision_id: str | None,
        expected_selection_id: str | None,
        software_version: str = VERSION,
    ) -> dict[str, Any]:
        """Explicitly materialize the CURRENT-ELIGIBLE sparse AI artifact (F18).

        CMR_FUNCV1_ADOPT_001 slice A — ONE explicit checked action. This
        method is NEVER called by inference completion: no auto-adopt exists
        anywhere; only an explicit physician intent reaches it.

        Preconditions verified BEFORE any row is written (fail-closed, typed,
        zero rows on refusal):

        1. an EXISTING legacy revision chain (this action appends to the
           physician lineage; it never bootstraps one);
        2. ``expected_head_revision_id`` equals the current chain head
           (:class:`~cmr_persistence.store.StaleHeadError` otherwise);
        3. a persisted ED/ES selection whose id equals
           ``expected_selection_id`` (``StaleHeadError`` otherwise);
        4. a CURRENT-ELIGIBLE sparse artifact via
           ``store.load_current_selected_phase_prediction`` bound to EXACTLY
           that head + selection (``ADOPT_INELIGIBLE`` otherwise — stale /
           unbound / cancelled / failed artifacts are audit-only);
        5. binding-coherent series, cine grid and DICOM workflow identity
           (``ADOPT_BINDING_MISMATCH`` otherwise);
        6. the COMPLETE five-group required key set
           (``cmr_workflow.coverage.selected_phase_required_keys`` over
           ``SELECTED_PHASE_GROUPS``, deduplicated for a flat ``ed_frame ==
           es_frame`` selection where the two roles share one plane) present
           in the artifact (``ADOPT_INCOMPLETE`` otherwise, naming the
           missing keys);
        7. pre-projection topology for EVERY key to adopt
           (:func:`check_pre_projection_topology`); a non-projectable key
           records its durable block and the action refuses with
           ``CMRError(CONTOUR_INVALID, "BLOCKED_NON_PROJECTABLE_TOPOLOGY:
           ...")`` — fail-visible, never silently flattened.

        Physician-wins: an existing chain key whose latest contour is
        physician-owned (:func:`_is_physician_owned_contour`: human-drawn
        ``source == "manual"`` or physician-``edited`` AI geometry) is
        PRESERVED verbatim and never overwritten. An unedited AI-origin
        chain contour is superseded by the current-eligible artifact (its own
        provenance stays in the append-only ancestors). If every required key
        is physician-owned the action refuses (``ADOPT_NOTHING_TO_ADOPT``) so
        a no-op head bump can never invalidate analysis.

        Materialization: the adopted keys are grouped deterministically by
        ``(slice, source frame)`` plane (canonical five-group order) and each
        plane appends exactly ONE
        :class:`~cmr_core.domain_types.ContourRevision` through
        ``store.save_revision_checked`` (the single-owner single-revision
        checked writer, head CAS re-verified inside its transaction). Each
        adopted :class:`~cmr_contours.model.Contour` keeps ``source="ai"``
        plus the artifact's ``model`` and ``provenance``, so AI-origin vs
        physician-edited stays auditable; the revision itself is composed
        with ``model=None`` (a physician adoption event is not an AI run).

        Atomic boundary (AMENDMENT-001): the whole multi-plane adoption
        persists through ``store.save_adopted_revision_batch_checked`` — ONE
        ``BEGIN IMMEDIATE`` transaction re-verifying the chain head AND the
        latest ED/ES selection in the same snapshot, then all plane
        revisions, then one ``adoption_bindings`` row. Any failure rolls
        back everything: zero revisions, zero binding rows (never a partial
        adoption).

        Returns the binding payload (identity + coverage + model identity +
        physician actor/timestamp + adopted/preserved key lists). Nothing is
        computed and no measurement/QC/approval row is touched.
        """
        if not isinstance(series_uid, str) or not series_uid.strip():
            raise CMRError(INVALID_ARGUMENT, "series_uid must be a non-empty string")
        if not isinstance(editor, str) or not editor.strip():
            raise CMRError(
                INVALID_ARGUMENT,
                "editor must be a non-empty physician identity",
            )
        chain = tuple(store.load_chain(series_uid))
        if not chain:
            raise CMRError(
                ADOPT_CHAIN_REQUIRED,
                f"series {series_uid!r} has no legacy revision chain; explicit "
                "sparse adoption materializes into the EXISTING physician "
                "revision lineage (bootstrap a chain first)",
            )
        head = chain[-1]
        if expected_head_revision_id != head.revision_id:
            raise StaleHeadError(
                f"stale head: expected {expected_head_revision_id!r} but the "
                f"persisted chain head is {head.revision_id!r}; zero rows written"
            )
        selection = store.load_latest_selection(series_uid)
        if selection is None:
            raise CMRError(
                ADOPT_SELECTION_REQUIRED,
                "a physician-confirmed ED/ES selection is required before "
                "sparse adoption; none is persisted for this series",
            )
        if expected_selection_id != selection.selection_id:
            raise StaleHeadError(
                f"selection mismatch: expected {expected_selection_id!r}, "
                f"current {selection.selection_id!r}; zero rows written"
            )
        for name, frame in (
            ("ed_frame", selection.ed_frame),
            ("es_frame", selection.es_frame),
        ):
            if (
                isinstance(frame, bool)
                or not isinstance(frame, int)
                or not 0 <= frame < cine.n_frames
            ):
                raise CMRError(
                    ADOPT_BINDING_MISMATCH,
                    f"persisted selection {name} {frame!r} is outside the cine "
                    f"frame grid [0, {cine.n_frames}); cannot bind the sparse "
                    "artifact to this cine",
                )
        sparse = store.load_current_selected_phase_prediction(
            series_uid,
            head_revision_id=head.revision_id,
            selection_id=selection.selection_id,
        )
        if sparse is None:
            raise CMRError(
                ADOPT_INELIGIBLE,
                f"no CURRENT-ELIGIBLE selected-phase prediction bound to head "
                f"{head.revision_id!r} and selection {selection.selection_id!r}; "
                "stale, unbound, cancelled and failed artifacts are audit-only — "
                "request selected-phase AI for this lineage and retry",
            )
        if sparse.series_uid != series_uid:
            raise CMRError(
                ADOPT_BINDING_MISMATCH,
                f"sparse artifact series {sparse.series_uid!r} does not match "
                f"requested series {series_uid!r}",
            )
        if sparse.slice_count != cine.n_slices or sparse.frame_count != cine.n_frames:
            raise CMRError(
                ADOPT_BINDING_MISMATCH,
                f"sparse artifact grid ({sparse.slice_count}x{sparse.frame_count}) "
                f"does not match the cine grid ({cine.n_slices}x{cine.n_frames})",
            )
        bound_workflow = head.provenance.workflow_id
        if bound_workflow is None:
            bound_workflow = cine.workflow_id
        sparse_workflow = sparse.provenance.workflow_id
        if (
            bound_workflow is None
            or sparse_workflow is None
            or str(sparse_workflow) != str(bound_workflow)
        ):
            raise CMRError(
                ADOPT_BINDING_MISMATCH,
                "sparse artifact DICOM workflow identity "
                f"{sparse_workflow!r} does not match the current lineage "
                f"workflow identity {bound_workflow!r}",
            )
        required = tuple(
            dict.fromkeys(
                selected_phase_required_keys(
                    cine.n_slices, selection.ed_frame, selection.es_frame
                )
            )
        )
        missing = tuple(key for key in required if not sparse.contours.get(key))
        # F22 (CMR_FUNCV1_F7_GENUINE_CLOSEOUT_001, C1 P1-1+P1-2): the
        # completeness gate consumes the EXISTING persisted applicability
        # authority — the latest APPLICABLE inclusion set for THIS head +
        # selection. Waivers stay in their persisted (slice, role,
        # structure) vocabulary; a required contour key
        # ``(slice, frame, structure)`` carrying ``role`` is exempted only
        # when that role's waiver covers it — role-exact (F19 pattern at
        # ``measurements.py``). EMPTY-consistent ``not get(key)``: an
        # artifact-PRESENT key with ``()`` is missing exactly like an
        # absent key, so the SAME waived leg exempts it (its adopt-loop
        # leg then skips it too — never adopted, never topology-blocked).
        # Shared-cavity on a flat ED==ES selection: the single shared
        # contour key is covered by two role keys (ED + ES); it is exempt
        # only when BOTH are waived. ED-only myocardium: only the ED role
        # key exists (ES myocardium is rejected by the grid validator).
        # No second authority is created: the waiver set is read from the
        # SAME persisted record (with head/selection/workflow coherence
        # verified) that analysis/QC consume. When no applicable set
        # exists the gate is byte-identical to the released behavior.
        # The exact set id + waiver-role coverage consumed here is
        # captured (``adopt_authority_*``) and re-verified INSIDE the
        # batch transaction (C1 P1-3) so a superseding inclusion persist
        # racing the adoption refuses with zero writes.
        waived_roles_by_key: dict[tuple[int, int, Structure], set[str]] = {}
        required_roles_by_key: dict[tuple[int, int, Structure], set[str]] = {}
        for role_token, group_structure in SELECTED_PHASE_GROUPS:
            role = (
                INCLUSION_ROLE_ED if role_token == "ed" else INCLUSION_ROLE_ES
            )
            frame = (
                selection.ed_frame
                if role_token == "ed"
                else selection.es_frame
            )
            for slice_index in range(cine.n_slices):
                try:
                    contour_key: tuple[int, int, Structure] = (
                        int(slice_index),
                        int(frame),
                        Structure(group_structure.value),
                    )
                except (TypeError, ValueError):  # pragma: no cover - enum guard
                    continue
                required_roles_by_key.setdefault(contour_key, set()).add(role)
        waived_contour_keys: set[tuple[int, int, Structure]] = set()
        applicable = store.load_latest_applicable_inclusion_set(
            series_uid,
            head_revision_id=head.revision_id,
            selection_id=selection.selection_id,
            workflow_id=bound_workflow,
            expected_n_slices=cine.n_slices,
        )
        if applicable is not None:
            applicable_set = applicable[0]
            if (
                applicable_set.series_uid == series_uid
                and applicable_set.selection_id == selection.selection_id
            ):
                policy = self.slice_inclusion_policy_from_record(
                    applicable_set,
                    ed_frame=selection.ed_frame,
                    es_frame=selection.es_frame,
                    n_slices=cine.n_slices,
                )
                role_frame = {
                    INCLUSION_ROLE_ED: selection.ed_frame,
                    INCLUSION_ROLE_ES: selection.es_frame,
                }
                for waived_slice, waived_role, waived_structure in applicability_waivers(
                    policy
                ):
                    frame = role_frame.get(waived_role)
                    if frame is None:
                        continue
                    try:
                        contour_key = (
                            int(waived_slice),
                            int(frame),
                            Structure(waived_structure),
                        )
                    except (TypeError, ValueError):
                        continue
                    # Role-exact (C1 P1-2): keep the waiver in its own
                    # (slice, role, structure) vocabulary; the contour key
                    # is exempted only when EVERY required role covering it
                    # is waived (flat ED==ES shared cavity needs both;
                    # non-flat keys carry exactly one role so this is the
                    # exact single-role waiver). A waiver for a
                    # non-required contour key is inert (no loosening).
                    waived_roles_by_key.setdefault(contour_key, set()).add(
                        waived_role
                    )
                for contour_key, roles in required_roles_by_key.items():
                    waived_here = waived_roles_by_key.get(contour_key, set())
                    if roles and roles <= waived_here:
                        waived_contour_keys.add(contour_key)
        # C1 P1-3: capture the EXACT authority consumed above so the batch
        # writer can re-verify it in-transaction (TOCTOU guard). Identity =
        # (latest applicable set id or None) + frozen waiver-role coverage.
        # ``None`` set id means "no applicable authority": any set racing
        # in before the batch write is a lineage change and must refuse.
        adopt_authority_set_id: str | None = None
        if applicable is not None:
            adopt_authority_set_id = applicable[0].inclusion_set_id
        adopt_authority_waivers: tuple[tuple[int, str, str], ...] = tuple(
            sorted(
                (
                    int(slice_index),
                    str(role),
                    str(structure.value),
                )
                for (slice_index, _frame, structure), roles in waived_roles_by_key.items()
                for role in sorted(roles)
            )
        )
        missing = tuple(key for key in missing if key not in waived_contour_keys)
        # NOTE: the waived leg of the partition (``missing`` keys covered by
        # the exact persisted waiver) is consumed by the adopt_keys loop
        # below (``key in waived_contour_keys`` branch); only the unwaived
        # remainder refuses here.
        if missing:
            preview = ", ".join(
                f"({key[0]},{key[1]},{key[2].value})" for key in missing[:8]
            )
            extra = len(missing) - 8
            suffix = f" (+{extra} more)" if extra > 0 else ""
            raise CMRError(
                ADOPT_INCOMPLETE,
                "incomplete selected-phase coverage: the artifact lacks "
                f"{len(missing)} required five-group key(s): {preview}{suffix}; "
                "refusing a partial adoption (no key is invented or projected)",
            )
        # Existing chain occupancy per exact key; later revisions win exactly
        # like ViewerState.merged_contours_at / the quantification index.
        # Tombstone rule (CMR_CREATE_A_001, R2 C1): keys whose newest chain
        # event is a remove tombstone are dropped, so a removed key is never
        # classified as physician-preserved (no resurrection into
        # preserved_keys).
        existing: dict[tuple[int, int, Structure], Contour] = {}
        for revision in chain:
            for structure, contour in revision.contours.items():
                existing[
                    (contour.slice_index, contour.frame_index, structure)
                ] = contour
        for tombstoned in head_tombstoned_keys(chain):
            existing.pop(tombstoned, None)
        adopt_keys: list[tuple[int, int, Structure]] = []
        preserved_keys: list[tuple[int, int, Structure]] = []
        waived_keys: list[tuple[int, int, Structure]] = []
        for key in required:
            if key in waived_contour_keys and not sparse.contours.get(key):
                # F22 (+ C1 P1-1): exactly-waived + artifact-missing —
                # missing is EMPTY-consistent (absent key OR present
                # ``()``), so a waived EMPTY-present key is exempt exactly
                # like a waived absent key: never adopted, never invented,
                # never topology-preflighted, never durably blocked (the
                # analysis/QC entries consume the same waiver downstream
                # as zero). C1 P1-2: the set membership above is already
                # role-exact (flat shared cavity needs both roles waived).
                waived_keys.append(key)
                continue
            carried = existing.get(key)
            if carried is not None and _is_physician_owned_contour(carried):
                preserved_keys.append(key)
            else:
                adopt_keys.append(key)
        if not adopt_keys:
            raise CMRError(
                ADOPT_NOTHING_TO_ADOPT,
                f"all {len(required)} required five-group key(s) are already "
                "carried by physician geometry in the current chain; nothing "
                "to adopt (refusing a no-op head bump)",
            )
        # Fail-visible topology preflight over EVERY key to adopt, BEFORE any
        # write: each non-projectable key records its durable block and the
        # whole action refuses with zero new revisions.
        reason_by_key: list[str] = []
        for key in adopt_keys:
            try:
                check_pre_projection_topology(key, sparse.contours[key])
            except CMRError as exc:
                reason = (
                    f"BLOCKED_NON_PROJECTABLE_TOPOLOGY: key (s={key[0]}, "
                    f"f={key[1]}, struct={key[2].value}) non-projectable "
                    f"({exc.message})"
                )
                store.record_topology_block(
                    series_uid=series_uid,
                    slice_index=key[0],
                    frame_index=key[1],
                    structure=key[2].value,
                    reason=reason,
                    created_at=created_at,
                )
                reason_by_key.append(reason)
        if reason_by_key:
            preview = "; ".join(reason_by_key[:4])
            extra = len(reason_by_key) - 4
            suffix = f" (+{extra} more)" if extra > 0 else ""
            raise CMRError(
                CONTOUR_INVALID,
                "BLOCKED_NON_PROJECTABLE_TOPOLOGY: "
                f"{len(reason_by_key)} adopted key(s) are not projectable and "
                "were durably blocked; refusing to flatten topology — no "
                f"revision was written: {preview}{suffix}",
            )
        planes: list[tuple[int, int]] = []
        for key in adopt_keys:
            plane = (key[0], key[1])
            if plane not in planes:
                planes.append(plane)
        keys_by_plane: dict[tuple[int, int], list[tuple[int, int, Structure]]] = {
            plane: [] for plane in planes
        }
        for key in adopt_keys:
            keys_by_plane[(key[0], key[1])].append(key)
        # C2 (AMENDMENT-001) + F22 C1 P1-3: build every revision WITHOUT
        # writing, then persist the whole batch atomically via
        # ``save_adopted_revision_batch_checked`` — head + selection +
        # the EXACT applicability authority consumed above (latest
        # applicable set id + waiver-role coverage) are re-verified INSIDE
        # the same transaction as the inserts, so a selection change OR a
        # superseding inclusion persist racing the adoption rolls back the
        # WHOLE batch (zero revisions, zero binding rows). The per-append
        # re-check is subsumed by the transactional check; no partial
        # adoption can ever persist.
        parent: ContourRevision = head
        new_revisions: list[ContourRevision] = []
        for plane in planes:
            plane_contours = {
                key[2]: _project_selected_phase_to_legacy_draft(
                    store, series_uid, sparse, key, created_at=created_at
                )
                for key in keys_by_plane[plane]
            }
            merged = dict(parent.contours)
            merged.update(plane_contours)
            provenance = Provenance(
                study_uid=(
                    parent.provenance.study_uid
                    if parent.provenance.study_uid is not None
                    else sparse.study_uid
                ),
                series_uid=series_uid,
                software_version=software_version,
                model=None,
                created_at=created_at,
                revision=parent.provenance.revision + 1,
                workflow_id=(
                    parent.provenance.workflow_id
                    if parent.provenance.workflow_id is not None
                    else sparse.provenance.workflow_id
                ),
            )
            revision = ContourRevision.from_parent(
                parent,
                series_uid=series_uid,
                contours=merged,
                editor=editor,
                created_at=created_at,
                provenance=provenance,
            )
            new_revisions.append(revision)
            parent = revision
        binding: dict[str, Any] = {
            "domain": "AdoptionBinding",
            "series_uid": series_uid,
            "head_revision_id": head.revision_id,
            "selection_id": selection.selection_id,
            "prediction_id": sparse.prediction_id,
            "model_run_id": sparse.model_run_id,
            "revision_ids": [],
            "adopted_by": editor,
            "adopted_at": created_at,
        }
        adopted_ids = store.save_adopted_revision_batch_checked(
            expected_head_revision_id=head.revision_id,
            expected_selection_id=selection.selection_id,
            expected_inclusion_set_id=adopt_authority_set_id,
            expected_inclusion_waivers=adopt_authority_waivers,
            expected_n_slices=cine.n_slices,
            revisions=new_revisions,
            binding={
                **binding,
                "revision_ids": [
                    revision.revision_id for revision in new_revisions
                ],
            },
        )
        assert adopted_ids == [revision.revision_id for revision in new_revisions]
        model = sparse.model
        return {
            "status": "adopted",
            "series_uid": series_uid,
            "workflow_id": str(bound_workflow),
            "prediction_id": sparse.prediction_id,
            "source": "ai",
            "head_revision_id": head.revision_id,
            "new_head_revision_id": parent.revision_id,
            "revision_ids": [revision.revision_id for revision in new_revisions],
            "selection_id": selection.selection_id,
            "ed_frame": selection.ed_frame,
            "es_frame": selection.es_frame,
            "slice_count": int(cine.n_slices),
            "frame_count": int(cine.n_frames),
            "source_frame_indices": [int(index) for index in sparse.source_frame_indices],
            "model": (
                None
                if model is None
                else {
                    "name": model.name,
                    "version": model.version,
                    "checksum": model.checksum,
                    "preprocessing_version": model.preprocessing_version,
                    "label_map": dict(model.label_map),
                }
            ),
            "adopted_by": editor,
            "adopted_at": created_at,
            "adopted_keys": [
                [key[0], key[1], key[2].value] for key in adopt_keys
            ],
            "physician_preserved_keys": [
                [key[0], key[1], key[2].value] for key in preserved_keys
            ],
            "waived_keys": [
                [key[0], key[1], key[2].value] for key in waived_keys
            ],
        }

    def reopen(
        self, store: SqliteCmrStore, series_uid: str, *, cine: CineVolume | None = None
    ) -> dict[str, Any]:
        """Load the persisted workflow state for one series as plain data.

        Returns a dict with keys ``series_uid``, ``chain`` (ordered
        revisions), ``latest_selection``, ``approvals``, ``qc_decisions``,
        plus (S2C-001) ``workflow_id``, ``source_manifest``,
        ``latest_historical_measurement`` and ``has_stale_analysis``.

        FROZEN PROJECTION (spec §5 / correction round 1 F4), STRENGTHENED by
        CMR_FUNCTION_V1_ANALYSIS_001: ``latest_measurement`` is present ONLY
        when the CURRENT coherent selected-phase analysis (head + latest
        selection + latest applicable inclusion + its exact 9-metric
        measurement + that measurement's QC digest) resolves. A measurement
        that is merely bound to the head/selection pair but is NOT part of the
        current analysis authority (unpaired QC, superseded inclusion, older
        selection) is exposed ONLY as ``latest_historical_measurement``.
        ``has_stale_analysis`` is True when a historical measurement exists
        that no longer resolves as the current analysis. "Latest" is STORE
        INSERTION SEQUENCE for measurements, selections and analyses (disclosed
        amendment: never a caller-controlled timestamp).

        The additive ``analysis`` key carries the current
        :class:`~cmr_persistence.records.SelectedPhaseAnalysisRecord` (or
        ``None``) so every reader resolves metrics + QC + bindings through the
        same authority; ``analysis_stale_reason`` carries the stable reason
        token when no current analysis resolves.

        Manifest-bound history (any persisted record carries a workflow id)
        REQUIRES the reloaded :class:`~cmr_core.cine.CineVolume`: it must
        match the persisted manifest's series UID, workflow id and cine
        digest before anything is returned. Legacy unbound synthetic/v2
        history remains readable with ``cine=None``.
        """
        # C1 (R1 P1a): chain + analysis come from ONE atomic snapshot —
        # no refresh commit can interleave between the two reads and mix a
        # stale chain head with a newer analysis. A snapshot whose analysis
        # does not bind the snapshot's own chain head is stale-by-construction
        # and returns analysis=None (never a mixed-head projection).
        snapshot = store.load_reopen_snapshot(
            series_uid,
            expected_n_slices=None if cine is None else cine.n_slices,
        )
        # C2 (R2 P1): the authority tuple comes FROM the snapshot — chain
        # ids, selection id and applicable inclusion id are the snapshot's
        # own reads, never re-reads around it. Post-snapshot loads below are
        # full objects for display only; the CURRENT-authority verdict uses
        # snapshot ids.
        chain = store.load_chain(series_uid)
        latest_historical_measurement = store.load_latest_measurement(series_uid)
        latest_selection = store.load_latest_selection(series_uid)
        approvals = store.load_approvals(series_uid)
        qc_decisions = store.load_qc_decisions(series_uid)

        # F14 frozen projection: the coherent current analysis binds EXACTLY
        # the current head, the latest selection, the latest applicable
        # inclusion set, its measurement and that measurement's QC payload.
        head = chain[-1] if chain else None
        snap_ids = snapshot["chain_ids"]
        if (
            head is not None
            and latest_selection is not None
            and snap_ids
            and head.revision_id == snap_ids[-1]
            and latest_selection.selection_id == snapshot["selection_id"]
        ):
            analysis, latest_measurement, analysis_stale_reason = (
                snapshot["analysis"],
                snapshot["measurement"],
                snapshot["analysis_stale_reason"],
            )
        elif not snap_ids or head is None or latest_selection is None:
            analysis, latest_measurement, analysis_stale_reason = (
                None, None, "no_head_or_selection"
            )
        else:
            analysis, latest_measurement, analysis_stale_reason = (
                None, None, "snapshot_authority_mismatch"
            )
            latest_measurement = None
        if (
            analysis is not None
            and head is not None
            and (
                analysis.head_revision_id != head.revision_id
                or analysis.selection_id != latest_selection.selection_id
                or (
                    analysis.inclusion_set_id
                    != snapshot["applicable_inclusion_set_id"]
                )
            )
        ):
            analysis, latest_measurement = None, None
            analysis_stale_reason = "snapshot_authority_mismatch"
            has_stale_analysis = True
        else:
            has_stale_analysis = None  # resolved below by the legacy rule
        if has_stale_analysis is None:
            has_stale_analysis = (
                latest_historical_measurement is not None
                and latest_measurement is None
            )

        # Correction round 2 R1: the workflow coherence scan covers EVERY
        # persisted row of the series (all revisions, ALL selections, ALL
        # measurements, approvals and decisions) via the store's dedicated
        # loader. Bound/unbound MIXING across the whole history fails closed,
        # exactly like two conflicting bound ids do: an A->B->A history or a
        # single legacy-unbound row beside manifest-bound rows can never be
        # opened as one coherent lineage. (None entries are NOT discarded:
        # their PRESENCE beside any bound id is the mixing signal.)
        persisted_workflows = store.load_series_workflow_ids(series_uid)
        bound_ids = {value for value in persisted_workflows if value is not None}
        unbound_present = any(value is None for value in persisted_workflows)
        if len(bound_ids) > 1:
            raise CMRError(
                INVALID_ARGUMENT,
                f"persisted history mixes workflow identities "
                f"{sorted(bound_ids)!r}; one series carries exactly one "
                "manifest-bound workflow",
            )
        if bound_ids and unbound_present:
            raise CMRError(
                INVALID_ARGUMENT,
                f"persisted history mixes manifest-bound workflow "
                f"{next(iter(bound_ids))!r} with UNBOUND rows; bound and "
                "unbound records may never be mixed in one lineage",
            )
        source_manifest: DicomSourceManifest | None = None
        workflow_id = next(iter(bound_ids)) if bound_ids else None
        if workflow_id is not None:
            source_manifest = store.load_source_manifest(workflow_id)
            if source_manifest is None:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"persisted history references workflow {workflow_id!r} but "
                    "its DICOM source manifest is missing from the store",
                )
            if source_manifest.series_uid != series_uid:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"source manifest {workflow_id!r} describes series "
                    f"{source_manifest.series_uid!r}, not {series_uid!r}",
                )
            # Manifest-bound history requires the reloaded decoded volume and
            # it must describe EXACTLY the persisted workflow.
            if cine is None:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"series {series_uid!r} has manifest-bound persisted "
                    "history; reopen requires the reloaded CineVolume that "
                    "matches the stored source manifest",
                )
            if (
                cine.series_uid != series_uid
                or cine.workflow_id != workflow_id
                or cine.cine_volume_digest != source_manifest.cine_volume_digest
                or cine.study_uid != source_manifest.study_uid
            ):
                raise CMRError(
                    INVALID_ARGUMENT,
                    "reloaded cine does not match the stored source manifest "
                    f"(series {cine.series_uid!r}, workflow {cine.workflow_id!r}"
                    f", digest {cine.cine_volume_digest!r}); refusing to open "
                    "history against a different source volume",
                )
        # CFM-001: the DISPLAYED applicable inclusion set is re-resolved
        # for the returned head/selection — but it must BELONG to the snapshot
        # authority. A newer set committed after the snapshot (same head +
        # selection) would otherwise mix old analysis/measurement with new
        # inclusion as current (R3 P1). Mismatch -> the whole projection
        # downgrades to snapshot_authority_mismatch (C3).
        applicable_inclusion_set: SliceInclusionSet | None = None
        if head is not None and latest_selection is not None and cine is not None:
            resolved = store.load_latest_applicable_inclusion_set(
                series_uid,
                head_revision_id=head.revision_id,
                selection_id=latest_selection.selection_id,
                workflow_id=workflow_id,
                expected_n_slices=cine.n_slices,
            )
            if resolved is not None:
                applicable_inclusion_set = resolved[0]
        if (
            analysis is not None
            and applicable_inclusion_set is not None
            and applicable_inclusion_set.inclusion_set_id
            != snapshot["applicable_inclusion_set_id"]
        ):
            analysis, latest_measurement = None, None
            analysis_stale_reason = "snapshot_authority_mismatch"
            has_stale_analysis = True
        whole_cine_head = store.load_latest_whole_cine_head(series_uid)
        whole_cine_head_id = (
            None
            if whole_cine_head is None
            else (
                whole_cine_head.prediction_id
                if isinstance(whole_cine_head, WholeCinePrediction)
                else whole_cine_head.revision_id
            )
        )
        topology_blocks = store.load_topology_blocks(series_uid)
        blocked_reason = topology_blocks[0].reason if topology_blocks else None
        return {
            "series_uid": series_uid,
            "chain": chain,
            "latest_measurement": latest_measurement,
            "latest_historical_measurement": latest_historical_measurement,
            "has_stale_analysis": has_stale_analysis,
            "analysis": analysis,
            "analysis_stale_reason": analysis_stale_reason,
            "latest_selection": latest_selection,
            "approvals": approvals,
            "qc_decisions": qc_decisions,
            "workflow_id": workflow_id,
            "source_manifest": source_manifest,
            "applicable_inclusion_set": applicable_inclusion_set,
            "whole_cine_head_id": whole_cine_head_id,
            "topology_blocks": [b.to_canonical() for b in topology_blocks],
            "blocked_reason": blocked_reason,
        }

    def resolve_topology(
        self,
        store: SqliteCmrStore,
        series_uid: str,
        *,
        selection: EDESSelection | None,
        measurement: Measurement | None = None,
        head_revision_id: str | None = None,
        cine: CineVolume | None = None,
    ) -> list[TopologyResolution]:
        """Resolve every persisted topology block key against the CURRENT state.

        Astra F1/F5: this is the relevance-and-resolution predicate that
        replaced the series-wide prohibition. For each persisted block it
        answers, in this order: carried as physician geometry by the current
        revision chain (resolved by replacement); covered by a physician
        disposition bound to the exact current head AND selection (resolved by
        disposition); not required by the selected metrics (visible, NON-
        blocking); otherwise metric-relevant and unresolved (blocking). A
        disposition bound to a different head or selection resolves NOTHING —
        it is reported as :data:`STALE_DISPOSITION` and blocks.

        ``cine`` supplies the authoritative grid extent (``n_slices`` /
        ``n_frames``). When it is omitted the extent is derived from the
        selection's ``n_frames`` plus the maximum slice/frame index present in
        the chain — a conservative superset, so an unknown grid NEVER makes a
        key look irrelevant.
        """
        blocks = store.load_topology_blocks(series_uid)
        if not blocks:
            return []
        chain = store.load_chain(series_uid)
        chain_keys: set[tuple[int, int, Structure]] = set()
        # R1 REVIEW F1 (P1, 2026-09-10): the AUTHORING stamp of a key is the
        # stamp of a revision whose contour content for that key ACTUALLY
        # DIFFERS from what the chain carried for it before — a real geometry-
        # authoring event. The previous rule took the latest stamp of ANY
        # revision carrying the key, so a ``snapshot_qc`` refresh (which
        # copies the parent's contours VERBATIM) gave the key a NEWER
        # authoring time with no physician geometry change and silently
        # flipped ``unresolved_blocking -> resolved_by_replacement``.
        # PRESENCE (``chain_keys``) is unchanged: a key copied forward by a
        # refresh IS carried by the chain. The shared helper keeps the
        # controller and the store's seal last line on ONE derivation.
        chain_keys: set[tuple[int, int, Structure]] = set()
        for revision in chain:
            for structure, contour in revision.contours.items():
                chain_keys.add((contour.slice_index, contour.frame_index, structure))
        # Tombstone rule (CMR_CREATE_A_001, R2 C1): keys whose newest chain
        # event is a remove tombstone read ABSENT here, so topology
        # presence can never resurrect removed geometry.
        chain_keys -= head_tombstoned_keys(chain)
        chain_key_times = chain_key_authoring_times(chain)
        head_id = head_revision_id or (chain[-1].revision_id if chain else "")
        selected = selection
        if selected is None and measurement is not None:
            selected = EDESSelection(
                series_uid=measurement.series_uid,
                ed_frame=measurement.ed_frame,
                es_frame=measurement.es_frame,
                selected_by="derived-from-measurement",
                selected_at=measurement.created_at,
                provenance=measurement.provenance,
            )
        # Without a selection (or a measurement carrying its frames) the
        # selected metrics are UNKNOWN, so relevance cannot be established:
        # every block key stays metric-relevant and therefore blocking.
        n_slices, n_frames = self._grid_extent(
            cine=cine,
            selection=selected,
            chain=chain,
            blocks=blocks,
        )
        # R1 REVIEW F2 (P1): the FULL append-only history is handed to the
        # resolver, which selects the APPLICABLE record per key (the one
        # bound to the CURRENT head + selection) and reports a key whose
        # only record is bound to an older state as STALE_DISPOSITION. A
        # superseded row is provenance, never a resolution of the state
        # being approved now.
        dispositions = list(store.load_topology_dispositions(series_uid))
        return resolve_topology_dispositions(
            blocks=blocks,
            chain_key_times=chain_key_times,
            dispositions=dispositions,
            head_revision_id=head_id,
            selection_id=selected.selection_id if selected is not None else "",
            n_slices=n_slices,
            n_frames=n_frames,
            ed_frame=selected.ed_frame if selected is not None else None,
            es_frame=selected.es_frame if selected is not None else 0,
        )

    @staticmethod
    def _grid_extent(
        *,
        cine: CineVolume | None,
        selection: EDESSelection,
        chain: list[ContourRevision],
        blocks: list[TopologyBlock],
    ) -> tuple[int | None, int | None]:
        """Authoritative grid extent for relevance, else ``(None, None)``.

        Only a real :class:`CineVolume` proves the measurement grid. Without
        one, the resolver treats EVERY block key as metric-relevant (fail
        closed) instead of guessing irrelevance from a partial view.
        """
        del selection, chain, blocks
        if cine is not None:
            return (cine.n_slices, cine.n_frames)
        return (None, None)

    def approve(
        self,
        store: SqliteCmrStore,
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
        cine: CineVolume | None = None,
    ) -> ApprovalSnapshot:
        """Enforce the approval policy and seal one immutable snapshot.

        The verdict is DERIVED from the supplied :class:`~cmr_qc.report
        .QCReport` object (``report.verdict``) — never taken from a
        caller-controlled string (review correction F0). Policy (binding):
        a non-``"PASS"`` verdict requires ``override=True`` AND a non-empty
        ``override_reason``; an explicit override on a ``"PASS"`` verdict is
        rejected as redundant. With a real ``measurement`` the snapshot
        embeds the override reason via the module-documented software_version
        suffix convention, propagates the REAL ``algorithm_version`` from the
        Measurement and is stored through ``store.save_approval`` (whose
        series/frame/revision coherence guards run inside the same
        transaction). With ``measurement=None`` on a REVIEW REQUIRED report,
        the owner-decided B-LITE mode seals a sentinel clinical-decision
        record WITHOUT numeric results (see module docstring "B-LITE
        no-measurement sealing") and persists it durably as a separate
        append-only :class:`~cmr_persistence.records.QCDecisionRecord`
        (milestone N7.1); the sentinel snapshot itself is returned in hand,
        while ``store.save_approval`` keeps refusing approval snapshots
        without a Measurement row.
        """
        # Step 0: Topology relevance-and-resolution guard (Astra F1-F5).
        #
        # REPLACES the former series-wide prohibition ("any persisted block
        # blocks the whole series"). Every persisted block key is resolved
        # against the CURRENT clinical state: resolved by physician geometry
        # in the chain head, resolved by a physician disposition bound to the
        # exact current head + selection, or non-blocking when the key is not
        # required by the selected metrics. A metric-relevant key that is
        # still unresolved (including a STALE disposition binding) fails
        # closed with a typed TOPOLOGY_BLOCKED naming the exact key.
        series_uid_resolved: str | None = None
        if measurement is not None:
            series_uid_resolved = measurement.series_uid
        elif selection is not None:
            series_uid_resolved = selection.series_uid
        else:
            with store._connect() as conn:
                row = conn.execute(
                    "SELECT series_uid FROM contour_revisions WHERE revision_id = ?",
                    (revision_id,),
                ).fetchone()
                if row is not None:
                    series_uid_resolved = str(row[0])

        if series_uid_resolved is not None:
            # The disposition set digest is bound into the sealed snapshot
            # below (Astra contract item 5 / required test 5): the approval
            # proves WHICH dispositions it relied on.
            #
            # R1 REVIEW F2 (P1): it binds the APPLICABLE set — the records
            # bound to the head being sealed AND the selection being sealed —
            # never every historical row of the series. A superseded row is
            # provenance and must not change the identity this approval
            # proves it relied on; the store re-verifies the same projection
            # atomically inside ``save_approval_checked``.
            approved_selection_id = (
                (selection.selection_id if selection is not None else None)
                or (measurement.selection_id if measurement is not None else None)
                or ""
            )
            approved_dispositions = applicable_topology_dispositions(
                store.load_topology_dispositions(series_uid_resolved),
                head_revision_id=revision_id,
                selection_id=approved_selection_id,
            )
            approved_disposition_digest = (
                topology_disposition_set_digest(approved_dispositions)
                if approved_dispositions
                else None
            )
            assert_topology_resolved_for_approval(
                self.resolve_topology(
                    store,
                    series_uid_resolved,
                    selection=selection,
                    measurement=measurement,
                    head_revision_id=revision_id,
                    cine=cine,
                ),
                series_uid=series_uid_resolved,
                head_revision_id=revision_id,
                selection_id=approved_selection_id,
            )
        else:
            approved_dispositions = []
            approved_disposition_digest = None
        if not isinstance(qc_report, QCReport):
            raise CMRError(
                INVALID_ARGUMENT,
                f"qc_report must be a QCReport instance, got "
                f"{type(qc_report).__name__}; the verdict is always derived "
                "from the report object, never from a caller string",
            )
        qc_verdict = qc_report.verdict
        if qc_verdict not in (PASS_VERDICT, REVIEW_REQUIRED_VERDICT):
            raise CMRError(
                INVALID_ARGUMENT,
                f"derived qc_verdict must be {PASS_VERDICT!r} or "
                f"{REVIEW_REQUIRED_VERDICT!r}, got {qc_verdict!r}",
            )
        override_reason_validated = ""
        if qc_verdict != PASS_VERDICT:
            if not override:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"qc_verdict {qc_verdict!r} requires an explicit physician "
                    "override: pass override=True with a non-empty override_reason",
                )
            if not isinstance(override_reason, str) or not override_reason.strip():
                raise CMRError(
                    INVALID_ARGUMENT,
                    "override=True requires a non-empty override_reason documenting "
                    "the physician's decision",
                )
            override_reason_validated = override_reason
        elif override:
            raise CMRError(
                INVALID_ARGUMENT,
                f"override was supplied for a {PASS_VERDICT!r} verdict; overrides "
                "only apply to review-required states",
            )
        if not isinstance(approved_by, str) or not approved_by.strip():
            raise CMRError(INVALID_ARGUMENT, "approved_by must be a non-empty string")

        if measurement is None:
            return self._seal_sentinel_decision(
                store,
                revision_id,
                measurement_id,
                qc_verdict,
                approved_by,
                approved_at,
                selection=selection,
                override_reason=override_reason_validated,
            )

        # S2C-001 cluster B (disclosed amendment): a measured approval MUST
        # reference the explicit EDESSelection it seals — frame-number
        # equality alone is no longer sufficient.
        if selection is None:
            raise CMRError(
                INVALID_ARGUMENT,
                "measured approval requires the explicit EDESSelection that "
                "identifies the reviewed state; frame-number equality alone is "
                "no longer sufficient",
            )
        if (
            selection.series_uid != measurement.series_uid
            or (selection.ed_frame, selection.es_frame)
            != (measurement.ed_frame, measurement.es_frame)
        ):
            raise CMRError(
                INVALID_ARGUMENT,
                "supplied selection disagrees with the measurement record; the "
                "approval must reference one coherent clinical state",
            )
        # S2C-001 cluster A: bound/unbound workflow mixing fails closed.
        if (measurement.provenance.workflow_id or None) != (
            selection.provenance.workflow_id or None
        ):
            raise CMRError(
                INVALID_ARGUMENT,
                f"measurement workflow {measurement.provenance.workflow_id!r} "
                f"disagrees with selection workflow "
                f"{selection.provenance.workflow_id!r}; bound and unbound "
                "records may never be mixed in one lineage",
            )
        if measurement.selection_id is not None and (
            measurement.selection_id != selection.selection_id
        ):
            raise CMRError(
                INVALID_ARGUMENT,
                "measurement binds selection "
                f"{measurement.selection_id!r} but the approval references "
                f"{selection.selection_id!r}; one coherent state required",
            )
        if measurement_id != measurement.measurement_id:
            raise CMRError(
                INVALID_ARGUMENT,
                f"measurement_id argument {measurement_id!r} does not match the "
                f"supplied measurement record {measurement.measurement_id!r}",
            )
        if revision_id != measurement.source_revision_id:
            raise CMRError(
                INVALID_ARGUMENT,
                f"revision_id argument {revision_id!r} does not match the "
                f"measurement's source_revision_id {measurement.source_revision_id!r}",
            )
        # CFM-001 §1.3 (owner policy §3): a NEW post-CFM measured clinical
        # approval may never seal a measurement with inclusion_set_id=None.
        # The null-bound legacy measurements stay readable and immutable;
        # sealing them again is refused here AND inside the atomic checked
        # store transaction (defense in depth, same deterministic verdict).
        if measurement.inclusion_set_id is None:
            raise CMRError(
                INVALID_ARGUMENT,
                "post-CFM measured approval requires the current applicable "
                "inclusion set: the supplied measurement binds none "
                "(inclusion_set_id=None); confirm and persist the complete "
                "inclusion grid, refresh the analysis, then approve",
            )
        supplied_qc_digest = canonical_digest(qc_report)
        # F14 (CMR_FUNCTION_V1_ANALYSIS_001): the store transaction below
        # atomically requires the CURRENT coherent analysis authority — the
        # exact current head, the latest selection, the latest applicable
        # inclusion set, this measurement and THIS QC payload digest — so a
        # measurement paired with a different/replaced/superseded state is
        # refused (typed ``AnalysisAuthorityError``/``StaleHeadError``, zero
        # staged rows) instead of silently mixing metric and QC authorities.
        base_software_version = measurement.software_version
        provenance_software_version = (
            embed_override_reason(base_software_version, override_reason_validated)
            if override
            else base_software_version
        )
        provenance = Provenance(
            study_uid=measurement.provenance.study_uid,
            series_uid=measurement.series_uid,
            software_version=provenance_software_version,
            model=None,
            created_at=approved_at,
            revision=measurement.provenance.revision,
            # S2C-001 cluster A: the sealed approval inherits the measured
            # workflow identity.
            workflow_id=measurement.provenance.workflow_id,
        )
        snapshot = ApprovalSnapshot(
            series_uid=measurement.series_uid,
            source_revision_id=revision_id,
            ed_frame=measurement.ed_frame,
            es_frame=measurement.es_frame,
            measurement_id=measurement.measurement_id,
            qc_verdict=qc_verdict,
            approved_by=approved_by,
            approved_at=approved_at,
            provenance=provenance,
            software_version=base_software_version,
            algorithm_version=measurement.algorithm_version,
            # S2C-001 clusters B+C: explicit selection binding + the exact QC
            # payload digest verified atomically inside the store transaction.
            selection_id=selection.selection_id,
            qc_report_digest=supplied_qc_digest,
            # CFM-001 §1.7: the approval snapshot binds EXACTLY the
            # measurement's applicable inclusion-set identity.
            inclusion_set_id=measurement.inclusion_set_id,
            # Astra F2/F5: the approval binds the topology-disposition set it
            # was resolved against (None when the series carries no
            # dispositions). The store re-verifies it atomically.
            topology_disposition_digest=approved_disposition_digest,
        )
        # S2C-001 clusters C+D: atomic head-check-and-seal. The unchecked
        # ``save_approval`` primitive is NEVER used by production paths.
        store.save_approval_checked(
            revision_id, snapshot, qc_report, approved_dispositions
        )
        return snapshot

    def _seal_sentinel_decision(
        self,
        store: SqliteCmrStore,
        revision_id: str,
        measurement_id: str,
        qc_verdict: str,
        approved_by: str,
        approved_at: str,
        *,
        selection: EDESSelection | None,
        override_reason: str,
    ) -> ApprovalSnapshot:
        """B-LITE no-measurement path: seal AND durably persist the decision.

        See the module docstring "B-LITE no-measurement sealing". Every
        input is validated fail-closed; the returned snapshot carries no
        numeric results and its identity is content-addressed as usual.
        Milestone N7.1: after sealing, the SAME decision is persisted as a
        separate append-only :class:`~cmr_persistence.records
        .QCDecisionRecord` via ``store.save_qc_decision`` — the immutable
        clinical ``approval_snapshots`` table and its foreign key to
        ``measurements`` stay untouched; the durable record anchors to the
        persisted contour-revision chain instead (verified above). The
        record's ``override_reason`` field carries the physician rationale
        FIRST-CLASS (the snapshot keeps embedding it in provenance via the
        documented suffix convention), and both artifacts share one
        provenance so a decision correlates across tables byte-for-byte.
        """
        if qc_verdict != REVIEW_REQUIRED_VERDICT:
            # PASS without a Measurement can never describe a verified state.
            raise CMRError(
                INVALID_ARGUMENT,
                "a real Measurement record is required to approve a PASS verdict "
                "(the snapshot propagates its identity, frames and REAL "
                "algorithm_version); only REVIEW REQUIRED decisions can be sealed "
                "without metrics under owner decision B-LITE",
            )
        if not isinstance(selection, EDESSelection):
            raise CMRError(
                INVALID_ARGUMENT,
                "the B-LITE no-measurement mode requires the explicit EDESSelection "
                "that identifies the series and reviewed ED/ES frames",
            )
        if measurement_id != SENTINEL_MEASUREMENT_ID:
            raise CMRError(
                INVALID_ARGUMENT,
                f"B-LITE sealing forces measurement_id={SENTINEL_MEASUREMENT_ID!r}; "
                f"got {measurement_id!r}",
            )
        if not isinstance(revision_id, str) or not revision_id.strip():
            raise CMRError(INVALID_ARGUMENT, "revision_id must be a non-empty string")
        # S2C-001 clusters C+D: the sentinel decision is DIGEST-EXEMPT but
        # still seals atomically against the CURRENT persisted head via
        # ``save_qc_decision_checked`` (stale heads fail with zero rows; the
        # explicit selection binding is verified inside the same transaction).
        embedded = (
            embed_override_reason(VERSION, override_reason) + SENTINEL_VERSION_SUFFIX
        )
        provenance = Provenance(
            study_uid=selection.provenance.study_uid,
            series_uid=selection.series_uid,
            software_version=embedded,
            model=None,
            created_at=approved_at,
            revision=selection.provenance.revision,
            workflow_id=selection.provenance.workflow_id,
        )
        snapshot = ApprovalSnapshot(
            series_uid=selection.series_uid,
            source_revision_id=revision_id,
            ed_frame=selection.ed_frame,
            es_frame=selection.es_frame,
            measurement_id=SENTINEL_MEASUREMENT_ID,
            qc_verdict=qc_verdict,
            approved_by=approved_by,
            approved_at=approved_at,
            provenance=provenance,
            software_version=VERSION,
            algorithm_version=SENTINEL_ALGORITHM_VERSION,
        )
        # N7.1 persistence, now through the atomic checked path: the separate
        # append-only qc_decision_records row. The store re-verifies the head
        # anchor and the selection binding inside its own transaction.
        store.save_qc_decision_checked(
            revision_id,
            QCDecisionRecord(
                series_uid=snapshot.series_uid,
                source_revision_id=snapshot.source_revision_id,
                ed_frame=snapshot.ed_frame,
                es_frame=snapshot.es_frame,
                qc_verdict=snapshot.qc_verdict,
                approved_by=snapshot.approved_by,
                approved_at=snapshot.approved_at,
                override_reason=override_reason,
                provenance=snapshot.provenance,
                software_version=snapshot.software_version,
                algorithm_version=snapshot.algorithm_version,
                selection_id=selection.selection_id,
            ),
        )
        return snapshot


def materialize_legacy_chain_from_whole_cine(
    store: SqliteCmrStore,
    cine: CineVolume,
    series_uid: str,
    *,
    ed_frame: int = 0,
    created_at: str,
    editor: str = "system:mv2-bootstrap",
    software_version: str = VERSION,
) -> str:
    """Execute two-phase per-plane materialization from whole-cine prediction (§4.4
    as amended by docs/decisions/BOOTSTRAP_PER_PLANE_AMENDMENT_2026-09-09.md).

    Phase 1: Per-Key Preflight over ALL Required Metric Keys.
    For each required key, U11 (``check_pre_projection_topology``) is evaluated.
    A failing key records a durable topology block for its own
    (slice, frame, structure) key (re-records are per-plane idempotent) and is
    then EXCLUDED from the chain; preflight CONTINUES with the remaining keys.
    Preflight aborts with CONTOUR_INVALID only when ZERO keys are projectable
    (nothing to chain) — the same error shape as the pre-amendment
    all-or-nothing abort.

    Phase 2: Transactional Store Write of the Projectable Subset.
    Drafts are constructed via the U12 pass branch for the projectable keys
    ONLY, and store.bootstrap_legacy_chain_checked commits P_0 and the linear
    revisions R_1 ... R_k inside a single write transaction — unchanged
    atomicity for the subset that is written (all rows or none).

    Excluded planes carry a per-plane durable block and are NOT part of the
    chain; the series-wide approval TOPOLOGY_BLOCKED guard is retained
    unchanged (out of scope for this amendment).
    """
    pred = store.load_latest_whole_cine_prediction(series_uid)
    if pred is None:
        raise CMRError(
            INVALID_ARGUMENT,
            f"no whole-cine prediction found for series {series_uid!r}",
        )

    # Required metric keys per §4.4:
    # LV_CAVITY and RV_CAVITY across all slices and frames,
    # plus LV_MYOCARDIUM across all slices at ed_frame.
    required_keys: list[tuple[int, int, Structure]] = []
    for s in range(cine.n_slices):
        for f in range(cine.n_frames):
            required_keys.append((s, f, Structure.LV_CAVITY))
            required_keys.append((s, f, Structure.RV_CAVITY))
        required_keys.append((s, ed_frame, Structure.LV_MYOCARDIUM))

    # Phase 1: Per-Key Preflight (per-plane blocks, no early abort).
    # Amended §4.4 (owner P0 2026-09-09): a non-projectable key records its
    # durable block and is EXCLUDED; preflight continues with the rest.
    projectable_keys: list[tuple[int, int, Structure]] = []
    for key in required_keys:
        source_contours = pred.contours.get(key, ())
        try:
            check_pre_projection_topology(key, source_contours)
        except CMRError as e:
            diagnostic = (
                f"BLOCKED_NON_PROJECTABLE_TOPOLOGY: key (s={key[0]}, f={key[1]}, "
                f"struct={key[2].value}) non-projectable ({e.message})"
            )
            store.record_topology_block(
                series_uid=series_uid,
                slice_index=key[0],
                frame_index=key[1],
                structure=key[2].value,
                reason=diagnostic,
                created_at=created_at,
            )
            continue
        projectable_keys.append(key)

    # Abort only when nothing is projectable: there is no chain to materialize.
    if not projectable_keys:
        raise CMRError(
            CONTOUR_INVALID,
            "BLOCKED_NON_PROJECTABLE_TOPOLOGY: no projectable metric key in "
            f"series {series_uid!r}; legacy chain materialization refused "
            f"({len(required_keys)} required keys, 0 projectable)",
        )

    # Phase 2: Materialization of the projectable subset.
    draft_contours: dict[tuple[int, int, Structure], Contour] = {}
    for key in projectable_keys:
        draft = _project_whole_cine_prediction_to_legacy_draft(
            store, series_uid, pred, key[0], key[1], key[2]
        )
        draft_contours[key] = draft

    planes: list[tuple[int, int]] = []
    for s in range(cine.n_slices):
        for f in range(cine.n_frames):
            planes.append((s, f))

    def _plane_mapping(s: int, f: int) -> dict[Structure, Contour]:
        m: dict[Structure, Contour] = {}
        if (s, f, Structure.LV_CAVITY) in draft_contours:
            m[Structure.LV_CAVITY] = draft_contours[(s, f, Structure.LV_CAVITY)]
        if (s, f, Structure.RV_CAVITY) in draft_contours:
            m[Structure.RV_CAVITY] = draft_contours[(s, f, Structure.RV_CAVITY)]
        if (s, f, Structure.LV_MYOCARDIUM) in draft_contours:
            m[Structure.LV_MYOCARDIUM] = draft_contours[(s, f, Structure.LV_MYOCARDIUM)]
        elif (s, ed_frame, Structure.LV_MYOCARDIUM) in draft_contours:
            m[Structure.LV_MYOCARDIUM] = draft_contours[(s, ed_frame, Structure.LV_MYOCARDIUM)]
        if not m:
            raise CMRError(
                CONTOUR_INVALID,
                f"BLOCKED_NON_PROJECTABLE_TOPOLOGY: plane (s={s}, f={f}) carries no "
                "projectable metric key; it is excluded from the legacy chain",
            )
        return m

    source = pred.source
    model = pred.model if source == "ai" else None
    prov_model = pred.provenance.model if source == "ai" else None

    # P_0 and R_1..R_k are built over the planes that still carry at least one
    # projectable key; fully-blocked planes are skipped (per-plane exclusion).
    planes = [(s, f) for s in range(cine.n_slices) for f in range(cine.n_frames)]
    chained_planes = [
        (s, f)
        for (s, f) in planes
        if (
            (s, f, Structure.LV_CAVITY) in draft_contours
            or (s, f, Structure.RV_CAVITY) in draft_contours
            or (s, f, Structure.LV_MYOCARDIUM) in draft_contours
            or (s, ed_frame, Structure.LV_MYOCARDIUM) in draft_contours
        )
    ]
    if not chained_planes:
        # Defensive: Phase 1 already guarantees at least one projectable key.
        raise CMRError(
            CONTOUR_INVALID,
            "BLOCKED_NON_PROJECTABLE_TOPOLOGY: no projectable plane to chain",
        )

    p0_contours = _plane_mapping(*chained_planes[0])
    p0 = Prediction(
        study_uid=pred.study_uid,
        series_uid=series_uid,
        source=source,
        contours=p0_contours,
        model=model,
        provenance=Provenance(
            study_uid=pred.study_uid,
            series_uid=series_uid,
            software_version=software_version,
            model=prov_model,
            created_at=created_at,
            revision=0,
            workflow_id=pred.provenance.workflow_id,
        ),
        created_at=created_at,
    )

    revisions: list[ContourRevision] = []
    r1 = ContourRevision.from_parent(
        p0,
        series_uid=series_uid,
        contours=p0_contours,
        editor=editor,
        created_at=created_at,
        provenance=Provenance(
            study_uid=pred.study_uid,
            series_uid=series_uid,
            software_version=software_version,
            model=None,
            created_at=created_at,
            revision=1,
            workflow_id=pred.provenance.workflow_id,
        ),
    )
    revisions.append(r1)

    for i, (s, f) in enumerate(chained_planes[1:], start=2):
        prev = revisions[-1]
        plane_contours = _plane_mapping(s, f)
        rev = ContourRevision.from_parent(
            prev,
            series_uid=series_uid,
            contours=plane_contours,
            editor=editor,
            created_at=created_at,
            provenance=Provenance(
                study_uid=pred.study_uid,
                series_uid=series_uid,
                software_version=software_version,
                model=None,
                created_at=created_at,
                revision=i,
                workflow_id=pred.provenance.workflow_id,
            ),
        )
        revisions.append(rev)

    return store.bootstrap_legacy_chain_checked(series_uid, p0, revisions)
