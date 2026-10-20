"""Node 6 QC snapshot adapter: cine contour index -> QCReport -> revision.

Bridges the frozen Node 5 cine-wide :class:`CineContourIndex` and the frozen
AI-QC rules into one physician-facing QC outcome per revision chain, WITHOUT
touching any frozen rule or domain type. Two public operations:

1. :func:`run_revision_qc` groups the index into per-``(slice, frame)``
   :class:`~cmr_contours.model.ContourSet` groups and evaluates the frozen QC
   bundle over the FULL cine stack with ``expected=[LV_CAVITY, RV_CAVITY]`` on
   every ``(slice, frame)`` and containment
   ``inner_outer=[(LV_CAVITY, LV_MYOCARDIUM)]`` evaluated ONLY at the ED frame
   (see "Containment scope" below). When ``volumes_by_frame`` is provided —
   the cavity volume curve indexed by frame, computed by the caller from the
   Node 5 measurement inputs or passed verbatim — the UNCHANGED
   :func:`cmr_qc.rules.check_volume_curve` result is appended, PLUS one
   supplemental ESV>EDV check anchored to the authoritative selection frames
   (see "ESV>EDV supplement" below). The function never rasterizes pixels,
   never calls an AI provider and never computes a clinical metric: it only
   re-runs existing pure rules over composed geometry.

   Source-field convention (contract: every flag records its location):
   geometry flags keep the location string already produced by the frozen
   rules (``"s{slice}f{frame}"``); volume-curve flags carry no inherent
   location, so this module stamps them with the full-stack range string
   ``"s[0:{n_slices}]f[0:{n_frames}]"`` (half-open, deterministic).

   Papillary/hole disclosure (S2C-001 cluster F): NO current QC rule detects
   papillary islands or topology lost during mask-to-single-loop conversion;
   the bundle below validates coverage/intersection/containment of the given
   contours only.

   Workflow coherence (S2C-001 cluster A/B): one exact manifest-bound
   workflow identity is verified across the index chain, the cine and the
   selection via :func:`~cmr_quantification.measurements
   .assert_coherent_workflow_identity`; bound/unbound mixing fails closed.

   Containment scope: ``run_qc_on_contour_set`` is called VERBATIM — full
   bundle including the explicit ``(LV_CAVITY, LV_MYOCARDIUM)`` pair through
   ``inner_outer`` — at ED-frame calls. At non-ED calls the SAME frozen
   bundle minus containment is composed from the frozen primitives
   (:func:`~cmr_qc.rules.check_missing_structures` +
   :func:`~cmr_qc.rules.check_self_intersection`), because the frozen rule
   resolves any falsy ``inner_outer`` argument to its built-in default pair,
   making "no containment" inexpressible through that parameter. Containment
   is therefore evaluated ONLY at the ED frame as required; contours at other
   frames are never dropped from the remaining checks.

   ESV>EDV supplement: the frozen ``check_volume_curve`` conditions its
   "ESV exceeds EDV" branch on ``min(volumes) > max(volumes)``, which is
   unsatisfiable for any finite list — provably dead code in a file this
   node must not modify. To keep the contract-required warning real, this
   module additionally compares the curve AT THE SELECTION FRAMES
   (``volumes[es_frame] > volumes[ed_frame]``, the authoritative manual
   EDESSelection) and emits a QC-severity ``qc.volume_curve`` flag when ESV
   exceeds EDV. ``cmr_qc/rules.py`` stays byte-identical.

2. :func:`run_selected_phase_qc` (FNC-B, additive) evaluates the SAME frozen
   primitives over the FIVE selected-phase geometry groups only (ED
   ``LV_CAVITY``/``LV_MYOCARDIUM``/``RV_CAVITY`` + ES ``LV_CAVITY``/
   ``RV_CAVITY``): missing structures, self-intersections, ED-only
   endo-in-epi containment and a per-ventricle ESV>EDV plausibility check
   anchored to the selection frames. It claims NO volume-curve QC and NO
   all-phase coverage, and it leaves :func:`run_revision_qc` untouched.

   Group-scope correction (R1 finding 1): each role's group is built from
   that role's :data:`SELECTED_PHASE_GROUPS` structures ONLY, via
   :func:`_selected_phase_group` — not from :func:`_group_slice_frame`,
   which collects EVERY structure present at the (slice, frame) and is
   shared with the legacy entry. A contour outside the five groups (a stray
   ES-frame ``LV_MYOCARDIUM``, an intermediate phase, ...) is IGNORED
   COMPLETELY: never required, never checked, never flagged — the same
   policy quantification applies to non-required keys. Without the filter a
   self-intersecting stray contour could flip the verdict to REVIEW
   REQUIRED on geometry the quantification path ignores entirely.

3. :func:`snapshot_qc` forks ONE child
   :class:`~cmr_core.domain_types.ContourRevision` that carries a
   :class:`~cmr_qc.report.QCReport` while keeping the parent's contour
   content verbatim unless the caller supplies an explicit contour map.
   Composition goes through the unchanged
   :meth:`~cmr_core.domain_types.ContourRevision.from_parent`, so version
   monotonicity, the provenance revision bump (+1), series coherence and the
   content-addressed ``revision_id`` semantics are inherited untouched. The
   result is fully deterministic: identical inputs and fixed timestamps yield
   an identical ``revision_id``.

Failure policy: malformed or incoherent inputs (series mismatch between the
index and the selection, selection ``n_frames`` disagreeing with the cine,
ED/ES frames outside the cine grid, curve length disagreeing with
``n_frames``, out-of-grid index keys) raise :class:`~cmr_core.errors.CMRError`
BEFORE any flag is collected — QC never guesses. Synthetic-data friendly:
no patient data, no GPU, no cloud, no AI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from cmr_contours.conversion import contour_to_mask
from cmr_contours.model import Contour, ContourSet
from cmr_core.cine import CineVolume
from cmr_core.domain_types import ContourRevision, EDESSelection, Prediction
from cmr_core.errors import (
    CONTOUR_INVALID,
    INVALID_ARGUMENT,
    QC_VOLUME_CURVE,
    CMRError,
    CMRWarning,
)
from cmr_core.provenance import VERSION, Provenance
from cmr_core.segmentation import Structure
from cmr_quantification.measurements import (
    INCLUSION_ROLE_ED,
    INCLUSION_ROLE_ES,
    INCLUSION_STRUCTURE_LV_CAVITY,
    INCLUSION_STRUCTURE_RV_CAVITY,
    CineContourIndex,
    SliceInclusionPolicy,
    _verify_inclusion_policy_coherence,
    assert_coherent_workflow_identity,
    effective_stack_gaps_mm,
)
from cmr_quantification.volume import slice_volume_ml

from cmr_qc.report import QCReport
from cmr_qc.rules import (
    check_missing_structures,
    check_self_intersection,
    check_volume_curve,
    run_qc_on_contour_set,
)

__all__ = [
    "EXPECTED_PER_FRAME",
    "INNER_OUTER_ED",
    "SELECTED_PHASE_GROUPS",
    "run_revision_qc",
    "run_selected_phase_qc",
    "snapshot_qc",
]

#: Structures required on every ``(slice, frame)`` of the stack (contract 1a).
EXPECTED_PER_FRAME: list[Structure] = [Structure.LV_CAVITY, Structure.RV_CAVITY]

#: Endo/epi pair supplied through ``inner_outer`` at the ED frame (contract 1a).
INNER_OUTER_ED: list[tuple[Structure, Structure]] = [
    (Structure.LV_CAVITY, Structure.LV_MYOCARDIUM)
]

#: FNC-B selected-phase geometry groups: ``(role, structure)`` pairs, ED first.
#: Only these groups are required, validated and checked by
#: :func:`run_selected_phase_qc`; every other index entry (intermediate
#: phases) is IGNORED.
SELECTED_PHASE_GROUPS: tuple[tuple[str, Structure], ...] = (
    ("ED", Structure.LV_CAVITY),
    ("ED", Structure.LV_MYOCARDIUM),
    ("ED", Structure.RV_CAVITY),
    ("ES", Structure.LV_CAVITY),
    ("ES", Structure.RV_CAVITY),
)

#: Temporal-coherence jump threshold (CMR_QC_TEMPORAL_COHERENCE_001): an
#: adjacent frame pair flags when ``|b - a| / min(a, b)`` exceeds this ratio
#: (a zero-to-nonzero step counts as an infinite jump and always flags).
#: 1.00 = a single-frame 100% swing (a doubling/halving), well above
#: physiological beat-to-beat variation on a per-frame curve and well below
#: a full segmentation-glitch spike. Calibrated against the seeded
#: workstation fixtures (gradual 16x14-vs-20x18 box tapers swing ~0.61 and
#: must stay silent) — the initial 0.50 fired on that legitimate taper.
TEMPORAL_JUMP_RATIO: float = 1.0


def _stack_location(n_slices: int, n_frames: int) -> str:
    """Location string stamped onto flags that have no location of their own."""
    return f"s[0:{n_slices}]f[0:{n_frames}]"


def _validate_bindings(
    index: CineContourIndex,
    cine: CineVolume,
    selection: EDESSelection,
) -> tuple[int, int]:
    """Fail closed BEFORE any flag collection on incoherent inputs."""
    if index.series_uid != selection.series_uid:
        raise CMRError(
            INVALID_ARGUMENT,
            f"index series {index.series_uid!r} does not match selection series "
            f"{selection.series_uid!r}",
        )
    n_slices = cine.n_slices
    n_frames = cine.n_frames
    if selection.n_frames is not None and selection.n_frames != n_frames:
        raise CMRError(
            INVALID_ARGUMENT,
            f"selection n_frames {selection.n_frames} disagrees with cine "
            f"n_frames {n_frames}",
        )
    if not (0 <= selection.ed_frame < n_frames) or not (0 <= selection.es_frame < n_frames):
        raise CMRError(
            INVALID_ARGUMENT,
            f"ED/ES frames ({selection.ed_frame}, {selection.es_frame}) are outside "
            f"0..{n_frames - 1}",
        )
    for slice_index, frame_index, structure in index:
        if slice_index >= n_slices or frame_index >= n_frames:
            raise CMRError(
                CONTOUR_INVALID,
                f"contour key ({slice_index},{frame_index},{structure.value}) is "
                f"outside the cine grid {n_slices}x{n_frames}; QC never silently "
                "ignores out-of-range contours",
            )
    return n_slices, n_frames


def _group_slice_frame(
    index: CineContourIndex,
    slice_index: int,
    frame_index: int,
) -> ContourSet:
    """Deterministic per-(slice, frame) group in ``Structure`` definition order."""
    group = ContourSet(slice_index=slice_index, frame_index=frame_index)
    for structure in Structure:
        contour = index.get((slice_index, frame_index, structure))
        if contour is not None:
            group.add(contour)
    return group


def run_revision_qc(
    index: CineContourIndex,
    cine: CineVolume,
    selection: EDESSelection,
    *,
    volumes_by_frame: list[float] | None = None,
) -> QCReport:
    """Run the frozen QC rule bundle over one full revision-chain stack.

    ``index`` is the assembled :class:`CineContourIndex` (see
    :func:`cmr_quantification.measurements.build_cine_contour_index` over the
    ordered ``load_chain()`` revisions), ``cine`` supplies the stack extent
    and ``selection`` the authoritative ED/ES frames. Every ``(slice, frame)``
    of the full stack is checked for missing cavities and self-intersections;
    the endo-in-epi containment pair is evaluated ONLY at the ED frame (see
    module docstring "Containment scope"). ``volumes_by_frame``, when
    provided, must be the cavity volume curve indexed by frame (length ==
    ``cine.n_frames``) — e.g. derived via the Node 5 measurement inputs or
    passed verbatim by the caller — and is fed through the UNCHANGED
    :func:`cmr_qc.rules.check_volume_curve` plus this module's supplemental
    ESV>EDV check at the selection frames (see module docstring "ESV>EDV
    supplement").

    Returns the aggregated :class:`QCReport`; ``report.passed`` is True only
    when no flag fired. Deterministic: two calls over equal inputs produce
    equal flags in equal order.
    """
    n_slices, n_frames = _validate_bindings(index, cine, selection)
    # S2C-001 cluster A/B: one exact workflow identity across chain/cine/
    # selection; mixing bound and unbound inputs fails closed.
    assert_coherent_workflow_identity(index, cine, selection)

    flags: list[CMRWarning] = []
    for slice_index in range(n_slices):
        for frame_index in range(n_frames):
            group = _group_slice_frame(index, slice_index, frame_index)
            if frame_index == selection.ed_frame:
                # Full frozen bundle VERBATIM, containment pair explicit.
                flags.extend(
                    run_qc_on_contour_set(
                        group,
                        expected=list(EXPECTED_PER_FRAME),
                        inner_outer=list(INNER_OUTER_ED),
                    )
                )
            else:
                # Review correction P1 — containment is evaluated ONLY at the
                # ED frame. The frozen rule resolves any falsy inner_outer to
                # its default pair, so omission cannot suppress it; compose
                # the SAME frozen bundle minus the containment leg instead
                # (missing structures first, then self-intersections — the
                # exact order run_qc_on_contour_set uses). No contour is
                # dropped from these remaining checks.
                flags.extend(check_missing_structures(group, list(EXPECTED_PER_FRAME)))
                for contour in group.contours.values():
                    flags.extend(check_self_intersection(contour))

    if volumes_by_frame is not None:
        curve = list(volumes_by_frame)
        if len(curve) != n_frames:
            raise CMRError(
                INVALID_ARGUMENT,
                f"volumes_by_frame carries {len(curve)} values but the cine has "
                f"{n_frames} frames; the curve must be indexed by frame",
            )
        location = _stack_location(n_slices, n_frames)
        for flag in check_volume_curve(curve):
            flags.append(flag if flag.source is not None else replace(flag, source=location))
        # Review correction P0 — ESV>EDV must not fail open: the frozen
        # check_volume_curve branch is dead code (min > max is unsatisfiable),
        # so compare AT THE SELECTION FRAMES here. rules.py stays untouched.
        esv_ml = curve[selection.es_frame]
        edv_ml = curve[selection.ed_frame]
        if esv_ml > edv_ml:
            flags.append(
                CMRWarning.qc(
                    QC_VOLUME_CURVE,
                    f"ESV {esv_ml} ml exceeds EDV {edv_ml} ml at selected frames "
                    f"{selection.ed_frame}/{selection.es_frame}",
                    source=location,
                )
            )
        # Temporal coherence (CMR_QC_TEMPORAL_COHERENCE_001, Phase 11): flag
        # implausible frame-to-frame jumps the per-frame geometry checks and
        # the flat/ESV>EDV curve checks cannot see — e.g. a single-frame
        # segmentation glitch inside an otherwise plausible curve. Additive
        # only: curves with fewer than 3 frames, or no curve at all, behave
        # byte-identically to before (no flags from this block).
        if len(curve) >= 3:
            for frame in range(len(curve) - 1):
                before_v, after_v = curve[frame], curve[frame + 1]
                floor = min(before_v, after_v)
                if floor <= 0.0:
                    # Zero-to-nonzero (or zero-to-zero handled by the flat
                    # check above) counts as an infinite jump and flags —
                    # unless both are exactly zero (no change at all).
                    if before_v == after_v:
                        continue
                    ratio = float("inf")
                else:
                    ratio = abs(after_v - before_v) / floor
                if ratio > TEMPORAL_JUMP_RATIO:
                    ratio_text = f"{ratio:.2f}" if ratio != float("inf") else "inf"
                    flags.append(
                        CMRWarning.qc(
                            QC_VOLUME_CURVE,
                            f"temporal jump f{frame}->{frame + 1}: {before_v} ml "
                            f"-> {after_v} ml (ratio {ratio_text} exceeds "
                            f"{TEMPORAL_JUMP_RATIO:.2f})",
                            source=location,
                        )
                    )

    return QCReport(flags=flags)


def _selected_phase_group(
    index: CineContourIndex,
    slice_index: int,
    frame_index: int,
    expected: list[Structure],
) -> ContourSet:
    """The (slice, frame) group restricted to EXACTLY ``expected``.

    R1 correction C1 (finding 1). Unlike :func:`_group_slice_frame` — which
    collects every structure present at that (slice, frame) and is left
    byte-untouched for :func:`run_revision_qc` — this builder takes the
    structures from the role's declared :data:`SELECTED_PHASE_GROUPS` list
    only. Any other contour at that (slice, frame) (an intermediate phase, a
    stray ES epicardium, ...) is therefore never required, never checked and
    never flagged, which is the same "ignored completely" policy the
    selected-phase quantification path applies to non-required keys.

    Order follows ``expected`` (the declared five-group order), so flag
    emission stays deterministic.
    """
    group = ContourSet(slice_index=slice_index, frame_index=frame_index)
    for structure in expected:
        contour = index.get((slice_index, frame_index, structure))
        if contour is not None:
            group.add(contour)
    return group


def _selected_phase_location(frame_index: int, n_slices: int) -> str:
    """Location string for a selected-phase (whole-slice) flag."""
    return f"s[0:{n_slices}]f{frame_index}"


def run_selected_phase_qc(
    index: CineContourIndex,
    cine: CineVolume,
    selection: EDESSelection,
    *,
    inclusion_policy: SliceInclusionPolicy | None = None,
) -> QCReport:
    """Run the frozen QC rule bundle over the FIVE selected-phase groups only.

    FNC-B additive entry (P1 clinical). It evaluates the SAME frozen
    primitives as :func:`run_revision_qc` —
    :func:`~cmr_qc.rules.check_missing_structures`,
    :func:`~cmr_qc.rules.check_self_intersection` and
    :func:`~cmr_qc.rules.check_containment` through
    :func:`~cmr_qc.rules.run_qc_on_contour_set` — but ONLY over the five
    required geometry groups (ED ``LV_CAVITY``/``LV_MYOCARDIUM``/
    ``RV_CAVITY`` plus ES ``LV_CAVITY``/``RV_CAVITY``) at the physician-
    authored ``selection`` frames:

    - missing-structure checks on exactly those five keys per slice;
    - self-intersection over the same five groups' contours;
    - containment ``(LV_CAVITY, LV_MYOCARDIUM)`` at the ED frame ONLY;
    - ESV>EDV plausibility for BOTH ventricles, computed from the
      selected-phase cavity slab volumes (``mask_area_mm2`` x the same
      ``effective_stack_gaps_mm`` gap tuple / 1000 — no new clinical
      formula and no pixel arithmetic beyond the existing helper).

    Any other contour in ``index`` (an intermediate phase, a stray ES-frame
    epicardium, ...) is IGNORED: it is never required, never checked and
    never flagged. R1 correction C1 (finding 1) enforces this by
    CONSTRUCTION — each role's group is built with
    :func:`_selected_phase_group` from that role's
    :data:`SELECTED_PHASE_GROUPS` structures alone, so an out-of-group
    contour is not merely absent from the expected list, it is absent from
    the group the frozen primitives iterate over (a self-intersecting stray
    ES ``LV_MYOCARDIUM`` therefore cannot flip the verdict). Consequently
    this entry makes NO volume-curve claim and NO all-phase coverage claim:
    it never calls :func:`~cmr_qc.rules.check_volume_curve` and never emits
    a whole-cine ``qc.volume_curve`` flag — ESV>EDV is reported as a
    selected-phase plausibility flag on ``qc.volume_curve`` anchored to the
    selection frames.

    Bindings fail closed exactly like :func:`run_revision_qc` (series
    mismatch, ``n_frames`` disagreement, ED/ES outside the grid, out-of-grid
    index keys) BEFORE any flag is collected.

    Returns the aggregated :class:`QCReport`; deterministic: two calls over
    equal inputs produce equal flags in equal order.

    CMR_FUNCV1_STACK_APPLICABILITY_001 (F19): ``inclusion_policy`` carries
    the SAME physician-authoritative applicability record the
    selected-phase measurement consumes (see
    :func:`~cmr_quantification.measurements.applicability_waivers`).
    The policy identity is verified with the SAME full coherence check
    the measurement applies (series, head/contour-preserving ancestor,
    selection, workflow, frames, slice count) BEFORE any waiver is read —
    a foreign-provenance grid fails closed here exactly as in measurement.
    A required key explicitly waived for its exact ``(slice, role,
    structure)`` is NON-APPLICABLE: it is exempted from the
    missing-structure check (and from containment/ESV>EDV evaluation,
    which consume zero for that slab) instead of flagged. A missing key
    with no waiver is UNKNOWN and is flagged exactly as before. ``None``
    (no authority) preserves the pre-change behavior byte-for-byte.
    """
    n_slices, n_frames = _validate_bindings(index, cine, selection)
    # Same S2C-001 cluster A/B rule as the legacy entry: one exact
    # manifest-bound workflow identity across chain/cine/selection.
    coherent_workflow_id = assert_coherent_workflow_identity(index, cine, selection)
    # C1 (P1-2): the waived-key set is derived from the SAME persisted record
    # the measurement consumes, AFTER the full policy-identity check below.
    # Without it a foreign-provenance grid (foreign revision, selection,
    # workflow, frames or slice count) could waive keys here while the
    # measurement refuses it.
    if inclusion_policy is not None:
        _verify_inclusion_policy_coherence(
            inclusion_policy, index, selection, cine, coherent_workflow_id
        )

    # Phase mapping is EXACT (ED role <-> selection.ed_frame, ES role <->
    # selection.es_frame). A flat ED==ES selection keeps role-exact
    # requiredness: the shared frame carries TWO required keys (one per
    # role), and each role's key is waived only by its OWN role waiver
    # (see _waived_for_role below — an ED waiver never covers an ES key).
    from cmr_quantification.measurements import applicability_waivers

    waived = applicability_waivers(inclusion_policy)

    def _waived_for_role(slice_index: int, role: str, structure: Structure) -> bool:
        return (slice_index, role, structure.value) in waived

    # Required structures per role, taken from the SAME declared five-group
    # vocabulary (ED group: LV_CAVITY, LV_MYOCARDIUM, RV_CAVITY; ES group:
    # LV_CAVITY, RV_CAVITY — declared order preserved).
    expected_by_role: dict[str, list[Structure]] = {"ED": [], "ES": []}
    for role, structure in SELECTED_PHASE_GROUPS:
        expected_by_role[role].append(structure)

    flags: list[CMRWarning] = []
    for slice_index in range(n_slices):
        # ED first (canonical role order), then ES.
        for role in ("ED", "ES"):
            frame_index = selection.ed_frame if role == "ED" else selection.es_frame
            # R1 correction C1 (finding 1): the group is built from the role's
            # declared SELECTED_PHASE_GROUPS structures ONLY. _group_slice_frame
            # includes EVERY structure present at that (slice, frame), so a
            # stray contour outside the five groups (e.g. an ES-frame
            # LV_MYOCARDIUM) would be self-intersection-checked and could flip
            # the verdict even though quantification ignores it. A filtered-out
            # structure is IGNORED COMPLETELY here — never required, never
            # checked, never flagged — exactly the quantification policy.
            group = _selected_phase_group(
                index, slice_index, frame_index, expected_by_role[role]
            )
            # F19: waived (non-applicable) structures leave the expected set
            # for THIS slice/role, so the frozen missing-structure rule
            # cannot flag an explicitly non-contributing key. Present-but-
            # waived geometry stays in the group (self-intersection and
            # containment still evaluate it — authority waives contribution,
            # not geometry validity). Role-exact (C1 P1-3): on a flat
            # ED==ES selection the two roles share one frame, but each
            # role's key is waived only by its OWN role waiver.
            waived_here = [
                structure
                for structure in expected_by_role[role]
                if _waived_for_role(slice_index, role, structure)
                and (slice_index, frame_index, structure) not in index
            ]
            expected_here = [
                structure
                for structure in expected_by_role[role]
                if structure not in waived_here
            ]
            if role == "ED":
                # Full frozen bundle VERBATIM (missing structures, then
                # self-intersections, then the containment pair) — identical
                # to the legacy ED-frame call.
                #
                # F19 containment note: when the ED cavity is waived-absent
                # there is no endocardium to contain, so the pair is
                # evaluated only when BOTH contours are present; a
                # waived-absent cavity with a present epicardium is the
                # SCMR-2020 epicardial-only mass state, not a containment
                # violation.
                flags.extend(
                    check_missing_structures(group, list(expected_here))
                )
                for contour in group.contours.values():
                    flags.extend(check_self_intersection(contour))
                if (slice_index, selection.ed_frame, Structure.LV_CAVITY) in index and (
                    slice_index,
                    selection.ed_frame,
                    Structure.LV_MYOCARDIUM,
                ) in index:
                    inner = group.get(Structure.LV_CAVITY)
                    outer = group.get(Structure.LV_MYOCARDIUM)
                    if inner is not None and outer is not None:
                        from cmr_qc.rules import check_containment

                        flags.extend(check_containment(inner, outer))
            else:
                # ES: the same frozen bundle minus containment (containment
                # is ED-only — review correction P1), composed from the
                # frozen primitives exactly like the legacy non-ED branch.
                flags.extend(check_missing_structures(group, list(expected_here)))
                for contour in group.contours.values():
                    flags.extend(check_self_intersection(contour))

    # --- ESV>EDV plausibility at the selected phases (both ventricles) -------
    # Reuses the existing slab primitives only: the SAME ordered gap tuple
    # and mask_area_mm2 that the quantification path consumes. No curve is
    # built, no intermediate phase is touched, no new clinical formula.
    geometries = list(cine.geometries)
    gaps = effective_stack_gaps_mm(geometries)
    # Role-specific structure vocabulary for the weighted plausibility
    # totals (C1 P1-4): each role's total consumes exactly its own role's
    # inclusion weights, the same weights the measurement's _gated_total
    # applies — an ES-excluded slab weights the ES total only, never ED.
    # Present-but-waived geometry: waiving contribution does NOT waive
    # validity — a present contour still rasterizes (see _cavity_slab),
    # and only the weight zeroes its plausibility contribution.
    _ROLE_STRUCTURE = {
        ("ED", Structure.LV_CAVITY): (INCLUSION_ROLE_ED, INCLUSION_STRUCTURE_LV_CAVITY),
        ("ED", Structure.RV_CAVITY): (INCLUSION_ROLE_ED, INCLUSION_STRUCTURE_RV_CAVITY),
        ("ES", Structure.LV_CAVITY): (INCLUSION_ROLE_ES, INCLUSION_STRUCTURE_LV_CAVITY),
        ("ES", Structure.RV_CAVITY): (INCLUSION_ROLE_ES, INCLUSION_STRUCTURE_RV_CAVITY),
    }
    for ventricle, structure in (("LV", Structure.LV_CAVITY), ("RV", Structure.RV_CAVITY)):
        # A missing cavity is already reported by the frozen missing-structure
        # rule above; plausibility cannot be evaluated without both phases.
        # F19: a WAIVED-ABSENT cavity contributes zero to its phase total
        # (same zero the measurement consumes), so plausibility stays
        # evaluable when authority covers the absence.
        def _cavity_slab(
            slice_index: int, role: str, frame_index: int
        ) -> float:
            # Role-exact (C1 P1-3): the caller passes the slab's own role,
            # so on a flat ED==ES selection each role's slab is waived only
            # by its OWN role waiver. Present geometry always rasterizes
            # (validity is checked above); a waived-absent key contributes
            # zero, an unwaived-absent key is NaN (plausibility skipped —
            # the missing-structure flag above is the verdict for it).
            if (slice_index, frame_index, structure) not in index:
                if _waived_for_role(slice_index, role, structure):
                    return 0.0
                return float("nan")
            mask = contour_to_mask(
                index[(slice_index, frame_index, structure)], geometries[slice_index]
            )
            return slice_volume_ml(mask, geometries[slice_index], gaps[slice_index])

        slabs_ed = [
            _cavity_slab(s, INCLUSION_ROLE_ED, selection.ed_frame)
            for s in range(n_slices)
        ]
        slabs_es = [
            _cavity_slab(s, INCLUSION_ROLE_ES, selection.es_frame)
            for s in range(n_slices)
        ]
        if any(value != value for value in (*slabs_ed, *slabs_es)):
            continue
        if inclusion_policy is None:
            edv_ml = float(sum(slabs_ed))
            esv_ml = float(sum(slabs_es))
        else:
            edv_ml = float(
                sum(
                    inclusion_policy.weight(
                        slice_index, *_ROLE_STRUCTURE[("ED", structure)]
                    )
                    * slab
                    for slice_index, slab in enumerate(slabs_ed)
                )
            )
            esv_ml = float(
                sum(
                    inclusion_policy.weight(
                        slice_index, *_ROLE_STRUCTURE[("ES", structure)]
                    )
                    * slab
                    for slice_index, slab in enumerate(slabs_es)
                )
            )
        if esv_ml > edv_ml:
            flags.append(
                CMRWarning.qc(
                    QC_VOLUME_CURVE,
                    f"selected-phase {ventricle} ESV {esv_ml} ml exceeds "
                    f"{ventricle} EDV {edv_ml} ml at selected frames "
                    f"{selection.ed_frame}/{selection.es_frame}; no "
                    "volume-curve QC is claimed for selected-phase coverage",
                    source=_selected_phase_location(selection.es_frame, n_slices),
                )
            )

    return QCReport(flags=flags)


def snapshot_qc(
    parent: Prediction | ContourRevision,
    qc_report: QCReport,
    *,
    series_uid: str,
    editor: str,
    created_at: str,
    contours: Mapping[Structure, Contour] | None = None,
    software_version: str = VERSION,
) -> ContourRevision:
    """Fork ONE child revision carrying ``qc_report`` (QC-only snapshot).

    The child keeps the parent's contour map VERBATIM when ``contours`` is
    omitted, so attaching a QC outcome changes no geometry; supplying an
    explicit map composes it exactly like an edit would (mapping-key coherence
    is re-validated by the Node 1 constructor). Composition goes through
    :meth:`ContourRevision.from_parent`: version = parent + 1, provenance
    revision = parent + 1 (explicit fields incl. caller-supplied
    ``created_at``; never wall-clock), ``model=None`` because a QC snapshot is
    not an AI result. ``qc_report`` must be a real :class:`QCReport` instance
    — passing ``None`` is rejected fail-closed because a QC-only snapshot
    without a report has no purpose (drop the helper call instead).

    Deterministic: identical inputs and fixed timestamps produce an identical
    content-addressed ``revision_id``. The parent object is never mutated.
    """
    if not isinstance(qc_report, QCReport):
        raise CMRError(
            INVALID_ARGUMENT,
            f"qc_report must be a QCReport instance, got {type(qc_report).__name__}",
        )
    merged: dict[Structure, Contour] = (
        dict(parent.contours) if contours is None else dict(contours)
    )
    provenance = Provenance(
        study_uid=parent.provenance.study_uid,
        series_uid=series_uid,
        software_version=software_version,
        model=None,
        created_at=created_at,
        revision=parent.provenance.revision + 1,
        workflow_id=parent.provenance.workflow_id,
    )
    return ContourRevision.from_parent(
        parent,
        series_uid=series_uid,
        contours=merged,
        editor=editor,
        created_at=created_at,
        provenance=provenance,
        qc_report=qc_report,
    )
