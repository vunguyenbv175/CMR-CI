"""Node 5 deterministic measurement adapter: contour revisions -> Measurement.

Bridges the frozen per-structure ``ContourRevision`` records into cine-wide
deterministic metrics WITHOUT touching any frozen domain type. The clinical
volume method is per-slice disc/slab summation over ephemeral masks:
``nonzero_pixels * pixel_area_mm2 * effective_gap_mm / 1000`` via the
unchanged primitives in :mod:`cmr_quantification.volume`.

Pipeline (all deterministic, no AI, no GPU, no cloud):

1. :func:`build_cine_contour_index` walks an ordered revision chain (the
   shape returned by ``SqliteCmrStore.load_chain``: ascending ``version``)
   and assembles the cine-wide ``(slice_index, frame_index, Structure)``
   index, taking each key from the contour fields themselves. A single
   ``ContourRevision`` holds at most one contour per structure, so a full
   stack is spread across SEVERAL chain revisions; a key repeated across
   revisions resolves to the HIGHEST-version record (later physician
   edits win). The resulting :class:`CineContourIndex` behaves exactly
   like ``dict[tuple[int, int, Structure], Contour]`` and additionally
   records, per key, which ``revision_id`` supplied it, plus the chain
   head identity used for measurement binding. Ambiguous duplicates at
   the same precedence level -- the same revision record supplied twice,
   or two chain entries sharing one version -- fail closed instead of
   guessing.
2. :func:`compute_measurement` validates required coverage against the
   full cine grid: ``LV_CAVITY`` and ``RV_CAVITY`` for every slice x
   frame used by the curve, plus ``LV_MYOCARDIUM`` at the ED mass frame.
   Missing or out-of-range keys are fatal; contours are never zero-filled.
3. Each contour is rasterized ephemerally with
   ``cmr_contours.conversion.contour_to_mask(contour, geometry)`` using
   exactly the geometry objects from ``cine.geometries[slice_index]``.
   Rasterization is transient: no segmentation provider, no AI result,
   no inferred mask and no persisted mask ever participates.
4. ``volume_curve_stack_ml`` returns ``[frame][slice]``; the dimensions
   are validated and the matrix is explicitly transposed to
   ``[slice][frame]`` BEFORE ``cavity_volume_curve_ml`` sums it
   (never rely on ``n_frames == n_slices`` coincidence).
5. ED/ES frames come ONLY from the manual :class:`EDESSelection`, which
   is authoritative: detection helpers are neither imported nor called
   here, and a manual selection is persisted verbatim into the
   Measurement.
6. Gap policy: multi-slice stacks use ONLY IPP-derived gaps
   (:func:`cmr_geometry.stack.stack_gaps_mm`); ``SpacingBetweenSlices``
   and ``SliceThickness`` never enter multi-slice gap math. A
   single-slice stack falls back to ``ImageGeometry.slice_gap_mm``
   (SpacingBetweenSlices, then SliceThickness), which must be positive
   and finite, and the fallback is recorded in the algorithm version.
   The effective gaps are stored in ``Measurement.stack_gaps_mm``.
7. Myocardium convention: ``LV_MYOCARDIUM`` traces the EPICARDIAL (outer)
   boundary of the left ventricle. The myocardium mask is
   ``epicardial_mask AND NOT endocardial_mask`` (the LV cavity mask).
   LV mass is computed at the ED frame only, via ``myocardial_mass`` on
   the disc/slab myocardium volume.

   Papillary/trabeculation disclosure (S2C-001 cluster F): whether
   papillary muscles are included in the mass depends on contour
   TOPOLOGY, which the single-outer-loop representation cannot express —
   internal holes are dropped by ``mask_to_contour`` before this code
   runs, so outer-minus-inner implements the include-in-mass target ONLY
   when that topology is representable. See
   :data:`PAPILLARY_TRABECULATION_CONVENTION_V1`; no geometry behavior
   changes in S2C-001 and the convention is NOT clinically frozen.
8. Failure policy: malformed, duplicate, non-finite, out-of-range or
   missing inputs raise :class:`CMRError` -- no guessed Measurement.
   ``ESV > EDV`` propagates the existing fail-closed
   ``quantification.invalid_input`` policy from ``function.py`` untouched.
   A flat manual curve (equal volumes at the selected frames, including
   ``ed_frame == es_frame``) with positive EDV is LEGAL and yields
   SV = 0 ml / EF = 0 percent plus ONE QC warning (``qc.volume_curve``)
   appended to the caller-supplied :class:`IssueSet` -- the Measurement
   has no QC container of its own (frozen Node 1 record), so the IssueSet
   IS its qc path for Node 5.
9. Provenance: ``software_version`` is required and non-empty;
   ``algorithm_version`` is canonical-only: it is built as
   ``disc-cylinder-v1/gaps=ipp`` or
   ``disc-cylinder-v1/gaps=single-slice-metadata-fallback`` according to
   the detected gap source, and a caller-supplied value is accepted ONLY
   when it equals the canonical string for that gap mode -- wrong-mode or
   arbitrary values fail closed (review correction F1a);
   ``created_at`` is the explicit caller-supplied ISO stamp;
   ``measurement_id`` remains the content-addressed digest of the
   canonical payload, bound to the index head revision id, the selected
   frames, the effective gaps and both version strings. Quantities keep
   raw unrounded floats; nothing in this pipeline rounds.
10. Workflow coherence (S2C-001 cluster A/B):
   :func:`build_cine_contour_index`, :func:`run_revision_qc` (via the
   shared helper) and :func:`compute_measurement` verify ONE exact
   manifest-bound workflow identity across the revision chain, the cine
   volume and the selection; mixing bound and unbound participants, or two
   different workflows, fails closed. When a coherent manifest-bound
   workflow exists, the produced Measurement carries ``workflow_id``
   (provenance), the persisted ``selection.selection_id`` binding and the
   study/series identity from the selection/cine instead of a hardcoded
   ``None`` study reference.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from cmr_contours.conversion import contour_to_mask
from cmr_contours.model import Contour
from cmr_core.cine import CineVolume
from cmr_core.domain_types import (
    ContourRevision,
    EDESSelection,
    Measurement,
    head_tombstoned_keys,
)
from cmr_core.errors import (
    CONTOUR_INVALID,
    QC_MISSING_STRUCTURE,
    QC_VOLUME_CURVE,
    QUANTIFICATION_INVALID_INPUT,
    CMRError,
    CMRWarning,
    IssueSet,
)
from cmr_core.geometry import ImageGeometry
from cmr_core.provenance import Provenance
from cmr_core.segmentation import Structure
from cmr_geometry.stack import stack_gaps_mm as ipp_stack_gaps_mm

from cmr_quantification.function import ejection_fraction, myocardial_mass, stroke_volume
from cmr_quantification.units import Quantity, Unit
from cmr_quantification.volume import (
    cavity_volume_curve_ml,
    mask_area_mm2,
    volume_curve_stack_ml,
)

ALGORITHM_VERSION_DISC_CYLINDER = "disc-cylinder-v1"
GAP_SOURCE_IPP = "gaps=ipp"
GAP_SOURCE_SINGLE_SLICE_FALLBACK = "gaps=single-slice-metadata-fallback"
ALGORITHM_VERSION_IPP = f"{ALGORITHM_VERSION_DISC_CYLINDER}/{GAP_SOURCE_IPP}"
ALGORITHM_VERSION_SINGLE_SLICE = (
    f"{ALGORITHM_VERSION_DISC_CYLINDER}/{GAP_SOURCE_SINGLE_SLICE_FALLBACK}"
)

#: CFM-001 explicit-inclusion algorithm identity family (spec §1.5). The
#: null-set legacy path keeps the ``disc-cylinder-v1/...`` strings above
#: byte-for-byte; an explicit set always seals ``disc-cylinder-v2/...``.
ALGORITHM_VERSION_DISC_CYLINDER_V2 = "disc-cylinder-v2"
INCLUSION_CONVENTION = "inclusion=whole-slice-v1"
ALGORITHM_VERSION_IPP_V2 = (
    f"{ALGORITHM_VERSION_DISC_CYLINDER_V2}/{GAP_SOURCE_IPP}/{INCLUSION_CONVENTION}"
)
ALGORITHM_VERSION_SINGLE_SLICE_V2 = (
    f"{ALGORITHM_VERSION_DISC_CYLINDER_V2}/"
    f"{GAP_SOURCE_SINGLE_SLICE_FALLBACK}/{INCLUSION_CONVENTION}"
)

#: FNC-B selected-phase coverage token (additive, P1 clinical). A
#: selected-phase measurement consumes exactly the five required geometry
#: groups (ED LV cavity / LV myocardium / RV cavity + ES LV cavity /
#: RV cavity) for every included SAX slice; intermediate phases are never
#: required and never consumed. Results sealed under this coverage token
#: must never be conflated with whole-cine (``disc-cylinder-v1/...``) or
#: whole-slice-inclusion (``.../inclusion=whole-slice-v1``) results.
SELECTED_PHASE_COVERAGE = "coverage=selected-phases-v1"
ALGORITHM_VERSION_IPP_SELECTED_PHASES = (
    f"{ALGORITHM_VERSION_DISC_CYLINDER_V2}/{GAP_SOURCE_IPP}/{SELECTED_PHASE_COVERAGE}"
)
ALGORITHM_VERSION_SINGLE_SLICE_SELECTED_PHASES = (
    f"{ALGORITHM_VERSION_DISC_CYLINDER_V2}/"
    f"{GAP_SOURCE_SINGLE_SLICE_FALLBACK}/{SELECTED_PHASE_COVERAGE}"
)

# --- CFM-001 basal-slice inclusion vocabulary ---------------------------------
#
# Dependency-neutral STRING vocabularies mirroring the persistence record
# enums WITHOUT importing cmr_persistence (quantification never imports
# persistence; the controller converts records into plain tuples first).

#: Correction round 2 (finding 6): dedicated structured reason prefix for the
#: post-inclusion ESV>EDV rejection. The viewer maps ONLY errors whose
#: message starts with this token to the frozen
#'ANALYSIS UNAVAILABLE' literal; other qc.volume_curve failures (e.g.
#: unusable single-slice gaps) keep their own messages.
POST_INCLUSION_ESV_EXCEEDS_EDV_REASON = "post-inclusion-esv-exceeds-edv"


class PostInclusionEsvExceedsEdvError(CMRError):
    """Correction round 3 (R7): TYPED reason for post-inclusion ESV>EDV.

    Still a :class:`CMRError` carrying the ``qc.volume_curve`` clinical
    finding class, but dispatchable by TYPE instead of message-prefix
    parsing. The structured reason rides on the instance.
    """

    reason = POST_INCLUSION_ESV_EXCEEDS_EDV_REASON

    def __init__(
        self,
        message: str,
        source: str | None = None,
        *,
        inclusion_set_id: str | None = None,
    ) -> None:
        super().__init__(QC_VOLUME_CURVE, message, source=source)
        self.inclusion_set_id = inclusion_set_id


class IncompleteVolumeCurveError(CMRError, KeyError):
    """Typed incomplete-data signal for :func:`compute_cavity_volume_curve_ml`.

    CMR_QUANT_TOMBSTONE_001 (Gap 4 verify-first): a tombstoned (or otherwise
    absent) ``(slice, frame, structure)`` key used to surface as a bare
    ``KeyError`` from the ``index[...]`` lookup inside the rasterizer — an
    unhandled crash type instead of the file's explicit-incomplete
    convention (``CMRError`` carrying ``qc.missing_structure`` with the
    "missing keys are never zero-filled" vocabulary). This error carries
    exactly that convention AND subclasses ``KeyError`` so the one
    pre-existing ``except KeyError`` consumer (the incomplete-fixture curve
    helper in the topology-disposition suite) keeps resolving absence to
    ``None`` byte-for-byte. It never carries a number: no silent metrics,
    no NaN, no zero-fill.
    """

INCLUSION_ROLE_ED = "ED"
INCLUSION_ROLE_ES = "ES"
INCLUSION_STRUCTURE_LV_CAVITY = "lv_cavity"
INCLUSION_STRUCTURE_RV_CAVITY = "rv_cavity"
INCLUSION_STRUCTURE_LV_MYOCARDIUM = "lv_myocardium"

#: Canonical entry order is (slice_index, role_order, structure_order) with
#: ED < ES and lv_cavity < rv_cavity < lv_myocardium (spec §1.2).
_INCLUSION_ROLE_ORDER = {INCLUSION_ROLE_ED: 0, INCLUSION_ROLE_ES: 1}
_INCLUSION_STRUCTURE_ORDER = {
    INCLUSION_STRUCTURE_LV_CAVITY: 0,
    INCLUSION_STRUCTURE_RV_CAVITY: 1,
    INCLUSION_STRUCTURE_LV_MYOCARDIUM: 2,
}

#: The five MANDATORY (role, structure) groups of a complete v1 grid. The
#: lv_myocardium decision applies ONLY to the ED role; an ES myocardium
#: entry is not applicable and fails closed (spec §1.2).
_MANDATORY_GROUPS: tuple[tuple[str, str], ...] = (
    # Canonical order per spec 1.2: role_order PRIMARY (ED < ES), then
    # structure_order (lv_cavity < rv_cavity < lv_myocardium).
    (INCLUSION_ROLE_ED, INCLUSION_STRUCTURE_LV_CAVITY),
    (INCLUSION_ROLE_ED, INCLUSION_STRUCTURE_RV_CAVITY),
    (INCLUSION_ROLE_ED, INCLUSION_STRUCTURE_LV_MYOCARDIUM),
    (INCLUSION_ROLE_ES, INCLUSION_STRUCTURE_LV_CAVITY),
    (INCLUSION_ROLE_ES, INCLUSION_STRUCTURE_RV_CAVITY),
)

#: Public canonical group order (ED < ES; lv_cavity < rv_cavity <
#: lv_myocardium) reused by controllers/viewers/tests when authoring grids.
INCLUSION_GRID_GROUP_ORDER = _MANDATORY_GROUPS

#: S2C-001 cluster F disclosure (metadata only, NOT part of any algorithm
#: version string): the current single-outer-loop contour representation
#: targets papillary/trabeculation inclusion in myocardial mass but cannot
#: represent internal holes, so the target is implemented ONLY when topology
#: is representable after mask conversion. Clinical status: NOT frozen
#: (CMR-01 owns hole-aware representation before any clinical freeze).
PAPILLARY_TRABECULATION_CONVENTION_V1 = (
    "target=include-in-myocardial-mass;"
    "representation=single-outer-loop;"
    "internal-holes=not-representable;"
    "clinical-status=not-frozen"
)

# Out-of-field tolerance for contour pixel coordinates (float noise only).
# ``contour_to_mask`` clips crossings to the image grid silently, so Node 5
# rejects any point outside the pixel field BEFORE rasterization
# (review correction F0).
PIXEL_BOUNDS_TOLERANCE_PX = 1e-6

# --- CMR_FUNCV1_STACK_APPLICABILITY_001 phase/structure applicability -------
#
# F19: the binary whole-slab inclusion grid doubles as the MINIMAL
# physician-authoritative phase/structure applicability record. An
# ``included=False`` decision on exactly one ``(slice, role, structure)``
# key is the applicability authority for that key: the structure is
# explicitly non-contributing at that selected phase (e.g. an apical slice
# whose cavity carries no blood pool while its myocardium still
# contributes mass, per SCMR 2020 LV-mass §iii). The authority is:
#
# * phase-specific — ``(s, ED, q)`` never waives ``(s, ES, q)``;
# * structure-specific — ``(s, r, lv_cavity)`` never waives ``rv_cavity``
#   or ``lv_myocardium``;
# * record-bound — it is read from the persisted ``SliceInclusionSet``
#   (via :class:`SliceInclusionPolicy`) with head/selection/workflow
#   coherence verified by :func:`_verify_inclusion_policy_coherence`;
#   it is NEVER inferred from AI EMPTY tuples, pixels, or geometry
#   absence.
#
# A key with neither geometry in the contour index NOR an explicit
# ``included=False`` waiver is UNKNOWN and fails closed
# (``qc.missing_structure``) in both measurement and QC. Waived keys
# contribute exactly zero cavity volume with NO fabricated contour; the
# waived endocardial mask used for cavity subtraction is the empty mask,
# so an epicardial-only apical slab keeps its full myocardial volume.


def applicability_waivers(
    policy: SliceInclusionPolicy | None,
) -> frozenset[tuple[int, str, str]]:
    """The exact ``(slice, role, structure)`` keys a policy waives.

    Returns the frozen set of keys whose binary weight is ``0.0``
    (physician-authoritative non-applicable decisions). ``None`` yields
    the empty set (no applicability authority). Keys are returned in the
    policy's own ``(slice_index, role, structure)`` string vocabulary.
    """
    if policy is None:
        return frozenset()
    return frozenset(
        key for key in policy._weights if policy._weights[key] == 0.0
    )


ContourKey = tuple[int, int, Structure]

__all__ = [
    "PostInclusionEsvExceedsEdvError",
    "IncompleteVolumeCurveError",
    "ALGORITHM_VERSION_DISC_CYLINDER",
    "ALGORITHM_VERSION_DISC_CYLINDER_V2",
    "ALGORITHM_VERSION_IPP",
    "ALGORITHM_VERSION_IPP_V2",
    "ALGORITHM_VERSION_IPP_SELECTED_PHASES",
    "ALGORITHM_VERSION_SINGLE_SLICE",
    "ALGORITHM_VERSION_SINGLE_SLICE_V2",
    "ALGORITHM_VERSION_SINGLE_SLICE_SELECTED_PHASES",
    "SELECTED_PHASE_COVERAGE",
    "CineContourIndex",
    "INCLUSION_CONVENTION",
    "INCLUSION_GRID_GROUP_ORDER",
    "POST_INCLUSION_ESV_EXCEEDS_EDV_REASON",
    "INCLUSION_ROLE_ED",
    "INCLUSION_ROLE_ES",
    "INCLUSION_STRUCTURE_LV_CAVITY",
    "INCLUSION_STRUCTURE_LV_MYOCARDIUM",
    "INCLUSION_STRUCTURE_RV_CAVITY",
    "PIXEL_BOUNDS_TOLERANCE_PX",
    "PAPILLARY_TRABECULATION_CONVENTION_V1",
    "SliceInclusionPolicy",
    "applicability_waivers",
    "assert_coherent_workflow_identity",
    "build_cine_contour_index",
    "build_slice_inclusion_policy",
    "compute_cavity_volume_curve_ml",
    "compute_measurement",
    "compute_measurement_from_contours",
    "compute_selected_phase_measurement",
    "effective_stack_gaps_mm",
    "validate_complete_grid_entries",
    "validated_transpose_frame_slice",
]


# --- CFM-001 dependency-neutral inclusion-policy object (spec §1.7) -----------


def validate_complete_grid_entries(rows: Any) -> int:
    """Validate a COMPLETE basal-inclusion entry grid and return ``N``.

    ``rows`` is an iterable of plain mappings carrying ``slice_index``,
    ``frame_role`` ("ED"|"ES"), ``structure``
    ("lv_cavity"|"rv_cavity"|"lv_myocardium") and ``included``. Extra keys
    (e.g. the optional persisted ``frame_index``) are ignored.

    Fail-closed rules (CFM-001 spec §1.2):

    - slices must cover EXACTLY ``0..N-1`` for each of the five mandatory
      ``(role, structure)`` groups; missing, extra, negative or duplicate
      keys fail;
    - ``lv_myocardium`` exists ONLY at the ED role; an ES myocardium entry
      is not applicable and fails;
    - ``included`` must be an ACTUAL Boolean: ``0``, ``1``, floats, null
      and fractional weights fail;
    - at least one slice stays included in EACH of the five groups.

    Returns the covered slice count ``N`` (a complete set carries exactly
    ``5 x N`` entries). Dependency-neutral: plain data in, plain int out.
    """
    if rows is None:
        raise ValueError("inclusion entries are required")
    seen: dict[tuple[int, str, str], bool] = {}
    for row in rows:
        try:
            slice_index = row["slice_index"]
            role = row["frame_role"]
            structure = row["structure"]
            included = row["included"]
        except (TypeError, KeyError) as exc:
            raise ValueError(
                f"inclusion entry {row!r} is missing required keys"
            ) from exc
        if isinstance(slice_index, bool) or not isinstance(slice_index, int):
            raise ValueError(f"slice_index must be an integer, got {slice_index!r}")
        if slice_index < 0:
            raise ValueError(
                f"slice_index must be non-negative, got {slice_index}"
            )
        if role not in _INCLUSION_ROLE_ORDER:
            raise ValueError(f"frame_role must be 'ED' or 'ES', got {role!r}")
        if structure not in _INCLUSION_STRUCTURE_ORDER:
            raise ValueError(
                "structure must be one of 'lv_cavity'|'rv_cavity'|"
                f"'lv_myocardium', got {structure!r}"
            )
        group = (role, structure)
        if group not in _MANDATORY_GROUPS:
            # Currently only lv_myocardium@ES can land here.
            raise ValueError(
                f"inclusion entry ({slice_index}, {role}, {structure}) is not "
                "applicable: the LV myocardium decision applies to the ED "
                "role only"
            )
        if not isinstance(included, bool):
            raise ValueError(
                f"included must be an actual Boolean, got {included!r}; "
                "integers, floats, null and fractional weights fail closed"
            )
        key = (slice_index, role, structure)
        if key in seen:
            raise ValueError(
                f"duplicate inclusion key {key}; exactly one decision per "
                "(slice, role, structure) is allowed"
            )
        seen[key] = included
    if not seen:
        raise ValueError("an inclusion set requires at least one entry")
    n_slices = max(key[0] for key in seen) + 1
    expected = 5 * n_slices
    if len(seen) != expected:
        missing: list[tuple[int, str, str]] = []
        for slice_index in range(n_slices):
            for role, structure in _MANDATORY_GROUPS:
                if (slice_index, role, structure) not in seen:
                    missing.append((slice_index, role, structure))
        preview = ", ".join(str(item) for item in missing[:6])
        raise ValueError(
            f"incomplete inclusion grid: {len(seen)} entries for {n_slices} "
            f"slices (expected exactly {expected}); first missing keys: "
            f"{preview}"
        )
    for role, structure in _MANDATORY_GROUPS:
        group_values = [
            included
            for (slice_index, g_role, g_structure), included in seen.items()
            if g_role == role and g_structure == structure
        ]
        if not any(group_values):
            raise ValueError(
                f"inclusion group (role={role}, structure={structure}) has no "
                "included slice; at least one slice must remain included in "
                "each of the five mandatory groups"
            )
    return n_slices


def _validated_weight_grid(
    weights: Any, n_slices: int
) -> MappingProxyType[tuple[int, str, str], float]:
    """Fail-closed validation of a COMPLETE binary weight grid (F1).

    Accepts ONLY exact ``1.0``/``0.0`` floats over exactly the five mandatory
    ``(role, structure)`` groups for every slice ``0..n_slices-1``, with at
    least one included slice per group. Returns a frozen MappingProxyType.
    """
    if not isinstance(weights, Mapping):
        raise ValueError("policy weights must be a mapping")
    validated: dict[tuple[int, str, str], float] = {}
    for key, value in weights.items():
        try:
            slice_index, role, structure = key
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid weight key {key!r}") from exc
        if (
            isinstance(slice_index, bool)
            or not isinstance(slice_index, int)
            or not 0 <= slice_index < n_slices
        ):
            raise ValueError(
                f"weight slice index {slice_index!r} outside 0..{n_slices - 1}"
            )
        if (role, structure) not in _MANDATORY_GROUPS:
            raise ValueError(
                f"weight group ({role!r}, {structure!r}) is not one of the "
                "five mandatory groups"
            )
        if isinstance(value, bool) or not isinstance(value, float) or value not in (
            1.0,
            0.0,
        ):
            raise ValueError(
                f"weight values must be exactly 1.0 or 0.0, got {value!r}; "
                "fractional half-slice weights are prohibited"
            )
        if key in validated:
            raise ValueError(f"duplicate weight key {key!r}")
        validated[key] = value
    expected = 5 * n_slices
    if len(validated) != expected:
        raise ValueError(
            f"policy carries {len(validated)} weights for {n_slices} slices "
            f"(expected exactly {expected})"
        )
    for role, structure in _MANDATORY_GROUPS:
        if not any(
            value == 1.0
            for (s, g_role, g_struct), value in validated.items()
            if g_role == role and g_struct == structure
        ):
            raise ValueError(
                f"inclusion group (role={role}, structure={structure}) has no "
                "included slice"
            )
    return MappingProxyType(validated)


@dataclass(frozen=True)
class SliceInclusionPolicy:
    """Validated, dependency-neutral binary whole-slab inclusion policy.

    CFM-001 spec §1.7: quantification must never import persistence, so the
    workflow/controller converts a persisted inclusion-set record into THIS
    object only after graph and completeness validation. The policy carries
    its full identity context (series / source revision / selection /
    workflow / resolved frames) so :func:`compute_measurement` can verify
    coherence fail-closed before any arithmetic runs.

    Weights are strictly binary (``1.0`` included, ``0.0`` excluded): a
    slice contributes its entire existing slab or zero. Fractional weights,
    interpolated valve planes and tapering are unrepresentable by design.
    """

    series_uid: str
    source_revision_id: str
    selection_id: str
    workflow_id: str | None
    ed_frame: int
    es_frame: int
    n_slices: int
    inclusion_set_id: str
    _weights: Mapping[tuple[int, str, str], float]

    def __post_init__(self) -> None:
        """F1 (P0): direct construction is as trustworthy as the builder.

        Re-validates the complete grid FROM THE WEIGHTS THEMSELVES and
        re-freezes the mapping, so a hand-built policy can never carry
        fractional weights, missing keys, extra keys, or a mutable mapping
        into :func:`compute_measurement`.
        """
        object.__setattr__(
            self, "_weights", _validated_weight_grid(self._weights, self.n_slices)
        )

    def weight(self, slice_index: int, role: str, structure: str) -> float:
        """Binary slab weight ``w(s, r, q)`` — 1.0 included / 0.0 excluded."""
        return self._weights[(slice_index, role, structure)]

    @property
    def algorithm_suffix_source(self) -> tuple[str, ...]:
        """Gap-source token pair used to build the v2 algorithm version."""
        return (GAP_SOURCE_IPP, GAP_SOURCE_SINGLE_SLICE_FALLBACK)


def build_slice_inclusion_policy(
    entries: Any,
    *,
    series_uid: str,
    source_revision_id: str,
    selection_id: str,
    inclusion_set_id: str,
    ed_frame: int,
    es_frame: int,
    n_slices: int,
    workflow_id: str | None = None,
) -> SliceInclusionPolicy:
    """Build a validated :class:`SliceInclusionPolicy` from plain entries.

    ``entries`` follows the same mapping shape as
    :func:`validate_complete_grid_entries`. The grid validator runs FIRST
    (fail-closed), then the covered slice count must equal ``n_slices`` — a
    policy for fewer/more slices than the cine would silently misweight the
    stack otherwise. Identity fields must be non-empty strings.
    """
    covered = validate_complete_grid_entries(entries)
    if covered != n_slices:
        raise ValueError(
            f"inclusion grid covers {covered} slices but the cine stack has "
            f"{n_slices}; the policy must cover every original cine slice "
            "exactly"
        )
    for name, value in (
        ("series_uid", series_uid),
        ("source_revision_id", source_revision_id),
        ("selection_id", selection_id),
        ("inclusion_set_id", inclusion_set_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    for frame_name, frame_value in (
        ("ed_frame", ed_frame),
        ("es_frame", es_frame),
    ):
        if (
            isinstance(frame_value, bool)
            or not isinstance(frame_value, int)
            or frame_value < 0
        ):
            raise ValueError(
                f"{frame_name} must be a non-negative integer, got {frame_value!r}"
            )
    if isinstance(n_slices, bool) or not isinstance(n_slices, int) or n_slices < 1:
        raise ValueError(f"n_slices must be a positive integer, got {n_slices!r}")
    if workflow_id is not None and (
        not isinstance(workflow_id, str) or not workflow_id.strip()
    ):
        raise ValueError("workflow_id must be a non-empty string or None")
    weights: dict[tuple[int, str, str], float] = {}
    for row in entries:
        key = (row["slice_index"], row["frame_role"], row["structure"])
        weights[key] = 1.0 if row["included"] else 0.0
    return SliceInclusionPolicy(
        series_uid=series_uid,
        source_revision_id=source_revision_id,
        selection_id=selection_id,
        workflow_id=workflow_id,
        ed_frame=ed_frame,
        es_frame=es_frame,
        n_slices=n_slices,
        inclusion_set_id=inclusion_set_id,
        _weights=MappingProxyType(weights),
    )


def _validate_contour_points(contour: Contour) -> None:
    """Fail closed on malformed contour geometry (malformed/non-finite)."""
    origin = f"({contour.slice_index},{contour.frame_index},{contour.structure.value})"
    if len(contour.points) < 3:
        raise CMRError(
            CONTOUR_INVALID,
            f"contour {origin} needs >= 3 points, got {len(contour.points)}",
            source=origin,
        )
    for index, point in enumerate(contour.points):
        x, y = point
        if not (math.isfinite(x) and math.isfinite(y)):
            raise CMRError(
                CONTOUR_INVALID,
                f"contour {origin} point #{index} is non-finite: {(x, y)}",
                source=origin,
            )


def _validate_contour_within_geometry(contour: Contour, geometry: ImageGeometry) -> None:
    """Fail closed BEFORE rasterization on out-of-field contour points.

    ``contour_to_mask`` clips scanline crossings to the image grid
    silently, so a partially-outside polygon would produce a positive but
    clinically WRONG slab volume instead of an error (review correction
    F0). Every point must lie within ``[0-tol, cols+tol] x [0-tol,
    rows+tol]`` pixel coordinates of its own slice geometry.

    Upper bound is ``cols``/``rows`` INCLUSIVE (review correction round 2,
    F0 off-by-one): the canonical grid convention of
    :mod:`cmr_contours.conversion` places boundary edges at ``x + 1``, so
    a contour touching the far edge of a full ``rows x cols`` field
    legitimately reaches exactly ``cols``/``rows``. Anything beyond that
    still fails closed.
    """
    origin = f"({contour.slice_index},{contour.frame_index},{contour.structure.value})"
    max_row = float(geometry.rows) + PIXEL_BOUNDS_TOLERANCE_PX
    max_col = float(geometry.cols) + PIXEL_BOUNDS_TOLERANCE_PX
    min_coord = -PIXEL_BOUNDS_TOLERANCE_PX
    for index, point in enumerate(contour.points):
        x, y = point
        if not (min_coord <= x <= max_col and min_coord <= y <= max_row):
            raise CMRError(
                CONTOUR_INVALID,
                f"contour {origin} point #{index} ({x}, {y}) lies outside the "
                f"{geometry.rows}x{geometry.cols} pixel field; out-of-field "
                "contours would be silently clipped by rasterization",
                source=origin,
            )


class CineContourIndex(Mapping[ContourKey, Contour]):
    """Cine-wide ``(slice, frame, structure) -> Contour`` collection.

    Behaves exactly like the plain index ``dict`` of the amended contract
    AND records provenance: which revision supplied each entry, the series
    it was composed for, the chain-head revision identity that
    measurements bind to, and (S2C-001) the single manifest-bound workflow
    identity shared by every chain revision when one exists. Immutable
    after construction.
    """

    def __init__(
        self,
        contours: Mapping[ContourKey, Contour],
        source_revision_by_key: Mapping[ContourKey, str],
        *,
        series_uid: str,
        head_revision_id: str,
        head_version: int,
        head_provenance_revision: int,
        workflow_id: str | None = None,
        lineage_revision_ids: frozenset[str] | None = None,
        contour_preserving_ancestor_ids: frozenset[str] | None = None,
    ) -> None:
        self._contours: Mapping[ContourKey, Contour] = MappingProxyType(dict(contours))
        self._sources: Mapping[ContourKey, str] = MappingProxyType(
            dict(source_revision_by_key)
        )
        self._series_uid = series_uid
        self._head_revision_id = head_revision_id
        self._head_version = head_version
        self._head_provenance_revision = head_provenance_revision
        self._workflow_id = workflow_id
        # CFM-001: every revision id of the walked chain.
        self._lineage_ids: frozenset[str] = (
            lineage_revision_ids
            if lineage_revision_ids is not None
            else frozenset({head_revision_id})
        )
        # CFM-001 correction F4: the maximal head-anchored prefix of revisions
        # EACH of which carries contour content IDENTICAL to its immediate
        # child (the head itself included). Only this prefix — general
        # ancestry is NOT enough — may anchor an applicable inclusion set,
        # because a physician contour edit must reset inclusion instead of
        # carrying it forward. Defaults to the head alone.
        self._preserving_ancestor_ids: frozenset[str] = (
            contour_preserving_ancestor_ids
            if contour_preserving_ancestor_ids is not None
            else frozenset({head_revision_id})
        )

    def __getitem__(self, key: ContourKey) -> Contour:
        return self._contours[key]

    def __iter__(self) -> Iterator[ContourKey]:
        return iter(self._contours)

    def __len__(self) -> int:
        return len(self._contours)

    def __contains__(self, key: object) -> bool:
        return key in self._contours

    @property
    def contours(self) -> Mapping[ContourKey, Contour]:
        """The assembled ``(slice, frame, structure) -> Contour`` mapping."""
        return self._contours

    @property
    def source_revision_by_key(self) -> Mapping[ContourKey, str]:
        """Supplying ``revision_id`` per key (later edits win, recorded)."""
        return self._sources

    @property
    def series_uid(self) -> str:
        return self._series_uid

    @property
    def head_revision_id(self) -> str:
        """Highest-version revision of the walked chain (measurement binder)."""
        return self._head_revision_id

    @property
    def head_version(self) -> int:
        return self._head_version

    @property
    def head_provenance_revision(self) -> int:
        return self._head_provenance_revision

    @property
    def workflow_id(self) -> str | None:
        """Single manifest-bound workflow identity of the chain, else None."""
        return self._workflow_id

    @property
    def lineage_revision_ids(self) -> frozenset[str]:
        """Every revision id of the walked chain (head included)."""
        return self._lineage_ids

    @property
    def contour_preserving_ancestor_ids(self) -> frozenset[str]:
        """Head-anchored prefix of contour-identical consecutive revisions."""
        return self._preserving_ancestor_ids


def assert_coherent_workflow_identity(
    index: CineContourIndex, cine: CineVolume, selection: EDESSelection
) -> str | None:
    """Verify ONE exact workflow identity across chain, cine and selection.

    S2C-001 cluster A/B: when any participant is manifest-bound (carries a
    workflow id) ALL must be bound to the SAME id — mixing bound with
    unbound participants, or two different workflows, fails closed. Returns
    the bound id or ``None`` for a fully legacy-unbound lineage.
    """
    candidates = {
        "chain": index.workflow_id,
        "cine": cine.workflow_id,
        "selection": selection.provenance.workflow_id,
    }
    bound = {name: value for name, value in candidates.items() if value is not None}
    if not bound:
        return None
    if len(bound) < len(candidates):
        unbound = sorted(name for name in candidates if name not in bound)
        bound_value = next(iter(bound.values()))
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"mixed manifest-bound/unbound inputs ({', '.join(unbound)} unbound "
            f"while others carry workflow {bound_value!r}); bound and unbound "
            "records may never be mixed in one lineage",
        )
    if len(set(bound.values())) > 1:
        detail = ", ".join(f"{name}={value!r}" for name, value in sorted(bound.items()))
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"conflicting workflow identities across cine/selection/chain: "
            f"{detail}; one measurement requires exactly one workflow identity",
        )
    return next(iter(bound.values()))


def build_cine_contour_index(
    revisions: Sequence[ContourRevision], series_uid: str
) -> CineContourIndex:
    """Assemble the cine-wide contour index across the ordered revision chain.

    ``revisions`` must be ordered oldest-first (ascending ``version``, the
    ``load_chain()`` order); several small revisions together carry the
    full ``(slice, frame, structure)`` coverage even though each one holds
    at most one contour per structure. Keys come from the contour fields
    themselves; a key carried by several revisions resolves to the
    highest-version record (later edits win) and its supplying
    ``revision_id`` is recorded. The chain must be a single LINEAGE: the
    first entry declares no ``parent_revision_id`` and every consecutive
    pair must satisfy ``revisions[i + 1].parent_revision_id ==
    revisions[i].revision_id`` -- sibling forks or disjoint chains fail
    closed (review correction F1b). Series mismatches and ambiguous
    duplicate records fail closed as well.
    """
    if not revisions:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            "at least one ContourRevision is required to compose a measurement",
        )
    contours: dict[ContourKey, Contour] = {}
    sources: dict[ContourKey, str] = {}
    # Tombstone rule (CMR_CREATE_A_001, R2 C1): keys whose newest chain event
    # is a remove tombstone read ABSENT here. The per-revision overwrite loop
    # below would otherwise resurrect the removed contour from the ancestor
    # that still carries it.
    suppressed = head_tombstoned_keys(revisions)
    seen_revision_ids: set[str] = set()
    seen_revision_ids: set[str] = set()
    chain_workflow_ids: set[str | None] = set()
    previous: ContourRevision | None = None
    head = revisions[-1]
    for revision in revisions:
        if revision.series_uid != series_uid:
            raise CMRError(
                QUANTIFICATION_INVALID_INPUT,
                f"revision series_uid {revision.series_uid!r} does not match "
                f"selection series {series_uid!r}",
            )
        # S2C-001 cluster A/B: the walked chain must share one workflow
        # identity (all bound to the same id, or all unbound).
        chain_workflow_ids.add(revision.provenance.workflow_id)
        if revision.revision_id in seen_revision_ids:
            raise CMRError(
                QUANTIFICATION_INVALID_INPUT,
                f"duplicate revision record {revision.revision_id!r} in the input "
                "chain; ambiguous duplicates fail closed",
            )
        seen_revision_ids.add(revision.revision_id)
        if previous is not None and revision.version <= previous.version:
            raise CMRError(
                QUANTIFICATION_INVALID_INPUT,
                f"revision versions must strictly increase along the chain, got "
                f"{previous.version} then {revision.version}; conflicting duplicate "
                "keys cannot be ordered",
            )
        # Single-lineage requirement (review correction F1b): a forked or
        # disjoint chain would silently merge competing edit histories.
        if previous is None:
            if revision.parent_revision_id is not None:
                raise CMRError(
                    QUANTIFICATION_INVALID_INPUT,
                    f"forked or disjoint revision chain: the first revision "
                    f"{revision.revision_id!r} (version {revision.version}) must "
                    f"not declare parent_revision_id "
                    f"{revision.parent_revision_id!r}",
                )
        elif revision.parent_revision_id != previous.revision_id:
            raise CMRError(
                QUANTIFICATION_INVALID_INPUT,
                f"forked or disjoint revision chain: revision {revision.revision_id!r} "
                f"(version {revision.version}) declares parent_revision_id "
                f"{revision.parent_revision_id!r}, but the preceding chain entry is "
                f"{previous.revision_id!r} (version {previous.version})",
            )
        for structure_key, contour in revision.contours.items():
            if structure_key != contour.structure:
                raise CMRError(
                    CONTOUR_INVALID,
                    f"contours key {structure_key.value!r} does not match contour "
                    f"structure {contour.structure.value!r}",
                )
            _validate_contour_points(contour)
            key: ContourKey = (contour.slice_index, contour.frame_index, structure_key)
            if key in suppressed:
                # A remove tombstone is the newest event for this key: drop
                # any ancestor-carried entry and never re-add it.
                contours.pop(key, None)
                sources.pop(key, None)
                continue
            # Later edits win: the highest-version revision seen last overwrites.
            contours[key] = contour
            sources[key] = revision.revision_id
        previous = revision
        head = revision
    # F4: walk backwards while each consecutive pair shares IDENTICAL
    # contour content; those revisions (plus the head) form the only
    # ancestor scope an inclusion set may anchor to.
    preserving: list[ContourRevision] = [head]
    for earlier in reversed(revisions[:-1]):
        last = preserving[-1]
        if dict(last.contours) != dict(earlier.contours):
            break
        preserving.append(earlier)
    preserving_ids = frozenset(rev.revision_id for rev in preserving)
    bound_chain_ids = {value for value in chain_workflow_ids if value is not None}
    if len(bound_chain_ids) > 1:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"revision chain mixes workflow identities {sorted(bound_chain_ids)!r}; "
            "one lineage requires exactly one manifest-bound workflow identity",
        )
    if 0 < len(bound_chain_ids) < len(chain_workflow_ids):
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            "revision chain mixes manifest-bound and unbound revisions; bound "
            "and unbound records may never be mixed in one lineage",
        )
    chain_workflow_id = next(iter(bound_chain_ids)) if bound_chain_ids else None
    return CineContourIndex(
        contours,
        sources,
        series_uid=series_uid,
        head_revision_id=head.revision_id,
        head_version=head.version,
        head_provenance_revision=head.provenance.revision,
        workflow_id=chain_workflow_id,
        lineage_revision_ids=frozenset(seen_revision_ids),
        contour_preserving_ancestor_ids=preserving_ids,
    )


def effective_stack_gaps_mm(geometries: Sequence[ImageGeometry]) -> tuple[float, ...]:
    """Effective per-slice slab gaps (mm) in sorted-position order.

    Multi-slice stacks use IPP-derived gaps exclusively. A single-slice
    stack falls back to ``ImageGeometry.slice_gap_mm`` (SpacingBetweenSlices,
    then SliceThickness), which must be positive and finite, otherwise
    ``CMRError("qc.volume_curve")`` is raised.
    """
    geoms = list(geometries)
    if not geoms:
        raise CMRError(QC_VOLUME_CURVE, "no slices available for gap determination")
    if len(geoms) == 1:
        gap = geoms[0].slice_gap_mm
        if gap is None or not math.isfinite(gap) or gap <= 0.0:
            raise CMRError(
                QC_VOLUME_CURVE,
                "single-slice stack requires a positive finite SpacingBetweenSlices "
                "or SliceThickness metadata fallback; volume integration impossible",
            )
        return (float(gap),)
    return tuple(ipp_stack_gaps_mm(geoms))


def validated_transpose_frame_slice(
    per_frame_per_slice: list[list[float]], n_slices: int
) -> list[list[float]]:
    """Validate the ``[frame][slice]`` layout and transpose to ``[slice][frame]``.

    ``volume_curve_stack_ml`` returns one row per frame. Every row must
    carry exactly ``n_slices`` values; otherwise a plain ``ValueError`` is
    raised BEFORE any summation, so a coincidental ``n_frames ==
    n_slices`` can never silently sum along the wrong axis.
    """
    n_frames = len(per_frame_per_slice)
    for frame_index, row in enumerate(per_frame_per_slice):
        if len(row) != n_slices:
            raise ValueError(
                f"volume_curve_stack_ml returned {len(row)} slice values at frame "
                f"{frame_index}, expected {n_slices}; refusing to guess the axis order"
            )
    return [
        [per_frame_per_slice[frame][slice_index] for frame in range(n_frames)]
        for slice_index in range(n_slices)
    ]


def compute_cavity_volume_curve_ml(
    index: CineContourIndex, cine: CineVolume, structure: Structure = Structure.LV_CAVITY
) -> list[float]:
    """Deterministic LV/RV cavity volume curve (ml) indexed by frame.

    S2C-001 cluster E additive helper: the exact rasterize->transpose->sum
    pipeline :func:`compute_measurement` uses internally, exposed so the
    controller's ``refresh_head_analysis`` can derive the physician-facing
    curve for QC without duplicating geometry math. Fails closed on
    non-finite values exactly like the internal path. A tombstoned (or
    otherwise absent) ``(slice, frame, structure)`` key raises
    :class:`IncompleteVolumeCurveError` — the file's explicit-incomplete
    ``qc.missing_structure`` convention ("missing keys are never
    zero-filled"), still catchable as ``KeyError`` — instead of a bare
    ``KeyError`` from the index lookup.
    """
    n_slices = cine.n_slices
    n_frames = cine.n_frames
    geometries = list(cine.geometries)
    required = {(s, f, structure) for s in range(n_slices) for f in range(n_frames)}
    absent = sorted(
        required.difference(index), key=lambda k: (k[0], k[1], k[2].value)
    )
    if absent:
        preview = ", ".join(f"({s},{f},{t.value})" for s, f, t in absent[:6])
        extra = len(absent) - 6
        suffix = f" (+{extra} more)" if extra > 0 else ""
        raise IncompleteVolumeCurveError(
            QC_MISSING_STRUCTURE,
            f"missing required {structure.value} contours: {preview}{suffix}; "
            "missing keys are never zero-filled",
            source=index.series_uid,
        )

    def _rasterize() -> list[list[np.ndarray]]:
        return [
            [
                contour_to_mask(index[(s, f, structure)], geometries[s])
                for f in range(n_frames)
            ]
            for s in range(n_slices)
        ]

    per_frame_per_slice = volume_curve_stack_ml(_rasterize(), geometries)
    per_slice_per_frame = validated_transpose_frame_slice(per_frame_per_slice, n_slices)
    curve = cavity_volume_curve_ml(per_slice_per_frame)
    for frame_index, value in enumerate(curve):
        if not math.isfinite(value):
            raise CMRError(
                QUANTIFICATION_INVALID_INPUT,
                f"non-finite {structure.value} volume derived at frame "
                f"{frame_index}",
            )
    return curve


def _verify_inclusion_policy_coherence(
    policy: SliceInclusionPolicy,
    index: CineContourIndex,
    selection: EDESSelection,
    cine: CineVolume,
    coherent_workflow_id: str | None,
) -> None:
    """Fail closed unless the policy identity matches the quantification inputs.

    CFM-001 spec §1.2: set, cine, selection, workflow, series and source
    revision identities must agree, and role resolution is EXACT
    (``ED -> selection.ed_frame``, ``ES -> selection.es_frame``).
    """
    mismatches: list[str] = []
    if policy.series_uid != index.series_uid:
        mismatches.append(
            f"policy series {policy.series_uid!r} != index series "
            f"{index.series_uid!r}"
        )
    # CFM-001 correction F4: the set must anchor to the head OR to an
    # ancestor reachable through CONTOUR-PRESERVING transitions only (QC
    # snapshot forks). General ancestry no longer suffices: a physician
    # contour edit resets inclusion instead of carrying it forward.
    if policy.source_revision_id != index.head_revision_id and (
        policy.source_revision_id not in index.contour_preserving_ancestor_ids
    ):
        mismatches.append(
            f"policy source revision {policy.source_revision_id!r} is neither "
            f"the chain head {index.head_revision_id!r} nor one of its "
            "contour-preserving ancestors"
        )
    if policy.selection_id != selection.selection_id:
        mismatches.append(
            f"policy selection {policy.selection_id!r} != selection "
            f"{selection.selection_id!r}"
        )
    if policy.ed_frame != selection.ed_frame or policy.es_frame != selection.es_frame:
        mismatches.append(
            f"policy frames (ED={policy.ed_frame}, ES={policy.es_frame}) != "
            f"selection frames (ED={selection.ed_frame}, ES={selection.es_frame})"
        )
    if policy.n_slices != cine.n_slices:
        mismatches.append(
            f"policy covers {policy.n_slices} slices != cine stack "
            f"{cine.n_slices}"
        )
    if policy.workflow_id != coherent_workflow_id:
        mismatches.append(
            f"policy workflow {policy.workflow_id!r} != lineage workflow "
            f"{coherent_workflow_id!r}"
        )
    if mismatches:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            "inclusion-policy identity disagrees with the quantification "
            f"inputs: {'; '.join(mismatches)}",
        )


def compute_measurement(
    index: CineContourIndex,
    selection: EDESSelection,
    cine: CineVolume,
    *,
    software_version: str,
    algorithm_version: str | None = None,
    created_at: str,
    issues: IssueSet | None = None,
    inclusion_policy: SliceInclusionPolicy | None = None,
    # CFM-001 correction F11: the LOW-LEVEL compatibility path defaults to
    # None so legacy-shaped arithmetic keeps its exact canonical bytes;
    # the new clinical workflow (apply-inclusion) stamps the table
    # version explicitly at the controller boundary.
    range_table_version: str | None = None,
) -> Measurement:
    """Compute the deterministic disc/slab Measurement for one cine series.

    ``index`` is the assembled :class:`CineContourIndex` (see
    :func:`build_cine_contour_index`). The manual ``selection`` frames are
    validated in-range against ``cine`` and persisted verbatim; the
    measurement binds to ``index.head_revision_id``. See the module
    docstring for the complete convention, gap and failure policy.

    CFM-001 additions:

    - ``inclusion_policy``: when supplied it MUST carry a complete binary
      grid (see :func:`build_slice_inclusion_policy`) whose identity matches
      the inputs; arithmetic becomes binary whole-slab (each included slice
      contributes its entire existing gap-derived slab, excluded slices
      contribute zero, gaps are NEVER recomputed) and the measurement seals
      a ``disc-cylinder-v2/...inclusion=whole-slice-v1`` algorithm identity
      plus the policy's ``inclusion_set_id``. When omitted, the virtual
      all-True compatibility path reproduces the legacy null-set arithmetic
      and ``disc-cylinder-v1/...`` bytes exactly.
    - ``range_table_version``: reference-table provenance stamped onto new
      clinical measurements (metadata only; never enters metric
      arithmetic or the algorithm version).
    """
    if not isinstance(software_version, str) or not software_version.strip():
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            "software_version is required and must be a non-empty string",
        )
    if index.series_uid != selection.series_uid:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"index series {index.series_uid!r} does not match selection series "
            f"{selection.series_uid!r}",
        )

    # --- S2C-001 cluster A/B: one exact workflow identity across the chain,
    # cine and selection; bound/unbound mixing fails closed. Study and
    # series identity are verified wherever present.
    coherent_workflow_id = assert_coherent_workflow_identity(index, cine, selection)
    study_candidates = {
        "selection": selection.provenance.study_uid,
        "cine": cine.study_uid,
        "index": None,
    }
    bound_studies = {value for value in study_candidates.values() if value is not None}
    if len(bound_studies) > 1:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"conflicting study identities across selection/cine: "
            f"{sorted(bound_studies)!r}; one measurement requires one study",
        )
    resolved_study_uid = next(iter(bound_studies)) if bound_studies else None

    # --- binding checks (fail closed BEFORE any calculation) -----------------
    n_slices = cine.n_slices
    n_frames = cine.n_frames
    if selection.n_frames is not None and selection.n_frames != n_frames:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"selection n_frames {selection.n_frames} disagrees with cine "
            f"n_frames {n_frames}",
        )
    if not (0 <= selection.ed_frame < n_frames) or not (0 <= selection.es_frame < n_frames):
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"ED/ES frames ({selection.ed_frame}, {selection.es_frame}) are outside "
            f"0..{n_frames - 1}",
        )

    # Fail closed on out-of-range keys AND out-of-field contour points
    # BEFORE any rasterization (review correction F0): conversion.py clips
    # crossings to the grid silently, so clipping must never be reachable.
    geometries = list(cine.geometries)
    for (slice_index, frame_index, structure), contour in index.items():
        if slice_index >= n_slices or frame_index >= n_frames:
            raise CMRError(
                CONTOUR_INVALID,
                f"contour key ({slice_index},{frame_index},{structure.value}) is "
                f"outside the cine grid {n_slices}x{n_frames}",
            )
        _validate_contour_within_geometry(contour, geometries[slice_index])

    required: set[ContourKey] = set()
    for slice_index in range(n_slices):
        for frame_index in range(n_frames):
            required.add((slice_index, frame_index, Structure.LV_CAVITY))
            required.add((slice_index, frame_index, Structure.RV_CAVITY))
        required.add((slice_index, selection.ed_frame, Structure.LV_MYOCARDIUM))
    missing = sorted(required.difference(index), key=lambda k: (k[0], k[1], k[2].value))
    if missing:
        preview = ", ".join(f"({s},{f},{t.value})" for s, f, t in missing[:6])
        extra = len(missing) - 6
        suffix = f" (+{extra} more)" if extra > 0 else ""
        raise CMRError(
            QC_MISSING_STRUCTURE,
            f"missing required contours: {preview}{suffix}; missing keys are never "
            "zero-filled",
            source=selection.series_uid,
        )

    # --- gap policy + provenance resolution ----------------------------------
    gaps = effective_stack_gaps_mm(geometries)
    # Provenance is canonical-only (review correction F1a): the caller may
    # omit algorithm_version (canonical value built from the detected gap
    # mode) or supply EXACTLY that canonical string; wrong-mode or
    # arbitrary values are rejected so provenance can never drift.
    # CFM-001: an explicit inclusion policy seals a disc-cylinder-v2
    # identity; the null-set compatibility path keeps disc-cylinder-v1.
    canonical_algorithm = (
        ALGORITHM_VERSION_SINGLE_SLICE if n_slices == 1 else ALGORITHM_VERSION_IPP
    )
    canonical_algorithm_v2 = (
        ALGORITHM_VERSION_SINGLE_SLICE_V2 if n_slices == 1 else ALGORITHM_VERSION_IPP_V2
    )
    if inclusion_policy is not None:
        _verify_inclusion_policy_coherence(
            inclusion_policy, index, selection, cine, coherent_workflow_id
        )
        canonical_for_mode = canonical_algorithm_v2
    else:
        canonical_for_mode = canonical_algorithm
    if algorithm_version is None or algorithm_version == canonical_for_mode:
        resolved_algorithm = canonical_for_mode
    else:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"algorithm_version {algorithm_version!r} is not the canonical value "
            f"for this gap mode; expected None or {canonical_for_mode!r}",
        )

    def _rasterize(structure: Structure) -> list[list[np.ndarray]]:
        return [
            [
                contour_to_mask(index[(s, f, structure)], geometries[s])
                for f in range(n_frames)
            ]
            for s in range(n_slices)
        ]

    def _slice_slab_volume(
        mask: np.ndarray, slice_index: int, gap: float
    ) -> float:
        """One slab term from THE shared ordered gap tuple (correction F3)."""
        return mask_area_mm2(mask, geometries[slice_index]) * gap / 1000.0

    def _slice_frame_volumes(structure: Structure) -> list[list[float]]:
        """``[slice][frame]`` slab volumes using EXACTLY the stored gaps.

        Correction F3: the ordered full-stack gap tuple is computed ONCE
        (``effective_stack_gaps_mm`` above — it IS ``stack_gaps_mm``) and
        every cavity and myocardial slab term consumes that same tuple;
        no nested helper re-derives or re-sorts gaps.
        """
        matrix: list[list[float]] = []
        for s in range(n_slices):
            row = [
                _slice_slab_volume(
                    contour_to_mask(index[(s, f, structure)], geometries[s]),
                    s,
                    gaps[s],
                )
                for f in range(n_frames)
            ]
            matrix.append(row)
        return matrix

    def _gated_total(matrix: list[list[float]], frame: int, role: str,
                     structure: str) -> float:
        """Σ_s w(s, role, structure) × A(s, frame) × g_s / 1000.

        CFM-001 spec §1.4 binary whole-slab arithmetic. With no policy this
        is the untouched legacy all-slices sum in the SAME slice order, so
        an explicit all-True set is numerically identical (bit-for-bit) to
        the null-set path while carrying a different algorithm identity.
        Weights are exact 1.0/0.0 floats; multiplication never perturbs an
        included slice's value.
        """
        if inclusion_policy is None:
            return sum(row[frame] for row in matrix)
        total = 0.0
        for slice_index, row in enumerate(matrix):
            total += (
                inclusion_policy.weight(slice_index, role, structure) * row[frame]
            )
        return total

    lv_matrix = _slice_frame_volumes(Structure.LV_CAVITY)
    rv_matrix = _slice_frame_volumes(Structure.RV_CAVITY)
    edv_ml = _gated_total(lv_matrix, selection.ed_frame,
                          INCLUSION_ROLE_ED, INCLUSION_STRUCTURE_LV_CAVITY)
    esv_ml = _gated_total(lv_matrix, selection.es_frame,
                          INCLUSION_ROLE_ES, INCLUSION_STRUCTURE_LV_CAVITY)
    rv_edv_ml = _gated_total(rv_matrix, selection.ed_frame,
                             INCLUSION_ROLE_ED, INCLUSION_STRUCTURE_RV_CAVITY)
    rv_esv_ml = _gated_total(rv_matrix, selection.es_frame,
                             INCLUSION_ROLE_ES, INCLUSION_STRUCTURE_RV_CAVITY)
    for label, value in (
        ("LV EDV", edv_ml), ("LV ESV", esv_ml),
        ("RV EDV", rv_edv_ml), ("RV ESV", rv_esv_ml),
    ):
        if not math.isfinite(value):
            raise CMRError(
                QUANTIFICATION_INVALID_INPUT,
                f"non-finite {label} derived after inclusion gating",
            )

    # --- CFM-001 §1.6: post-inclusion volume-curve QC -------------------------
    # Included ESV > EDV is checked PER VENTRICLE before stroke-volume or EF
    # validation. No phase swap, no toggle modification, no SV clamping and
    # no reuse of any all-slices result: a structured qc.volume_curve finding
    # naming the ventricle and the inclusion-set candidate is raised and NO
    # Measurement is constructed. The null-set path keeps the legacy
    # fail-closed behavior of function.py untouched.
    if inclusion_policy is not None:
        for ventricle, v_edv, v_esv in (
            ("LV", edv_ml, esv_ml),
            ("RV", rv_edv_ml, rv_esv_ml),
        ):
            if v_esv > v_edv:
                raise PostInclusionEsvExceedsEdvError(
                    f"{POST_INCLUSION_ESV_EXCEEDS_EDV_REASON}: {ventricle} ESV "
                    f"({v_esv} ml) exceeds {ventricle} EDV "
                    f"({v_edv} ml) after whole-slice inclusion; analysis "
                    f"unavailable, no Measurement is constructed",
                    source=selection.series_uid,
                    inclusion_set_id=inclusion_policy.inclusion_set_id,
                )

    # Flat-curve owner policy (N5-R1 E as corrected by the contract addendum):
    # legal with positive EDV; the QC path is the caller-supplied IssueSet
    # (Measurement is a frozen Node 1 record without its own QC container).
    # ESV > EDV and EDV <= 0 are NOT pre-filtered: they propagate
    # fail-closed from function.py.
    if esv_ml == edv_ml and issues is not None:
        issues.add(
            CMRWarning.qc(
                QC_VOLUME_CURVE,
                "flat manual volume curve (equal volumes at the selected ED/ES "
                "frames); recording SV=0 ml and EF=0% for physician review",
                source=f"{selection.series_uid} frames "
                f"{selection.ed_frame}/{selection.es_frame}",
            )
        )
    sv_ml = stroke_volume(edv_ml, esv_ml)
    ef_pct = ejection_fraction(edv_ml, esv_ml)
    rv_sv_ml = stroke_volume(rv_edv_ml, rv_esv_ml)
    rv_ef_pct = ejection_fraction(rv_edv_ml, rv_esv_ml)

    # --- LV mass at the ED frame: epicardial MINUS endocardial ---------------
    # Independence rule (CFM-001 §1.4): the lv_cavity/ED inclusion toggle
    # does NOT gate the endocardial mask used for cavity subtraction — only
    # the lv_myocardium/ED weight gates the resulting myocardial slab.
    myocardium_masks: list[np.ndarray] = []
    for slice_index in range(n_slices):
        epicardial = contour_to_mask(
            index[(slice_index, selection.ed_frame, Structure.LV_MYOCARDIUM)],
            geometries[slice_index],
        )
        endocardial = contour_to_mask(
            index[(slice_index, selection.ed_frame, Structure.LV_CAVITY)],
            geometries[slice_index],
        )
        myocardium_masks.append(epicardial & ~endocardial)
    # Same primitive and ordering machinery as the curves so the slab gaps
    # agree exactly with the stored stack_gaps_mm. With an explicit policy
    # each slab is multiplied by the binary lv_myocardium/ED weight AFTER
    # the gap-derived volume is formed (excluded slices contribute exactly
    # zero; gaps are never recomputed or redistributed).
    # Correction F3: mass slabs consume the SAME shared gap tuple.
    myocardium_slabs = [
        _slice_slab_volume(myocardium_masks[s], s, gaps[s])
        for s in range(n_slices)
    ]
    if inclusion_policy is None:
        myocardium_volume_ml = float(sum(myocardium_slabs))
    else:
        myocardium_volume_ml = float(
            sum(
                inclusion_policy.weight(
                    slice_index, INCLUSION_ROLE_ED,
                    INCLUSION_STRUCTURE_LV_MYOCARDIUM,
                ) * slab
                for slice_index, slab in enumerate(myocardium_slabs)
            )
        )
    if not math.isfinite(myocardium_volume_ml):
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            "non-finite myocardium volume derived at the ED frame",
        )
    lv_mass_g = myocardial_mass(myocardium_volume_ml)

    provenance = Provenance(
        study_uid=resolved_study_uid,
        series_uid=selection.series_uid,
        software_version=software_version,
        model=None,
        created_at=created_at,
        revision=index.head_provenance_revision,
        workflow_id=coherent_workflow_id,
    )
    return Measurement(
        series_uid=selection.series_uid,
        ed_frame=selection.ed_frame,
        es_frame=selection.es_frame,
        source_revision_id=index.head_revision_id,
        edv_ml=Quantity(edv_ml, Unit.ML),
        esv_ml=Quantity(esv_ml, Unit.ML),
        sv_ml=sv_ml,
        ef_pct=ef_pct,
        lv_mass_g=lv_mass_g,
        rv_edv_ml=Quantity(rv_edv_ml, Unit.ML),
        rv_esv_ml=Quantity(rv_esv_ml, Unit.ML),
        rv_sv_ml=rv_sv_ml,
        rv_ef_pct=rv_ef_pct,
        stack_gaps_mm=gaps,
        software_version=software_version,
        algorithm_version=resolved_algorithm,
        provenance=provenance,
        created_at=created_at,
        selection_id=selection.selection_id,
        # CFM-001: explicit inclusion binds the set identity; the null-set
        # compatibility path keeps inclusion_set_id=None (legacy bytes).
        inclusion_set_id=(
            None if inclusion_policy is None
            else inclusion_policy.inclusion_set_id
        ),
        range_table_version=range_table_version,
    )


RevisionLike = ContourRevision | Sequence[ContourRevision]


def compute_measurement_from_contours(
    revision: RevisionLike,
    selection: EDESSelection,
    cine: CineVolume,
    *,
    software_version: str,
    algorithm_version: str | None = None,
    created_at: str,
    issues: IssueSet | None = None,
    inclusion_policy: SliceInclusionPolicy | None = None,
    # CFM-001 correction F11: the LOW-LEVEL compatibility path defaults to
    # None so legacy-shaped arithmetic keeps its exact canonical bytes;
    # the new clinical workflow (apply-inclusion) stamps the table
    # version explicitly at the controller boundary.
    range_table_version: str | None = None,
) -> Measurement:
    """Convenience entry composing the bridge and :func:`compute_measurement`.

    ``revision`` is either one latest :class:`ContourRevision` or the
    ordered revision chain (ascending ``version``). This wrapper exists
    for callers holding revisions directly; the canonical Node 5 flow is
    ``build_cine_contour_index(load_chain(), series)`` followed by
    :func:`compute_measurement`.
    """
    revisions: list[ContourRevision] = (
        [revision] if isinstance(revision, ContourRevision) else list(revision)
    )
    index = build_cine_contour_index(revisions, selection.series_uid)
    return compute_measurement(
        index,
        selection,
        cine,
        software_version=software_version,
        algorithm_version=algorithm_version,
        created_at=created_at,
        issues=issues,
        inclusion_policy=inclusion_policy,
        range_table_version=range_table_version,
    )


# --- FNC-B additive selected-phase quantification (P1 clinical) ---------------
#
# The legacy :func:`compute_measurement` above is byte-untouched: it keeps
# requiring LV/RV cavities on EVERY (slice, frame) plus LV myocardium at ED.
# The entry below is the separate selected-phase path over the existing
# five-group inclusion contract. It reuses the SAME primitives
# (``contour_to_mask``, ``mask_area_mm2``, ``effective_stack_gaps_mm``,
# ``stroke_volume``, ``ejection_fraction``, ``myocardial_mass``), the SAME
# binary whole-slice inclusion semantics (weights exactly 1.0/0.0, gaps
# never recomputed) and the SAME head/selection/inclusion binding — but it
# consumes exactly the five required geometry groups and seals its own
# canonical algorithm identity carrying ``coverage=selected-phases-v1``.


def _selected_phase_required_keys(
    n_slices: int, ed_frame: int, es_frame: int
) -> set[ContourKey]:
    """Exact five-group required set: ED LV/RV cavities + LV myocardium, ES LV/RV."""
    required: set[ContourKey] = set()
    for slice_index in range(n_slices):
        required.add((slice_index, ed_frame, Structure.LV_CAVITY))
        required.add((slice_index, ed_frame, Structure.LV_MYOCARDIUM))
        required.add((slice_index, ed_frame, Structure.RV_CAVITY))
        required.add((slice_index, es_frame, Structure.LV_CAVITY))
        required.add((slice_index, es_frame, Structure.RV_CAVITY))
    return required


def compute_selected_phase_measurement(
    index: CineContourIndex,
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
    """Compute the deterministic selected-phase Measurement for one cine series.

    FNC-B additive path: consumes exactly the five required geometry groups
    (ED ``LV_CAVITY``/``LV_MYOCARDIUM``/``RV_CAVITY`` plus ES
    ``LV_CAVITY``/``RV_CAVITY``) for every included SAX slice, at the
    physician-authored ``selection`` frames. Any extra contour present in
    ``index`` (e.g. intermediate-phase cavities) is IGNORED: it is never
    required, never validated and never rasterized.

    CMR_FUNCV1_STACK_APPLICABILITY_001 (F19): a required key that is
    absent from ``index`` but explicitly waived by ``inclusion_policy``
    (``included=False`` on exactly that ``(slice, role, structure)`` key)
    is NON-APPLICABLE, not UNKNOWN: it contributes exactly zero cavity
    volume with NO fabricated contour, and the waived endocardial mask
    used for cavity subtraction is the empty mask (so an epicardial-only
    apical slab keeps its full myocardial volume, per SCMR 2020 LV-mass
    §iii). A missing key with NO waiver is UNKNOWN and fails closed
    with a typed ``CMRError`` naming the exact keys; contours are never
    zero-filled and waivers are never inferred (phase-specific and
    structure-specific; see :func:`applicability_waivers`).

    Equivalence: on an all-phase fixture projected to the five groups this
    yields the same nine scalar value/unit pairs as the legacy
    :func:`compute_measurement` on the same index/selection, because every
    slab term uses the same rasterization, the same ordered gap tuple, the
    same slice order and the same formula primitives.

    Provenance: ``algorithm_version`` is canonical-only for the
    selected-phase family (``disc-cylinder-v2/<gap-source>/
    coverage=selected-phases-v1``). Legacy ``disc-cylinder-v1/...`` and
    ``.../inclusion=whole-slice-v1`` identities are rejected fail-closed so
    a selected-phase result can never be sealed as a whole-cine one. When
    ``inclusion_policy`` is supplied its identity is verified exactly as in
    the legacy path and its ``inclusion_set_id`` is bound; gaps are never
    recomputed. LV mass is computed at the ED frame only
    (epicardial-minus-endocardial, same convention as the legacy path).
    """
    if not isinstance(software_version, str) or not software_version.strip():
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            "software_version is required and must be a non-empty string",
        )
    if index.series_uid != selection.series_uid:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"index series {index.series_uid!r} does not match selection series "
            f"{selection.series_uid!r}",
        )

    # One exact workflow identity across chain/cine/selection (same rule as
    # the legacy path); study/series coherence verified wherever present.
    coherent_workflow_id = assert_coherent_workflow_identity(index, cine, selection)
    study_candidates = {
        "selection": selection.provenance.study_uid,
        "cine": cine.study_uid,
        "index": None,
    }
    bound_studies = {value for value in study_candidates.values() if value is not None}
    if len(bound_studies) > 1:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"conflicting study identities across selection/cine: "
            f"{sorted(bound_studies)!r}; one measurement requires one study",
        )
    resolved_study_uid = next(iter(bound_studies)) if bound_studies else None

    # --- binding checks (fail closed BEFORE any calculation) -----------------
    n_slices = cine.n_slices
    n_frames = cine.n_frames
    if selection.n_frames is not None and selection.n_frames != n_frames:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"selection n_frames {selection.n_frames} disagrees with cine "
            f"n_frames {n_frames}",
        )
    if not (0 <= selection.ed_frame < n_frames) or not (0 <= selection.es_frame < n_frames):
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"ED/ES frames ({selection.ed_frame}, {selection.es_frame}) are outside "
            f"0..{n_frames - 1}",
        )

    # Required-set enforcement (fail-closed): exactly the five groups at the
    # selected frames. Extra index entries (intermediate phases) are ignored
    # entirely — only required-key contours are range/field validated.
    # F19: a missing key explicitly waived by the inclusion policy
    # (included=False on exactly that (slice, role, structure) key) is
    # NON-APPLICABLE and is exempted from the missing-key refusal; a
    # missing key with no waiver is UNKNOWN and fails closed.
    geometries = list(cine.geometries)
    required = _selected_phase_required_keys(n_slices, selection.ed_frame, selection.es_frame)
    waived = applicability_waivers(inclusion_policy)

    def _waiver_key(
        slice_index: int, frame_index: int, structure: Structure
    ) -> tuple[int, str, str] | None:
        """The exact waiver key for a required contour, or None.

        Phase mapping is EXACT (ED role <-> selection.ed_frame, ES role <->
        selection.es_frame). Structure mapping uses the inclusion vocabulary
        values (``lv_cavity``/``rv_cavity``/``lv_myocardium``).

        Role-exact requiredness (C1 P1-3): even on a flat ED==ES selection
        the shared frame carries TWO required keys (one per role — see
        :func:`_selected_phase_required_keys`, which emits both). Each
        required key is waived only by its OWN role waiver: an ED waiver
        never covers the ES key at the same frame and vice versa.
        """
        role: str | None = None
        if frame_index == selection.ed_frame and frame_index == selection.es_frame:
            # Flat selection: the caller disambiguates by role via the
            # required key's own role — this helper keeps the legacy
            # (frame, structure) signature for the non-flat fast path only.
            # The role-exact checks below (missing-set filter, _slab_volume,
            # _myocardium_masks) pass the required key's role explicitly.
            return None
        if frame_index == selection.ed_frame:
            role = INCLUSION_ROLE_ED
        elif frame_index == selection.es_frame:
            role = INCLUSION_ROLE_ES
        else:
            return None
        key = (slice_index, role, structure.value)
        return key if key in waived else None

    def _waiver_key_for_role(
        slice_index: int, role: str, structure: Structure
    ) -> tuple[int, str, str] | None:
        """Role-exact waiver lookup for one ``(slice, role, structure)`` key."""
        key = (slice_index, role, structure.value)
        return key if key in waived else None

    def _required_role(frame_index: int) -> str | None:
        """The inclusion role of one required contour key, or None.

        Non-flat selections map by frame; a flat ED==ES selection is
        ambiguous by frame alone, so callers iterate the required key's
        own role explicitly (see the missing-set filter below).
        """
        if frame_index == selection.ed_frame and frame_index == selection.es_frame:
            return None
        if frame_index == selection.ed_frame:
            return INCLUSION_ROLE_ED
        if frame_index == selection.es_frame:
            return INCLUSION_ROLE_ES
        return None

    _ = _required_role  # documented helper; the flat path below is explicit.

    for slice_index, frame_index, structure in sorted(
        required, key=lambda k: (k[0], k[1], k[2].value)
    ):
        if slice_index >= n_slices or frame_index >= n_frames:
            raise CMRError(
                CONTOUR_INVALID,
                f"contour key ({slice_index},{frame_index},{structure.value}) is "
                f"outside the cine grid {n_slices}x{n_frames}",
            )
        if (slice_index, frame_index, structure) in index:
            _validate_contour_within_geometry(
                index[(slice_index, frame_index, structure)],
                geometries[slice_index],
            )
    # Role-exact missing-key filter (C1 P1-3): each required key is
    # identified by its OWN (slice, role, structure). On a flat ED==ES
    # selection the shared frame yields two required keys per structure;
    # each is waived only by its own role waiver. Non-flat keys map to
    # their role by frame (ED frame -> ED role, ES frame -> ES role).
    def _missing_unwaived() -> list[ContourKey]:
        absent = required.difference(index)
        if selection.ed_frame != selection.es_frame:
            return sorted(
                (
                    key
                    for key in absent
                    if _waiver_key(key[0], key[1], key[2]) is None
                ),
                key=lambda k: (k[0], k[1], k[2].value),
            )
        # Flat ED==ES: expand the shared frame per role. The required set
        # holds ONE contour key per (slice, shared frame, structure), but
        # TWO role keys; the contour covers both roles iff present, and a
        # missing contour is refused unless BOTH role waivers cover it —
        # equivalently, each role key is checked role-exactly.
        unwaived: list[ContourKey] = []
        for slice_index, frame_index, structure in sorted(
            absent, key=lambda k: (k[0], k[1], k[2].value)
        ):
            if frame_index != selection.ed_frame:
                if _waiver_key(slice_index, frame_index, structure) is None:
                    unwaived.append((slice_index, frame_index, structure))
                continue
            ed_waived = (
                _waiver_key_for_role(slice_index, INCLUSION_ROLE_ED, structure)
                is not None
            )
            es_waived = (
                _waiver_key_for_role(slice_index, INCLUSION_ROLE_ES, structure)
                is not None
            )
            if structure is Structure.LV_MYOCARDIUM:
                # Myocardium is an ED-only group: only the ED role key
                # exists for it (the grid validator rejects ES myocardium).
                if not ed_waived:
                    unwaived.append((slice_index, frame_index, structure))
            elif not (ed_waived and es_waived):
                # Cavity shared-frame contour missing: refuse unless the
                # STILL-applicable role's key is waived. In practice the
                # contour either exists (both roles covered) or is absent;
                # naming the shared key once is exact.
                unwaived.append((slice_index, frame_index, structure))
        return sorted(unwaived, key=lambda k: (k[0], k[1], k[2].value))

    missing = _missing_unwaived()
    if missing:
        preview = ", ".join(f"({s},{f},{t.value})" for s, f, t in missing[:6])
        extra = len(missing) - 6
        suffix = f" (+{extra} more)" if extra > 0 else ""
        raise CMRError(
            QC_MISSING_STRUCTURE,
            f"missing required selected-phase contours: {preview}{suffix}; "
            "missing keys are never zero-filled",
            source=selection.series_uid,
        )

    # --- gap policy + provenance resolution ----------------------------------
    gaps = effective_stack_gaps_mm(geometries)
    # Canonical-only enforcement (same style as the legacy path): the caller
    # may omit algorithm_version or supply EXACTLY the canonical
    # selected-phase string for the detected gap mode. Legacy whole-cine
    # (disc-cylinder-v1/...) and whole-slice-inclusion
    # (.../inclusion=whole-slice-v1) identities are rejected so a
    # selected-phase result is never sealed under a whole-cine identity.
    canonical_for_mode = (
        ALGORITHM_VERSION_SINGLE_SLICE_SELECTED_PHASES
        if n_slices == 1
        else ALGORITHM_VERSION_IPP_SELECTED_PHASES
    )
    if inclusion_policy is not None:
        _verify_inclusion_policy_coherence(
            inclusion_policy, index, selection, cine, coherent_workflow_id
        )
    if algorithm_version is None or algorithm_version == canonical_for_mode:
        resolved_algorithm = canonical_for_mode
    else:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"algorithm_version {algorithm_version!r} is not the canonical "
            f"selected-phase value for this gap mode; expected None or "
            f"{canonical_for_mode!r}",
        )

    def _slab_volume(
        structure: Structure, slice_index: int, frame: int, *, role: str | None = None
    ) -> float:
        """One slab term from THE shared ordered gap tuple (same as legacy).

        F19: a waived (non-applicable) key contributes exactly zero with NO
        fabricated contour — the slab is skipped, never rasterized.
        Role-exact (C1 P1-3): on a flat ED==ES selection the caller passes
        the slab's own role, so each role's slab is waived only by its own
        role waiver.
        """
        if (slice_index, frame, structure) not in index:
            if role is not None:
                waiver = _waiver_key_for_role(slice_index, role, structure)
            else:
                waiver = _waiver_key(slice_index, frame, structure)
            if waiver is not None:
                return 0.0
            raise CMRError(
                QC_MISSING_STRUCTURE,
                f"missing required selected-phase contour: "
                f"({slice_index},{frame},{structure.value}); missing keys are "
                "never zero-filled",
                source=selection.series_uid,
            )
        mask = contour_to_mask(
            index[(slice_index, frame, structure)], geometries[slice_index]
        )
        return mask_area_mm2(mask, geometries[slice_index]) * gaps[slice_index] / 1000.0

    def _gated_total(
        values_by_slice: list[float], role: str, structure: str
    ) -> float:
        """Binary whole-slab gating with the exact legacy summation order."""
        if inclusion_policy is None:
            return sum(values_by_slice)
        total = 0.0
        for slice_index, value in enumerate(values_by_slice):
            total += inclusion_policy.weight(slice_index, role, structure) * value
        return total

    _flat = selection.ed_frame == selection.es_frame
    lv_edv_ml = _gated_total(
        [
            _slab_volume(
                Structure.LV_CAVITY, s, selection.ed_frame,
                role=INCLUSION_ROLE_ED if _flat else None,
            )
            for s in range(n_slices)
        ],
        INCLUSION_ROLE_ED,
        INCLUSION_STRUCTURE_LV_CAVITY,
    )
    lv_esv_ml = _gated_total(
        [
            _slab_volume(
                Structure.LV_CAVITY, s, selection.es_frame,
                role=INCLUSION_ROLE_ES if _flat else None,
            )
            for s in range(n_slices)
        ],
        INCLUSION_ROLE_ES,
        INCLUSION_STRUCTURE_LV_CAVITY,
    )
    rv_edv_ml = _gated_total(
        [
            _slab_volume(
                Structure.RV_CAVITY, s, selection.ed_frame,
                role=INCLUSION_ROLE_ED if _flat else None,
            )
            for s in range(n_slices)
        ],
        INCLUSION_ROLE_ED,
        INCLUSION_STRUCTURE_RV_CAVITY,
    )
    rv_esv_ml = _gated_total(
        [
            _slab_volume(
                Structure.RV_CAVITY, s, selection.es_frame,
                role=INCLUSION_ROLE_ES if _flat else None,
            )
            for s in range(n_slices)
        ],
        INCLUSION_ROLE_ES,
        INCLUSION_STRUCTURE_RV_CAVITY,
    )
    for label, value in (
        ("LV EDV", lv_edv_ml),
        ("LV ESV", lv_esv_ml),
        ("RV EDV", rv_edv_ml),
        ("RV ESV", rv_esv_ml),
    ):
        if not math.isfinite(value):
            raise CMRError(
                QUANTIFICATION_INVALID_INPUT,
                f"non-finite {label} derived after inclusion gating",
            )

    # Post-inclusion ESV>EDV per ventricle (same typed rejection as legacy;
    # no phase swap, no SV clamping, no Measurement constructed).
    if inclusion_policy is not None:
        for ventricle, v_edv, v_esv in (
            ("LV", lv_edv_ml, lv_esv_ml),
            ("RV", rv_edv_ml, rv_esv_ml),
        ):
            if v_esv > v_edv:
                raise PostInclusionEsvExceedsEdvError(
                    f"{POST_INCLUSION_ESV_EXCEEDS_EDV_REASON}: {ventricle} ESV "
                    f"({v_esv} ml) exceeds {ventricle} EDV "
                    f"({v_edv} ml) after whole-slice inclusion; analysis "
                    f"unavailable, no Measurement is constructed",
                    source=selection.series_uid,
                    inclusion_set_id=inclusion_policy.inclusion_set_id,
                )

    # Flat-curve owner policy (same as legacy): legal with positive EDV; the
    # QC path is the caller-supplied IssueSet. ESV > EDV and EDV <= 0
    # propagate fail-closed from function.py untouched.
    if lv_esv_ml == lv_edv_ml and issues is not None:
        issues.add(
            CMRWarning.qc(
                QC_VOLUME_CURVE,
                "flat manual volume curve (equal volumes at the selected ED/ES "
                "frames); recording SV=0 ml and EF=0% for physician review",
                source=f"{selection.series_uid} frames "
                f"{selection.ed_frame}/{selection.es_frame}",
            )
        )
    sv_ml = stroke_volume(lv_edv_ml, lv_esv_ml)
    ef_pct = ejection_fraction(lv_edv_ml, lv_esv_ml)
    rv_sv_ml = stroke_volume(rv_edv_ml, rv_esv_ml)
    rv_ef_pct = ejection_fraction(rv_edv_ml, rv_esv_ml)

    # --- LV mass at the ED frame: epicardial MINUS endocardial ---------------
    # Same convention as the legacy path: the lv_cavity/ED inclusion toggle
    # does NOT gate the endocardial mask used for cavity subtraction — only
    # the lv_myocardium/ED weight gates the resulting myocardial slab.
    # F19: a WAIVED (non-applicable) cavity key subtracts the EMPTY mask
    # (zero endocardial area), so an epicardial-only apical slab keeps its
    # full myocardial volume with no fabricated endocardial contour.
    def _myocardium_masks(
        slice_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        epicardial_key = (slice_index, selection.ed_frame, Structure.LV_MYOCARDIUM)
        endocardial_key = (slice_index, selection.ed_frame, Structure.LV_CAVITY)
        field_shape = (geometries[slice_index].rows, geometries[slice_index].cols)
        if epicardial_key not in index:
            if _waiver_key_for_role(
                slice_index, INCLUSION_ROLE_ED, Structure.LV_MYOCARDIUM
            ) is None:
                raise CMRError(
                    QC_MISSING_STRUCTURE,
                    "missing required selected-phase contour: "
                    f"({slice_index},{selection.ed_frame},"
                    f"{Structure.LV_MYOCARDIUM.value}); missing keys are "
                    "never zero-filled",
                    source=selection.series_uid,
                )
            epicardial = np.zeros(field_shape, dtype=bool)
        else:
            epicardial = contour_to_mask(
                index[epicardial_key],
                geometries[slice_index],
            )
        if endocardial_key not in index:
            if _waiver_key_for_role(
                slice_index, INCLUSION_ROLE_ED, Structure.LV_CAVITY
            ) is None:
                raise CMRError(
                    QC_MISSING_STRUCTURE,
                    "missing required selected-phase contour: "
                    f"({slice_index},{selection.ed_frame},"
                    f"{Structure.LV_CAVITY.value}); missing keys are "
                    "never zero-filled",
                    source=selection.series_uid,
                )
            endocardial = np.zeros(field_shape, dtype=bool)
        else:
            endocardial = contour_to_mask(
                index[endocardial_key],
                geometries[slice_index],
            )
        return epicardial, endocardial

    myocardium_slabs = []
    for s in range(n_slices):
        _epi_mask, _endo_mask = _myocardium_masks(s)
        myocardium_slabs.append(
            mask_area_mm2(_epi_mask & ~_endo_mask, geometries[s])
            * gaps[s]
            / 1000.0
        )
    if inclusion_policy is None:
        myocardium_volume_ml = float(sum(myocardium_slabs))
    else:
        myocardium_volume_ml = float(
            sum(
                inclusion_policy.weight(
                    slice_index,
                    INCLUSION_ROLE_ED,
                    INCLUSION_STRUCTURE_LV_MYOCARDIUM,
                )
                * slab
                for slice_index, slab in enumerate(myocardium_slabs)
            )
        )
    if not math.isfinite(myocardium_volume_ml):
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            "non-finite myocardium volume derived at the ED frame",
        )
    lv_mass_g = myocardial_mass(myocardium_volume_ml)

    provenance = Provenance(
        study_uid=resolved_study_uid,
        series_uid=selection.series_uid,
        software_version=software_version,
        model=None,
        created_at=created_at,
        revision=index.head_provenance_revision,
        workflow_id=coherent_workflow_id,
    )
    return Measurement(
        series_uid=selection.series_uid,
        ed_frame=selection.ed_frame,
        es_frame=selection.es_frame,
        source_revision_id=index.head_revision_id,
        edv_ml=Quantity(lv_edv_ml, Unit.ML),
        esv_ml=Quantity(lv_esv_ml, Unit.ML),
        sv_ml=sv_ml,
        ef_pct=ef_pct,
        lv_mass_g=lv_mass_g,
        rv_edv_ml=Quantity(rv_edv_ml, Unit.ML),
        rv_esv_ml=Quantity(rv_esv_ml, Unit.ML),
        rv_sv_ml=rv_sv_ml,
        rv_ef_pct=rv_ef_pct,
        stack_gaps_mm=gaps,
        software_version=software_version,
        algorithm_version=resolved_algorithm,
        provenance=provenance,
        created_at=created_at,
        selection_id=selection.selection_id,
        inclusion_set_id=(
            None if inclusion_policy is None else inclusion_policy.inclusion_set_id
        ),
        range_table_version=range_table_version,
    )
