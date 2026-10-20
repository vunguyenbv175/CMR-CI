"""Point-level manual contour editing with an undo/redo command stack (Node 4).

Every edit operation in this module is NON-MUTATING: it copies the input
contour's points (fresh outer list plus fresh inner tuples, so ``_FrozenContour``
inputs taken from persisted :class:`~cmr_core.domain_types.Prediction` /
:class:`~cmr_core.domain_types.ContourRevision` records work unchanged —
review amendment N4-R1 B) and returns a NEW
:class:`~cmr_contours.model.Contour` with ``edited=True`` and a ``revision``
bumped by one, mirroring the existing :meth:`Contour.mark_edited` semantics.
``Contour`` instances are never mutated in place and the input object is left
byte-identical.

Provenance rule (amendment N4-R1 A): whenever a derived :class:`Provenance` is
needed it is constructed EXPLICITLY with every field including ``created_at``
(preserved from the source record so results stay deterministic and test
friendly). :meth:`Provenance.bumped` is never called here: it drops
``created_at`` and stamps ``datetime.now()``.

``ensure_closed`` semantics (amendment N4-R1 G): appending the first point when
``is_closed`` is False closes the polygon. Calling it on an ALREADY-closed
contour still returns a NEW instance with ``edited=True`` and a bumped revision,
per the contract wording "every operation returns a NEW Contour instance with
edited=True and bumped revision". Callers needing a true no-op must compare the
returned instance themselves (``result is not contour``, equal points).

``delete_vertex`` failure semantics (amendment N4-R1 C): removing a vertex from
a 3-point contour leaves two points and the :class:`Contour` constructor raises
:class:`CMRError` with code ``contour.invalid``; this module lets that error
propagate — no catching, no partial results. Out-of-range indices are rejected
with the same code before any list surgery.

Revision persistence (:func:`save_to_revision`) MERGES edited structures into a
copy of the parent contour map (amendment N4-R1 D): unedited structures are
carried through verbatim because a revision serializes its whole contour map.
The child revision's :class:`Provenance` is likewise constructed explicitly
with all fields including the caller-supplied ``created_at``; ``model`` stays
``None`` because a physician edit is not an AI result (the AI origin remains on
the contours themselves).

Commands (:class:`EditCommand` subclasses) are pure frozen data describing ONE
operation; :class:`EditCommandStack` applies them snapshot-based — entries are
``(command, before_contour)`` pairs and ``undo()`` returns the exact prior
immutable object (amendment N4-R1 E). Executing after an undo clears the redo
tail (standard branch semantics). The stack holds no UI wiring, no controller
and no persistence calls.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from cmr_core.cine import CineVolume
from cmr_core.domain_types import (
    _S3_COORD_LIMIT,
    EDIT_OP_CREATE,
    EDIT_OP_REMOVE,
    ContourRevision,
    Prediction,
)
from cmr_core.errors import CONTOUR_INVALID, INVALID_ARGUMENT, CMRError
from cmr_core.geometry import (
    _AREA_EPSILON,
    _ring_winding_normalized,
    _signed_area,
    _validate_no_self_intersection,
)
from cmr_core.provenance import VERSION, ModelMetadata, Provenance
from cmr_core.segmentation import Structure
from cmr_qc.report import QCReport

from cmr_contours.model import Contour

__all__ = [
    "DeleteVertex",
    "EditCommand",
    "EditCommandStack",
    "EnsureClosed",
    "InsertVertex",
    "MAX_REPLACEMENT_VERTICES",
    "ReplaceContour",
    "TranslateContour",
    "apply_command",
    "compose_create_revision",
    "compose_remove_revision",
    "copy_contour_across_frames",
    "copy_contour_across_slices",
    "delete_vertex",
    "ensure_closed",
    "insert_vertex",
    "replace_contour",
    "save_to_revision",
    "to_implicit_closure_points",
    "translate_contour",
    "validate_and_normalize_replacement_ring",
]


# --- non-mutating construction helpers --------------------------------------


def _copy_points(contour: Contour) -> list[tuple[float, float]]:
    """Fresh outer list AND fresh inner tuples (amendment N4-R1 B copy rule).

    Persisted contours arrive as tuple-of-tuples (or list-of-lists after JSON
    reconstruction); copying per pair guarantees the new instance shares no
    mutable structure with the source record.
    """
    return [(x, y) for x, y in contour.points]


def _clone_metadata(metadata: ModelMetadata) -> ModelMetadata:
    """Fresh :class:`ModelMetadata` with a fresh mutable ``label_map`` dict.

    ``ModelMetadata`` is frozen but its ``label_map`` is a MUTABLE dict, so
    sharing one instance across a parent contour, a stored history snapshot and
    a derived edit result lets an in-place ``label_map`` mutation leak across
    every boundary (review e5614a6 P1-1, probes STACK_ALIAS_RESTORED_LABEL_MAP
    and EDIT_RESULT_MUTATES_PARENT_MODEL). Every detachment boundary clones.
    """
    return ModelMetadata(
        name=metadata.name,
        version=metadata.version,
        label_map=dict(metadata.label_map),
        preprocessing_version=metadata.preprocessing_version,
        checksum=metadata.checksum,
    )


def _detach_model(model: ModelMetadata | None) -> ModelMetadata | None:
    """None-safe metadata clone (fresh mutable ``label_map``)."""
    return None if model is None else _clone_metadata(model)


def _snapshot_contour(contour: Contour) -> Contour:
    """Fully detached deep snapshot of ``contour`` (reconciliation correction).

    The result shares no mutable structure with the input: fresh outer points
    list, fresh inner tuples, a cloned :class:`Provenance` (Provenance is a
    MUTABLE dataclass), and cloned :class:`~cmr_core.provenance.ModelMetadata`
    instances for BOTH ``contour.model`` and ``provenance.model`` — the
    metadata is frozen but its ``label_map`` is a mutable dict. Immutable
    fields (:class:`~cmr_core.segmentation.Structure`, strings, ints, bools)
    are shared safely. History boundaries own snapshots built by this helper so
    external in-place mutation of retained references can never rewrite the
    past.
    """
    provenance = None if contour.provenance is None else replace(contour.provenance)
    if provenance is not None:
        provenance.model = _detach_model(provenance.model)
    return Contour(
        slice_index=contour.slice_index,
        frame_index=contour.frame_index,
        structure=contour.structure,
        points=_copy_points(contour),
        source=contour.source,
        edited=contour.edited,
        approved=contour.approved,
        revision=contour.revision,
        model=_detach_model(contour.model),
        provenance=provenance,
    )


def _bump_provenance(provenance: Provenance | None) -> Provenance | None:
    """Explicit provenance bump that PRESERVES ``created_at`` (N4-R1 A).

    Never uses :meth:`Provenance.bumped`, which silently drops ``created_at``
    and stamps the current wall-clock time.
    """
    if provenance is None:
        return None
    # Review e5614a6 P1-1: the bumped provenance must not retain the source
    # metadata by reference, or a later mutation of the derived result would
    # rewrite the parent/snapshot label_map.
    return Provenance(
        study_uid=provenance.study_uid,
        series_uid=provenance.series_uid,
        software_version=provenance.software_version,
        model=None if provenance.model is None else _clone_metadata(provenance.model),
        created_at=provenance.created_at,
        revision=provenance.revision + 1,
    )


def _edited_copy(contour: Contour, points: list[tuple[float, float]]) -> Contour:
    """Build the edited result: NEW instance, ``edited=True``, revision bumped.

    Approval policy (owner decision FORK_UNAPPROVED_DERIVED_REVISION_V1):
    changed contour content is ALWAYS unapproved. The approved revision and
    its approval snapshot remain immutable upstream; the first edit forks an
    unapproved derived contour that must pass a new approval lifecycle. No
    edit operation may silently preserve or transfer ``approved=True``.

    Review e5614a6 P1-1: the result deep-clones ``contour.model`` and
    ``provenance.model`` (mutable ``label_map`` included) so mutating the
    returned edit can never mutate the approved parent or a stored undo
    snapshot; metadata FIELD VALUES are preserved for traceability (which AI
    model produced the polygon being corrected); only the edit flags change.
    """
    model = _detach_model(contour.model)
    bumped = _bump_provenance(contour.provenance)
    return Contour(
        slice_index=contour.slice_index,
        frame_index=contour.frame_index,
        structure=contour.structure,
        points=list(points),
        source=contour.source,
        edited=True,
        approved=False,
        revision=contour.revision + 1,
        model=model,
        provenance=bumped,
    )


# --- argument validation -----------------------------------------------------


def _invalid(detail: str) -> CMRError:
    return CMRError(CONTOUR_INVALID, detail)


def _finite_coordinate(value: float, name: str) -> float:
    """Validate one coordinate/offset; reject bools, non-numbers, non-finites.

    The value is returned unchanged (ints stay ints) so canonical serializations
    keep their numeric types stable across edits.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"{name} must be a real number, got {value!r}")
    if isinstance(value, int) and abs(value) > _S3_COORD_LIMIT:
        raise _invalid(f"{name} coordinate magnitude exceeds safety limit")
    try:
        fval = float(value)
    except OverflowError:
        raise _invalid(f"{name} coordinate magnitude exceeds safety limit")
    if not math.isfinite(fval):
        raise _invalid(f"{name} must be finite, got {value!r}")
    return value


def _position_pair(value: object) -> tuple[float, float]:
    """Validate an ``(x, y)`` pixel position given as a 2-sequence."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        x = _finite_coordinate(value[0], "position.x")
        y = _finite_coordinate(value[1], "position.y")
        return (x, y)
    raise _invalid(f"position must be a finite (x, y) pair, got {value!r}")


# --- geometric helpers --------------------------------------------------------


def _segment_distance_sq(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Squared Euclidean distance from point P to segment AB."""
    abx = bx - ax
    aby = by - ay
    length_sq = abx * abx + aby * aby
    if length_sq == 0:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy
    t = ((px - ax) * abx + (py - ay) * aby) / length_sq
    t = max(0.0, min(1.0, t))
    dx = px - (ax + t * abx)
    dy = py - (ay + t * aby)
    return dx * dx + dy * dy


def _nearest_edge_index(
    position: tuple[float, float], points: list[tuple[float, float]], *, open_polygon: bool
) -> int:
    """Index of the edge nearest to ``position`` (ties -> lowest edge index).

    Edges are consecutive pairs ``(points[i], points[i+1])`` for
    ``i in [0, len(points)-2]``. For an OPEN polygon the implicit closing edge
    ``(points[-1], points[0])`` participates as candidate index
    ``len(points)-1``; for closed polygons the stored duplicate first point
    already covers it as edge ``len(points)-2``.
    """
    px, py = position
    best_edge = 0
    best_distance_sq = math.inf
    for edge in range(len(points) - 1):
        ax, ay = points[edge]
        bx, by = points[edge + 1]
        distance_sq = _segment_distance_sq(px, py, ax, ay, bx, by)
        if distance_sq < best_distance_sq:  # strict '<' keeps the lowest index on ties
            best_distance_sq = distance_sq
            best_edge = edge
    if open_polygon:
        last = len(points) - 1
        ax, ay = points[last]
        bx, by = points[0]
        distance_sq = _segment_distance_sq(px, py, ax, ay, bx, by)
        if distance_sq < best_distance_sq:
            best_edge = last  # insertion index becomes len(points): append at end
    return best_edge


# --- edit operations (all return NEW Contour instances, never mutate) --------


def insert_vertex(
    contour: Contour, index_or_position: int | tuple[float, float] | list[float]
) -> Contour:
    """Insert one vertex and return the NEW edited contour.

    Two deterministic addressing modes:

    - **Edge index** (``int``): split the edge STARTING at vertex ``i`` at its
      midpoint; valid domain ``[0, len(points)-2]`` (the explicit consecutive
      edges). Midpoint arithmetic always yields floats. For closed polygons
      this covers every side including the closing one; for open polygons the
      implicit closing edge is addressable only via the position mode.
    - **Pixel position** (an ``(x, y)`` pair): the vertex is inserted exactly at
      the given position into the nearest edge (squared-distance to the
      segment; ties resolve to the lowest edge index).

    Raises:
        CMRError: code ``contour.invalid`` for bool/non-int indices,
            out-of-range indices, malformed positions or non-finite
            coordinates.
    """
    points = _copy_points(contour)
    value = index_or_position
    if isinstance(value, bool):
        raise _invalid(f"edge index must be an integer, got bool {value!r}")
    if isinstance(value, int):
        max_edge_index = len(points) - 2
        if not 0 <= value <= max_edge_index:
            raise _invalid(
                f"edge index for vertex insertion must be within [0, {max_edge_index}], "
                f"got {value}"
            )
        ax, ay = points[value]
        bx, by = points[value + 1]
        points.insert(value + 1, ((ax + bx) / 2, (ay + by) / 2))
        return _edited_copy(contour, points)
    position = _position_pair(value)
    edge_index = _nearest_edge_index(position, points, open_polygon=not contour.is_closed)
    points.insert(edge_index + 1, position)
    return _edited_copy(contour, points)


def delete_vertex(contour: Contour, vertex_index: int) -> Contour:
    """Remove vertex ``vertex_index`` and return the NEW edited contour.

    Negative indices are rejected (no Python-style wrap-around: silent deletion
    of the "last" vertex would be surprising in a clinical editing context).
    If fewer than three points remain, the :class:`Contour` constructor raises
    :class:`CMRError` (``contour.invalid``); per amendment N4-R1 C this error
    propagates unchanged — no catching, no partial results.
    """
    points = _copy_points(contour)
    if isinstance(vertex_index, bool) or not isinstance(vertex_index, int):
        raise _invalid(f"vertex_index must be an integer, got {vertex_index!r}")
    if not 0 <= vertex_index < len(points):
        raise _invalid(f"vertex_index out of range [0, {len(points) - 1}]: {vertex_index}")
    del points[vertex_index]
    return _edited_copy(contour, points)


def translate_contour(contour: Contour, dx: float, dy: float) -> Contour:
    """Shift every point by ``(dx, dy)`` pixels and return the NEW contour."""
    offset_x = _finite_coordinate(dx, "dx")
    offset_y = _finite_coordinate(dy, "dy")
    points = [(x + offset_x, y + offset_y) for x, y in contour.points]
    return _edited_copy(contour, points)


def ensure_closed(contour: Contour) -> Contour:
    """Append the first point when the contour is open; return the NEW contour.

    Per amendment N4-R1 G an ALREADY-closed input still produces a NEW instance
    with ``edited=True`` and a bumped revision (see module docstring).
    """
    points = _copy_points(contour)
    if not contour.is_closed:
        points.append(points[0])
    return _edited_copy(contour, points)


#: Maximum vertex limit for freehand replacement rings (O(n^2) intersection check cap).
MAX_REPLACEMENT_VERTICES: int = 2000


def to_implicit_closure_points(
    points: Sequence[tuple[float, float]] | Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    """Convert an explicitly closed ring to implicit closure by dropping the terminal duplicate."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        return pts[:-1]
    return pts


def validate_and_normalize_replacement_ring(
    points: Sequence[Any],
    *,
    max_vertices: int = MAX_REPLACEMENT_VERTICES,
    cine: CineVolume | None = None,
    slice_index: int | None = None,
) -> tuple[tuple[float, float], ...]:
    """Importer-family PolygonRing validation and normalization for replacement rings.

    Adjudication contract C2 & C4:
    - strict numeric pairs EXCLUDING booleans; finite, bounded coordinates (abs <= 1e150)
    - >= 3 distinct vertices; non-degenerate area; SIMPLE ring (no self-intersection)
    - implicit closure (reject repeated terminal first vertex); reject consecutive duplicates
    - winding: normalized to counter-clockwise (CCW) per existing convention
      (_ring_winding_normalized(ccw=True))
    - MAX vertex limit: caps at 2000 vertices (quadratic intersection check guard)
    - out-of-field behavior: reject (not clip) when cine slice bounds are provided
    - no silent clipping, smoothing, or topology repair

    Documented winding/closure convention (C2/C4):
    Submitted points vs normalized saved geometry:
    Saved geometry has implicit closure (no duplicate closing vertex) and is normalized
    to counter-clockwise (CCW) winding via _ring_winding_normalized(..., ccw=True).
    Tests assert normalized saved geometry against these exact invariants.
    Provenance invariant (C4):
    Provenance is snapshot-and-lineage based. Persisted canonical records store
    geometry snapshots and parent revision lineage (no command log in ContourRevision).
    Before/after state is fully reproducible from the snapshot and its parent.
    """
    if isinstance(points, (str, bytes, Mapping)) or not isinstance(points, Sequence):
        raise _invalid(
            "replacement points must be a sequence of (x, y) pairs, "
            f"got {type(points).__name__}"
        )
    if len(points) > max_vertices:
        raise _invalid(
            f"replacement points count ({len(points)}) exceeds maximum limit of {max_vertices}"
        )
    if len(points) < 3:
        raise _invalid(f"replacement ring must have >= 3 points, got {len(points)}")

    pts: list[tuple[float, float]] = []
    for i, pt in enumerate(points):
        if isinstance(pt, (str, bytes, Mapping)) or not isinstance(pt, Sequence) or len(pt) != 2:
            raise _invalid(f"replacement vertex {i} must be an (x, y) pair, got {pt!r}")
        x, y = pt[0], pt[1]
        # Strict numeric pairs EXCLUDING booleans
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise _invalid(f"replacement vertex {i} x coordinate must be a real number, got {x!r}")
        if isinstance(y, bool) or not isinstance(y, (int, float)):
            raise _invalid(f"replacement vertex {i} y coordinate must be a real number, got {y!r}")
        if isinstance(x, int) and abs(x) > _S3_COORD_LIMIT:
            raise _invalid(f"replacement vertex {i} coordinate magnitude exceeds safety limit")
        if isinstance(y, int) and abs(y) > _S3_COORD_LIMIT:
            raise _invalid(f"replacement vertex {i} coordinate magnitude exceeds safety limit")
        try:
            fx, fy = float(x), float(y)
        except OverflowError:
            raise _invalid(f"replacement vertex {i} coordinate magnitude exceeds safety limit")
        if not (math.isfinite(fx) and math.isfinite(fy)):
            raise _invalid(f"replacement vertex {i} coordinates must be finite, got ({x!r}, {y!r})")
        if abs(fx) > _S3_COORD_LIMIT or abs(fy) > _S3_COORD_LIMIT:
            raise _invalid(f"replacement vertex {i} coordinate magnitude exceeds safety limit")
        pts.append((fx, fy))

    # Consecutive duplicate check
    for i in range(len(pts) - 1):
        if pts[i] == pts[i + 1]:
            raise _invalid(f"consecutive duplicate vertices forbidden at index {i}: {pts[i]}")

    # Implicit closure check (reject repeated terminal first vertex)
    if pts[0] == pts[-1]:
        raise _invalid("do not repeat the first vertex at the end; closure is implicit")

    # >= 3 distinct vertices
    if len(set(pts)) < 3:
        raise _invalid(f"replacement ring must have >= 3 distinct vertices, got {len(set(pts))}")

    # Out-of-field check: reject, not clip
    if cine is not None and slice_index is not None:
        if 0 <= slice_index < cine.n_slices:
            geom = cine.geometries[slice_index]
            rows, cols = geom.rows, geom.cols
            for i, (x, y) in enumerate(pts):
                if not (0.0 <= x <= float(cols) and 0.0 <= y <= float(rows)):
                    raise _invalid(
                        f"replacement vertex {i} ({x}, {y}) out of image "
                        f"field (0..{cols}, 0..{rows})"
                    )

    # Degenerate area check
    area = _signed_area(pts)
    if abs(area) < _AREA_EPSILON:
        raise _invalid(f"degenerate ring: signed area below epsilon ({area})")

    # Simple ring check (no self-intersection)
    try:
        _validate_no_self_intersection(pts)
    except ValueError as exc:
        raise _invalid(f"replacement ring self-intersection: {exc}") from exc

    # Winding normalization to CCW per existing convention
    normalized = _ring_winding_normalized(pts, ccw=True)
    return tuple((float(x), float(y)) for x, y in normalized)


def replace_contour(
    contour: Contour,
    points: Sequence[Any],
    *,
    cine: CineVolume | None = None,
) -> Contour:
    """Replace a contour's geometry with a validated, CCW-normalized simple ring.

    Adjudication contract C2/C3/C4:
    - Importer-family PolygonRing validation: simple ring, non-degenerate,
      no self-intersection, implicit closure, strict numeric pairs excluding
      booleans, capped vertex count, out-of-field rejection.
    - Preserves target's origin source ("ai" or "manual", never "edited"), structure,
      slice_index, frame_index, model metadata identity.
    - Sets edited=True, approved=False, increments revision.
    - Winding normalized to counter-clockwise (CCW).
    """
    normalized_points = validate_and_normalize_replacement_ring(
        points, cine=cine, slice_index=contour.slice_index
    )
    return _edited_copy(contour, list(normalized_points))


def copy_contour_across_frames(
    contour: Contour,
    target_frames: Sequence[Any],
) -> list[Contour]:
    """Duplicate one contour's geometry onto empty neighbor frames (copy, Phase 6).

    Returns one NEW edited :class:`Contour` per target frame: same
    ``slice_index`` + ``structure`` + point geometry (fresh copies via
    :func:`_copy_points`), ``frame_index`` set to the target, ``edited=True``,
    ``approved=False``, ``revision`` bumped by one, provenance bumped with
    ``created_at`` preserved and model deep-cloned (all inherited from
    :func:`_edited_copy`).

    This is a verbatim DUPLICATE — no interpolation between frames, no
    propagation of later edits, no smoothing. The physician reviews each copy
    and corrects it with the existing edit operations; QC/myo gates evaluate
    the copies exactly like any other contour.

    Fail-closed validation (code ``contour.invalid``):

    - ``target_frames`` must be a non-empty sequence of integers (bools
      rejected, matching the module's existing integer discipline);
    - every target must be ``>= 0``;
    - no duplicate targets;
    - no target may equal the source ``frame_index`` (a copy onto itself is
      not a copy).
    - COPY-R3-001: the source ring must satisfy
      :func:`validate_and_normalize_replacement_ring` (simple
      non-degenerate ring — the same gate `replace`/`create` enforce), so a
      degenerate/self-intersecting source refuses HERE with zero copies
      instead of propagating invalid geometry onto empty frames. The
      normalized (CCW) points are what gets duplicated.
    """
    if isinstance(target_frames, (str, bytes, Mapping)) or not isinstance(
        target_frames, Sequence
    ):
        raise _invalid(
            "copy target frames must be a non-empty sequence of frame indices, "
            f"got {type(target_frames).__name__}"
        )
    targets = list(target_frames)
    if not targets:
        raise _invalid("copy target frames must be non-empty")
    seen: set[int] = set()
    for position, target in enumerate(targets):
        if isinstance(target, bool) or not isinstance(target, int):
            raise _invalid(
                f"copy target frame {position} must be an integer, got {target!r}"
            )
        if target < 0:
            raise _invalid(
                f"copy target frame {position} must be >= 0, got {target}"
            )
        if target == contour.frame_index:
            raise _invalid(
                f"copy target frame {position} equals the source frame "
                f"{contour.frame_index}; a copy onto itself is not a copy"
            )
        if target in seen:
            raise _invalid(
                f"duplicate copy target frame {target} at position {position}"
            )
        seen.add(target)
    # COPY-R3-001: source-ring topology gate — a degenerate or
    # self-intersecting source refuses here (zero copies) instead of
    # persisting invalid geometry onto empty frames. Normalized CCW points
    # are duplicated (same convention as replace/create).
    normalized_source = list(
        validate_and_normalize_replacement_ring(contour.points)
    )
    copies: list[Contour] = []
    for target in targets:
        sibling = _edited_copy(contour, list(normalized_source))
        sibling.frame_index = target
        copies.append(sibling)
    return copies


def copy_contour_across_slices(
    contour: Contour,
    target_slices: Sequence[Any],
) -> list[Contour]:
    """Duplicate one contour's geometry onto empty neighbor slices (copy, Phase 6).

    S16-transpose of :func:`copy_contour_across_frames`: returns one NEW
    edited :class:`Contour` per target slice with the same ``frame_index`` +
    ``structure`` + CCW-normalized point geometry, ``slice_index`` set to the
    target, ``edited=True``, ``approved=False``, ``revision`` bumped by one,
    provenance bumped with ``created_at`` preserved and model deep-cloned
    (all inherited from :func:`_edited_copy`).

    Verbatim DUPLICATE — no interpolation between slices, no propagation of
    later edits, no smoothing. The physician reviews each copy and corrects
    it with the existing edit operations; QC/myo gates evaluate the copies
    exactly like any other contour.

    Fail-closed validation (code ``contour.invalid``): ``target_slices``
    must be a non-empty sequence of non-bool integers ``>= 0`` with no
    duplicates and never the source ``slice_index``; the source ring must
    satisfy :func:`validate_and_normalize_replacement_ring` (COPY-R3-001 —
    zero copies on refusal; the normalized CCW points are duplicated).
    """
    if isinstance(target_slices, (str, bytes, Mapping)) or not isinstance(
        target_slices, Sequence
    ):
        raise _invalid(
            "copy target slices must be a non-empty sequence of slice indices, "
            f"got {type(target_slices).__name__}"
        )
    targets = list(target_slices)
    if not targets:
        raise _invalid("copy target slices must be non-empty")
    seen: set[int] = set()
    for position, target in enumerate(targets):
        if isinstance(target, bool) or not isinstance(target, int):
            raise _invalid(
                f"copy target slice {position} must be an integer, got {target!r}"
            )
        if target < 0:
            raise _invalid(
                f"copy target slice {position} must be >= 0, got {target}"
            )
        if target == contour.slice_index:
            raise _invalid(
                f"copy target slice {position} equals the source slice "
                f"{contour.slice_index}; a copy onto itself is not a copy"
            )
        if target in seen:
            raise _invalid(
                f"duplicate copy target slice {target} at position {position}"
            )
        seen.add(target)
    normalized_source = list(
        validate_and_normalize_replacement_ring(contour.points)
    )
    copies: list[Contour] = []
    for target in targets:
        sibling = _edited_copy(contour, list(normalized_source))
        sibling.slice_index = target
        copies.append(sibling)
    return copies


# --- pure-data commands (one operation each) ----------------------------------


@dataclass(frozen=True)
class EditCommand:
    """Base class for pure-data descriptions of ONE contour edit operation."""


@dataclass(frozen=True)
class InsertVertex(EditCommand):
    """Insert a vertex by edge index (midpoint split) or pixel position.

    A ``list`` position is SNAPSHOT-normalized to an immutable ``(x, y)``
    tuple at construction (correction round 1, amendment N4-R1 E): the command
    stores its own validated copy, so later in-place mutation of the caller's
    list can never change what ``redo()`` replays.
    """

    index_or_position: int | tuple[float, float] | list[float]

    def __post_init__(self) -> None:
        # Pure-data rule (N4-R1 E): commands are immutable DATA. Retaining a
        # caller's mutable list BY REFERENCE let an in-place mutation between
        # execute() and redo() silently change the replayed edit (Terra
        # DEFECT-1). Snapshot list positions through the canonical validator;
        # malformed values raise contour.invalid here instead of at apply time.
        value = self.index_or_position
        if isinstance(value, list):
            object.__setattr__(self, "index_or_position", _position_pair(value))


@dataclass(frozen=True)
class DeleteVertex(EditCommand):
    """Delete the vertex at ``vertex_index``."""

    vertex_index: int


@dataclass(frozen=True)
class TranslateContour(EditCommand):
    """Translate every vertex by ``(dx, dy)`` pixels."""

    dx: float
    dy: float


@dataclass(frozen=True)
class ReplaceContour(EditCommand):
    """Replace the entire polygon ring with validated new vertices."""

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        normalized = validate_and_normalize_replacement_ring(self.points)
        object.__setattr__(self, "points", normalized)


@dataclass(frozen=True)
class EnsureClosed(EditCommand):
    """Close the polygon (append first point when open; see amendment G)."""


def apply_command(command: EditCommand, contour: Contour) -> Contour:
    """Apply one pure-data command to ``contour`` and return the NEW result."""
    if isinstance(command, InsertVertex):
        return insert_vertex(contour, command.index_or_position)
    if isinstance(command, DeleteVertex):
        return delete_vertex(contour, command.vertex_index)
    if isinstance(command, TranslateContour):
        return translate_contour(contour, command.dx, command.dy)
    if isinstance(command, ReplaceContour):
        return replace_contour(contour, command.points)
    if isinstance(command, EnsureClosed):
        return ensure_closed(contour)
    raise CMRError(INVALID_ARGUMENT, f"unknown edit command type: {type(command).__name__}")


class EditCommandStack:
    """In-memory snapshot-based undo/redo stack over one contour lineage.

    Undo entries are ``(command, before_snapshot)`` pairs (amendment N4-R1 E),
    where every snapshot is a fully detached deep copy owned by the stack
    (reconciliation correction): commands apply to stack-owned snapshots, and
    ``undo()``/``current`` never hand out a mutable alias of recorded history.
    Mutating a retained caller reference after a later ``execute()`` therefore
    cannot corrupt undo or redo history — restored states compare equal by
    canonical content, not by object identity. ``redo()`` re-applies the
    undone command to the current state through the same pure functions.
    Executing any command after an undo clears the redo tail (standard branch
    semantics). No UI wiring, no controller, no persistence calls live here.
    """

    def __init__(self, initial: Contour) -> None:
        # The stack owns its own deep snapshot of the starting state; a later
        # in-place mutation of the caller's object cannot rewrite history.
        self._initial = _snapshot_contour(initial)
        self._current = self._initial
        self._undo_entries: list[tuple[EditCommand, Contour]] = []
        self._redo_commands: list[EditCommand] = []

    # --- introspection ---

    @property
    def initial(self) -> Contour:
        """The starting contour the stack was constructed with."""
        return self._initial

    @property
    def current(self) -> Contour:
        """The contour produced by all applied (and not undone) commands."""
        return self._current

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_entries)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_commands)

    @property
    def undo_count(self) -> int:
        """Number of executed-and-not-undone commands."""
        return len(self._undo_entries)

    @property
    def redo_count(self) -> int:
        """Number of undone commands available for re-application."""
        return len(self._redo_commands)

    # --- operations ---

    def execute(self, command: EditCommand) -> Contour:
        """Apply ``command``, record the prior state, clear the redo tail."""
        # Snapshot at the history boundary: the entry stores a fully detached
        # copy, so later external mutation of any retained reference (including
        # the object this call returns) cannot rewrite the recorded past.
        before = _snapshot_contour(self._current)
        after = apply_command(command, before)
        self._undo_entries.append((command, before))
        self._redo_commands.clear()  # branch-after-undo slices the redo tail
        self._current = after
        return after

    def undo(self) -> Contour:
        """Revert to the recorded prior state and return it.

        The returned snapshot is the popped history entry itself: it is no
        longer part of the undo tail, so caller-side mutation of it cannot
        poison remaining history; the next ``execute()`` re-snapshots anyway.
        """
        if not self._undo_entries:
            raise CMRError(INVALID_ARGUMENT, "nothing to undo")
        command, before = self._undo_entries.pop()
        self._redo_commands.append(command)
        self._current = before
        return before

    def redo(self) -> Contour:
        """Re-apply the most recently undone command to the current state."""
        if not self._redo_commands:
            raise CMRError(INVALID_ARGUMENT, "nothing to redo")
        command = self._redo_commands.pop()
        before = _snapshot_contour(self._current)
        after = apply_command(command, before)
        self._undo_entries.append((command, before))
        self._current = after
        return after


# --- persistence composition ---------------------------------------------------


def save_to_revision(
    parent: Prediction | ContourRevision,
    *,
    series_uid: str,
    edits: Mapping[Structure, Contour],
    editor: str,
    created_at: str,
    software_version: str = VERSION,
    qc_report: QCReport | None = None,
) -> ContourRevision:
    """Compose edited contours into a NEW child :class:`ContourRevision`.

    - **Merge** (amendment N4-R1 D): starts from ``dict(parent.contours)`` and
      overlays ``edits`` so every UNEDITED structure persists verbatim in the
      serialized revision; editing a structure absent from the parent fails
      closed instead of silently adding structures.
    - **Mapping-key coherence** (reconciliation correction): every ``edits``
      key must equal its contour's own ``structure``; a contour stored under a
      foreign key corrupts canonical state and fails closed.
    - **Approval fork** (owner policy FORK_UNAPPROVED_DERIVED_REVISION_V1):
      every changed contour produced by an edit operation is unapproved, and
      an explicitly supplied edited contour that still carries
      ``approved=True`` is rejected fail-closed. The parent record and its
      approval state remain immutable; the derived revision links to it and
      must pass a fresh approval lifecycle.
    - **Provenance** (amendment N4-R1 A): built explicitly with ALL fields;
      ``created_at`` is the caller-supplied value (fixed timestamps in tests),
      ``revision`` is ``parent.provenance.revision + 1`` (re-checked by
      :meth:`ContourRevision.from_parent`) and ``model`` is ``None`` because a
      physician edit is not an AI result.
    - **Monotonic version / parent links**: delegated to
      :meth:`ContourRevision.from_parent` (version 1 from a
      :class:`Prediction`, ``parent.version + 1`` from a revision).
    - The parent :class:`Prediction`/:class:`ContourRevision` objects are never
      mutated.
    """
    merged: dict[Structure, Contour] = dict(parent.contours)
    for structure, contour in edits.items():
        if structure != contour.structure:
            raise _invalid(
                f"edits key {structure.value!r} does not match contour "
                f"structure {contour.structure.value!r}; mapping keys must "
                "equal Contour.structure"
            )
        if contour.approved:
            raise _invalid(
                "edited contour for structure "
                f"{structure.value!r} still has approved=True; changed content "
                "must be unapproved (FORK_UNAPPROVED_DERIVED_REVISION_V1) and "
                "pass a new approval lifecycle"
            )
        if structure not in parent.contours:
            known = ", ".join(sorted(item.value for item in parent.contours))
            raise _invalid(
                f"cannot edit unknown structure {structure.value!r}; "
                f"parent contains [{known}]"
            )
        merged[structure] = contour
    provenance = Provenance(
        study_uid=parent.provenance.study_uid,
        series_uid=series_uid,
        software_version=software_version,
        model=None,
        created_at=created_at,
        revision=parent.provenance.revision + 1,
        # S2C-001 cluster A: the manifest-bound workflow identity propagates
        # through every derived revision so one lineage stays one workflow.
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


def _create_remove_provenance(
    parent: Prediction | ContourRevision,
    *,
    series_uid: str,
    created_at: str,
    software_version: str,
) -> Provenance:
    """Explicit caller-supplied provenance for create/remove revisions.

    Same contract as :func:`save_to_revision` (all fields explicit,
    ``model=None`` because a physician edit is not an AI result, manifest
    workflow identity propagated).
    """
    return Provenance(
        study_uid=parent.provenance.study_uid,
        series_uid=series_uid,
        software_version=software_version,
        model=None,
        created_at=created_at,
        revision=parent.provenance.revision + 1,
        workflow_id=parent.provenance.workflow_id,
    )


def compose_create_revision(
    parent: Prediction | ContourRevision,
    *,
    slice_index: int,
    frame_index: int,
    structure: Structure,
    points: Sequence[Any],
    editor: str,
    created_at: str,
    series_uid: str,
    software_version: str = VERSION,
    qc_report: QCReport | None = None,
    cine: CineVolume | None = None,
) -> ContourRevision:
    """Compose a NEW child revision carrying one physician-created contour.

    Create-path composition (CMR_CREATE_A_001, R2 C1–C2): deliberately
    DISTINCT from :func:`save_to_revision` so the unknown-structure check
    there is never relaxed. The created contour is ``source="manual"``,
    ``edited=True``, ``approved=False`` with ``model=None`` provenance; the
    ring passes the existing replacement-ring validator (strict pairs,
    simple non-degenerate ring, in-field rejection when ``cine`` is
    supplied). Effective-key absence (the key must be genuinely missing from
    the chain, not merely absent from ``parent.contours``) is the
    controller's eligibility business and is NOT re-decided here. The
    revision carries the durable ``(edit_op="create", edit_key)`` identity;
    the parent record is never mutated.
    """
    for name, value in (("slice_index", slice_index), ("frame_index", frame_index)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise _invalid(f"create {name} must be an integer, got {value!r}")
        if value < 0:
            raise _invalid(f"create {name} must be non-negative, got {value!r}")
    if not isinstance(structure, Structure):
        raise _invalid(
            f"create structure must be a Structure member, got {structure!r}"
        )
    normalized = validate_and_normalize_replacement_ring(
        points, cine=cine, slice_index=slice_index
    )
    contour = Contour(
        slice_index=slice_index,
        frame_index=frame_index,
        structure=structure,
        points=list(normalized),
        source="manual",
        edited=True,
        approved=False,
        revision=0,
        model=None,
        provenance=None,
    )
    merged: dict[Structure, Contour] = dict(parent.contours)
    merged[structure] = contour
    provenance = _create_remove_provenance(
        parent,
        series_uid=series_uid,
        created_at=created_at,
        software_version=software_version,
    )
    return ContourRevision.from_parent(
        parent,
        series_uid=series_uid,
        contours=merged,
        editor=editor,
        created_at=created_at,
        provenance=provenance,
        qc_report=qc_report,
        edit_op=EDIT_OP_CREATE,
        edit_key=(slice_index, frame_index, structure),
    )


def compose_remove_revision(
    parent: Prediction | ContourRevision,
    *,
    slice_index: int,
    frame_index: int,
    structure: Structure,
    inverse_of: str,
    editor: str,
    created_at: str,
    series_uid: str,
    software_version: str = VERSION,
    qc_report: QCReport | None = None,
) -> ContourRevision:
    """Compose a NEW parent-linked TOMBSTONE revision removing one contour.

    Remove-path composition (CMR_CREATE_A_001, R2 C1): the returned revision
    omits exactly the ``(slice, frame, structure)`` key from its contour map
    AND carries the explicit ``(edit_op="remove", edit_key, inverse_of)``
    identity, so the omission is distinguishable from a missing snapshot
    entry. Inverse authority (immediate-head single-key create by the same
    physician) is the controller's business and is NOT re-decided here. The
    removal creates no approval and carries no fabricated contour; the
    parent record is never mutated.
    """
    for name, value in (("slice_index", slice_index), ("frame_index", frame_index)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise _invalid(f"remove {name} must be an integer, got {value!r}")
        if value < 0:
            raise _invalid(f"remove {name} must be non-negative, got {value!r}")
    if not isinstance(structure, Structure):
        raise _invalid(
            f"remove structure must be a Structure member, got {structure!r}"
        )
    if not isinstance(inverse_of, str) or not inverse_of:
        raise _invalid("remove inverse_of must be a non-empty revision identity")
    current = parent.contours.get(structure) if isinstance(parent.contours, Mapping) else None
    if (
        current is None
        or current.slice_index != slice_index
        or current.frame_index != frame_index
    ):
        raise _invalid(
            f"cannot remove ({slice_index}, {frame_index}, {structure.value}): "
            "the immediate parent head does not carry that contour"
        )
    merged: dict[Structure, Contour] = {
        map_structure: contour
        for map_structure, contour in parent.contours.items()
        if not (
            map_structure == structure
            and contour.slice_index == slice_index
            and contour.frame_index == frame_index
        )
    }
    provenance = _create_remove_provenance(
        parent,
        series_uid=series_uid,
        created_at=created_at,
        software_version=software_version,
    )
    return ContourRevision.from_parent(
        parent,
        series_uid=series_uid,
        contours=merged,
        editor=editor,
        created_at=created_at,
        provenance=provenance,
        qc_report=qc_report,
        edit_op=EDIT_OP_REMOVE,
        edit_key=(slice_index, frame_index, structure),
        inverse_of=inverse_of,
    )
