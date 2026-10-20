"""Closed workstation action intents and state projections."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from cmr_contours.editing import (
    delete_vertex,
    insert_vertex,
    replace_contour,
    translate_contour,
    validate_and_normalize_replacement_ring,
)
from cmr_contours.model import Contour
from cmr_core.cine import CineVolume
from cmr_core.domain_types import (
    HoleAwareContour,
    Measurement,
    SelectedPhasePrediction,
    WholeCinePrediction,
    effective_chain_contour,
    head_tombstoned_keys,
)
from cmr_core.errors import CONTOUR_INVALID, INVALID_ARGUMENT, CMRError
from cmr_core.provenance import VERSION, Provenance
from cmr_core.result_reference import ApprovedResultReference
from cmr_core.segmentation import Structure
from cmr_persistence.records import (
    TopologyDisposition,
    TopologyDispositionRecord,
    applicable_topology_dispositions,
)
from cmr_persistence.store import SqliteCmrStore, StaleHeadError
from cmr_quantification.measurements import validate_complete_grid_entries
from cmr_quantification.reference_ranges import ReferenceSex
from cmr_viewer.state import (
    DEFAULT_FPS,
    INCLUSION_GRID_GROUPS,
    NUMBER_LEX_RE,
    DisplayWindowLevel,
    InclusionToggle,
    ViewerState,
    all_included_toggles,
    canonical_number_lex,
    canonical_visible_structures_json,
    toggles_of_inclusion_set,
    window_level_json,
)
from cmr_workflow.controller import (
    WorkflowController,
    materialize_legacy_chain_from_whole_cine,
    project_persisted_whole_cine_to_legacy_draft,
)
from cmr_workflow.coverage import SELECTED_PHASE_GROUPS, resolve_selected_phase_coverage
from cmr_workflow.topology_disposition import (
    assert_topology_resolved_for_approval,
    disposition_role_for_frame,
    resolution_projection,
)

from .jobs import _typed_diagnostic, enqueue_selected_phase_recompute


def _verified_x1_checkpoints(root: Path | None) -> tuple[str, str, str, str]:
    """Torch-free pinned-digest verification of the two X1 files (A5/A8).

    Returns the pinned RELATIVE artifact names (NOT absolute host paths)
    plus both digests. The absolute weights root is local runtime config:
    it is NEVER persisted, NEVER returned by the API, NEVER logged by this
    surface. Size + sha256 of BOTH files are verified BEFORE enqueue (fail
    closed); the execution boundary re-verifies (files may change while
    queued). No torch import happens here.
    """
    if root is None:
        raise CMRError("x1_weights_unavailable", "pinned X1 weights are not configured")
    try:
        from inference.providers.baai.artifacts import (
            STAGE1_PATH,
            STAGE2_PATH,
            verify_both,
        )
        stage1 = root / STAGE1_PATH
        stage2 = root / STAGE2_PATH
        if not stage1.is_file() or not stage2.is_file():
            raise ValueError("one or more pinned X1 checkpoints are missing")
        verify_both({STAGE1_PATH: stage1.read_bytes(), STAGE2_PATH: stage2.read_bytes()})
    except Exception as exc:
        if isinstance(exc, CMRError) and exc.code == "x1_weights_unavailable":
            raise
        raise CMRError(
            "x1_weights_unavailable", "pinned X1 weights failed verification"
        ) from exc
    from inference.providers.baai.artifacts import STAGE1_SHA256, STAGE2_SHA256
    return STAGE1_PATH, STAGE2_PATH, STAGE1_SHA256, STAGE2_SHA256

LOGGER = logging.getLogger(__name__)


__all__ = [
    "ActionConflict",
    "ActionIntent",
    "InvalidPayload",
    "ViewIntent",
    "_best_effort_analysis_jobs",
    "_best_effort_combined_jobs",
    "apply_checked_action",
    "all_included_wire",
    "build_topology_resolution_projection",
    "build_viewer_state",
    "display_query_params",
    "reject_duplicate_json_keys",
    "parse_action_intent",
    "parse_visible_structures",
    "parse_window_level",
    "reject_non_canonical_encoding",
    "resolve_study_contours",
    "state_projection",
]

#: The exact accepted ``Structure.value`` strings in frozen enum order.
STRUCTURE_VALUES: tuple[str, ...] = tuple(structure.value for structure in Structure)

#: Slice 2 (C-WIRE) finite module-action alias table: ``function.<name>``
#: normalizes to the canonical action once at ``parse_action_intent`` entry.
#: Canonical names stay valid; unknown/dotted actions pass through and are
#: rejected by the closed grammar below. No envelope change, no new action.
MODULE_ACTION_ALIASES: dict[str, str] = {
    "function.select_ed_es": "select_ed_es",
    "function.apply_inclusion_and_refresh": "apply_inclusion_and_refresh",
    "function.record_applicability_decision": "record_applicability_decision",
    "function.record_topology_disposition": "record_topology_disposition",
    "function.adopt_selected_phase_ai": "adopt_selected_phase_ai",
    "function.approve": "approve",
    "function.cancel_analysis_job": "cancel_analysis_job",
}


def reject_non_grammar_number_text(raw: str, name: str) -> None:
    """Reject exponent/sign spellings in the RAW JSON text (§3.2 negatives).

    A number's VALUE cannot reveal its spelling: ``1e2``, ``1E2`` and ``100``
    all parse to the same legal ``100.0``, and ``-0.0`` parses to ``-0.0``
    whose canonical form is ``-0``. The frozen grammar rejects those
    SPELLINGS, so the canonical-bytes check must run on the raw JSON text of
    the incoming param, before/independently of ``json.loads``.

    The scan finds every JSON number token and requires each to match the
    grammar. Structure names are quoted strings and are never matched.
    """
    for match in _JSON_NUMBER_TOKEN_RE.finditer(raw):
        if not NUMBER_LEX_RE.fullmatch(match.group(0)):
            raise _invalid(
                f"{name} must use the frozen decimal number grammar (no exponent "
                f"forms, no -0.0, no leading zeros, no trailing '.'), got "
                f"{match.group(0)!r}"
            )


#: One JSON number token, including an optional exponent/sign spelling.
_JSON_NUMBER_TOKEN_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def reject_duplicate_json_keys(raw: str, name: str) -> None:
    """Reject duplicate JSON OBJECT KEYS in the raw param text (§3.2 negative).

    ``json.loads`` silently keeps the LAST occurrence of a duplicated key, so
    ``{"center":40,"width":400,"center":50}`` parses to ``center=50`` and the
    duplicate is UNOBSERVABLE once the value is a Python object — the existing
    :func:`reject_duplicate_keys` therefore cannot see it. The frozen canonical
    grammar admits exactly one spelling of each key, so the check must run on
    the RAW text, exactly like :func:`reject_non_grammar_number_text`.

    Keys are collected with a string/escape-aware scan so a duplicated key
    inside a JSON string value can never be mistaken for an object key.
    """
    depth_stack: list[set[str]] = []
    index = 0
    length = len(raw)
    while index < length:
        char = raw[index]
        if char == '"':
            end = index + 1
            while end < length:
                if raw[end] == "\\":
                    end += 2
                    continue
                if raw[end] == '"':
                    break
                end += 1
            if end >= length:
                raise _invalid(f"{name} is not valid JSON")
            token = raw[index + 1 : end]
            after = raw[end + 1 :].lstrip()
            # A key is a string immediately followed by ':' at this depth.
            if after.startswith(":") and depth_stack:
                keys = depth_stack[-1]
                if token in keys:
                    raise _invalid(
                        f"{name} must not contain duplicate JSON object keys, got "
                        f"{token!r} twice"
                    )
                keys.add(token)
            index = end + 1
            continue
        if char == "{":
            depth_stack.append(set())
        elif char == "}":
            if depth_stack:
                depth_stack.pop()
        index += 1


def reject_non_canonical_encoding(raw_query: str, name: str) -> None:
    """Reject a query string that is not EXACTLY canonically encoded (§3.2).

    §3.2 step 3 freezes ONE encoding for the wire — ``urllib.parse.quote(s,
    safe="")`` (Python) / ``encodeURIComponent(s)`` (JS), "every byte
    encoded" — and the §3.2 negative list names "unencoded or
    partially-encoded params" and "malformed percent/UTF-8" as 400s.
    ``parse_qs`` DECODES before anyone can look: both
    ``window_level={"center":40,"width":400}`` and
    ``window_level=%7B%22center%22%3A40%2C%22width%22%3A400%7D`` reach the
    parser as the identical Python string, so the spelling is UNOBSERVABLE
    after decoding — exactly the situation that forces
    :func:`reject_non_grammar_number_text` and
    :func:`reject_duplicate_json_keys` to run on raw text. This check is the
    encoding counterpart and must run on the RAW query, before ``parse_qs``.

    Canonical means, for every ``&``-separated field:

      * a literal (unencoded) parameter NAME — the frozen encoder emits
        parameter names as-is, so an encoded name (``window%5Flevel=``) is a
        second spelling of the same parameter and is rejected;
      * a value that survives the round trip ``quote(unquote(v, safe=""),
        safe="") == v`` byte-for-byte.

    That single round-trip rule is exactly "every byte percent-encoded in
    upper-case canonical form" and therefore rejects: unencoded JSON;
    partially-encoded JSON (encoded braces with literal commas/quotes);
    lower-case escapes (``%7b``); redundant escapes (``n%75ll``); ``+`` for a
    space (the frozen encoder emits ``%20``); bare ``%`` and truncated
    escapes (malformed percent); and a field with no ``=``. A double-encoded
    value is a WELL-FORMED encoding, so it is deliberately NOT rejected here
    — it decodes to non-JSON text and stays a 400 at the JSON layer, which is
    what the §5.7 fix (b34486d/53b5f64) already establishes.

    An EMPTY query string carries no field and is accepted (omission is the
    canonical default for every view parameter).
    """
    if not raw_query:
        return
    if "\x00" in raw_query:
        raise _invalid(f"{name} contains a NUL byte")
    for position, field in enumerate(raw_query.split("&")):
        if not field:
            raise _invalid(f"{name} field {position} is empty")
        if "=" not in field:
            raise _invalid(f"{name} field {position} has no '=' separator")
        raw_key, _, raw_value = field.partition("=")
        if not raw_key:
            raise _invalid(f"{name} field {position} has an empty parameter name")
        if raw_key != unquote(raw_key):
            raise _invalid(
                f"{name} parameter name {raw_key!r} must be unencoded, exactly as "
                f"the frozen encoder emits it"
            )
        try:
            decoded = unquote(raw_value, errors="strict")
        except UnicodeDecodeError as exc:
            raise _invalid(
                f"{name} parameter {raw_key!r} is not valid percent-encoded UTF-8"
            ) from exc
        if quote(decoded, safe="") != raw_value:
            raise _invalid(
                f"{name} parameter {raw_key!r} is not canonically percent-encoded "
                f"(§3.2 step 3: quote(s, safe='') / encodeURIComponent — every byte "
                f"encoded, upper-case escapes, no partial or redundant escaping, "
                f"no '+' for a space)"
            )


def parse_window_level(value: Any) -> DisplayWindowLevel | None:
    """Parse one ``window_level`` view value (S3-008B §3.1/§3.2).

    Accepts ``None`` (the default/omission spelling) or an object with
    EXACTLY the keys ``center`` and ``width`` carrying actual JSON numbers.
    Everything else — duplicate keys, unsorted keys, exponent/sign spellings,
    strings where numbers belong, NaN/inf, JSON ``null`` inside the object,
    non-object values — fails closed with :class:`InvalidPayload`, which the
    HTTP layer maps to 400.

    Booleans are rejected (an actual JSON number is required) and the domain
    type performs the remaining finite/width validation (§2.1).
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _invalid("view.window_level must be an object with center and width")
    if set(value) != {"center", "width"}:
        raise _invalid("view.window_level must contain exactly center and width")
    center = value["center"]
    width = value["width"]
    for name, number in (("center", center), ("width", width)):
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise _invalid(f"view.window_level.{name} must be an actual number")
        token = canonical_number_lex(number)
        if not NUMBER_LEX_RE.fullmatch(token):
            # §3.2 grammar: no exponent forms (1e2/1E2), no "-0.0", no
            # trailing ".". Python's json accepts several spellings the
            # frozen grammar excludes, so check the LEXICAL FORM, not just
            # the numeric value.
            raise _invalid(
                f"view.window_level.{name} must use the frozen decimal number "
                f"grammar (no exponent forms, no -0.0), got {token!r}"
            )
    return DisplayWindowLevel(center, width)


def parse_visible_structures(value: Any) -> tuple[Structure, ...] | None:
    """Parse one ``visible_structures`` view value (S3-008B §3.1/§3.2).

    Accepts ``None`` (ALL structures), ``[]`` (no contours) or an array of
    unique known ``Structure.value`` strings in EXACT enum-definition order.
    Unknown values, duplicates, out-of-order arrays, non-arrays, non-string
    entries and nested ``null`` fail closed with :class:`InvalidPayload`.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise _invalid("view.visible_structures must be an array of structure names")
    resolved: list[Structure] = []
    for entry in value:
        if not isinstance(entry, str):
            raise _invalid("view.visible_structures entries must be strings")
        try:
            structure = Structure(entry)
        except ValueError as exc:
            known = ", ".join(STRUCTURE_VALUES)
            raise _invalid(
                f"view.visible_structures contains an unknown structure {entry!r}; "
                f"known structures: {known}"
            ) from exc
        if structure in resolved:
            raise _invalid(
                "view.visible_structures must not contain duplicate structures"
            )
        resolved.append(structure)
    # §2.1/§3.2: exact enum-definition order. Map each RESOLVED member (in the
    # order it arrived) to its enum index and require a strictly ascending
    # sequence — an out-of-order array therefore fails closed. (Iterating the
    # enum and filtering would always produce a sorted list and prove nothing.)
    order = [list(Structure).index(member) for member in resolved]
    if order != sorted(order):
        raise _invalid(
            "view.visible_structures must be in Structure enum definition order: "
            + ", ".join(STRUCTURE_VALUES)
        )
    return tuple(resolved)


def display_query_params(
    window_level: DisplayWindowLevel | None,
    visible_structures: tuple[Structure, ...] | None,
) -> dict[str, str]:
    """Canonical query params for the two display inputs (S3-008B §3.2).

    Omission is the canonical default, so ``None`` contributes NO parameter.
    An explicit value is the canonical compact sorted-key JSON rendered with
    the shared numeric lexical grammar. These bytes are the cine CACHE-KEY
    input: one distinct canonical query ⇒ one distinct body ⇒ one distinct
    exact-body ETag (§3.1).
    """
    params: dict[str, str] = {}
    if window_level is not None:
        params["window_level"] = window_level_json(window_level)
    if visible_structures is not None:
        params["visible_structures"] = canonical_visible_structures_json(
            visible_structures
        )
    return params


class InvalidPayload(ValueError):
    """The HTTP envelope is malformed or outside the closed wire schema."""


class ActionConflict(Exception):
    """Raised when an action cannot be applied due to conflicting state (HTTP 409)."""

    def __init__(self, message: str, *, error: str = "conflict") -> None:
        super().__init__(message)
        self.message = message
        self.error = error


@dataclass(frozen=True)
class ViewIntent:
    slice_index: int
    frame_index: int
    fps: int
    playing: bool
    reference_sex: ReferenceSex | None
    inclusion_draft: tuple[dict[str, Any], ...] | None
    # --- S3-008B display-only view context (§3.1) -------------------------
    #: Request-scoped window/level, carried exactly like ``inclusion_draft``.
    #: NEVER an action field, NEVER persisted, NEVER clinical truth.
    window_level: DisplayWindowLevel | None = None
    #: Request-scoped contour visibility filter (None = ALL structures).
    visible_structures: tuple[Structure, ...] | None = None


@dataclass(frozen=True)
class ActionIntent:
    action: str
    params: Mapping[str, Any]
    client_nonce: str
    view: ViewIntent
    expected_head_revision_id: str | None
    expected_selection_id: str | None
    expected_inclusion_set_id: str | None
    expected_whole_cine_head_id: str | None = None
    legacy_clinical_path: bool = False


def _invalid(message: str) -> InvalidPayload:
    return InvalidPayload(message)


def _int_field(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(f"{name} must be an integer")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise _invalid(f"{name} must be an actual Boolean")
    return value


def reject_duplicate_keys(value: Any, name: str) -> None:
    """Reject duplicate JSON object keys anywhere in a display value (§3.2).

    ``json.loads`` silently keeps the LAST occurrence of a duplicated key, so
    ``{"center":1,"width":2,"center":3}`` would parse as ``center=3`` while
    remaining a SECOND spelling of one logical value. The frozen canonical
    grammar admits exactly one key order with no duplicates, so duplicates
    (and, by the same rule, any object whose keys are not already in canonical
    sorted order) fail closed before any domain value is built.
    """
    if isinstance(value, Mapping):
        keys = [str(key) for key in value]
        if len(set(keys)) != len(keys):
            raise _invalid(f"{name} must not contain duplicate JSON object keys")
        if keys != sorted(keys):
            raise _invalid(f"{name} object keys must be in canonical sorted order")
        for key, item in value.items():
            reject_duplicate_keys(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_duplicate_keys(item, f"{name}[{index}]")


def _identity(mapping: Mapping[str, Any], name: str) -> str | None:
    if name not in mapping:
        raise _invalid(f"lineage.{name} is required")
    value = mapping[name]
    if value is not None and (not isinstance(value, str) or not value):
        raise _invalid(f"lineage.{name} must be a non-empty string or null")
    return value


def _canonical_grid(rows: list[dict[str, Any]], n_slices: int) -> tuple[dict[str, Any], ...]:
    try:
        covered = validate_complete_grid_entries(rows)
    except (TypeError, KeyError, ValueError) as exc:
        raise _invalid(f"inclusion_draft is not a complete 5xN grid: {exc}") from exc
    if covered != n_slices:
        raise _invalid(
            f"inclusion_draft covers {covered} slices but the cine has {n_slices}"
        )
    expected_keys = {"slice_index", "frame_role", "structure", "included"}
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise _invalid(
                "each inclusion_draft entry must contain exactly slice_index, "
                "frame_role, structure and included"
            )
        normalized.append(
            {
                "slice_index": row["slice_index"],
                "frame_role": row["frame_role"],
                "structure": row["structure"],
                "included": row["included"],
            }
        )
    group_order = {group: index for index, group in enumerate(INCLUSION_GRID_GROUPS)}
    normalized.sort(
        key=lambda row: (
            int(row["slice_index"]),
            group_order[(str(row["frame_role"]), str(row["structure"]))],
        )
    )
    return tuple(normalized)


def parse_view_intent(
    value: Mapping[str, Any],
    cine: CineVolume,
    *,
    require_all: bool = True,
    allow_display: bool = True,
) -> ViewIntent:
    if not isinstance(value, Mapping):
        raise _invalid("view must be an object")
    required = {"slice", "frame", "fps", "playing", "reference_sex", "inclusion_draft"}
    if require_all and not required.issubset(value):
        missing = sorted(required - set(value))
        raise _invalid(f"view is missing required fields: {', '.join(missing)}")
    # S3-008B §3.2 negatives: duplicate/unsorted JSON object keys are rejected
    # for the display params BEFORE any domain validation. json.loads keeps the
    # LAST duplicate, which would let two spellings of one cache key collide;
    # the frozen grammar admits exactly one canonical key order.
    for name in ("window_level", "visible_structures"):
        if name in value:
            reject_duplicate_keys(value[name], name)
    if not allow_display and (
        "window_level" in value or "visible_structures" in value
    ):
        # S3-008B §3.1: display inputs are NOT action fields and are NOT
        # parsed by the action envelope. Fail closed rather than silently
        # ignore them, so a client can never mistake a display choice for an
        # accepted mutation input.
        raise _invalid(
            "view inside an action envelope must not carry display fields "
            "window_level/visible_structures (display-only, never action input)"
        )
    slice_index = _int_field(value, "slice") if "slice" in value else 0
    frame_index = _int_field(value, "frame") if "frame" in value else 0
    fps = _int_field(value, "fps") if "fps" in value else DEFAULT_FPS
    playing = _strict_bool(value.get("playing", False), "view.playing")
    if not 0 <= slice_index < cine.n_slices:
        raise _invalid("view.slice is outside the cine stack")
    if not 0 <= frame_index < cine.n_frames:
        raise _invalid("view.frame is outside the cine frame range")
    if not 1 <= fps <= 60:
        raise _invalid("view.fps must be between 1 and 60")
    reference = value.get("reference_sex")
    if reference is not None:
        if reference == ReferenceSex.MEN.value:
            reference_sex = ReferenceSex.MEN
        elif reference == ReferenceSex.WOMEN.value:
            reference_sex = ReferenceSex.WOMEN
        else:
            raise _invalid("view.reference_sex must be men, women or null")
    else:
        reference_sex = None
    draft_value = value.get("inclusion_draft")
    if draft_value is None:
        # ``None`` means that this stateless request has no client-owned
        # draft yet. ``build_viewer_state`` resolves it to the persisted
        # applicable set (or the frozen all-included default); an explicit
        # array remains the only way to submit a client draft.
        draft = None
    elif isinstance(draft_value, list):
        rows = draft_value
        draft = _canonical_grid(rows, cine.n_slices)
    else:
        raise _invalid("view.inclusion_draft must be an array")
    # S3-008B §3.1: the two display inputs are OPTIONAL view context, parsed
    # exactly like inclusion_draft. Absent means default (None). They are
    # never action fields and never reach persistence.
    window_level = (
        parse_window_level(value["window_level"]) if "window_level" in value else None
    )
    visible_structures = (
        parse_visible_structures(value["visible_structures"])
        if "visible_structures" in value
        else None
    )
    return ViewIntent(
        slice_index,
        frame_index,
        fps,
        playing,
        reference_sex,
        draft,
        window_level,
        visible_structures,
    )


def parse_action_intent(
    value: Any,
    cine: CineVolume,
    *,
    legacy_clinical_path: bool = False,
) -> ActionIntent:
    if not isinstance(value, Mapping):
        raise _invalid("action envelope must be an object")
    required = {"action", "params", "client_nonce", "view", "lineage"}
    if set(value) != required:
        raise _invalid(
            "action envelope must contain exactly action, params, client_nonce, "
            "view and lineage"
        )
    action = value["action"]
    if isinstance(action, str):
        action = MODULE_ACTION_ALIASES.get(action, action)
    if not isinstance(action, str) or not action:
        raise _invalid("action must be a non-empty string")
    params = value["params"]
    if not isinstance(params, Mapping):
        raise _invalid("params must be an object")
    nonce = value["client_nonce"]
    if not isinstance(nonce, str) or not nonce:
        raise _invalid("client_nonce must be a non-empty string")
    lineage = value["lineage"]
    if not isinstance(lineage, Mapping):
        raise _invalid("lineage must be an object")
    valid_lineages = (
        {"head_revision_id", "selection_id", "inclusion_set_id"},
        {"head_revision_id", "whole_cine_head_id", "selection_id", "inclusion_set_id"},
    )
    if set(lineage) not in valid_lineages:
        raise _invalid(
            "lineage must contain the expected identities "
            "(head_revision_id, selection_id, inclusion_set_id and optional whole_cine_head_id)"
        )
    intent = ActionIntent(
        action=action,
        params=params,
        client_nonce=nonce,
        view=parse_view_intent(value["view"], cine, allow_display=False),
        expected_head_revision_id=_identity(lineage, "head_revision_id"),
        expected_selection_id=_identity(lineage, "selection_id"),
        expected_inclusion_set_id=_identity(lineage, "inclusion_set_id"),
        expected_whole_cine_head_id=(
            _identity(lineage, "whole_cine_head_id") if "whole_cine_head_id" in lineage else None
        ),
        legacy_clinical_path=legacy_clinical_path,
    )
    _validate_action_params(intent, cine=cine)
    return intent


def _validate_action_params(intent: ActionIntent, cine: CineVolume | None = None) -> None:
    if intent.action == "select_ed_es":
        if set(intent.params) != {"ed_frame", "es_frame"}:
            raise _invalid("select_ed_es params must contain exactly ed_frame and es_frame")
        _int_field(intent.params, "ed_frame")
        _int_field(intent.params, "es_frame")
    elif intent.action == "apply_inclusion_and_refresh":
        if intent.params:
            raise _invalid("apply_inclusion_and_refresh params must be empty")
        if intent.view.inclusion_draft is None:
            raise _invalid(
                "apply_inclusion_and_refresh requires a complete inclusion_draft array"
            )
    elif intent.action == "record_applicability_decision":
        # FUNCV1 physician applicability loop C1 (P1-1): decision-persist
        # WITHOUT recompute. Same complete-grid draft requirement as the
        # compound refresh (the draft rides the view, params stay empty),
        # plus the displayed-phase binding (P1-2): params MUST carry the
        # exact displayed viewport frame the physician confirmed
        # (``displayed_frame``), so the server can refuse a submit whose
        # role frame disagrees with what was on screen. ED==ES keeps role
        # isolation: the ROLE (not the frame alone) selects the key, so the
        # shared frame still binds role-exactly.
        if set(intent.params) != {"displayed_frame", "frame_role"}:
            raise _invalid(
                "record_applicability_decision params must contain exactly "
                "displayed_frame and frame_role"
            )
        _int_field(intent.params, "displayed_frame")
        if intent.params["frame_role"] not in ("ED", "ES"):
            raise _invalid("frame_role must be 'ED' or 'ES'")
        if intent.view.inclusion_draft is None:
            raise _invalid(
                "record_applicability_decision requires a complete inclusion_draft array"
            )
        if cine is not None and not (
            0 <= int(intent.params["displayed_frame"]) < cine.n_frames
        ):
            raise _invalid(
                f"displayed_frame {intent.params['displayed_frame']} out of cine "
                f"frame range (0..{cine.n_frames - 1})"
            )
    elif intent.action == "save_contour_edits":
        _validate_contour_edit_params(intent, cine=cine)
    elif intent.action == "bootstrap_legacy_chain":
        if set(intent.params) - {"ed_frame"}:
            raise _invalid("bootstrap_legacy_chain params may only contain optional ed_frame")
        if "ed_frame" in intent.params:
            ed = _int_field(intent.params, "ed_frame")
            if cine is not None and not (0 <= ed < cine.n_frames):
                raise _invalid(f"ed_frame {ed} out of cine frame range (0..{cine.n_frames - 1})")
    elif intent.action == "cancel_analysis_job":
        if set(intent.params) != {"job_id"} or not isinstance(intent.params["job_id"], str):
            raise _invalid("cancel_analysis_job params must contain exactly job_id")
    elif intent.action == "request_selected_phase_ai":
        if intent.params:
            raise _invalid("request_selected_phase_ai params must be empty")
    elif intent.action == "adopt_selected_phase_ai":
        if intent.params:
            raise _invalid("adopt_selected_phase_ai params must be empty")
    elif intent.action == "approve":
        if intent.params:
            raise _invalid("approve action takes no params; params must be empty")
    elif intent.action == "record_topology_disposition":
        _validate_topology_disposition_params(intent, cine=cine)
    else:
        raise _invalid(f"unknown action: {intent.action}")


#: Closed disposition vocabulary (Astra contract item 4).
TOPOLOGY_DISPOSITION_VALUES: tuple[str, ...] = tuple(
    disposition.value for disposition in TopologyDisposition
)

#: The exact accepted key set of ONE record_topology_disposition params object.
_TOPOLOGY_DISPOSITION_PARAMS_KEYS: frozenset[str] = frozenset(
    {"slice_index", "frame_index", "structure", "disposition", "reason"}
)

def _validate_topology_disposition_params(
    intent: ActionIntent, cine: CineVolume | None = None
) -> None:
    """Strict grammar for ``record_topology_disposition``.

    Mirrors :data:`CONTOUR_EDIT_COMMANDS`-style validation: an EXACT key set,
    strict types, closed enums, and a non-empty reason. A disposition for a
    key with NO persisted block (nothing to resolve) is rejected at apply
    time, where the store transaction also rejects a stale head/selection.
    """
    params = intent.params
    if set(params) != _TOPOLOGY_DISPOSITION_PARAMS_KEYS:
        raise _invalid(
            "record_topology_disposition params must contain exactly "
            f"{sorted(_TOPOLOGY_DISPOSITION_PARAMS_KEYS)}"
        )
    slice_index = _int_field(params, "slice_index")
    frame_index = _int_field(params, "frame_index")
    if slice_index < 0 or frame_index < 0:
        raise _invalid("slice_index and frame_index must be non-negative")
    if cine is not None:
        if slice_index >= cine.n_slices:
            raise _invalid(
                f"slice_index {slice_index} out of cine range (0..{cine.n_slices - 1})"
            )
        if frame_index >= cine.n_frames:
            raise _invalid(
                f"frame_index {frame_index} out of cine range (0..{cine.n_frames - 1})"
            )
    structure = params["structure"]
    if structure not in STRUCTURE_VALUES:
        raise _invalid(f"structure must be one of {list(STRUCTURE_VALUES)}")
    disposition = params["disposition"]
    if disposition not in TOPOLOGY_DISPOSITION_VALUES:
        raise _invalid(
            f"disposition must be one of {list(TOPOLOGY_DISPOSITION_VALUES)}, "
            f"got {disposition!r}"
        )
    reason = params["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise _invalid("reason must be a non-empty string")

# --- S3-009 D7: the checked contour-save web action -----------------------------
#
# The frozen wire grammar for ONE contour-save action. The edit commands are
# the Node 4 (cmr_contours.editing) operations spelled as closed JSON objects;
# the action applies them to the persisted WHOLE-CINE record and appends ONE
# append-only ``whole_cine_revisions`` row (edited_keys + supersedes
# lineage). The viewer-facing ContourRevision chain (overlay/approval)
# stays owned by the existing controller flows.

#: The closed edit-command vocabulary (cmr_contours.editing, Node 4).
#: ``create``/``remove`` are the bounded legacy-only single-key operations
#: (CMR_CREATE_A_001, R2 C1–C2): exact key sets below, exactly one command
#: per request when either is present.
CONTOUR_EDIT_COMMANDS: tuple[str, ...] = (
    "translate",
    "insert",
    "delete",
    "replace",
    "create",
    "remove",
    "copy",
    "copyslice",
)

#: The exact accepted key set of one edit-command object per command kind.
_EDIT_COMMAND_KEYS: dict[str, frozenset[str]] = {
    "translate": frozenset({"kind", "slice_index", "frame_index", "structure", "dx", "dy"}),
    "insert": frozenset(
        {"kind", "slice_index", "frame_index", "structure", "index_or_position"}
    ),
    "delete": frozenset(
        {"kind", "slice_index", "frame_index", "structure", "vertex_index"}
    ),
    "replace": frozenset(
        {"kind", "slice_index", "frame_index", "structure", "points"}
    ),
    "create": frozenset(
        {"kind", "slice_index", "frame_index", "structure", "points"}
    ),
    "remove": frozenset({"kind", "slice_index", "frame_index", "structure"}),
    "copy": frozenset(
        {
            "kind",
            "src_slice_index",
            "src_frame_index",
            "dst_frame_indices",
            "structure",
        }
    ),
    "copyslice": frozenset(
        {
            "kind",
            "src_slice_index",
            "src_frame_index",
            "dst_slice_indices",
            "structure",
        }
    ),
}


def _edit_target(
    mapping: Mapping[str, Any], cine: CineVolume | None = None
) -> tuple[int, int, Structure]:
    """Resolve and validate one edit command's (slice, frame, structure)."""
    slice_index = mapping.get("slice_index")
    frame_index = mapping.get("frame_index")
    structure_value = mapping.get("structure")
    for name, value in (("slice_index", slice_index), ("frame_index", frame_index)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise _invalid(f"contour edit {name} must be an integer")
    if not isinstance(structure_value, str):
        raise _invalid("contour edit structure must be a structure name string")
    # The two index guards above REJECT a missing/non-integer index before it
    # can reach ``int()``, so this is the type-narrowing the checker needs —
    # not a second validation path (unreachable at runtime).
    if slice_index is None or frame_index is None:  # pragma: no cover
        raise _invalid("contour edit slice_index and frame_index are required")
    s_idx = int(slice_index)
    f_idx = int(frame_index)
    if cine is not None:
        if not (0 <= s_idx < cine.n_slices and 0 <= f_idx < cine.n_frames):
            raise _invalid(
                f"contour edit target ({s_idx}, {f_idx}) out of cine grid bounds "
                f"(n_slices={cine.n_slices}, n_frames={cine.n_frames})"
            )
    try:
        structure = Structure(structure_value)
    except ValueError as exc:
        known = ", ".join(STRUCTURE_VALUES)
        raise _invalid(
            f"contour edit structure {structure_value!r} is unknown; known: {known}"
        ) from exc
    return s_idx, f_idx, structure


def _validate_copy_command(
    edit: Mapping[str, Any], position: int, cine: CineVolume | None = None
) -> None:
    """Fail-closed validation of one ``copy`` wire command.

    Exact keys ``{kind, src_slice_index, src_frame_index, dst_frame_indices,
    structure}`` (checked by the caller). ``structure`` uses the same enum
    vocabulary as every other edit kind; indices are non-bool integers;
    ``dst_frame_indices`` is a non-empty list of non-bool integers with no
    duplicates and never the source frame. Cine grid bounds apply when a
    cine is supplied. Copy never overwrites — destination occupancy is
    refused at apply time (zero rows), where the lineage head is known.
    """
    src_slice = edit.get("src_slice_index")
    src_frame = edit.get("src_frame_index")
    for name, value in (("src_slice_index", src_slice), ("src_frame_index", src_frame)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise _invalid(f"contour edit {position} {name} must be an integer")
    structure_value = edit.get("structure")
    if not isinstance(structure_value, str):
        raise _invalid("contour edit structure must be a structure name string")
    try:
        Structure(structure_value)
    except ValueError as exc:
        raise _invalid(
            f"contour edit structure {structure_value!r} is unknown; known: "
            + ", ".join(STRUCTURE_VALUES)
        ) from exc
    dst = edit.get("dst_frame_indices")
    if not isinstance(dst, list) or not dst:
        raise _invalid(
            f"contour edit {position} dst_frame_indices must be a non-empty list"
        )
    seen: set[int] = set()
    for i, frame in enumerate(dst):
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise _invalid(
                f"contour edit {position} dst_frame_indices[{i}] must be an integer"
            )
        if frame < 0:
            raise _invalid(
                f"contour edit {position} dst_frame_indices[{i}] must be >= 0"
            )
        if frame == src_frame:
            raise _invalid(
                f"contour edit {position} dst_frame_indices[{i}] equals the "
                "source frame; a copy onto itself is not a copy"
            )
        if frame in seen:
            raise _invalid(
                f"contour edit {position} duplicate dst_frame_indices entry {frame}"
            )
        seen.add(frame)
    if cine is not None:
        if not (0 <= int(src_slice) < cine.n_slices):
            raise _invalid(
                f"contour edit {position} src_slice_index {src_slice} out of cine "
                f"slice range (0..{cine.n_slices - 1})"
            )
        if not (0 <= int(src_frame) < cine.n_frames):
            raise _invalid(
                f"contour edit {position} src_frame_index {src_frame} out of cine "
                f"frame range (0..{cine.n_frames - 1})"
            )
        for frame in seen:
            if not (0 <= frame < cine.n_frames):
                raise _invalid(
                    f"contour edit {position} dst frame {frame} out of cine "
                    f"frame range (0..{cine.n_frames - 1})"
                )


def _validate_copyslice_command(
    edit: Mapping[str, Any], position: int, cine: CineVolume | None = None
) -> None:
    """Fail-closed validation of one ``copyslice`` wire command.

    S16-transpose of :func:`_validate_copy_command`: exact keys
    ``{kind, src_slice_index, src_frame_index, dst_slice_indices,
    structure}`` (checked by the caller); ``dst_slice_indices`` is a
    non-empty list of non-bool integers ``>= 0`` with no duplicates and
    never the source slice. Cine grid bounds apply to source slice/frame
    and every destination slice when a cine is supplied. Copy never
    overwrites — destination occupancy is refused at apply time (zero
    rows), where the lineage head is known.
    """
    src_slice = edit.get("src_slice_index")
    src_frame = edit.get("src_frame_index")
    for name, value in (("src_slice_index", src_slice), ("src_frame_index", src_frame)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise _invalid(f"contour edit {position} {name} must be an integer")
    structure_value = edit.get("structure")
    if not isinstance(structure_value, str):
        raise _invalid("contour edit structure must be a structure name string")
    try:
        Structure(structure_value)
    except ValueError as exc:
        raise _invalid(
            f"contour edit structure {structure_value!r} is unknown; known: "
            + ", ".join(STRUCTURE_VALUES)
        ) from exc
    dst = edit.get("dst_slice_indices")
    if not isinstance(dst, list) or not dst:
        raise _invalid(
            f"contour edit {position} dst_slice_indices must be a non-empty list"
        )
    seen: set[int] = set()
    for i, sl in enumerate(dst):
        if isinstance(sl, bool) or not isinstance(sl, int):
            raise _invalid(
                f"contour edit {position} dst_slice_indices[{i}] must be an integer"
            )
        if sl < 0:
            raise _invalid(
                f"contour edit {position} dst_slice_indices[{i}] must be >= 0"
            )
        if sl == src_slice:
            raise _invalid(
                f"contour edit {position} dst_slice_indices[{i}] equals the "
                "source slice; a copy onto itself is not a copy"
            )
        if sl in seen:
            raise _invalid(
                f"contour edit {position} duplicate dst_slice_indices entry {sl}"
            )
        seen.add(sl)
    if cine is not None:
        if not (0 <= int(src_slice) < cine.n_slices):
            raise _invalid(
                f"contour edit {position} src_slice_index {src_slice} out of cine "
                f"slice range (0..{cine.n_slices - 1})"
            )
        if not (0 <= int(src_frame) < cine.n_frames):
            raise _invalid(
                f"contour edit {position} src_frame_index {src_frame} out of cine "
                f"frame range (0..{cine.n_frames - 1})"
            )
        for sl in seen:
            if not (0 <= sl < cine.n_slices):
                raise _invalid(
                    f"contour edit {position} dst slice {sl} out of cine "
                    f"slice range (0..{cine.n_slices - 1})"
                )


def _validate_contour_edit_params(
    intent: ActionIntent, cine: CineVolume | None = None
) -> None:
    """Fail-closed validation of the ``save_contour_edits`` wire grammar."""
    if set(intent.params) != {"edits"}:
        raise _invalid("save_contour_edits params must contain exactly edits")
    edits = intent.params["edits"]
    if not isinstance(edits, list) or not edits:
        raise _invalid("save_contour_edits requires a non-empty edits array")
    seen_targets: set[tuple[int, int, Structure]] = set()
    seen_planes: set[tuple[int, int]] = set()
    for position, edit in enumerate(edits):
        if not isinstance(edit, Mapping):
            raise _invalid(f"contour edit {position} must be an object")
        kind = edit.get("kind")
        if kind not in CONTOUR_EDIT_COMMANDS:
            raise _invalid(
                f"contour edit {position} kind must be one of "
                f"{CONTOUR_EDIT_COMMANDS}, got {kind!r}"
            )
        if set(edit) != _EDIT_COMMAND_KEYS[str(kind)]:
            raise _invalid(
                f"contour edit {position} ({kind}) must contain exactly "
                f"{sorted(_EDIT_COMMAND_KEYS[str(kind)])}"
            )
        if kind == "copy":
            _validate_copy_command(edit, position, cine=cine)
            continue
        if kind == "copyslice":
            _validate_copyslice_command(edit, position, cine=cine)
            continue
        target = _edit_target(edit, cine=cine)
        seen_planes.add((target[0], target[1]))
        if target in seen_targets:
            raise _invalid(
                "save_contour_edits applies at most ONE command per "
                f"(slice, frame, structure); duplicate target {target!r}"
            )
        seen_targets.add(target)
        if kind == "translate":
            for name in ("dx", "dy"):
                value = edit[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise _invalid(
                        f"contour edit {position} {name} must be a real number"
                    )
        elif kind == "insert":
            value = edit["index_or_position"]
            valid = (
                isinstance(value, int)
                and not isinstance(value, bool)
                or (
                    isinstance(value, list)
                    and len(value) == 2
                    and all(
                        isinstance(component, (int, float))
                        and not isinstance(component, bool)
                        for component in value
                    )
                )
            )
            if not valid:
                raise _invalid(
                    f"contour edit {position} index_or_position must be an integer "
                    "edge index or a finite [x, y] pair"
                )
        elif kind == "delete":
            value = edit["vertex_index"]
            if isinstance(value, bool) or not isinstance(value, int):
                raise _invalid(
                    f"contour edit {position} vertex_index must be an integer"
                )
        elif kind == "replace" or kind == "create":
            points = edit.get("points")
            if not isinstance(points, list):
                raise _invalid(
                    f"contour edit {position} points must be a list of coordinates"
                )
            try:
                validate_and_normalize_replacement_ring(
                    points, cine=cine, slice_index=target[0]
                )
            except CMRError as exc:
                raise _invalid(
                    f"contour edit {position} replacement points invalid: {exc.message}"
                ) from exc
    if len(seen_planes) > 1:
        raise _invalid("all edit commands must target the same active plane")
    if any(
        isinstance(edit, Mapping) and edit.get("kind") in ("create", "remove", "copy", "copyslice")
        for edit in edits
    ):
        # R2 C1 (+ copy/copyslice discipline): a request containing a
        # create/remove/copy/copyslice command contains exactly one command.
        # Mixed requests and batch create/remove/copy/copyslice are refused;
        # the indices/structures of that single command were validated above.
        if len(edits) != 1:
            raise _invalid(
                "save_contour_edits with a create/remove/copy/copyslice command must "
                "contain exactly one command; mixed requests and batch "
                "create/remove/copy/copyslice are refused"
            )


def all_included_wire(n_slices: int) -> list[dict[str, Any]]:
    return [
        {
            "slice_index": toggle.slice_index,
            "frame_role": toggle.role,
            "structure": toggle.structure,
            "included": True,
        }
        for toggle in all_included_toggles(n_slices)
    ]


def _wire_from_toggles(toggles: tuple[InclusionToggle, ...]) -> list[dict[str, Any]]:
    return [
        {
            "slice_index": toggle.slice_index,
            "frame_role": toggle.role,
            "structure": toggle.structure,
            "included": toggle.included,
        }
        for toggle in toggles
    ]


def build_viewer_state(
    controller: WorkflowController,
    store: SqliteCmrStore,
    series_uid: str,
    cine: CineVolume,
    view: ViewIntent | None = None,
    *,
    inclusion_draft: tuple[dict[str, Any], ...] | None = None,
    data: Mapping[str, Any] | None = None,
) -> tuple[ViewerState, dict[str, Any]]:
    reopened = dict(data) if data is not None else controller.reopen(store, series_uid, cine=cine)
    chain = tuple(reopened["chain"])
    if view is None:
        draft_toggles = (
            toggles_of_inclusion_set(reopened["applicable_inclusion_set"])
            if reopened.get("applicable_inclusion_set") is not None
            else all_included_toggles(cine.n_slices)
        )
        state = ViewIntent(
            0, 0, DEFAULT_FPS, False, None, tuple(_wire_from_toggles(draft_toggles))
        )
    else:
        state = view
    if inclusion_draft is not None:
        rows = inclusion_draft
    elif state.inclusion_draft is not None:
        rows = state.inclusion_draft
    else:
        persisted = reopened.get("applicable_inclusion_set")
        draft_toggles = (
            toggles_of_inclusion_set(persisted)
            if persisted is not None
            else all_included_toggles(cine.n_slices)
        )
        rows = tuple(_wire_from_toggles(draft_toggles))
    toggles = tuple(
        InclusionToggle(
            int(row["slice_index"]),
            str(row["frame_role"]),
            str(row["structure"]),
            row["included"],
        )
        for row in rows
    )
    if "whole_cine_head_id" in reopened:
        whole_cine_head_id = reopened["whole_cine_head_id"]
    else:
        from cmr_core.domain_types import WholeCinePrediction

        wc_head = store.load_latest_whole_cine_head(series_uid)
        whole_cine_head_id = (
            None
            if wc_head is None
            else (
                wc_head.prediction_id
                if isinstance(wc_head, WholeCinePrediction)
                else wc_head.revision_id
            )
        )
    viewer_state = ViewerState(
        series_uid=series_uid,
        cine=cine,
        current_slice_index=state.slice_index,
        current_frame_index=state.frame_index,
        selection=reopened["latest_selection"],
        latest_measurement=reopened["latest_measurement"],
        qc_report=chain[-1].qc_report if chain else None,
        approvals=tuple(reopened["approvals"]),
        qc_decisions=tuple(reopened["qc_decisions"]),
        revision_chain=chain,
        playing=state.playing,
        fps=state.fps,
        inclusion_draft=toggles,
        persisted_inclusion_set=reopened.get("applicable_inclusion_set"),
        reference_sex=state.reference_sex,
        # S3-008B §3.1: display context flows into the two trailing ViewerState
        # fields ONLY. It is display-only: never persisted, never clinical.
        window_level=state.window_level,
        visible_structures=state.visible_structures,
        whole_cine_head_id=whole_cine_head_id,
        # Astra F1-F5: durable dispositions are surfaced for display/audit so
        # the viewer can show resolved vs unresolved status per key.
        # R1 review F2 (P1): the append-only table can hold several rows for
        # one key (one per head/selection binding); the viewer speaks about
        # the CURRENT state, so it shows the APPLICABLE set — exactly the
        # projection the approval relevance predicate resolves by and the
        # sealed digest binds.
        topology_dispositions=tuple(applicable_topology_dispositions(
            store.load_topology_dispositions(series_uid),
            head_revision_id=(chain[-1].revision_id if chain else ""),
            selection_id=(
                reopened["latest_selection"].selection_id
                if reopened.get("latest_selection") is not None
                else ""
            ),
        )),
        # Viewer disposition controls v4 (additive): precomputed resolutions
        # for the CURRENT head + selection via the SAME predicate the approve
        # action uses. Fail-closed relevance (no selection => all blocking)
        # is handled inside ``resolve_topology``. An empty tuple means no
        # persisted blocks (or a legacy construction); the wire projection
        # renders an empty key list in that case.
        topology_resolutions=tuple(controller.resolve_topology(
            store,
            series_uid,
            selection=reopened.get("latest_selection"),
            head_revision_id=(chain[-1].revision_id if chain else ""),
            cine=cine,
        )),
    )
    return viewer_state, reopened


def _measurement_projection(measurement: Measurement | None) -> dict[str, Any] | None:
    if measurement is None:
        return None
    names = (
        "edv_ml",
        "esv_ml",
        "sv_ml",
        "ef_pct",
        "lv_mass_g",
        "rv_edv_ml",
        "rv_esv_ml",
        "rv_sv_ml",
        "rv_ef_pct",
    )
    return {
        name: {
            "value": float(getattr(measurement, name).value),
            "unit": getattr(measurement, name).unit.value,
        }
        for name in names
    }


#: Viewer disposition controls v4 — read-only per-action refusal vocabulary.
#: Reuses the refusal predicates of ``_apply_topology_disposition`` as
#: READ-ONLY checks (never executes writes).
_TOPOLOGY_UNAVAILABLE_REASONS: tuple[str, ...] = (
    "selection_required",
    "non_selected_phase",
    "chain_missing",
    "replaced_without_geometry",
    "excluded_with_geometry",
)

#: Independent physician-disposition fact (never conflated with resolver kind).
_DISPOSITION_STATUS_VALUES: tuple[str, ...] = (
    "recorded_replaced",
    "recorded_excluded",
    "not_recorded",
)


def _chain_carries_key(
    state: ViewerState,
    *,
    slice_index: int,
    frame_index: int,
    structure: str,
) -> bool:
    """Whether the CURRENT revision chain carries geometry for the key.

    Read-only mirror of the ``carried`` scan in
    :func:`_apply_topology_disposition` (which scans
    ``store.load_chain``; the state's chain comes from the same rows via
    ``reopen``). No geometry bytes leave this predicate — only presence.
    """
    try:
        wanted = Structure(structure)
    except ValueError:
        return False
    for revision in state.revision_chain:
        for contour in revision.contours.values():
            if (
                contour.slice_index == slice_index
                and contour.frame_index == frame_index
                and contour.structure is wanted
            ):
                return True
    return False


def _record_eligibility(
    state: ViewerState,
    *,
    slice_index: int,
    frame_index: int,
    structure: str,
    disposition: str,
) -> tuple[bool, str | None]:
    """Read-only ``can_record`` + ``unavailable_reason`` for one action+key.

    Order: ``selection_required`` → ``chain_missing`` →
    ``non_selected_phase`` → geometry checks. The same predicates the write
    path refuses on; the write path would refuse on either phase or geometry,
    so this order is conservative and deterministic when both apply.
    """
    if state.selection is None:
        return (False, "selection_required")
    if state.head_revision_id is None:
        return (False, "chain_missing")
    try:
        disposition_role_for_frame(
            frame_index=frame_index,
            ed_frame=state.selection.ed_frame,
            es_frame=state.selection.es_frame,
        )
    except CMRError:
        return (False, "non_selected_phase")
    carried = _chain_carries_key(
        state,
        slice_index=slice_index,
        frame_index=frame_index,
        structure=structure,
    )
    if disposition == TopologyDisposition.REPLACED.value and not carried:
        return (False, "replaced_without_geometry")
    if disposition == TopologyDisposition.EXCLUDED.value and carried:
        return (False, "excluded_with_geometry")
    return (True, None)


def _approval_eligibility(
    state: ViewerState,
    resolutions: list[Any],
    *,
    series_uid: str,
) -> dict[str, Any]:
    """Read-only approval eligibility mirroring the FULL approve predicate.

    Order and messages match the ``approve`` branch of
    :func:`apply_checked_action` verbatim: topology resolution →
    selection presence → stale-analysis/measurement → QC verdict. Lineage
    CAS mismatches are per-request (not state eligibility) and are not part
    of this projection. Returns ``{can_approve, refusal_code, refusal}``.
    """
    head_id = state.head_revision_id or ""
    selection_id = (
        state.selection.selection_id if state.selection is not None else ""
    )
    if resolutions:
        try:
            assert_topology_resolved_for_approval(
                resolutions,
                series_uid=series_uid,
                head_revision_id=head_id,
                selection_id=selection_id,
            )
        except CMRError as exc:
            return {
                "can_approve": False,
                "refusal_code": "topology_blocked",
                "refusal": f"topology block active: {exc.message}",
            }
    if state.selection is None:
        return {
            "can_approve": False,
            "refusal_code": "no_selection",
            "refusal": "no selection",
        }
    if state.analysis_is_stale or state.latest_measurement is None:
        return {
            "can_approve": False,
            "refusal_code": "stale_analysis",
            "refusal": "refresh required before approval",
        }
    if state.qc_report is None:
        return {
            "can_approve": False,
            "refusal_code": "missing_qc_report",
            "refusal": "qc_report is missing",
        }
    verdict = state.qc_report.verdict
    if verdict != "PASS":
        return {
            "can_approve": False,
            "refusal_code": "qc_verdict_rejected",
            "refusal": f"approval requires PASS verdict, got {verdict}",
        }
    return {"can_approve": True, "refusal_code": None, "refusal": None}


def build_topology_resolution_projection(
    state: ViewerState,
    resolutions: list[Any],
) -> dict[str, Any]:
    """Additive server projection of topology resolution + disposition facts.

    Wires the EXISTING :func:`resolution_projection` per persisted blocked
    key (all five kinds) and adds two INDEPENDENT facts the viewer renders
    separately: ``disposition_status`` (from the APPLICABLE set only — never
    conflated with the resolver kind) and per-action ``eligibility``. No
    geometry bytes are emitted; history stays separately retrievable via the
    existing store loaders.
    """
    applicable_by_key: dict[tuple[int, int, str], Any] = {}
    for record in state.topology_dispositions:
        applicable_by_key[
            (record.slice_index, record.frame_index, record.structure)
        ] = record
    head_id = state.head_revision_id
    selection_id = (
        state.selection.selection_id if state.selection is not None else None
    )
    keys: list[dict[str, Any]] = []
    for resolution in resolutions:
        base = resolution_projection(resolution)
        key = (
            resolution.block.slice_index,
            resolution.block.frame_index,
            resolution.block.structure,
        )
        applicable = applicable_by_key.get(key)
        if applicable is None:
            disposition_status = "not_recorded"
            applicable_payload: dict[str, Any] | None = None
        elif applicable.disposition is TopologyDisposition.REPLACED:
            disposition_status = "recorded_replaced"
            applicable_payload = {
                "disposition_id": applicable.disposition_id,
                "disposition": applicable.disposition.value,
                "physician": applicable.physician,
                "reason": applicable.reason,
                "source_revision_id": applicable.source_revision_id,
                "selection_id": applicable.selection_id,
            }
        else:
            disposition_status = "recorded_excluded"
            applicable_payload = {
                "disposition_id": applicable.disposition_id,
                "disposition": applicable.disposition.value,
                "physician": applicable.physician,
                "reason": applicable.reason,
                "source_revision_id": applicable.source_revision_id,
                "selection_id": applicable.selection_id,
            }
        replaced_can, replaced_reason = _record_eligibility(
            state,
            slice_index=resolution.block.slice_index,
            frame_index=resolution.block.frame_index,
            structure=resolution.block.structure,
            disposition=TopologyDisposition.REPLACED.value,
        )
        excluded_can, excluded_reason = _record_eligibility(
            state,
            slice_index=resolution.block.slice_index,
            frame_index=resolution.block.frame_index,
            structure=resolution.block.structure,
            disposition=TopologyDisposition.EXCLUDED.value,
        )
        entry: dict[str, Any] = dict(base)
        entry["head_revision_id"] = head_id
        entry["selection_id"] = selection_id
        entry["disposition_status"] = disposition_status
        entry["applicable_disposition"] = applicable_payload
        entry["eligibility"] = {
            "replaced": {
                "can_record": replaced_can,
                "unavailable_reason": replaced_reason,
            },
            "excluded": {
                "can_record": excluded_can,
                "unavailable_reason": excluded_reason,
            },
        }
        keys.append(entry)
    keys.sort(
        key=lambda item: (
            int(item["slice_index"]),
            int(item["frame_index"]),
            str(item["structure"]),
        )
    )
    return {
        "head_revision_id": head_id,
        "selection_id": selection_id,
        "keys": keys,
        "approval_eligibility": _approval_eligibility(
            state, resolutions, series_uid=state.series_uid
        ),
    }


def _best_effort_analysis_jobs(
    store: SqliteCmrStore, series_uid: str
) -> tuple[Any, ...]:
    """Read optional job metadata without weakening a clinical response.

    Covers BOTH job tables (A6): analysis recompute rows plus inference
    rows, so the counts/analysis_state projection is server-owned over the
    merged feed. The webapp stays kind-agnostic and arithmetic-free.
    """
    return _best_effort_combined_jobs(store, series_uid)


def _best_effort_combined_jobs(
    store: SqliteCmrStore, series_uid: str
) -> tuple[Any, ...]:
    """Read the merged series-scoped job feed (A6: BOTH tables)."""
    try:
        analysis = tuple(store.load_analysis_jobs(series_uid))
    except Exception as exc:
        LOGGER.warning(
            "analysis-job projection read failed: %s",
            _typed_diagnostic(exc, "analysis_job_projection_failed"),
        )
        analysis = ()
    try:
        inference = tuple(store.load_inference_jobs(series_uid))
    except Exception as exc:
        LOGGER.warning(
            "inference-job projection read failed: %s",
            _typed_diagnostic(exc, "inference_job_projection_failed"),
        )
        inference = ()
    merged = (*analysis, *inference)
    return tuple(sorted(merged, key=lambda job: (job.created_at, job.job_id)))


def state_projection(
    state: ViewerState,
    *,
    include_manifest: bool = False,
    analysis_jobs: tuple[Any, ...] | None = None,
    analysis: Any | None = None,
    analysis_stale_reason: str | None = None,
) -> dict[str, Any]:
    """Serialize a projection only; no new clinical computation is performed.

    ``analysis_jobs`` is the merged series-scoped feed over BOTH job tables
    (A6); counts + ``analysis_state`` are server-owned over both kinds so
    the webapp stays kind-agnostic and arithmetic-free.

    ``analysis`` is the CURRENT coherent
    :class:`~cmr_persistence.records.SelectedPhaseAnalysisRecord` (F14) as
    resolved by ``reopen``/``load_current_analysis``; ``None`` means no
    current analysis resolves and ``analysis_stale_reason`` carries the stable
    reason token. The additive trailing ``analysis`` object is the ONE place
    the wire exposes which head/selection/inclusion/measurement/QC binding the
    displayed metrics belong to.
    """
    head = state.head_revision
    measurement = state.latest_measurement
    qc = state.qc_report
    selection = state.selection
    persisted = state.persisted_inclusion_set
    base: dict[str, Any] = {
        "series_uid": state.series_uid,
        "slice": state.current_slice_index,
        "frame": state.current_frame_index,
        "fps": state.fps,
        "playing": state.playing,
        "reference_sex": None if state.reference_sex is None else state.reference_sex.value,
        "selection": (
            None
            if selection is None
            else {
                "ed_frame": selection.ed_frame,
                "es_frame": selection.es_frame,
                "selection_id": selection.selection_id,
            }
        ),
        "head_revision_id": state.head_revision_id,
        "whole_cine_head_id": state.whole_cine_head_id,
        "selection_id": None if selection is None else selection.selection_id,
        "inclusion_set_id": None if persisted is None else persisted.inclusion_set_id,
        "measurement_id": None if measurement is None else measurement.measurement_id,
        "qc_digest": None if qc is None else _qc_digest(qc),
        "qc_verdict": None if qc is None else qc.verdict,
        "qc_flags": []
        if qc is None
        else [
            {
                "code": flag.code,
                "message": flag.message,
                "source": flag.source,
                "severity": flag.severity.value,
            }
            for flag in qc.flags
        ],
        "measurement_summary": _measurement_projection(measurement),
        "approvals": [
            {
                "approval_id": snapshot.approval_id,
                "source_revision_id": snapshot.source_revision_id,
                "measurement_id": snapshot.measurement_id,
                "approved_by": snapshot.approved_by,
                "approved_at": snapshot.approved_at,
                "qc_verdict": snapshot.qc_verdict,
            }
            for snapshot in state.approvals
        ],
        "approval_label": state.approval_label,
        "analysis_is_stale": state.analysis_is_stale,
        "inclusion_grid": [
            {"frame_role": role, "structure": structure}
            for role, structure in INCLUSION_GRID_GROUPS
        ],
        "inclusion_draft": _wire_from_toggles(state.inclusion_draft),
        "inclusion_status": state.inclusion_status_label,
        "workflow_id": None if state.cine.workflow_id is None else state.cine.workflow_id,
        "study_uid": state.cine.study_uid,
        "manifest_digest": (
            None
            if state.cine.source_manifest is None
            else state.cine.source_manifest.workflow_id
        ),
        "frames": state.cine.n_frames,
        "slices": state.cine.n_slices,
        "modality": "MR",
        "head_version": None if head is None else head.version,
        # --- S3-008B §3.1: EXACTLY two new FLAT keys, appended LAST -------
        # No key is renamed or reordered; no 'view' object and no manifest
        # section exists in the accepted projection and none is invented.
        "window_level": (
            None
            if state.window_level is None
            else {
                "center": state.window_level.center,
                "width": state.window_level.width,
            }
        ),
        "visible_structures": (
            None
            if state.visible_structures is None
            else [structure.value for structure in state.visible_structures]
        ),
        # Astra F1-F5 (additive, appended LAST): durable per-key topology
        # dispositions for display/audit. Never clinical geometry.
        "topology_dispositions": [
            {
                "disposition_id": record.disposition_id,
                "slice_index": record.slice_index,
                "frame_index": record.frame_index,
                "structure": record.structure,
                "disposition": record.disposition.value,
                "physician": record.physician,
                "reason": record.reason,
                "created_at": record.created_at,
                "source_revision_id": record.source_revision_id,
                "selection_id": record.selection_id,
            }
            for record in state.topology_dispositions
        ],
        # Viewer disposition controls v4 (additive, trailing): server
        # projection of resolver kind + independent disposition fact +
        # per-action eligibility + approval eligibility. Wires the EXISTING
        # ``resolution_projection()`` per persisted blocked key; legacy
        # consumers ignore the unknown trailing key. No geometry bytes.
        "topology_resolution": build_topology_resolution_projection(
            state, list(state.topology_resolutions)
        ),
        # F14 (CMR_FUNCTION_V1_ANALYSIS_001, additive trailing): the ONE
        # coherent selected-phase analysis authority the displayed metrics and
        # QC resolve to. ``None`` means no current analysis resolves; the
        # stable ``analysis_stale_reason`` token says why (no_head,
        # no_selection, head_qc_missing, no_analysis, not_current).
        "analysis": (
            None
            if analysis is None
            else {
                "analysis_id": analysis.analysis_id,
                "series_uid": analysis.series_uid,
                "study_uid": analysis.study_uid,
                "workflow_id": analysis.workflow_id,
                "head_revision_id": analysis.head_revision_id,
                "selection_id": analysis.selection_id,
                "ed_frame": analysis.ed_frame,
                "es_frame": analysis.es_frame,
                "inclusion_set_id": analysis.inclusion_set_id,
                "measurement_id": analysis.measurement_id,
                "qc_report_digest": analysis.qc_report_digest,
                "qc_verdict": analysis.qc_verdict,
                "algorithm_version": analysis.algorithm_version,
                "software_version": analysis.software_version,
                "range_table_version": analysis.range_table_version,
                "created_at": analysis.provenance.created_at,
            }
        ),
        "analysis_stale_reason": (
            analysis_stale_reason if analysis is None else None
        ),
    }
    if analysis_jobs is not None:
        base["analysis_jobs"] = {
            state_name: sum(1 for job in analysis_jobs if job.state == state_name)
            for state_name in ("queued", "running", "succeeded", "failed", "cancelled")
        }
        base["analysis_state"] = (
            "updating"
            if any(job.state in {"queued", "running"} for job in analysis_jobs)
            else ("stale" if state.analysis_is_stale else "current")
        )
    if not include_manifest:
        return base

    geom = state.cine.geometries[state.current_slice_index]

    def _frame_sort_key(item: tuple[int, int, Any]) -> tuple[bool, float, int, int]:
        s_idx, orig_idx, frame = item
        tt = frame.trigger_time_ms
        return (tt is None, float(tt) if tt is not None else 0.0, s_idx, orig_idx)

    all_frames: list[tuple[int, int, Any]] = [
        (s_idx, orig_f_idx, frame)
        for s_idx, slice_frames in enumerate(state.cine.frames)
        for orig_f_idx, frame in enumerate(slice_frames)
    ]
    all_frames_sorted = sorted(all_frames, key=_frame_sort_key)

    canonical_phase_vector = [
        float(frame.trigger_time_ms)
        for _, _, frame in all_frames_sorted
        if frame.trigger_time_ms is not None
    ]

    nominal_dt = (1000.0 / state.fps) if state.fps > 0 else 100.0
    if len(canonical_phase_vector) > 1:
        frame_time_vector = []
        for i in range(len(canonical_phase_vector) - 1):
            dt = canonical_phase_vector[i + 1] - canonical_phase_vector[i]
            frame_time_vector.append(float(dt) if dt > 0 else nominal_dt)
        mean_dt = (
            sum(frame_time_vector) / len(frame_time_vector)
            if frame_time_vector
            else nominal_dt
        )
        frame_time_vector.append(float(mean_dt))
    else:
        frame_time_vector = [nominal_dt] * len(all_frames_sorted)

    frames_manifest = [
        {
            "sop_instance_uid": frame.sop_instance_uid or f"{state.series_uid}.{s_idx}.{f_idx}",
            "slice_index": s_idx,
            "frame_index": seq_idx,
            "trigger_time_ms": (
                float(frame.trigger_time_ms) if frame.trigger_time_ms is not None else None
            ),
            "image_id": (
                f"cmr-study://{state.series_uid}/"
                f"{frame.sop_instance_uid or f'{state.series_uid}.{s_idx}.{f_idx}'}"
            ),
        }
        for seq_idx, (s_idx, f_idx, frame) in enumerate(all_frames_sorted)
    ]

    geometry_manifest = {
        "rows": geom.rows,
        "cols": geom.cols,
        "pixel_spacing": [float(geom.row_spacing_mm), float(geom.col_spacing_mm)],
        "row_pixel_spacing": float(geom.row_spacing_mm),
        "column_pixel_spacing": float(geom.col_spacing_mm),
        "slice_thickness": (
            float(geom.slice_thickness) if geom.slice_thickness is not None else None
        ),
        "spacing_between_slices": (
            float(geom.spacing_between_slices)
            if geom.spacing_between_slices is not None
            else None
        ),
        "image_position_patient": [float(v) for v in geom.image_position_patient],
        "image_orientation_patient": [float(v) for v in geom.image_orientation_patient],
        "row_cosines": [float(v) for v in geom.row_direction],
        "column_cosines": [float(v) for v in geom.col_direction],
        "frame_of_reference_uid": getattr(
            geom, "frame_of_reference_uid", f"FOR-{state.series_uid}"
        ),
    }

    import numpy as np

    current_slice_frames = state.cine.frames[state.current_slice_index]
    raw_frame_data = np.asarray(
        current_slice_frames[state.current_frame_index].pixel_data, dtype=np.float64
    )
    # Display W/L must come from a ROBUST intensity window, not raw min/max:
    # raw min/max is dominated by a few saturated/air pixels (e.g. a 16-bit
    # volume with max 65535) which flattens the whole image to grey mush.
    # Percentile (p1..p99) is the standard DICOM-ish display heuristic.
    finite = raw_frame_data[np.isfinite(raw_frame_data)]
    if finite.size == 0:
        finite = raw_frame_data
    p_low = float(np.percentile(finite, 1.0))
    p_high = float(np.percentile(finite, 99.0))
    if p_high <= p_low:
        p_low = float(finite.min())
        p_high = float(finite.max())
    if p_high <= p_low:
        p_high = p_low + 1.0
    auto_center = (p_low + p_high) / 2.0
    auto_width = max((p_high - p_low), 1.0)

    adapted = dict(base)
    adapted["geometry"] = geometry_manifest
    adapted["canonical_phase_vector"] = canonical_phase_vector
    adapted["frame_time_vector"] = frame_time_vector
    adapted["frames_manifest"] = frames_manifest
    adapted["default_window_level"] = {
        "center": auto_center,
        "width": auto_width,
    }
    return adapted


def _qc_digest(report: Any) -> str:
    from cmr_core.domain_types import canonical_digest

    return canonical_digest(report)


def _provenance_for_action(
    state: ViewerState,
    cine: CineVolume,
    series_uid: str,
    timestamp: str,
) -> Provenance:
    head = state.head_revision
    return Provenance(
        study_uid=(head.provenance.study_uid if head is not None else cine.study_uid),
        series_uid=series_uid,
        software_version=VERSION,
        model=None,
        created_at=timestamp,
        revision=head.provenance.revision if head is not None else 0,
        workflow_id=(head.provenance.workflow_id if head is not None else cine.workflow_id),
    )


def _apply_topology_disposition(
    controller: WorkflowController,
    store: SqliteCmrStore,
    series_uid: str,
    cine: CineVolume,
    intent: ActionIntent,
    *,
    state: ViewerState,
    user: str,
    timestamp: str,
) -> dict[str, Any]:
    """Record ONE physician topology disposition (Astra contract item 6).

    Backend surface only — the viewer controls themselves are deferred to a
    later round. Strict, additive and fail-closed:

    - the exact head and selection MUST be supplied (the lineage CAS) and must
      match the CURRENT persisted state;
    - the key MUST carry a persisted topology block (nothing to resolve -> 400);
    - ``REPLACED`` requires the CURRENT chain to carry physician geometry for
      the key (a disposition may not claim a replacement that did not happen);
    - ``EXCLUDED`` requires the key to remain ABSENT from the chain (an
      excluded key may not silently ride along in the metrics).
    """
    params = intent.params
    slice_index = int(params["slice_index"])
    frame_index = int(params["frame_index"])
    structure = str(params["structure"])
    disposition = TopologyDisposition(str(params["disposition"]))
    reason = str(params["reason"])

    if state.selection is None:
        raise ActionConflict("no selection", error="no_selection")
    if state.head_revision_id is None:
        raise ActionConflict("no revision chain", error="no_chain")
    if intent.expected_head_revision_id != state.head_revision_id:
        raise StaleHeadError(
            f"head revision mismatch: expected {intent.expected_head_revision_id!r}, "
            f"current {state.head_revision_id!r}"
        )
    if intent.expected_selection_id != state.selection.selection_id:
        raise StaleHeadError(
            f"selection mismatch: expected {intent.expected_selection_id!r}, "
            f"current {state.selection.selection_id!r}"
        )
    blocks = [
        block
        for block in store.load_topology_blocks(series_uid)
        if block.slice_index == slice_index
        and block.frame_index == frame_index
        and block.structure == structure
    ]
    if not blocks:
        raise CMRError(
            INVALID_ARGUMENT,
            f"no persisted topology block for key (slice={slice_index}, "
            f"frame={frame_index}, structure={structure}); there is nothing to "
            "resolve",
        )
    chain = tuple(store.load_chain(series_uid))
    carried = [
        revision
        for revision in chain
        if any(
            contour.slice_index == slice_index
            and contour.frame_index == frame_index
            and contour.structure is Structure(structure)
            for contour in revision.contours.values()
        )
    ]
    if disposition is TopologyDisposition.REPLACED and not carried:
        raise CMRError(
            INVALID_ARGUMENT,
            f"REPLACED disposition for key (slice={slice_index}, "
            f"frame={frame_index}, structure={structure}) refused: the current "
            "revision chain does not carry that key as physician geometry",
        )
    if disposition is TopologyDisposition.EXCLUDED and carried:
        raise CMRError(
            INVALID_ARGUMENT,
            f"EXCLUDED disposition for key (slice={slice_index}, "
            f"frame={frame_index}, structure={structure}) refused: the current "
            "revision chain already carries that key; exclude the geometry or "
            "record REPLACED instead of claiming an exclusion that is not true",
        )
    role = disposition_role_for_frame(
        frame_index=frame_index,
        ed_frame=state.selection.ed_frame,
        es_frame=state.selection.es_frame,
    )
    record = TopologyDispositionRecord(
        series_uid=series_uid,
        slice_index=slice_index,
        frame_index=frame_index,
        structure=structure,
        source_block_id=blocks[0].block_id or "",
        source_revision_id=state.head_revision_id,
        selection_id=state.selection.selection_id,
        frame_role=role,
        disposition=disposition,
        physician=user,
        created_at=timestamp,
        reason=reason,
        provenance=_provenance_for_action(state, cine, series_uid, timestamp),
    )
    store.save_topology_disposition_set_checked(
        [record],
        expected_head_revision_id=state.head_revision_id,
        selection_id=state.selection.selection_id,
    )
    return {
        "status": "recorded",
        "disposition_id": record.disposition_id,
        "series_uid": series_uid,
        "slice_index": slice_index,
        "frame_index": frame_index,
        "structure": structure,
        "disposition": disposition.value,
        "head_revision_id": state.head_revision_id,
        "selection_id": state.selection.selection_id,
    }

def apply_checked_action(
    controller: WorkflowController,
    store: SqliteCmrStore,
    series_uid: str,
    cine: CineVolume,
    intent: ActionIntent,
    *,
    user: str,
    timestamp: str,
    legacy_clinical_path: bool = False,
    jobs_enabled: bool = True,
    cancel_already_applied: bool = False,
    x1_weights_root: Path | None = None,
) -> dict[str, Any]:
    """Apply one of the three closed mutation intents through checked APIs."""
    _validate_action_params(intent, cine=cine)
    is_legacy = legacy_clinical_path or getattr(intent, "legacy_clinical_path", False)
    state, _reopened = build_viewer_state(
        controller, store, series_uid, cine, intent.view
    )
    contour_save_result: dict[str, Any] | None = None
    bootstrap_result: dict[str, Any] | None = None
    approval_result: dict[str, Any] | None = None
    disposition_result: dict[str, Any] | None = None
    inference_result: dict[str, str] | None = None
    adoption_result: dict[str, Any] | None = None
    enqueue_lineage = False
    if intent.action == "cancel_analysis_job":
        # R2 review correction: a pre-applied (runner-scoped) cancellation
        # still enters THIS branch; only the store mutation is skipped so a
        # cancellation request can never fall through into an unrelated
        # inclusion mutation.
        if not cancel_already_applied:
            store.cancel_analysis_job(
                str(intent.params["job_id"]), updated_at=timestamp, series_uid=series_uid
            )
        final_draft = (
            intent.view.inclusion_draft
            if intent.view.inclusion_draft is not None
            else tuple(_wire_from_toggles(state.inclusion_draft))
        )
    elif intent.action == "request_selected_phase_ai":
        # Request-only action (A1/A4/A8): frames come from the CURRENT
        # physician-confirmed EDESSelection (never inferred); NO clinical
        # state (revision/selection/inclusion/measurement/approval) is
        # written here — the enqueue IS the requested effect. The bound
        # identity (head_revision_id + selection_id) is recorded in the
        # payload for the A1 eligibility predicate. The payload carries
        # portable identity only: relative pinned names + digests + device +
        # provider/algorithm version (A4/A8); the absolute weights root is
        # local runtime config and is never persisted/returned/logged.
        if state.selection is None:
            raise CMRError(
                "edes_selection_required",
                "a physician-confirmed ED/ES selection is required",
            )
        try:
            stage1_rel, stage2_rel, stage1_sha256, stage2_sha256 = _verified_x1_checkpoints(
                x1_weights_root
            )
        except CMRError:
            raise
        except Exception as exc:
            # Path-safe (F5): never propagate raw checkpoint text.
            raise CMRError(
                "x1_weights_unavailable", "pinned X1 weights failed verification"
            ) from exc
        payload = {
            "series_uid": series_uid,
            "head_revision_id": state.head_revision_id,
            "selection_id": state.selection.selection_id,
            "ed_frame": state.selection.ed_frame,
            "es_frame": state.selection.es_frame,
            "selected_frames": sorted(
                [state.selection.ed_frame, state.selection.es_frame]
            ),
            "stage1_checkpoint": stage1_rel,
            "stage2_checkpoint": stage2_rel,
            "stage1_sha256": stage1_sha256,
            "stage2_sha256": stage2_sha256,
            "device": "cpu",
            "provider": "baai_cine_sax",
            "algorithm_version": "x1-selected-phase-v1",
        }
        try:
            inference_job = store.enqueue_inference_job(
                series_uid=series_uid,
                stage1_checkpoint=stage1_rel,
                stage2_checkpoint=stage2_rel,
                selected_frames=payload["selected_frames"],
                request_payload=payload,
                created_at=timestamp,
                device="cpu",
            )
        except CMRError:
            raise
        except Exception as exc:
            # Path-safe (F5): an enqueue failure must never propagate raw
            # exception text (SQLite errors spell host paths) to the HTTP
            # handler or the logs — typed code only.
            raise CMRError(
                "inference_enqueue_failed", "background inference request was rejected"
            ) from exc
        inference_result = {
            "inference_job_id": inference_job.job_id,
            "inference_job_state": inference_job.state,
            "head_revision_id": state.head_revision_id,
            "selection_id": state.selection.selection_id,
        }
        final_draft = (
            intent.view.inclusion_draft
            if intent.view.inclusion_draft is not None
            else tuple(_wire_from_toggles(state.inclusion_draft))
        )
    elif intent.action == "adopt_selected_phase_ai":
        # CMR_FUNCV1_ADOPT_001 slice A: the EXPLICIT checked physician
        # adoption of the CURRENT-eligible sparse AI artifact into the
        # EXISTING legacy revision lineage. This branch is reachable ONLY
        # through an explicitly posted action intent — inference completion
        # never calls it (no auto-adopt). The controller owns every refusal
        # (stale head/selection via the store CAS shape, ineligible binding,
        # incomplete five-group coverage, binding mismatch, unsafe
        # topology) BEFORE appending any revision; this surface only routes
        # the intent and the physician identity/timestamp. NO analysis job is
        # enqueued (slice B owns coherent measurement/QC; nothing is
        # computed here) and no inclusion/approval state is written.
        adoption_result = controller.adopt_selected_phase_ai_checked(
            store,
            cine,
            series_uid,
            editor=user,
            created_at=timestamp,
            expected_head_revision_id=intent.expected_head_revision_id,
            expected_selection_id=intent.expected_selection_id,
        )
        final_draft = (
            intent.view.inclusion_draft
            if intent.view.inclusion_draft is not None
            else tuple(_wire_from_toggles(state.inclusion_draft))
        )
    elif intent.action == "record_topology_disposition":
        disposition_result = _apply_topology_disposition(
            controller,
            store,
            series_uid,
            cine,
            intent,
            state=state,
            user=user,
            timestamp=timestamp,
        )
        final_draft = (
            intent.view.inclusion_draft
            if intent.view.inclusion_draft is not None
            else tuple(_wire_from_toggles(state.inclusion_draft))
        )
        enqueue_lineage = True
    elif intent.action == "select_ed_es":
        provenance = _provenance_for_action(state, cine, series_uid, timestamp)

        def provenance_factory() -> Provenance:
            return provenance

        controller.record_ed_es_checked(
            store,
            cine,
            int(intent.params["ed_frame"]),
            int(intent.params["es_frame"]),
            user,
            timestamp,
            provenance_factory,
            expected_head_revision_id=intent.expected_head_revision_id,
            expected_selection_id=intent.expected_selection_id,
            expected_inclusion_set_id=intent.expected_inclusion_set_id,
        )
        final_draft = tuple(
            {
                "slice_index": toggle.slice_index,
                "frame_role": toggle.role,
                "structure": toggle.structure,
                "included": True,
            }
            for toggle in all_included_toggles(cine.n_slices)
        )
        enqueue_lineage = True
    elif intent.action == "save_contour_edits":
        if is_legacy:
            expected_head_revision_id = intent.expected_head_revision_id
            if expected_head_revision_id is None:
                raise _invalid(
                    "save_contour_edits requires a non-null lineage.head_revision_id "
                    "carrying the current legacy head revision identity to supersede"
                )
            chain = tuple(_reopened.get("chain", ()))
            if not chain:
                chain = tuple(store.load_chain(series_uid))
            if not chain:
                raise _invalid(
                    f"no legacy revision chain found for series {series_uid!r}; "
                    "bootstrap required before saving edits"
                )
            parent = chain[-1]
            requested_edits = list(intent.params["edits"])
            if len(requested_edits) == 1 and str(requested_edits[0].get("kind")) in (
                "create",
                "remove",
                "copy",
                "copyslice",
            ):
                # Dedicated legacy create/remove/copy/copyslice dispatch
                # (CMR_CREATE_A_001, R2 C2 + copy + copyslice): mirrors the
                # replace structure below but routes through the
                # controller-owned checked operations, which own eligibility,
                # composition, the myo gate and the checked append. The
                # single-command rule was enforced by the wire grammar;
                # whole-cine create/remove/copy/copyslice is refused on its
                # own path below.
                single = requested_edits[0]
                if str(single["kind"]) == "copyslice":
                    saved_rev = controller.copy_contours_across_slices_checked(
                        store,
                        series_uid,
                        src_slice_index=int(single["src_slice_index"]),
                        src_frame_index=int(single["src_frame_index"]),
                        dst_slice_indices=list(single["dst_slice_indices"]),
                        structure=Structure(single["structure"]),
                        editor=user,
                        created_at=timestamp,
                        cine=cine,
                        expected_head_revision_id=expected_head_revision_id,
                        expected_selection_id=intent.expected_selection_id,
                        expected_inclusion_set_id=intent.expected_inclusion_set_id,
                    )
                    copied_slices = list(single["dst_slice_indices"])
                    contour_save_result = {
                        "status": "saved",
                        "series_uid": series_uid,
                        "head_revision_id": saved_rev.revision_id,
                        "revision_id": saved_rev.revision_id,
                        "version": saved_rev.version,
                        "edited_keys": [
                            f"{s}:{single['src_frame_index']}:{single['structure']}"
                            for s in copied_slices
                        ],
                        "parent_head_id": expected_head_revision_id,
                    }
                    final_draft = (
                        intent.view.inclusion_draft
                        if intent.view.inclusion_draft is not None
                        else tuple(
                            {
                                "slice_index": toggle.slice_index,
                                "frame_role": toggle.role,
                                "structure": toggle.structure,
                                "included": True,
                            }
                            for toggle in all_included_toggles(cine.n_slices)
                        )
                    )
                    enqueue_lineage = True
                elif str(single["kind"]) == "copy":
                    saved_rev = controller.copy_contours_checked(
                        store,
                        series_uid,
                        src_slice_index=int(single["src_slice_index"]),
                        src_frame_index=int(single["src_frame_index"]),
                        dst_frame_indices=list(single["dst_frame_indices"]),
                        structure=Structure(single["structure"]),
                        editor=user,
                        created_at=timestamp,
                        cine=cine,
                        expected_head_revision_id=expected_head_revision_id,
                        expected_selection_id=intent.expected_selection_id,
                        expected_inclusion_set_id=intent.expected_inclusion_set_id,
                    )
                    copied_frames = list(single["dst_frame_indices"])
                    contour_save_result = {
                        "status": "saved",
                        "series_uid": series_uid,
                        "head_revision_id": saved_rev.revision_id,
                        "revision_id": saved_rev.revision_id,
                        "version": saved_rev.version,
                        "edited_keys": [
                            f"{single['src_slice_index']}:{f}:{single['structure']}"
                            for f in copied_frames
                        ],
                        "parent_head_id": expected_head_revision_id,
                    }
                    final_draft = (
                        intent.view.inclusion_draft
                        if intent.view.inclusion_draft is not None
                        else tuple(
                            {
                                "slice_index": toggle.slice_index,
                                "frame_role": toggle.role,
                                "structure": toggle.structure,
                                "included": True,
                            }
                            for toggle in all_included_toggles(cine.n_slices)
                        )
                    )
                    enqueue_lineage = True
                else:
                    s_idx = int(single["slice_index"])
                    f_idx = int(single["frame_index"])
                    single_structure = Structure(single["structure"])
                if str(single["kind"]) in ("copy", "copyslice"):
                    pass  # saved_rev + contour_save_result already composed above
                elif str(single["kind"]) == "create":
                    saved_rev = controller.create_contour_checked(
                        store,
                        series_uid,
                        slice_index=s_idx,
                        frame_index=f_idx,
                        structure=single_structure,
                        points=single["points"],
                        editor=user,
                        created_at=timestamp,
                        cine=cine,
                        expected_head_revision_id=expected_head_revision_id,
                        expected_selection_id=intent.expected_selection_id,
                        expected_inclusion_set_id=intent.expected_inclusion_set_id,
                    )
                    contour_save_result = {
                        "status": "saved",
                        "series_uid": series_uid,
                        "head_revision_id": saved_rev.revision_id,
                        "revision_id": saved_rev.revision_id,
                        "version": saved_rev.version,
                        "edited_keys": [f"{s_idx}:{f_idx}:{single_structure.value}"],
                        "parent_head_id": expected_head_revision_id,
                    }
                else:
                    saved_rev = controller.remove_contour_checked(
                        store,
                        series_uid,
                        slice_index=s_idx,
                        frame_index=f_idx,
                        structure=single_structure,
                        editor=user,
                        created_at=timestamp,
                        cine=cine,
                        expected_head_revision_id=expected_head_revision_id,
                    )
                    contour_save_result = {
                        "status": "saved",
                        "series_uid": series_uid,
                        "head_revision_id": saved_rev.revision_id,
                        "revision_id": saved_rev.revision_id,
                        "version": saved_rev.version,
                        "edited_keys": [f"{s_idx}:{f_idx}:{single_structure.value}"],
                        "parent_head_id": expected_head_revision_id,
                    }
                final_draft = (
                    intent.view.inclusion_draft
                    if intent.view.inclusion_draft is not None
                    else tuple(
                        {
                            "slice_index": toggle.slice_index,
                            "frame_role": toggle.role,
                            "structure": toggle.structure,
                            "included": True,
                        }
                        for toggle in all_included_toggles(cine.n_slices)
                    )
                )
                enqueue_lineage = True
            else:
                edits_map: dict[Structure, Contour] = {}
                for edit in requested_edits:
                    s_idx = int(edit["slice_index"])
                    f_idx = int(edit["frame_index"])
                    structure = Structure(edit["structure"])
                    # Tombstone-aware edit-target lookup (CMR_CREATE_A_001, R2
                    # C1): a remove tombstone for the exact key stops the
                    # backwards search, so a later edit of a removed key
                    # reads "missing" instead of silently resurrecting an
                    # ancestor contour.
                    target_contour = effective_chain_contour(
                        chain, (s_idx, f_idx, structure)
                    )
                    kind = str(edit["kind"])
                    if kind == "replace":
                        wc_pred = store.load_latest_whole_cine_prediction(series_uid)
                        if wc_pred is not None:
                            source = wc_pred.contours.get((s_idx, f_idx, structure), ())
                            if len(source) > 1:
                                raise CMRError(
                                    CONTOUR_INVALID,
                                    f"target contour ({s_idx}, {f_idx}, {structure.value}) is "
                                    f"multi-component ({len(source)} components); replace "
                                    "requires a single simple polygon",
                                )
                            if len(source) == 1 and len(source[0].holes) > 0:
                                raise CMRError(
                                    CONTOUR_INVALID,
                                    f"target contour ({s_idx}, {f_idx}, {structure.value}) is "
                                    f"annular ({len(source[0].holes)} holes); replace requires "
                                    "a single simple polygon without holes",
                                )
                        if target_contour is None:
                            raise CMRError(
                                CONTOUR_INVALID,
                                f"no eligible legacy contour target found for ({s_idx}, {f_idx}, "
                                f"{structure.value}); replace cannot implicitly create or adopt "
                                "contours",
                            )
                        edited_c = replace_contour(target_contour, edit["points"], cine=cine)
                    else:
                        if target_contour is None:
                            if (s_idx, f_idx, structure) in head_tombstoned_keys(chain):
                                raise CMRError(
                                    CONTOUR_INVALID,
                                    f"BLOCKED_EDIT_REMOVED_TARGET: no eligible legacy contour "
                                    f"target found for ({s_idx}, {f_idx}, {structure.value}); "
                                    f"key was removed and {kind} cannot resurrect removed "
                                    "contours via whole-cine projection; "
                                    f"next: create the ({s_idx}, {f_idx}, {structure.value}) "
                                    "key anew on the legacy clinical path, then retry",
                                )
                            target_contour = project_persisted_whole_cine_to_legacy_draft(
                                store, series_uid, s_idx, f_idx, structure
                            )
                        if kind == "translate":
                            edited_c = translate_contour(
                                target_contour, float(edit["dx"]), float(edit["dy"])
                            )
                        elif kind == "insert":
                            pos = edit["index_or_position"]
                            edited_c = insert_vertex(
                                target_contour,
                                pos if isinstance(pos, int) else [float(coord) for coord in pos],
                            )
                        elif kind == "delete":
                            edited_c = delete_vertex(target_contour, int(edit["vertex_index"]))
                        else:
                            raise _invalid(f"unsupported edit kind: {kind!r}")
                    edits_map[structure] = edited_c

                saved_rev = controller.save(
                    store,
                    parent=parent,
                    edits=edits_map,
                    editor=user,
                    created_at=timestamp,
                    cine=cine,
                    expected_head_revision_id=expected_head_revision_id,
                    # F4 (CMR_CREATE_A_F4_SCOPE_001): the full pre-save
                    # lineage lets the containment gate resolve unchanged
                    # counterpart geometry on the affected planes (C1
                    # derivation); without it counterparts living in older
                    # revisions would read absent and bypass the gate.
                    chain=chain,
                    # PENDING_EDIT_BIND_001 (R1 P1-005): bind the CAPTURED
                    # (authored) selection/inclusion authority. The ordinary
                    # path does NOT opt into the source/topology CAS block
                    # (expected_n_slices=None keeps the historical head-only
                    # CAS shape): source/topology CAS is create/remove-scoped
                    # per the C1 contract, and ordinary fixtures legitimately
                    # persist source predictions. Selection/inclusion are
                    # re-verified by the controller preflight below instead.
                    expected_selection_id=intent.expected_selection_id,
                    expected_inclusion_set_id=intent.expected_inclusion_set_id,
                )
                contour_save_result = {
                    "status": "saved",
                    "series_uid": series_uid,
                    "head_revision_id": saved_rev.revision_id,
                    "revision_id": saved_rev.revision_id,
                    "version": saved_rev.version,
                    "edited_keys": [
                        f"{edit['slice_index']}:{edit['frame_index']}:{edit['structure']}"
                        for edit in intent.params["edits"]
                    ],
                    "parent_head_id": expected_head_revision_id,
                }
                final_draft = (
                    intent.view.inclusion_draft
                    if intent.view.inclusion_draft is not None
                    else tuple(
                        {
                            "slice_index": toggle.slice_index,
                            "frame_role": toggle.role,
                            "structure": toggle.structure,
                            "included": True,
                        }
                        for toggle in all_included_toggles(cine.n_slices)
                    )
                )
                enqueue_lineage = True
        else:
            for edit in intent.params["edits"]:
                if edit.get("kind") == "replace":
                    raise CMRError(
                        INVALID_ARGUMENT,
                        "replace edit kind is not supported on the whole-cine path; "
                        "only legacy clinical path is supported",
                    )
                if edit.get("kind") in ("create", "remove"):
                    # R2 C2: create and remove are legacy-clinical-path only;
                    # the whole-cine allow-list stays exactly
                    # translate/insert/delete.
                    raise CMRError(
                        INVALID_ARGUMENT,
                        "BLOCKED_CREATE_REMOVE_PATH: legacy clinical path required",
                    )
            # S3-009 D7: the whole-cine lineage identity the client expects to
            # supersede is carried in expected_whole_cine_head_id (exposed in
            # state_projection and action lineage).
            expected_whole_cine_head_id = intent.expected_whole_cine_head_id
            if expected_whole_cine_head_id is None:
                raise _invalid(
                    "save_contour_edits requires a non-null lineage.whole_cine_head_id "
                    "carrying the current whole-cine head identity to supersede"
                )
            contour_save_result = controller.save_contour_edits_checked(
                store,
                series_uid,
                edits=[
                    {
                        "kind": str(edit["kind"]),
                        "slice_index": edit["slice_index"],
                        "frame_index": edit["frame_index"],
                        "structure": edit["structure"],
                        **(
                            {"dx": edit["dx"], "dy": edit["dy"]}
                            if edit["kind"] == "translate"
                            else {}
                        ),
                        **(
                            {"index_or_position": edit["index_or_position"]}
                            if edit["kind"] == "insert"
                            else {}
                        ),
                        **(
                            {"vertex_index": edit["vertex_index"]}
                            if edit["kind"] == "delete"
                            else {}
                        ),
                        **(
                            {"points": edit["points"]}
                            if edit["kind"] == "replace"
                            else {}
                        ),
                    }
                    for edit in intent.params["edits"]
                ],
                editor=user,
                created_at=timestamp,
                expected_whole_cine_head_id=expected_whole_cine_head_id,
            )
            final_draft = tuple(
                {
                    "slice_index": toggle.slice_index,
                    "frame_role": toggle.role,
                    "structure": toggle.structure,
                    "included": True,
                }
                for toggle in all_included_toggles(cine.n_slices)
            )
            enqueue_lineage = True
    elif intent.action == "bootstrap_legacy_chain":
        if state.head_revision_id is not None:
            raise ActionConflict("already bootstrapped", error="already_bootstrapped")
        pred = store.load_latest_whole_cine_prediction(series_uid)
        if pred is None:
            raise CMRError(
                INVALID_ARGUMENT,
                f"no whole-cine prediction found for series {series_uid!r}",
            )
        # BOOTSTRAP_ED_BIND_001 (R1 P1-008 + ChatGPT-1802): materialization
        # must bind the CURRENT persisted physician ED selection. A missing
        # param is only acceptable when a persisted selection exists; the
        # historical silent default of ED=0 is removed.
        if "ed_frame" in intent.params:
            ed_frame = int(intent.params["ed_frame"])
        else:
            latest_selection = store.load_latest_selection(series_uid)
            if latest_selection is None:
                raise CMRError(
                    INVALID_ARGUMENT,
                    "bootstrap_legacy_chain requires ed_frame when no persisted "
                    "ED/ES selection exists; refusing to default ED=0",
                )
            ed_frame = int(latest_selection.ed_frame)
        new_head_id = materialize_legacy_chain_from_whole_cine(
            store,
            cine,
            series_uid,
            ed_frame=ed_frame,
            created_at=timestamp,
            editor=user,
        )
        bootstrap_result = {
            "head_revision_id": new_head_id,
            "status": "ok",
        }
        final_draft = (
            intent.view.inclusion_draft
            if intent.view.inclusion_draft is not None
            else tuple(
                {
                    "slice_index": toggle.slice_index,
                    "frame_role": toggle.role,
                    "structure": toggle.structure,
                    "included": True,
                }
                for toggle in all_included_toggles(cine.n_slices)
            )
        )
        enqueue_lineage = True
    elif intent.action == "approve":
        # Astra F1/F5: the series-wide "any block -> refuse" guard is REPLACED
        # by the same relevance-and-resolution predicate the controller uses.
        # Only a metric-relevant, unresolved key blocks; irrelevant keys stay
        # visible and non-blocking. The typed reason names the exact key.
        # When no selection exists the selected metrics are UNKNOWN, so every
        # block key is treated as metric-relevant (fail closed) and the
        # topology refusal is reported before the "no selection" 409.
        # Viewer v4: the eligibility mirror (``_approval_eligibility``) is the
        # SAME predicate the projection serves; it is consulted here so the
        # wire refusal and the projected refusal cannot drift. Messages are
        # preserved verbatim (existing behavior unchanged).
        resolutions = controller.resolve_topology(
            store,
            series_uid,
            selection=state.selection,
            head_revision_id=state.head_revision_id,
            cine=cine,
        )
        eligibility = _approval_eligibility(
            state, resolutions, series_uid=series_uid
        )
        if not eligibility["can_approve"]:
            raise ActionConflict(
                str(eligibility["refusal"]),
                error=str(eligibility["refusal_code"]),
            )
        if intent.expected_head_revision_id != state.head_revision_id:
            raise StaleHeadError(
                f"head revision mismatch: expected {intent.expected_head_revision_id!r}, "
                f"current {state.head_revision_id!r}"
            )
        if intent.expected_selection_id != state.selection.selection_id:
            raise StaleHeadError(
                f"selection mismatch: expected {intent.expected_selection_id!r}, "
                f"current {state.selection.selection_id!r}"
            )
        expected_inclusion_id = (
            state.persisted_inclusion_set.inclusion_set_id
            if state.persisted_inclusion_set is not None
            else None
        )
        if intent.expected_inclusion_set_id != expected_inclusion_id:
            raise StaleHeadError(
                f"inclusion set mismatch: expected {intent.expected_inclusion_set_id!r}, "
                f"current {expected_inclusion_id!r}"
            )

        assert state.head_revision_id is not None
        assert state.latest_measurement is not None
        assert state.qc_report is not None
        snapshot = controller.approve(
            store,
            state.head_revision_id,
            state.latest_measurement.measurement_id,
            state.qc_report,
            approved_by=user,
            approved_at=timestamp,
            measurement=state.latest_measurement,
            selection=state.selection,
            cine=cine,
        )
        approval_result = {
            "approval_id": snapshot.approval_id,
            "status": "sealed",
        }
        final_draft = (
            intent.view.inclusion_draft
            if intent.view.inclusion_draft is not None
            else tuple(_wire_from_toggles(state.inclusion_draft))
        )
    elif intent.action == "record_applicability_decision":
        # FUNCV1 physician applicability loop C2 (R2 P1 residual): persist
        # the physician decision WITHOUT recompute. The server binds the
        # displayed frame the physician confirmed to the ACTUALLY-MUTATED
        # grid key — NOT to the declared ``params.frame_role``. The submitted
        # draft is diffed against the CURRENT applicable grid (the persisted
        # applicable set when one exists, else the frozen all-included
        # default — the same source ``build_viewer_state`` resolves a
        # missing draft from): every key whose value differs must have its
        # OWN role frame equal to ``displayed_frame`` (ED role <->
        # selection.ed_frame, ES <-> selection.es_frame). A submit whose
        # mutated key disagrees — e.g. params claim ED while the draft
        # waives an ES entry — is refused BEFORE any write (zero rows), so
        # viewing ED can never author ES authority (and vice versa).
        # ED==ES keeps role isolation by construction: the bind uses each
        # MUTATED key's OWN role value (equal frames here still resolve
        # role-exactly), so no cross-role inference is possible.
        # Persistence itself reuses the checked SliceInclusionSet surface
        # (three-way nullable CAS, zero partial rows); the new applicable
        # set invalidates the old analysis/approval (reopen resolves
        # analysis=None/not_current; the seal refuses) while missing
        # geometry keeps refusing recompute/seal fail-closed (nothing
        # computed here, no contour fabricated).
        if state.selection is None:
            raise ActionConflict("no selection", error="no_selection")
        displayed_frame = int(intent.params["displayed_frame"])
        frame_role = str(intent.params["frame_role"])
        expected_declared_frame = (
            state.selection.ed_frame if frame_role == "ED" else state.selection.es_frame
        )
        if displayed_frame != expected_declared_frame:
            raise CMRError(
                INVALID_ARGUMENT,
                f"displayed frame {displayed_frame} does not match the "
                f"selected {frame_role} frame {expected_declared_frame}; navigate to "
                f"the {frame_role} phase before confirming the decision",
            )
        assert intent.view.inclusion_draft is not None
        # C2 baseline: the CURRENT applicable grid — the persisted
        # applicable set when one exists, else the frozen all-included
        # default (the same source ``build_viewer_state`` resolves a
        # missing draft from). This is NOT ``state.inclusion_draft``: that
        # field echoes the SUBMITTED draft (built from ``intent.view``), so
        # diffing against it would always be empty. A concurrent commit
        # between this read and the persist below stays fail-closed via
        # the store three-way CAS (``expected_inclusion_set_id``) inside
        # ``record_applicability_decision_checked`` (stale -> refuse,
        # zero rows). The comparison uses the canonical mutation key
        # (slice, role, structure) and BOOLEAN values only; unknown extra
        # draft keys diff against the empty baseline (no silent adoption).
        if state.persisted_inclusion_set is not None:
            current_grid = {
                (
                    entry.slice_index,
                    entry.frame_role.value,
                    entry.structure.value,
                ): bool(entry.included)
                for entry in state.persisted_inclusion_set.entries
            }
        else:
            current_grid = {
                (toggle.slice_index, toggle.role, toggle.structure): True
                for toggle in all_included_toggles(cine.n_slices)
            }
        mutated_roles: list[str] = []
        for row in intent.view.inclusion_draft:
            row_key = (
                int(row["slice_index"]),
                str(row["frame_role"]),
                str(row["structure"]),
            )
            submitted = row["included"]
            if not isinstance(submitted, bool):
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"inclusion draft entry {row_key!r} must carry an actual Boolean",
                )
            if row_key[1] not in ("ED", "ES"):
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"inclusion draft entry {row_key!r} carries an unknown role",
                )
            baseline = current_grid.get(row_key)
            if baseline is None or bool(baseline) is not submitted:
                mutated_roles.append(row_key[1])
        if not mutated_roles:
            raise CMRError(
                INVALID_ARGUMENT,
                "applicability decision changes no inclusion key; nothing to record",
            )
        for mutated_role in mutated_roles:
            expected_mutated_frame = (
                state.selection.ed_frame
                if mutated_role == "ED"
                else state.selection.es_frame
            )
            if displayed_frame != expected_mutated_frame:
                raise CMRError(
                    INVALID_ARGUMENT,
                    f"displayed frame {displayed_frame} does not match the "
                    f"mutated {mutated_role} frame {expected_mutated_frame}; navigate to "
                    f"the {mutated_role} phase before confirming the decision",
                )
        controller.record_applicability_decision_checked(
            store,
            cine,
            series_uid,
            entries=list(intent.view.inclusion_draft),
            selected_by=user,
            software_version=VERSION,
            created_at=timestamp,
            expected_head_revision_id=intent.expected_head_revision_id,
            expected_selection_id=intent.expected_selection_id,
            expected_inclusion_set_id=intent.expected_inclusion_set_id,
        )
        final_draft = intent.view.inclusion_draft
        # The decision changed the applicable grid, so the selected-phase
        # recompute SHOULD re-run against the new authority (when coverage
        # artifacts can serve it). Unlike the compound refresh (which
        # recomputes synchronously), this path persists only — the
        # background job owns the recompute.
        enqueue_lineage = True
    else:
        assert intent.view.inclusion_draft is not None
        controller.apply_inclusion_and_refresh_analysis_checked(
            store,
            cine,
            series_uid,
            entries=list(intent.view.inclusion_draft),
            selected_by=user,
            software_version=VERSION,
            created_at=timestamp,
            editor="workstation",
            expected_head_revision_id=intent.expected_head_revision_id,
            expected_selection_id=intent.expected_selection_id,
            expected_inclusion_set_id=intent.expected_inclusion_set_id,
        )
        final_draft = intent.view.inclusion_draft
    # S3-008B §3.1: display context never survives an action round-trip —
    # actions carry only the accepted vocabulary (display fields stay None).
    final_view = ViewIntent(
        intent.view.slice_index,
        intent.view.frame_index,
        intent.view.fps,
        intent.view.playing,
        intent.view.reference_sex,
        tuple(final_draft),
    )
    final_state, final_reopened = build_viewer_state(
        controller, store, series_uid, cine, final_view
    )
    if enqueue_lineage and final_state.selection is not None and jobs_enabled:
        try:
            coverage = resolve_selected_phase_coverage(
                store,
                series_uid,
                final_state.selection.ed_frame,
                final_state.selection.es_frame,
                slice_count=cine.n_slices,
            )
            coverage_identity = {
                "ed_frame": final_state.selection.ed_frame,
                "es_frame": final_state.selection.es_frame,
                "slice_count": coverage.slice_count,
                "frame_count": coverage.frame_count,
                "ai_artifact_id": coverage.ai_artifact_id,
                "ai_artifact_domain": coverage.ai_artifact_domain,
                "physician_artifact_id": coverage.physician_artifact_id,
                "resolved_keys": [
                    {
                        "key": [key[0], key[1], key[2].value],
                        "artifact_id": value.artifact_id,
                        "artifact_domain": value.artifact_domain,
                        "authority": value.authority,
                        "projection": value.projection,
                    }
                    for key, value in sorted(
                        coverage.resolved.items(),
                        key=lambda item: (
                            item[0][0], item[0][1], item[0][2].value
                        ),
                    )
                ],
                "missing": [
                    [key[0], key[1], key[2].value] for key in coverage.missing
                ],
                "unknown_grid": coverage.unknown_grid,
            }
            enqueue_selected_phase_recompute(
                store,
                series_uid=series_uid,
                head_revision_id=final_state.head_revision_id,
                selection_id=final_state.selection_id,
                inclusion_set_id=final_state.inclusion_set_id,
                coverage_identity=coverage_identity,
                created_at=timestamp,
            )
        except Exception as exc:
            # Job scheduling is deliberately non-clinical and best effort. A
            # persistence/coverage failure must never roll back the mutation.
            # Path-safe (F5): log the exception TYPE + stable code, never raw
            # text (checkpoint spellings must not reach logs).
            LOGGER.warning(
                "selected-phase recompute enqueue failed: %s",
                _typed_diagnostic(exc, "selected_phase_recompute_enqueue_failed"),
            )
    result = {
        "client_nonce": intent.client_nonce,
        "action": intent.action,
        "state": state_projection(
            final_state,
            analysis_jobs=(
                _best_effort_analysis_jobs(store, series_uid)
                if jobs_enabled
                else None
            ),
            analysis=final_reopened.get("analysis"),
            analysis_stale_reason=final_reopened.get("analysis_stale_reason"),
        ),
        "inclusion_draft": list(final_draft),
    }
    if contour_save_result is not None:
        # The contour-save response carries the whole-cine lineage facts the
        # client needs for its next CAS (revision id + edited keys +
        # supersedes lineage), exactly one action per response.
        result["contour_save"] = contour_save_result
    if bootstrap_result is not None:
        result["bootstrap"] = bootstrap_result
    if approval_result is not None:
        result["approval"] = approval_result
    if disposition_result is not None:
        result["topology_disposition"] = disposition_result
    if inference_result is not None:
        result.update(inference_result)
    if adoption_result is not None:
        result["adoption"] = adoption_result
    return result


def _structure_name(structure: Structure | str) -> str:
    """Format structure name for JSON wire contract (e.g. 'LV_CAVITY')."""
    if isinstance(structure, Structure):
        return structure.name
    return Structure(structure).name


def _experimental_contour_dict(
    structure: Structure,
    source_contours: tuple[HoleAwareContour, ...],
    *,
    source: str,
) -> dict[str, Any]:
    """Emit ONE structure's contours in the experimental pre-contour wire shape.

    Shared by the prediction-only branch and (per the 2026-09-09 per-plane
    amendment) the chain branch's fill-in for structures the chain does NOT
    carry, so both paths emit byte-identical geometry:

    * ``points_pixel``: the FIRST component's exterior (unchanged contract for
      existing consumers);
    * ``components_pixel``: the FULL component collection (exterior + holes per
      component), so multi-component topology survives display;
    * ``holes_pixel``: the FIRST component's holes — and STRICTLY ABSENT when
      the first component carries none (absence is the contract, not ``[]``).

    Display-only: this builds response dicts. It writes NOTHING to revisions,
    predictions or the whole-cine record.
    """
    first_comp = source_contours[0]
    first_pts = [[float(x), float(y)] for x, y in first_comp.exterior.points]
    first_holes = [
        [[float(x), float(y)] for x, y in hole.points] for hole in first_comp.holes
    ]
    components_pixel = [
        {
            "exterior": [[float(x), float(y)] for x, y in comp.exterior.points],
            "holes": [[[float(x), float(y)] for x, y in hole.points] for hole in comp.holes],
        }
        for comp in source_contours
    ]
    contour_dict: dict[str, Any] = {
        "structure": _structure_name(structure),
        "source": source,
        "edited": False,
        "draft": False,
        "points_pixel": first_pts,
        "components_pixel": components_pixel,
    }
    if first_holes:
        contour_dict["holes_pixel"] = first_holes
    return contour_dict


def _chain_absent_experimental_contours(
    store: SqliteCmrStore,
    series_uid: str,
    *,
    slice_index: int,
    frame_index: int,
    present_structures: set[Structure],
) -> list[dict[str, Any]]:
    """Experimental geometry for chain-ABSENT structures on one plane (§1a, display-only).

    Called from the chain branch of :func:`resolve_study_contours` after the
    chain-merged contours are built. Per-plane bootstrap (2026-09-09) EXCLUDES
    non-projectable keys (annular / multi-component) from the legacy chain, so
    without this fill-in that geometry disappears from ``GET /contours`` — a
    display regression versus the pre-bootstrap prediction-only view.

    Contract:
    * Only structures ABSENT from ``present_structures`` (the chain merge) and
      PRESENT in the experimental whole-cine prediction at this
      ``(slice, frame)`` are emitted — chain geometry always WINS and is
      never overridden here.
    * Emission shape is IDENTICAL to the prediction-only branch (shared
      :func:`_experimental_contour_dict` helper), plus the per-contour
      ``contour_source: experimental_precontour`` marker so the viewer still
      warns that this particular structure is research-only AI geometry even
      though the response as a whole is chain-merged (the top-level flag stays
      ABSENT: the plane is NOT overall an experimental-only display).
    * Read-only: no revision, prediction or whole-cine write.
    """
    pred = store.load_latest_whole_cine_prediction(series_uid)
    if pred is None:
        return []
    filled: list[dict[str, Any]] = []
    for structure in Structure:
        if structure in present_structures:
            continue  # chain wins: never override a chained/edited key
        source_contours = pred.contours.get((slice_index, frame_index, structure), ())
        if not source_contours:
            continue
        contour_dict = _experimental_contour_dict(
            structure,
            source_contours,
            source=pred.source if pred.source == "ai" else "ai",
        )
        # Per-contour provenance marker: this ONE structure is experimental
        # pre-contour geometry, not physician chain data.
        contour_dict["contour_source"] = "experimental_precontour"
        filled.append(contour_dict)
    return filled


def resolve_study_contours(
    controller: WorkflowController,
    store: SqliteCmrStore,
    series_uid: str,
    cine: CineVolume,
    *,
    slice_index: int,
    frame_index: int,
) -> dict[str, Any]:
    """Resolve display contours for non-clinical read endpoint GET /api/study/<uid>/contours.

    Resolution order:
    1. If legacy chain head exists: chain-merged contours at (slice, frame)
       (ViewerState.merged_contours_at semantics — latest per structure across
       the whole chain) with draft: false, PLUS the experimental fill-in of
       §1a below for structures the chain does not carry.
    2. Else if persisted whole-cine prediction exists: for each structure present
       at that key, U12-project a display contour (exterior points); non-projectable
       keys are omitted and blocked_reason carries the reason.
    3. Else: contours: [].

    §1a Chain-absent experimental fill-in (display merge, 2026-09-09 C1):
    after per-plane bootstrap, a plane whose structure was NON-PROJECTABLE
    (annular / multi-component) is EXCLUDED from the legacy chain, so the
    chain-merged view alone would drop that geometry from the display — a
    regression vs the pre-bootstrap (prediction-only) view. For structures
    ABSENT from the chain merge but PRESENT in the experimental whole-cine
    prediction at this (slice, frame), the resolver now re-emits them in the
    EXACT prediction-only emission shape (points_pixel = first-component
    exterior, components_pixel = full collection, holes_pixel when the first
    component carries holes) with the per-contour
    ``contour_source: experimental_precontour`` marker preserved. Chain
    geometry WINS wherever present: an edited/projectable key keeps its chain
    form and provenance, and experimental geometry NEVER overrides a chain
    key. This is DISPLAY-ONLY — no experimental key is ever written into a
    revision, prediction or the whole-cine record.

    B10-live Option B (owner-ratified 2026-09-06): when the display source is the
    AI pre-contour (whole-cine prediction projection with NO legacy chain head),
    the payload carries the additive top-level flag
    ``"contour_source": "experimental_precontour"``. The key is ABSENT for
    chain-merged responses and when neither source exists, so the field is a
    purely additive, backward-compatible warning signal for the viewer
    (pre-existing pinned consumers ignore unknown keys). No clinical
    computation, no persistence, no approval-semantics change.
    """
    if not (0 <= slice_index < cine.n_slices):
        raise ValueError(
            f"slice index {slice_index} out of cine grid bounds (0..{cine.n_slices - 1})"
        )
    if not (0 <= frame_index < cine.n_frames):
        raise ValueError(
            f"frame index {frame_index} out of cine grid bounds (0..{cine.n_frames - 1})"
        )

    topology_blocks = store.load_topology_blocks(series_uid)
    # Per-plane blocked status (owner P0 2026-09-09): only a block recorded for
    # THIS (slice, frame) marks the requested plane blocked. Other planes'
    # blocks stay visible in the full ``topology_blocks`` audit list (reopen /
    # state) but must not poison an unblocked plane's read.
    plane_blocks = [
        b
        for b in topology_blocks
        if b.slice_index == slice_index and b.frame_index == frame_index
    ]
    blocked_reason: str | None = plane_blocks[0].reason if plane_blocks else None
    #: B10-live Option B additive marker; set only for the prediction-only
    #: (pre-bootstrap AI pre-contour) display source, None otherwise.
    contour_source: str | None = None

    chain = tuple(store.load_chain(series_uid))
    # A1: current-eligibility. Only the sparse artifact bound (via a
    # SUCCEEDED inference_jobs row) to the CURRENT head + selection is
    # eligible for the current viewer plane and current coverage.
    # Cancelled/failed/interrupted/unbound/stale artifacts are audit-only.
    _chain_head_id = chain[-1].revision_id if chain else None
    _latest_selection = store.load_latest_selection(series_uid)
    _current_selection_id = (
        None if _latest_selection is None else _latest_selection.selection_id
    )
    # Chain-absent whole-cine series bind head None: the A1 predicate
    # compares (head None, selection id) exactly, so those bindings resolve.
    if _current_selection_id is not None:
        sparse_pred = store.load_current_selected_phase_prediction(
            series_uid,
            head_revision_id=_chain_head_id,
            selection_id=_current_selection_id,
        )
    else:
        sparse_pred = None
    wc_head = store.load_latest_whole_cine_head(series_uid)
    whole_cine_head_id: str | None = (
        None
        if wc_head is None
        else (
            wc_head.prediction_id
            if isinstance(wc_head, WholeCinePrediction)
            else wc_head.revision_id
        )
    )

    contours_list: list[dict[str, Any]] = []
    sparse_contributed = False

    if chain:
        head_revision_id: str | None = chain[-1].revision_id
        viewer_state, reopened = build_viewer_state(controller, store, series_uid, cine)
        # Per-plane (owner P0 2026-09-09): the reopened state's blocked_reason
        # stays SERIES-wide for audit, so it is only adopted when it belongs to
        # the requested (slice, frame). Other planes' blocks must not poison an
        # unblocked plane's read.
        if blocked_reason is None and reopened.get("blocked_reason") is not None:
            series_plane_blocks = [
                b
                for b in topology_blocks
                if b.slice_index == slice_index and b.frame_index == frame_index
            ]
            if series_plane_blocks:
                blocked_reason = reopened["blocked_reason"]
        merged = viewer_state.merged_contours_at(slice_index, frame_index)
        for contour in merged:
            contours_list.append(
                {
                    "structure": _structure_name(contour.structure),
                    "source": contour.source,
                    "edited": bool(contour.edited),
                    "draft": False,
                    "points_pixel": [[float(x), float(y)] for x, y in contour.points],
                }
            )
        # §1a display merge (2026-09-09 C1): fill in structures the chain does
        # NOT carry but the experimental whole-cine prediction does, using the
        # SAME emission helper as the prediction-only branch. Chain geometry
        # wins wherever present (filled structures are exactly the
        # chain-absent ones); nothing is written to revisions.
        fill_in = _chain_absent_experimental_contours(
            store,
            series_uid,
            slice_index=slice_index,
            frame_index=frame_index,
            present_structures={contour.structure for contour in merged},
        )
        contours_list.extend(fill_in)
        occupied = {contour.structure for contour in merged}
        occupied.update(
            next(
                structure for structure in Structure
                if structure.name == item.get("structure")
            )
            for item in fill_in
            if item.get("structure") in {structure.name for structure in Structure}
        )
        if sparse_pred is not None and frame_index in sparse_pred.source_frame_indices:
            contributed = False
            for structure in Structure:
                if structure in occupied:
                    continue
                source_contours = sparse_pred.contours.get(
                    (slice_index, frame_index, structure), ()
                )
                if not source_contours:
                    continue
                item = _experimental_contour_dict(
                    structure, source_contours, source="ai"
                )
                item["contour_source"] = "experimental_sparse_ai"
                contours_list.append(item)
                occupied.add(structure)
                contributed = True
            if contributed:
                sparse_contributed = True

    else:
        head_revision_id = None
        pred = store.load_latest_whole_cine_prediction(series_uid)
        occupied: set[Structure] = set()
        whole_cine_contributed = False
        if pred is not None:
            # B10-live Option B: pre-bootstrap AI pre-contour display is
            # EXPERIMENTAL (research only). The flag is additive; absence of
            # the key means the source is NOT the experimental pre-contour.
            contour_source = "experimental_precontour"
            for structure in Structure:
                key = (slice_index, frame_index, structure)
                source_contours = pred.contours.get(key, ())
                if not source_contours:
                    continue
                contours_list.append(
                    _experimental_contour_dict(
                        structure,
                        source_contours,
                        source=pred.source if pred.source == "ai" else "ai",
                    )
                )
                occupied.add(structure)
                whole_cine_contributed = True
        if sparse_pred is not None and frame_index in sparse_pred.source_frame_indices:
            for structure in Structure:
                if structure in occupied:
                    continue
                source_contours = sparse_pred.contours.get(
                    (slice_index, frame_index, structure), ()
                )
                if not source_contours:
                    continue
                item = _experimental_contour_dict(
                    structure, source_contours, source="ai"
                )
                item["contour_source"] = "experimental_sparse_ai"
                contours_list.append(item)
                sparse_contributed = True
        # F7: a genuinely sparse-only plane (no whole-cine geometry on it,
        # only eligible sparse geometry) carries the top-level sparse
        # marker, so the renderer resolves `full` — exactly like the
        # whole-cine pre-contour plane. Mixed whole-cine+sparse planes keep
        # the whole-cine marker; chain+sparse planes keep per-contour
        # markers only and stay `mixed`.
        if sparse_contributed and not whole_cine_contributed and contour_source is None:
            contour_source = "experimental_sparse_ai"
        if pred is None and not sparse_contributed:
            # Neither exists -> contours: []
            pass

    payload: dict[str, Any] = {
        "series_uid": series_uid,
        "slice": slice_index,
        "frame": frame_index,
        "head_revision_id": head_revision_id,
        "whole_cine_head_id": whole_cine_head_id,
        "contours": contours_list,
        "blocked_reason": blocked_reason,
    }
    if contour_source is not None:
        # Additive B10-live Option B marker: only present when the displayed
        # contours come from the AI pre-contour (prediction-only projection).
        payload["contour_source"] = contour_source
    if sparse_contributed and sparse_pred is not None:
        payload["sparse_ai_prediction_id"] = sparse_pred.prediction_id
        payload["sparse_ai_source_frames"] = list(sparse_pred.source_frame_indices)
    return payload


# ---------------------------------------------------------------------------
# FNC-G1 (CMR_FNCG_EVIDENCE_001): read-only evidence projection.
#
# This surface composes EXISTING persisted/projected records only. It never
# writes, never computes a clinical quantity, never emits geometry bytes and
# never emits absolute host paths. The per-key authority tally is derived
# from the SAME reads ``resolve_study_contours`` uses:
#
#   chain occupancy (``ViewerState.merged_contours_at``)
#   > latest whole-cine prediction
#   > A1-ELIGIBLE sparse artifact
#
# The sparse read uses ONLY the current-binding accessor
# (``load_current_selected_phase_prediction``): an unbound or stale sparse
# artifact is audit-only and reads ``missing``. The audit-only
# ``load_latest_selected_phase_prediction`` reader is NEVER consulted here.
# ---------------------------------------------------------------------------

#: Stable schema version of the evidence response envelope.
EVIDENCE_VERSION = 1

#: Authority source vocabulary (deterministic; tests pin the spellings).
EVIDENCE_SOURCE_PHYSICIAN = "physician"
EVIDENCE_SOURCE_WHOLE_CINE_AI = "whole_cine_ai"
EVIDENCE_SOURCE_SPARSE_AI = "sparse_ai"
EVIDENCE_SOURCE_MISSING = "missing"

#: Per-key provenance markers. The whole-cine and sparse spellings mirror the
#: ``resolve_study_contours`` per-contour markers EXACTLY; ``physician_chain``
#: labels a key served by the legacy merged physician chain.
EVIDENCE_CONTOUR_PHYSICIAN = "physician_chain"
EVIDENCE_CONTOUR_WHOLE_CINE = "experimental_precontour"
EVIDENCE_CONTOUR_SPARSE = "experimental_sparse_ai"

_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _evidence_relative_name(value: Any) -> str | None:
    """A relative provenance label (A8), or ``None`` for anything path-like.

    Only relative spellings pass; an absolute POSIX/Windows path, a ``\\\\``
    separator, a NUL byte or a ``..`` traversal segment is DROPPED (never
    stripped into a guessed basename), so host paths cannot leave this
    projection.
    """
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("/") or "\\" in value or "\x00" in value:
        return None
    if _WINDOWS_ABSOLUTE_RE.match(value):
        return None
    if any(segment == ".." for segment in value.split("/")):
        return None
    return value


def _evidence_digest_label(value: Any) -> str | None:
    """A digest label, or ``None``; path-shaped spellings never pass."""
    if not isinstance(value, str) or not value:
        return None
    if "/" in value or "\\" in value or "\x00" in value:
        return None
    return value


def _evidence_ai_row(
    store: SqliteCmrStore,
    prediction: WholeCinePrediction | SelectedPhasePrediction,
    *,
    artifact_domain: str,
    bound_job: Any | None,
) -> dict[str, Any]:
    """One AI provenance row (provenance, never authority).

    Relative checkpoint names + digests ONLY (A8): the persisted
    ``stage*_artifact_path`` convention is reused and every path-shaped value
    is dropped. Model-run enrichment is best effort with a typed, path-safe
    log line; the prediction's own content identity is always present.
    """
    model = (
        prediction.model
        if prediction.model is not None
        else prediction.provenance.model
    )
    model_run = None
    if prediction.model_run_id:
        # A missing/unreadable model-run row is a STORE FAILURE, not an
        # absent provenance: the AI row would otherwise present a complete
        # projection with silently dropped model identity (R1 P2). Let it
        # propagate to the sanitized endpoint boundary.
        model_run = store.load_model_run(prediction.model_run_id)
    runtime: Mapping[str, Any] = {}
    if model_run is not None and isinstance(model_run.spatial_transforms, Mapping):
        candidate = model_run.spatial_transforms.get("x1_runtime")
        if isinstance(candidate, Mapping):
            runtime = candidate
    stage1_name = _evidence_relative_name(
        runtime.get("stage1_artifact_path")
    ) or _evidence_relative_name(runtime.get("stage1_checkpoint"))
    stage2_name = _evidence_relative_name(
        runtime.get("stage2_artifact_path")
    ) or _evidence_relative_name(runtime.get("stage2_checkpoint"))
    device = runtime.get("device")
    # R2 P1: the runtime device is an untrusted persisted string — it must
    # pass the same path-safe label projection as every other exported
    # metadata field (an absolute-path device would otherwise leave the
    # payload verbatim). The job-device fallback was already filtered.
    device = _evidence_relative_name(device)
    if device is None and bound_job is not None:
        device = _evidence_relative_name(bound_job.device)
    algorithm_version = None
    if bound_job is not None:
        try:
            job_payload = json.loads(bound_job.request_payload)
        except (TypeError, ValueError):
            job_payload = None
        if isinstance(job_payload, Mapping):
            candidate_version = job_payload.get("algorithm_version")
            # Same path-safe rule: persisted job payload strings are
            # untrusted; a path-shaped version label is dropped, not echoed.
            algorithm_version = _evidence_relative_name(candidate_version)
    model_sha256 = None
    if model_run is not None:
        model_sha256 = _evidence_digest_label(model_run.model_sha256)
    if model_sha256 is None and model is not None:
        model_sha256 = _evidence_digest_label(model.checksum)
    # R2 P1: every other raw model/framework string is untrusted persisted
    # metadata — project it through the same path-safe labels. Structured
    # identity (name/version/digests) is preserved; free-text spellings that
    # are path-shaped are dropped to None rather than echoed.
    model_name = None if model is None else _evidence_relative_name(model.name)
    model_version = None if model is None else _evidence_relative_name(model.version)
    model_preprocessing = (
        None if model is None else _evidence_relative_name(model.preprocessing_version)
    )
    model_checksum_label = None if model is None else _evidence_digest_label(model.checksum)
    framework_label = (
        None if model_run is None else _evidence_relative_name(model_run.framework)
    )
    return {
        "prediction_id": prediction.prediction_id,
        "artifact_domain": artifact_domain,
        "source": prediction.source,
        "model_run_id": prediction.model_run_id,
        "model": (
            None
            if model is None
            else {
                "name": model_name,
                "version": model_version,
                "sha256": model_sha256,
                "preprocessing_version": model_preprocessing,
                "checksum": model_checksum_label,
            }
        ),
        "framework": framework_label,
        "device": device,
        "algorithm_version": algorithm_version,
        "checkpoints": {
            "stage1": {
                "relative_name": stage1_name,
                "sha256": _evidence_digest_label(runtime.get("stage1_sha256")),
            },
            "stage2": {
                "relative_name": stage2_name,
                "sha256": _evidence_digest_label(runtime.get("stage2_sha256")),
            },
        },
        "software_version": prediction.provenance.software_version,
        "created_at": prediction.created_at,
    }


def build_evidence_projection(
    controller: WorkflowController,
    store: SqliteCmrStore,
    series_uid: str,
    cine: CineVolume,
) -> dict[str, Any]:
    """Compose the read-only FNC-G1 evidence projection (``evidence_version`` 1).

    One NEW response shape: no frozen-projection ordering constraint applies,
    but every key is insertion-ordered and pinned by tests. All reads are
    EXISTING persisted/projected records — no writes, no clinical arithmetic,
    no geometry bytes, no absolute paths.

    Authority ranks chain > whole-cine AI > A1-eligible sparse AI per required
    five-group key (``SELECTED_PHASE_GROUPS`` over the cine slice extent).
    Without a persisted selection the ED/ES frames are undefined: every entry
    is reported ``missing`` with ``frame: null`` and no frame is ever invented.
    """
    reopened = controller.reopen(store, series_uid, cine=cine)
    chain = tuple(reopened["chain"])
    selection = reopened["latest_selection"]
    viewer_state, _ = build_viewer_state(
        controller, store, series_uid, cine, data=reopened
    )
    head = chain[-1] if chain else None

    # --- 1. per-required-key authority (same reads as the resolver) --------
    whole_prediction = store.load_latest_whole_cine_prediction(series_uid)
    sparse_prediction: SelectedPhasePrediction | None = None
    if selection is not None:
        # A1: exactly the eligibility identity ``resolve_study_contours``
        # resolves (current legacy head + latest selection). The audit-only
        # latest reader is never used here.
        sparse_prediction = store.load_current_selected_phase_prediction(
            series_uid,
            head_revision_id=(None if head is None else head.revision_id),
            selection_id=selection.selection_id,
        )

    merged_cache: dict[tuple[int, int], set[Structure]] = {}

    def _merged_structures(slice_index: int, frame_index: int) -> set[Structure]:
        cache_key = (slice_index, frame_index)
        cached = merged_cache.get(cache_key)
        if cached is None:
            cached = {
                contour.structure
                for contour in viewer_state.merged_contours_at(slice_index, frame_index)
            }
            merged_cache[cache_key] = cached
        return cached

    authority: list[dict[str, Any]] = []
    serving_whole_cine = False
    serving_sparse = False
    for slice_index in range(cine.n_slices):
        for role, structure in SELECTED_PHASE_GROUPS:
            frame_index = (
                None
                if selection is None
                else (selection.ed_frame if role == "ed" else selection.es_frame)
            )
            source = EVIDENCE_SOURCE_MISSING
            contour_source: str | None = None
            if frame_index is not None:
                required_key = (slice_index, frame_index, structure)
                if structure in _merged_structures(slice_index, frame_index):
                    source = EVIDENCE_SOURCE_PHYSICIAN
                    contour_source = EVIDENCE_CONTOUR_PHYSICIAN
                elif (
                    whole_prediction is not None
                    and whole_prediction.contours.get(required_key)
                ):
                    source = EVIDENCE_SOURCE_WHOLE_CINE_AI
                    contour_source = EVIDENCE_CONTOUR_WHOLE_CINE
                    serving_whole_cine = True
                elif (
                    sparse_prediction is not None
                    and frame_index in sparse_prediction.source_frame_indices
                    and sparse_prediction.contours.get(required_key)
                ):
                    source = EVIDENCE_SOURCE_SPARSE_AI
                    contour_source = EVIDENCE_CONTOUR_SPARSE
                    serving_sparse = True
            authority.append(
                {
                    "group": role,
                    "structure": structure.name,
                    "slice": slice_index,
                    "frame": frame_index,
                    "source": source,
                    "contour_source": contour_source,
                }
            )

    # --- 7. inference jobs (same row shape as the merged /jobs feed) -------
    # A loader failure is a STORE FAILURE, not "no jobs": substituting []
    # would present unavailable evidence as absent history (R1 P2). Let it
    # propagate to the sanitized endpoint boundary.
    inference_jobs = list(store.load_inference_jobs(series_uid))
    inference_jobs.sort(key=lambda job: (job.created_at, job.job_id))
    inference_rows = [
        {
            "job_id": job.job_id,
            "kind": "selected_phase_inference",
            "state": job.state,
            "error_code": job.error_code,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "timings": json.loads(job.timings),
        }
        for job in inference_jobs
    ]

    def _bound_succeeded_job(prediction_id: str) -> Any | None:
        candidates = [
            job
            for job in inference_jobs
            if job.prediction_id == prediction_id and job.state == "succeeded"
        ]
        return candidates[-1] if candidates else None

    # --- 6. AI provenance rows (for each source currently serving keys) ----
    ai_rows: list[dict[str, Any]] = []
    if serving_whole_cine and whole_prediction is not None:
        ai_rows.append(
            _evidence_ai_row(
                store,
                whole_prediction,
                artifact_domain="WholeCinePrediction",
                bound_job=None,
            )
        )
    if serving_sparse and sparse_prediction is not None:
        ai_rows.append(
            _evidence_ai_row(
                store,
                sparse_prediction,
                artifact_domain="SelectedPhasePrediction",
                bound_job=_bound_succeeded_job(sparse_prediction.prediction_id),
            )
        )

    # --- 4. QC rows (head gateway report + durable decisions) --------------
    report = viewer_state.qc_report
    qc_section = {
        "qc_digest": None if report is None else _qc_digest(report),
        "qc_verdict": None if report is None else report.verdict,
        "qc_flags": (
            []
            if report is None
            else [
                {
                    "code": flag.code,
                    "message": flag.message,
                    "source": flag.source,
                    "severity": flag.severity.value,
                }
                for flag in report.flags
            ]
        ),
        "qc_decisions": [
            {
                "decision_id": record.decision_id,
                "source_revision_id": record.source_revision_id,
                "selection_id": record.selection_id,
                "qc_verdict": record.qc_verdict,
                "approved_by": record.approved_by,
                "approved_at": record.approved_at,
                # The rationale payload stays server-side; only the presence
                # of a recorded override is exposed.
                "override": bool(record.override_reason.strip()),
            }
            for record in viewer_state.qc_decisions
        ],
    }

    # --- 5. approvals (content-addressed identity + sealed binding) --------
    # Each row also carries the FNC-G2 derivation-only ``reference`` view: the
    # sealed content identity (``approval_id``) plus the binding pointers that
    # identity covers, derived through
    # ``ApprovedResultReference.from_snapshot``. No new seal and no new
    # persistence: the source is the same digest-re-verified snapshot row.
    approvals = []
    for snapshot in viewer_state.approvals:
        reference = ApprovedResultReference.from_snapshot(snapshot)
        approvals.append(
            {
                "approval_id": snapshot.approval_id,
                "measurement_id": snapshot.measurement_id,
                "source_revision_id": snapshot.source_revision_id,
                "selection_id": snapshot.selection_id,
                "inclusion_set_id": snapshot.inclusion_set_id,
                "qc_verdict": snapshot.qc_verdict,
                "qc_report_digest": snapshot.qc_report_digest,
                "topology_disposition_digest": snapshot.topology_disposition_digest,
                "approved_by": snapshot.approved_by,
                "approved_at": snapshot.approved_at,
                "binding": {
                    "head_revision_id": snapshot.source_revision_id,
                    "selection_id": snapshot.selection_id,
                    "inclusion_set_id": snapshot.inclusion_set_id,
                    "measurement_id": snapshot.measurement_id,
                    "qc_verdict": snapshot.qc_verdict,
                    "qc_report_digest": snapshot.qc_report_digest,
                    "topology_disposition_digest": snapshot.topology_disposition_digest,
                },
                "reference": {
                    "approval_id": reference.approval_id,
                    "series_uid": reference.series_uid,
                    "head_revision_id": reference.head_revision_id,
                    "selection_id": reference.selection_id,
                    "measurement_id": reference.measurement_id,
                    "qc_verdict": reference.qc_verdict,
                    "qc_report_digest": reference.qc_report_digest,
                    "inclusion_set_id": reference.inclusion_set_id,
                    "topology_disposition_digest": (
                        reference.topology_disposition_digest
                    ),
                },
            }
        )

    # --- 2. DICOM / provenance identity ------------------------------------
    manifest = reopened.get("source_manifest")
    provenance_section = {
        "workflow_id": reopened.get("workflow_id"),
        "series_uid": series_uid,
        "study_uid": cine.study_uid,
        "cine_digest": cine.cine_volume_digest,
        "manifest_digest": None if manifest is None else manifest.workflow_id,
        "n_frames": cine.n_frames,
        "n_slices": cine.n_slices,
        "source_manifest_file_count": (
            None if manifest is None else len(manifest.files)
        ),
    }

    # --- 3. phase / revision / inclusion rows (reopen semantics) -----------
    persisted_inclusion = reopened.get("applicable_inclusion_set")
    measurement = reopened.get("latest_measurement")
    historical_measurement = reopened.get("latest_historical_measurement")
    lineage_section = {
        "head_revision_id": None if head is None else head.revision_id,
        "head_version": None if head is None else head.version,
        "selection": (
            None
            if selection is None
            else {
                "ed_frame": selection.ed_frame,
                "es_frame": selection.es_frame,
                "selection_id": selection.selection_id,
            }
        ),
        "inclusion_set_id": (
            None
            if persisted_inclusion is None
            else persisted_inclusion.inclusion_set_id
        ),
        "inclusion_status": viewer_state.inclusion_status_label,
        "measurement_id": None if measurement is None else measurement.measurement_id,
        "latest_historical_measurement_id": (
            None
            if historical_measurement is None
            else historical_measurement.measurement_id
        ),
        "has_stale_analysis": bool(reopened.get("has_stale_analysis")),
    }

    return {
        "series_uid": series_uid,
        "evidence_version": EVIDENCE_VERSION,
        "sections": {
            "authority": authority,
            "provenance": provenance_section,
            "lineage": lineage_section,
            "qc": qc_section,
            "approvals": approvals,
            "ai": ai_rows,
            "inference_jobs": inference_rows,
        },
    }
