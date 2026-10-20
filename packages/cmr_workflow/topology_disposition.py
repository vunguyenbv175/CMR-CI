"""Topology disposition + approval relevance resolver (Astra F1-F5, 2026-09-09).

This module replaces the former SERIES-WIDE approval prohibition ("any
persisted :class:`~cmr_core.domain_types.TopologyBlock` blocks the whole
series") with the key-level relevance-and-resolution predicate the owner-level
Astra review requires, and it is the ONE place that predicate lives so the
controller, the workstation wire action and the tests cannot drift.

The predicate (Astra "Recommended Minimum Safe Semantics")::

    For each persisted TopologyBlock of the series:
      relevant = the block's key is required by the SELECTED Cine measurement
                 (LV_CAVITY / RV_CAVITY at every (slice, frame) of the cine
                 grid, plus LV_MYOCARDIUM at the selected ED frame — exactly
                 the required-key set ``compute_measurement`` validates).
      if the CURRENT revision-chain head carries physician geometry for the
         exact key            -> RESOLVED_BY_REPLACEMENT
      elif a TopologyDisposition exists for the exact key, bound to the exact
         current head AND selection_id, with disposition EXCLUDED or REPLACED
                               -> RESOLVED_BY_DISPOSITION
      elif NOT relevant        -> UNRESOLVED_IRRELEVANT (visible, non-approved,
                                does NOT block)
      else                     -> TOPOLOGY_BLOCKED (typed, names the key)

Two hard invariants:

* Relevance is derived from :func:`required_metric_keys` — the SAME source of
  truth :func:`cmr_quantification.measurements.build_cine_contour_index` /
  ``compute_measurement`` use. No second notion of required keys exists here.
* A disposition whose bound head or selection no longer matches the CURRENT
  head/selection is STALE: it is NOT a resolution and the approval fails
  closed naming the stale binding (Astra F5 / required test 3).

Nothing in this module supplies geometry. ``EXCLUDED`` means "intentionally
absent from quantification", never "zero" and never "use AI/fill-in instead".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from cmr_core.domain_types import TopologyBlock
from cmr_core.errors import CMRError, TOPOLOGY_BLOCKED
from cmr_core.segmentation import Structure
from cmr_persistence.records import (
    QC_TOPOLOGY_MISSING_DATA,
    QC_TOPOLOGY_PHYSICIAN_EXCLUDED,
    QC_TOPOLOGY_UNRESOLVED,
    SliceInclusionFrameRole,
    TopologyDisposition,
    TopologyDispositionRecord,
)

__all__ = [
    "TopologyResolution",
    "RESOLVED_BY_DISPOSITION",
    "RESOLVED_BY_REPLACEMENT",
    "UNRESOLVED_BLOCKING",
    "UNRESOLVED_IRRELEVANT",
    "STALE_DISPOSITION",
    "required_metric_keys",
    "resolve_topology_dispositions",
    "assert_topology_resolved_for_approval",
    "resolution_projection",
    "disposition_role_for_frame",
]


# --- resolution kinds (typed strings, contract item 7) -----------------------

#: The exact current chain head carries physician geometry for the key.
RESOLVED_BY_REPLACEMENT = "topology.resolved_by_replacement"
#: A physician disposition bound to the exact head + selection resolves it.
RESOLVED_BY_DISPOSITION = "topology.resolved_by_disposition"
#: Visible and non-blocking: the key is not required by the selected metrics.
UNRESOLVED_IRRELEVANT = "topology.unresolved_irrelevant"
#: Metric-relevant and NOT resolved -> approval fails closed.
UNRESOLVED_BLOCKING = "topology.unresolved_blocking"
#: A disposition exists but is bound to a different head/selection -> stale.
STALE_DISPOSITION = "topology.stale_disposition"


def required_metric_keys(
    *,
    n_slices: int,
    n_frames: int,
    ed_frame: int,
) -> frozenset[tuple[int, int, Structure]]:
    """The exact required-key set of the selected Cine measurement.

    Single source of truth, mirrored from
    :func:`cmr_quantification.measurements.compute_measurement` (which fails
    closed on ``required.difference(index)``) and from
    ``materialize_legacy_chain_from_whole_cine``'s ``required_keys`` preflight:
    ``LV_CAVITY`` and ``RV_CAVITY`` at every (slice, frame), plus
    ``LV_MYOCARDIUM`` at every slice at the selected ED frame.
    """
    keys: set[tuple[int, int, Structure]] = set()
    for slice_index in range(n_slices):
        for frame_index in range(n_frames):
            keys.add((slice_index, frame_index, Structure.LV_CAVITY))
            keys.add((slice_index, frame_index, Structure.RV_CAVITY))
        keys.add((slice_index, ed_frame, Structure.LV_MYOCARDIUM))
    return frozenset(keys)


@dataclass(frozen=True)
class TopologyResolution:
    """The resolution verdict of ONE persisted topology block key."""

    block: TopologyBlock
    relevant: bool
    kind: str
    #: The disposition that resolved the key (``RESOLVED_BY_DISPOSITION``),
    #: or the stale one that did NOT resolve it (``STALE_DISPOSITION``).
    disposition: TopologyDispositionRecord | None = None
    #: Why this key is (or is not) metric-relevant; human-readable.
    detail: str = ""

    @property
    def key(self) -> tuple[int, int, str]:
        return (
            self.block.slice_index,
            self.block.frame_index,
            self.block.structure,
        )

    @property
    def key_text(self) -> str:
        return (
            f"(slice={self.block.slice_index}, frame={self.block.frame_index}, "
            f"structure={self.block.structure})"
        )

    @property
    def blocks_approval(self) -> bool:
        """True only for a metric-relevant, unresolved key (or a stale one)."""
        return self.kind in (UNRESOLVED_BLOCKING, STALE_DISPOSITION)

    @property
    def qc_code(self) -> str:
        """Typed QC distinction for this key (contract item 7)."""
        if self.kind == RESOLVED_BY_DISPOSITION and (
            self.disposition is not None
            and self.disposition.disposition is TopologyDisposition.EXCLUDED
        ):
            return QC_TOPOLOGY_PHYSICIAN_EXCLUDED
        if self.kind == RESOLVED_BY_REPLACEMENT:
            return QC_TOPOLOGY_PHYSICIAN_EXCLUDED
        if self.kind == UNRESOLVED_IRRELEVANT:
            return QC_TOPOLOGY_MISSING_DATA
        return QC_TOPOLOGY_UNRESOLVED


def resolve_topology_dispositions(
    *,
    blocks: list[TopologyBlock],
    chain_key_times: Mapping[tuple[int, int, Structure], str],
    dispositions: list[TopologyDispositionRecord],
    head_revision_id: str,
    selection_id: str,
    n_slices: int | None,
    n_frames: int | None,
    ed_frame: int | None,
    es_frame: int = 0,
) -> list[TopologyResolution]:
    """Resolve every persisted block key against the CURRENT clinical state.

    ``chain_key_times`` maps every key whose geometry the CURRENT revision
    chain AUTHORED to the ``created_at`` stamp of that authoring event. It is
    NOT "the latest stamp of any revision carrying the key" (R1 review F1,
    P1): a revision that merely re-carries identical contours — a
    ``snapshot_qc`` refresh copies its parent's contour map VERBATIM — is not
    a geometry-authoring event, and a newer stamp derived from it must never
    resolve a block. The caller derives the map by comparing each revision's
    contour content for the key against what the chain carried before it.

    ``dispositions`` may be the series' whole append-only history (R1 review
    F2, P1): the resolver first selects the APPLICABLE record per key — the
    one bound to the current ``head_revision_id`` and ``selection_id`` — and
    only that record can resolve a key. A row bound to a different
    head/selection is provenance, never a resolution, and a key whose ONLY
    record is such a row is reported as :data:`STALE_DISPOSITION` (visible
    and blocking: the physician must re-dispose the key against the current
    state, which the F2 re-disposition path now permits).

    Resolved-by-replacement requires physician geometry authored AFTER the
    block was recorded: a revision that merely predates (or coincides with)
    the durable block is not a replacement of it, it is the state the block
    was recorded against. An unparseable timestamp therefore fails CLOSED
    (never counts as a replacement).

    Fail-closed grid rule: when the measurement grid (``n_slices`` /
    ``n_frames`` / ``ed_frame``) is NOT available at the call site, relevance
    CANNOT be proven, so EVERY block key is treated as metric-relevant.
    Irrelevance is a positive claim about the selected metrics and is never
    inferred from a missing grid.
    """
    grid_known = n_slices is not None and n_frames is not None and ed_frame is not None
    required = (
        required_metric_keys(n_slices=n_slices, n_frames=n_frames, ed_frame=ed_frame)
        if grid_known
        else None
    )
    # R1 REVIEW F2 (P1): the APPLICABLE disposition of a key is the one bound
    # to the exact CURRENT head + selection. The append-only table may carry
    # an older row for the same key (one row per binding at most), so the
    # resolution predicate selects the current binding here — while KEEPING
    # the superseded rows addressable, so a key whose only record is stale is
    # reported as STALE_DISPOSITION (visible, blocking) instead of silently
    # resolving against a description of a different clinical state.
    applicable_by_key: dict[tuple[int, int, str], TopologyDispositionRecord] = {}
    history_by_key: dict[tuple[int, int, str], TopologyDispositionRecord] = {}
    for record in dispositions:
        record_key = (record.slice_index, record.frame_index, record.structure)
        if (
            record.source_revision_id == head_revision_id
            and record.selection_id == selection_id
        ):
            applicable_by_key[record_key] = record
        elif record_key not in history_by_key:
            # Deterministic fallback for the STALE report: the most recent
            # superseded row wins (``dispositions`` arrives in the store's
            # canonical ascending order, so later entries are newer).
            history_by_key[record_key] = record
    by_key: dict[tuple[int, int, str], TopologyDispositionRecord] = {
        **history_by_key,
        **applicable_by_key,
    }
    resolutions: list[TopologyResolution] = []
    for block in blocks:
        key = (block.slice_index, block.frame_index, block.structure)
        structure: Structure | None = _structure_or_none(block.structure)
        # R1 COMPLETION DEFECT FIX (2026-09-09): ``required_metric_keys`` is
        # keyed by :class:`Structure` ENUM members while a persisted
        # ``TopologyBlock.structure`` is its STRING value, and ``Structure`` is
        # a plain :class:`~enum.Enum` (NOT a str-enum): ``key in required``
        # therefore compared a str against an enum member and was ALWAYS
        # False, so every block key resolved as "not required" and the
        # relevance predicate never blocked anything. Relevance is evaluated
        # on the normalized ENUM key so both spellings mean the same key.
        enum_key = (
            None
            if structure is None
            else (block.slice_index, block.frame_index, structure)
        )
        relevant = (
            True
            if required is None
            else (enum_key is not None and enum_key in required)
        )
        if required is None:
            relevance_detail = (
                "relevance to the selected Cine measurement cannot be established "
                "(no CineVolume / measurement grid available), so the key is "
                "treated as metric-relevant"
            )
        elif relevant:
            relevance_detail = (
                f"required by the selected Cine measurement "
                f"({_relevance_reason(key[2], key[1], ed_frame, es_frame)})"
            )
        else:
            relevance_detail = (
                "not required by the selected Cine measurement "
                f"(n_slices={n_slices}, n_frames={n_frames}, ED frame {ed_frame})"
            )
        if structure is not None and _supersedes_block(chain_key_times, key, block):
            resolutions.append(
                TopologyResolution(
                    block=block,
                    relevant=relevant,
                    kind=RESOLVED_BY_REPLACEMENT,
                    detail=(
                        "the current revision chain carries physician geometry "
                        f"for key {key!r} authored after the block was recorded "
                        # Same enum/str normalization as the relevance lookup:
                        # ``chain_key_times`` is keyed by Structure members.
                        f"({chain_key_times[enum_key]!r} > {block.created_at!r})"
                    ),
                )
            )
            continue
        disposition = by_key.get(key)
        if disposition is not None:
            if (
                disposition.source_revision_id != head_revision_id
                or disposition.selection_id != selection_id
            ):
                resolutions.append(
                    TopologyResolution(
                        block=block,
                        relevant=relevant,
                        kind=STALE_DISPOSITION,
                        disposition=disposition,
                        detail=(
                            "disposition "
                            f"{disposition.disposition_id[:12]}.. is bound to head "
                            f"{disposition.source_revision_id[:12]}.. / selection "
                            f"{disposition.selection_id[:12]}.. but the current head "
                            f"is {head_revision_id[:12]}.. / selection "
                            f"{selection_id[:12]}..; the disposition is STALE and "
                            "does not resolve this key"
                        ),
                    )
                )
                continue
            resolutions.append(
                TopologyResolution(
                    block=block,
                    relevant=relevant,
                    kind=RESOLVED_BY_DISPOSITION,
                    disposition=disposition,
                    detail=(
                        f"physician {disposition.physician!r} recorded "
                        f"{disposition.disposition.value} for key {key!r} on the "
                        f"current head/selection: {disposition.reason}"
                    ),
                )
            )
            continue
        if not relevant:
            resolutions.append(
                TopologyResolution(
                    block=block,
                    relevant=False,
                    kind=UNRESOLVED_IRRELEVANT,
                    detail=relevance_detail,
                )
            )
            continue
        resolutions.append(
            TopologyResolution(
                block=block,
                relevant=True,
                kind=UNRESOLVED_BLOCKING,
                detail=(
                    f"key {key!r} is {relevance_detail} and is neither present as "
                    "physician geometry in the current chain head nor covered by a "
                    "physician topology disposition bound to the current head and "
                    "selection"
                ),
            )
        )
    resolutions.sort(key=lambda item: item.key)
    return resolutions


def assert_topology_resolved_for_approval(
    resolutions: list[TopologyResolution],
    *,
    series_uid: str,
    head_revision_id: str,
    selection_id: str,
) -> None:
    """Fail closed (typed :data:`TOPOLOGY_BLOCKED`) on any blocking resolution.

    The message names the EXACT unresolved key and why it is relevant, so the
    physician sees a visible, actionable reason (Astra required test 8).
    """
    blocking = [item for item in resolutions if item.blocks_approval]
    if not blocking:
        return
    first = blocking[0]
    if first.kind == STALE_DISPOSITION:
        raise CMRError(
            TOPOLOGY_BLOCKED,
            f"approval blocked for series {series_uid!r}: topology-disposition for "
            f"key {first.key_text} is STALE ({first.detail}); recording a "
            f"disposition against the current head {head_revision_id!r} and "
            f"selection {selection_id!r} is required",
        )
    names = ", ".join(item.key_text for item in blocking[:4])
    extra = len(blocking) - 4
    suffix = f" (+{extra} more)" if extra > 0 else ""
    raise CMRError(
        TOPOLOGY_BLOCKED,
        f"approval blocked for series {series_uid!r} head {head_revision_id!r} / "
        f"selection {selection_id!r}: {len(blocking)} metric-relevant topology "
        f"key(s) remain unresolved: {names}{suffix}; unresolved key "
        f"unresolved: {names}{suffix}; unresolved key "
        f"{first.key_text} is {first.detail}; each such key must be replaced by "
        "physician geometry in the current chain head or explicitly excluded by a "
        "physician topology disposition bound to this exact head and selection",
    )


def resolution_projection(resolution: TopologyResolution) -> dict[str, Any]:
    """Wire-safe projection of one resolution (additive, viewer-facing)."""
    payload: dict[str, Any] = {
        "slice_index": resolution.block.slice_index,
        "frame_index": resolution.block.frame_index,
        "structure": resolution.block.structure,
        "relevant": resolution.relevant,
        "kind": resolution.kind,
        "qc_code": resolution.qc_code,
        "blocks_approval": resolution.blocks_approval,
        "detail": resolution.detail,
        "block_reason": resolution.block.reason,
    }
    if resolution.disposition is not None:
        payload["disposition"] = {
            "disposition_id": resolution.disposition.disposition_id,
            "disposition": resolution.disposition.disposition.value,
            "physician": resolution.disposition.physician,
            "reason": resolution.disposition.reason,
            "source_revision_id": resolution.disposition.source_revision_id,
            "selection_id": resolution.disposition.selection_id,
        }
    return payload


def disposition_role_for_frame(
    *, frame_index: int, ed_frame: int, es_frame: int
) -> SliceInclusionFrameRole:
    """Frame role one disposition key speaks about (exact, never inferred)."""
    if frame_index == ed_frame:
        return SliceInclusionFrameRole.ED
    if frame_index == es_frame:
        return SliceInclusionFrameRole.ES
    # A key that is neither the ED nor the ES frame still has to declare a
    # role; ED is the conservative default ONLY when the frames coincide.
    if ed_frame == es_frame:
        return SliceInclusionFrameRole.ED
    raise CMRError(
        "app.invalid_argument",
        f"frame {frame_index} is neither the selected ED frame ({ed_frame}) nor "
        f"the ES frame ({es_frame}); a disposition must speak about a selected "
        "phase frame",
    )


def _supersedes_block(
    chain_key_times: Mapping[tuple[int, int, Structure], str],
    key: tuple[int, int, str],
    block: TopologyBlock,
) -> bool:
    """True when chain geometry for ``key`` was authored AFTER the block."""
    structure: Structure | None = _structure_or_none(key[2])
    if structure is None:
        return False
    authored = chain_key_times.get((key[0], key[1], structure))
    if authored is None:
        return False
    try:
        return _parse_utc(authored) > _parse_utc(block.created_at)
    except ValueError:
        # Fail closed: an unorderable stamp is never a replacement.
        return False


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp is not timezone-aware: {value!r}")
    return parsed.astimezone(timezone.utc)


def _structure_or_none(value: str) -> Structure | None:
    try:
        return Structure(value)
    except ValueError:
        return None


def _relevance_reason(
    structure: str, frame_index: int, ed_frame: int, es_frame: int
) -> str:
    if structure == Structure.LV_MYOCARDIUM.value:
        return f"LV_MYOCARDIUM at the selected ED frame {ed_frame}"
    phases = []
    if frame_index == ed_frame:
        phases.append("ED")
    if frame_index == es_frame:
        phases.append("ES")
    phase_text = "/".join(phases) if phases else f"frame {frame_index}"
    return f"{structure} at {phase_text}"
