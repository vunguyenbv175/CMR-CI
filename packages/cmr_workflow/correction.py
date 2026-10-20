"""Whole-cine physician correction coordinator.

Corrections are full-state immutable revisions.  This coordinator performs
all parent and state checks before constructing a revision, then delegates the
single persistence operation to the frozen append-only store API.  It does
not create QC decisions, approvals, measurements, or new failure codes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from cmr_core.domain_types import (
    CineContourMapping,
    WholeCinePrediction,
    WholeCineRevision,
    canonical_json,
)
from cmr_core.provenance import Provenance
from cmr_core.segmentation import Structure
from cmr_persistence.store import SqliteCmrStore

MaskKey = tuple[int, int, Structure]


@dataclass(frozen=True)
class CorrectionOutcome:
    """The only two outcomes exposed by the M6 correction coordinator."""

    outcome: Literal["revision-persisted-v1", "rejected-v1"]
    revision_id: str | None
    rejection: str | None

    def __post_init__(self) -> None:
        if self.outcome not in ("revision-persisted-v1", "rejected-v1"):
            raise ValueError(f"unknown correction outcome {self.outcome!r}")
        if self.outcome == "revision-persisted-v1":
            if not isinstance(self.revision_id, str) or not self.revision_id:
                raise ValueError("persisted correction outcome requires revision_id")
            if self.rejection is not None:
                raise ValueError("persisted correction outcome cannot carry a rejection")
        else:
            if self.revision_id is not None:
                raise ValueError("rejected correction outcome cannot carry revision_id")
            if not isinstance(self.rejection, str) or not self.rejection:
                raise ValueError("rejected correction outcome requires rejection")


def _key_string(key: object) -> str:
    if (
        not isinstance(key, tuple)
        or len(key) != 3
        or isinstance(key[0], bool)
        or not isinstance(key[0], int)
        or key[0] < 0
        or isinstance(key[1], bool)
        or not isinstance(key[1], int)
        or key[1] < 0
        or not isinstance(key[2], Structure)
    ):
        return repr(key)
    return f"{key[0]}:{key[1]}:{key[2].value}"


def _rejected(message: str) -> CorrectionOutcome:
    return CorrectionOutcome(
        outcome="rejected-v1",
        revision_id=None,
        rejection=message or "correction rejected",
    )


def _projection(values: object) -> str:
    # canonical_json recursively handles the immutable HoleAwareContour tuple
    # and is the same projection used by the frozen record digest path.
    return canonical_json(values)


def _load_parent(
    store: SqliteCmrStore,
    prediction_id: str,
    parent_revision_id: str | None,
) -> (
    tuple[WholeCinePrediction, WholeCinePrediction | WholeCineRevision, int, str | None]
    | CorrectionOutcome
):
    """Load and bind all parents before inspecting candidate edit state."""

    try:
        prediction = store.load_whole_cine_prediction(prediction_id)
    except Exception as exc:
        return _rejected(str(exc))
    if prediction is None:
        return _rejected(f"prediction {prediction_id!r} is not present")

    if parent_revision_id is None:
        return prediction, prediction, 1, None

    try:
        parent_revision = store.load_whole_cine_revision(parent_revision_id)
    except Exception as exc:
        return _rejected(str(exc))
    if parent_revision is None:
        return _rejected(f"parent revision {parent_revision_id!r} is not present")
    if parent_revision.parent_prediction_id != prediction.prediction_id:
        # Keep the store's frozen coherence wording, which is also what a
        # direct append attempt would report for a lineage splice.
        return _rejected(
            "parent_prediction_id switch rejected: child anchors "
            f"{prediction.prediction_id!r} but parent revision anchors "
            f"{parent_revision.parent_prediction_id!r}"
        )
    if parent_revision.study_uid != prediction.study_uid:
        return _rejected(
            "revision study_uid "
            f"{prediction.study_uid!r} does not match parent revision study_uid "
            f"{parent_revision.study_uid!r}"
        )
    if parent_revision.series_uid != prediction.series_uid:
        return _rejected(
            "revision series_uid "
            f"{prediction.series_uid!r} does not match parent revision series_uid "
            f"{parent_revision.series_uid!r}"
        )
    return prediction, parent_revision, parent_revision.version + 1, parent_revision_id


def _validate_full_state(
    *,
    prediction: WholeCinePrediction,
    parent: WholeCinePrediction | WholeCineRevision,
    edited_keys: tuple[MaskKey, ...],
    contours: CineContourMapping,
) -> str | None:
    if not isinstance(contours, Mapping):
        return "correction contours must be a mapping"
    expected_keys = set(prediction.contours)
    try:
        provided_keys = set(contours)
        parent_keys = set(parent.contours)
    except (TypeError, ValueError) as exc:
        return str(exc)
    if provided_keys != expected_keys:
        return "correction contours must cover exactly the prediction key set"
    if parent_keys != expected_keys:
        return "parent state does not cover exactly the prediction key set"

    if not isinstance(edited_keys, tuple):
        # WholeCineRevision accepts any iterable but the public M6 contract is
        # a tuple.  Report the same frozen non-empty/duplicate semantics below
        # after converting only the caller-owned key container.
        try:
            edited_keys = tuple(edited_keys)
        except TypeError as exc:
            return str(exc)
    if not edited_keys:
        return "edited_keys must be non-empty"
    seen: set[MaskKey] = set()
    for key in edited_keys:
        try:
            already_seen = key in seen
        except TypeError:
            return f"edited key {_key_string(key)!r} does not exist"
        if already_seen:
            return f"duplicate edited key {_key_string(key)!r}"
        try:
            seen.add(key)
        except TypeError:
            return f"edited key {_key_string(key)!r} does not exist"
        if key not in expected_keys:
            return f"edited key {_key_string(key)!r} does not exist"

    for key in expected_keys:
        try:
            current_projection = _projection(contours[key])
            parent_projection = _projection(parent.contours[key])
        except Exception as exc:
            return str(exc)
        if key in seen:
            if current_projection == parent_projection:
                return f"edited key {_key_string(key)!r} is unchanged from parent"
        elif current_projection != parent_projection:
            return f"unedited key {_key_string(key)!r} differs from parent"
    return None


def apply_physician_correction(
    *,
    store: SqliteCmrStore,
    prediction_id: str,
    parent_revision_id: str | None,
    edited_keys: tuple[MaskKey, ...],
    contours: CineContourMapping,
    editor: str,
    provenance: Provenance,
    created_at: str,
) -> CorrectionOutcome:
    """Create and append one full-state physician revision, or reject it."""

    loaded = _load_parent(store, prediction_id, parent_revision_id)
    if isinstance(loaded, CorrectionOutcome):
        return loaded
    prediction, parent, version, bound_parent_revision_id = loaded

    validation_error = _validate_full_state(
        prediction=prediction,
        parent=parent,
        edited_keys=edited_keys,
        contours=contours,
    )
    if validation_error is not None:
        return _rejected(validation_error)

    try:
        # Preserve the caller-supplied provenance fields while deriving only
        # the chain position.  In particular, do not call the legacy
        # ``Provenance.bumped`` helper here: it assigns a new wall-clock
        # timestamp, which would make a fixed-timestamp correction identity
        # nondeterministic.
        derived_provenance = Provenance(
            study_uid=provenance.study_uid,
            series_uid=provenance.series_uid,
            software_version=provenance.software_version,
            model=provenance.model,
            created_at=provenance.created_at,
            revision=version - 1,
            workflow_id=provenance.workflow_id,
        )
        revision = WholeCineRevision(
            study_uid=prediction.study_uid,
            series_uid=prediction.series_uid,
            version=version,
            parent_prediction_id=prediction.prediction_id,
            parent_revision_id=bound_parent_revision_id,
            editor=editor,
            edited_keys=tuple(edited_keys),
            contours=contours,
            provenance=derived_provenance,
            created_at=created_at,
        )
        persisted_id = store.save_whole_cine_revision(revision)
    except Exception as exc:
        # In particular, preserve SQLite's exact ``append-only`` abort text;
        # the store transaction has already rolled back before it returns.
        return _rejected(str(exc))
    return CorrectionOutcome(
        outcome="revision-persisted-v1",
        revision_id=persisted_id,
        rejection=None,
    )


__all__ = ["CorrectionOutcome", "apply_physician_correction"]
