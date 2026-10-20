"""Route frozen contour-QC warnings back to whole-cine prediction keys.

The legacy QC rule bundle accepts one :class:`Contour` per structure.  A
whole-cine prediction can retain multiple hole-aware components, so this
module supplies a read-only invocation adapter: it selects the largest
exterior component for the rule call and retains the exact key context for
each resulting warning.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from cmr_contours.conversion import _signed_area as contour_signed_area
from cmr_contours.model import Contour, ContourSet
from cmr_core.domain_types import HoleAwareContour, WholeCinePrediction, canonical_json
from cmr_core.errors import (
    QC_CONTAINMENT,
    QC_MISSING_STRUCTURE,
    QC_SELF_INTERSECTION,
    QC_VOLUME_CURVE,
    CMRWarning,
    Severity,
)
from cmr_core.segmentation import Structure

from . import rules as _rules

MaskKey = tuple[int, int, Structure]


@dataclass(frozen=True)
class KeyedQCObservation:
    """One frozen QC warning and the exact prediction keys it affects."""

    warning: CMRWarning
    keys: tuple[MaskKey, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.warning, CMRWarning):
            raise ValueError("warning must be a CMRWarning")
        if self.warning.severity is not Severity.QC:
            raise ValueError("prediction QC observations require QC severity")
        if not isinstance(self.keys, tuple) or not self.keys:
            raise ValueError("prediction QC observations require at least one key")
        normalized: list[MaskKey] = []
        for key in self.keys:
            _validate_key(key)
            if key not in normalized:
                normalized.append(key)
        if len(normalized) != len(self.keys):
            raise ValueError("prediction QC observation keys must be unique")
        object.__setattr__(self, "keys", tuple(normalized))


def _validate_key(key: object) -> None:
    if not isinstance(key, tuple) or len(key) != 3:
        raise ValueError(
            "prediction QC keys must be (slice_index, frame_index, structure) tuples"
        )
    if isinstance(key[0], bool) or not isinstance(key[0], int) or key[0] < 0:
        raise ValueError(f"invalid prediction QC slice index {key[0]!r}")
    if isinstance(key[1], bool) or not isinstance(key[1], int) or key[1] < 0:
        raise ValueError(f"invalid prediction QC frame index {key[1]!r}")
    if not isinstance(key[2], Structure):
        raise ValueError(f"invalid prediction QC structure {key[2]!r}")


def _largest_exterior(contours: tuple[HoleAwareContour, ...]) -> HoleAwareContour | None:
    if not contours:
        return None
    # ``cmr_contours.conversion`` and the core geometry use the same
    # shoelace/signed-area convention: the normalized exterior is CCW and its
    # absolute area is the primary-component size.  Do not use net_area here;
    # holes do not change which exterior component is primary.
    return max(
        contours,
        key=lambda contour: (
            abs(contour_signed_area(list(contour.exterior.points))),
            canonical_json(contour),
        ),
    )


def _legacy_contour(
    contour: HoleAwareContour, slice_index: int, frame_index: int
) -> Contour:
    """Adapt one primary exterior without modifying the frozen source contour."""

    points: list[tuple[float, float]] = []
    for x, y in contour.exterior.points:
        points.append((float(x), float(y)))
    return Contour(
        slice_index=slice_index,
        frame_index=frame_index,
        structure=contour.structure,
        points=points,
        source="ai",
    )


class _ExplicitlySkipContainment(list[tuple[Structure, Structure]]):
    """Truthy empty list so the frozen rule's ``or`` default is not selected."""

    def __bool__(self) -> bool:
        return True


def _known_ed_frame(
    volume_curve_ml: Mapping[tuple[int, Structure], list[float]] | None,
) -> int | None:
    """Infer ED only when all supplied LV curves agree on their argmax."""

    if volume_curve_ml is None:
        return None
    lv_curves: list[list[float]] = []
    for key, curve in volume_curve_ml.items():
        _validate_curve_key(key)
        if key[1] is not Structure.LV_CAVITY:
            continue
        if not isinstance(curve, list) or not curve:
            continue
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in curve
        ):
            raise ValueError("volume curve values must be finite real numbers")
        lv_curves.append(curve)
    if not lv_curves:
        return None
    candidate_frames = {
        max(range(len(curve)), key=lambda index: curve[index]) for curve in lv_curves
    }
    if len(candidate_frames) != 1:
        return None
    return next(iter(candidate_frames))


def _validate_curve_key(key: object) -> None:
    if not isinstance(key, tuple) or len(key) != 2:
        raise ValueError(
            "volume curve keys must be (slice_index, structure) tuples"
        )
    if isinstance(key[0], bool) or not isinstance(key[0], int) or key[0] < 0:
        raise ValueError(f"invalid volume curve slice index {key[0]!r}")
    if not isinstance(key[1], Structure):
        raise ValueError(f"invalid volume curve structure {key[1]!r}")


def _structure_from_message(warning: CMRWarning, *, code: str) -> Structure | None:
    if warning.code != code:
        return None
    if code == QC_MISSING_STRUCTURE:
        match = re.match(r"^structure ([a-z_]+) missing on slice ", warning.message)
    elif code == QC_SELF_INTERSECTION:
        match = re.match(
            r"^([a-z_]+) contour self-intersects on slice ", warning.message
        )
    else:
        match = re.match(
            r"^([a-z_]+) contour is not fully inside [a-z_]+ on slice ",
            warning.message,
        )
    if match is None:
        raise ValueError("QC warning cannot be keyed to exactly one structure")
    try:
        return Structure(match.group(1))
    except ValueError as exc:
        raise ValueError("QC warning cannot be keyed to exactly one structure") from exc


def _key_rule_warning(
    warning: CMRWarning,
    *,
    slice_index: int,
    frame_index: int,
    expected: tuple[Structure, ...],
) -> KeyedQCObservation:
    if warning.severity is not Severity.QC:
        raise ValueError("QC rule emitted a warning without QC severity")

    if warning.code == QC_MISSING_STRUCTURE:
        structure = _structure_from_message(warning, code=QC_MISSING_STRUCTURE)
        if structure is None or structure not in expected:
            raise ValueError("QC missing-structure warning cannot be keyed")
        key = (slice_index, frame_index, structure)
        return KeyedQCObservation(warning=warning, keys=(key,))

    if warning.code in (QC_SELF_INTERSECTION, QC_CONTAINMENT):
        structure = _structure_from_message(warning, code=warning.code)
        if structure is None:
            raise ValueError("QC geometry warning cannot be keyed")
        key = (slice_index, frame_index, structure)
        return KeyedQCObservation(warning=warning, keys=(key,))

    raise ValueError("QC warning cannot be keyed to a prediction key")


def _group_contours(
    prediction: WholeCinePrediction, slice_index: int, frame_index: int
) -> ContourSet:
    group = ContourSet(slice_index=slice_index, frame_index=frame_index)
    for key, values in sorted(
        prediction.contours.items(), key=lambda item: item[0][2].value
    ):
        if key[0] != slice_index or key[1] != frame_index:
            continue
        primary = _largest_exterior(values)
        if primary is not None:
            group.add(_legacy_contour(primary, slice_index, frame_index))
    return group


def _route_group(
    group: ContourSet,
    *,
    expected: tuple[Structure, ...],
    inner_outer: list[tuple[Structure, Structure]] | _ExplicitlySkipContainment,
) -> list[KeyedQCObservation]:
    # Use the frozen bundle as the sole rule invocation.  The truthy empty
    # sentinel is needed because that bundle intentionally treats a falsy
    # inner_outer argument as its default containment pair.
    warnings = _rules.run_qc_on_contour_set(
        group,
        expected=list(expected),
        inner_outer=inner_outer,
    )
    return [
        _key_rule_warning(
            warning,
            slice_index=group.slice_index,
            frame_index=group.frame_index,
            expected=expected,
        )
        for warning in warnings
    ]


def _route_volume_curve(
    prediction: WholeCinePrediction,
    volume_curve_ml: Mapping[tuple[int, Structure], list[float]],
) -> list[KeyedQCObservation]:
    observations: list[KeyedQCObservation] = []
    for key, curve in sorted(
        volume_curve_ml.items(), key=lambda item: (item[0][0], item[0][1].value)
    ):
        _validate_curve_key(key)
        if key[0] >= prediction.slice_count:
            raise ValueError("volume curve slice is outside the prediction grid")
        if not isinstance(curve, list):
            raise ValueError("volume curves must be lists of finite real numbers")
        copied_curve = list(curve)
        if len(copied_curve) != prediction.frame_count:
            raise ValueError("volume curve must contain one value for every prediction frame")
        if any(
            (key[0], frame_index, key[1]) not in prediction.contours
            for frame_index in range(len(copied_curve))
        ):
            raise ValueError("volume curve cannot be keyed to the prediction contour keys")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in copied_curve
        ):
            raise ValueError("volume curve values must be finite real numbers")
        warnings = _rules.check_volume_curve(copied_curve)
        if warnings and not copied_curve:
            raise ValueError("QC volume-curve warning cannot be keyed to any frame")
        keys = tuple((key[0], frame_index, key[1]) for frame_index in range(len(copied_curve)))
        for warning in warnings:
            if warning.severity is not Severity.QC or warning.code != QC_VOLUME_CURVE:
                raise ValueError("QC volume-curve warning cannot be keyed")
            observations.append(KeyedQCObservation(warning=warning, keys=keys))
    return observations


def route_prediction_qc(
    prediction: WholeCinePrediction,
    *,
    expected_structures: tuple[Structure, ...],
    volume_curve_ml: Mapping[tuple[int, Structure], list[float]] | None = None,
) -> tuple[KeyedQCObservation, ...]:
    """Run frozen QC over every prediction grid location and key its warnings."""

    if not isinstance(prediction, WholeCinePrediction):
        raise ValueError("prediction must be a WholeCinePrediction")
    if not isinstance(expected_structures, tuple) or not expected_structures:
        raise ValueError("expected_structures must be a non-empty tuple")
    if len(set(expected_structures)) != len(expected_structures):
        raise ValueError("expected_structures must not contain duplicates")
    if any(not isinstance(structure, Structure) for structure in expected_structures):
        raise ValueError("expected_structures must contain only Structure members")
    if volume_curve_ml is not None and not isinstance(volume_curve_ml, Mapping):
        raise ValueError("volume_curve_ml must be a mapping or None")

    ed_frame = _known_ed_frame(volume_curve_ml)
    observations: list[KeyedQCObservation] = []
    # WholeCinePrediction validates the distinct index coverage and bounds;
    # visiting the complete Cartesian grid also catches a group with no
    # contour entries at all as a set of missing structures.
    for slice_index in range(prediction.slice_count):
        for frame_index in range(prediction.frame_count):
            group = _group_contours(prediction, slice_index, frame_index)
            if ed_frame is not None and frame_index == ed_frame:
                inner_outer: list[tuple[Structure, Structure]] | _ExplicitlySkipContainment = [
                    (Structure.LV_CAVITY, Structure.LV_MYOCARDIUM)
                ]
            else:
                inner_outer = _ExplicitlySkipContainment()
            observations.extend(
                _route_group(
                    group,
                    expected=expected_structures,
                    inner_outer=inner_outer,
                )
            )

    if volume_curve_ml is not None:
        observations.extend(_route_volume_curve(prediction, volume_curve_ml))
    return tuple(observations)


__all__ = ["KeyedQCObservation", "route_prediction_qc"]
