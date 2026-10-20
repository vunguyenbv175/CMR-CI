"""Explicit units for clinical quantities (Round 1/2).

All quantification functions return :class:`Quantity` values so callers can
never silently mix units. The unit attached to a value must always agree
with its numeric scaling (e.g. ``cardiac_output`` returns ``L/min`` with a
value in litres per minute). Conversion helpers are central and pure.

Round 2 additions (per ROUND1_REVIEW 3.1/3.2):
- ``L_PER_MIN``: cardiac output is expressed in L/min, value scaled as such.
- ``ML_PER_M2`` / ``G_PER_M2``: indexed volumes/mass divided by BSA.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Unit(Enum):
    ML = "ml"
    L = "l"
    G = "g"
    ML_PER_BEAT = "ml/beat"
    PERCENT = "%"
    ML_PER_MIN = "ml/min"
    L_PER_MIN = "l/min"
    BPM = "bpm"
    G_ML = "g/ml"
    MM2 = "mm^2"
    MM3 = "mm^3"
    MM = "mm"
    MS = "ms"
    ML_PER_M2 = "ml/m^2"
    G_PER_M2 = "g/m^2"
    ML_ML = "ml/ml"


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: Unit

    def __post_init__(self) -> None:
        if self.value != self.value or self.value in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite quantity {self.value} {self.unit.value}")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.value:.3f} {self.unit.value}"

    def round_to(self, digits: int = 1) -> Quantity:
        return Quantity(round(self.value, digits), self.unit)

    def convert_to(self, unit: Unit) -> Quantity:
        """Convert this quantity to ``unit`` when a known linear relation
        exists between the two units; raises otherwise."""
        if unit is self.unit:
            return self
        if self.unit is Unit.ML and unit is Unit.L:
            return Quantity(self.value / 1000.0, Unit.L)
        if self.unit is Unit.L and unit is Unit.ML:
            return Quantity(self.value * 1000.0, Unit.ML)
        raise ValueError(f"no conversion defined from {self.unit.value} to {unit.value}")


# --- central conversion helpers (no scattered arithmetic) -------------------


def ml_to_l(ml: float) -> float:
    """Convert millilitres to litres (1 L = 1000 mL)."""
    return ml / 1000.0


def l_to_ml(litres: float) -> float:
    """Convert litres to millilitres (1 L = 1000 mL)."""
    return litres * 1000.0


def index_by_bsa(value_ml: float, bsa_m2: float) -> float:
    """Index a millilitre quantity by body surface area (mL/m^2)."""
    if bsa_m2 <= 0:
        raise ValueError(f"BSA must be > 0, got {bsa_m2}")
    return value_ml / bsa_m2
