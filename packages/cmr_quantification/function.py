"""Deterministic cardiac-function primitives (Tasks G, Round 2 corrections).

Pure functions with explicit units and strict input validation. These are
foundation math — NOT clinically validated (see reports).

Formulas (value and unit always agree):

- SV   = EDV - ESV                    (ml)
- EF   = (EDV - ESV) / EDV            (fraction -> %)
- CO   = SV * HR                      (l/min — value scaled in litres/min)
- mass = myocardium_volume_ml * 1.05 g/ml    (g)
- SVi  = SV / BSA                     (ml/m^2)
- mass_index = mass / BSA             (g/m^2)

Round 2 corrections (per ROUND1_REVIEW 3.1/3.2):
- ``cardiac_output`` now returns ``Unit.L_PER_MIN`` with the value in
  litres per minute (previously the value was L/min while tagged mL/min).
- indexed outputs use ``Unit.ML_PER_M2`` / ``Unit.G_PER_M2``.
"""

from __future__ import annotations

from cmr_core.errors import QUANTIFICATION_INVALID_INPUT, CMRError

from .units import Quantity, Unit, index_by_bsa, ml_to_l

MYOCARDIUM_DENSITY_G_ML = 1.05  # standard myocardial tissue density assumption


def stroke_volume(edv_ml: float, esv_ml: float) -> Quantity:
    _validate(edv_ml, esv_ml)
    return Quantity(edv_ml - esv_ml, Unit.ML)


def ejection_fraction(edv_ml: float, esv_ml: float) -> Quantity:
    _validate(edv_ml, esv_ml)
    if edv_ml <= 0:
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"EDV must be > 0 for EF, got {edv_ml}")
    return Quantity((edv_ml - esv_ml) / edv_ml * 100.0, Unit.PERCENT)


def cardiac_output(sv_ml: float, heart_rate_bpm: float) -> Quantity:
    """Cardiac output: SV (ml) x HR (bpm), expressed in L/min.

    Dimensional check: SV=60 ml and HR=70 bpm give 60*70/1000 = 4.2 L/min.
    """
    if sv_ml < 0:
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"SV must be >= 0, got {sv_ml}")
    if not (heart_rate_bpm > 0):
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT, f"heart rate must be > 0, got {heart_rate_bpm}"
        )
    return Quantity(ml_to_l(sv_ml * heart_rate_bpm), Unit.L_PER_MIN)


def cardiac_output_ml_per_min(sv_ml: float, heart_rate_bpm: float) -> Quantity:
    """Cardiac output explicitly expressed in mL/min (value scaled as such)."""
    if sv_ml < 0:
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"SV must be >= 0, got {sv_ml}")
    if not (heart_rate_bpm > 0):
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT, f"heart rate must be > 0, got {heart_rate_bpm}"
        )
    return Quantity(sv_ml * heart_rate_bpm, Unit.ML_PER_MIN)


def myocardial_mass(
    myocardium_volume_ml: float, density_g_ml: float = MYOCARDIUM_DENSITY_G_ML
) -> Quantity:
    if myocardium_volume_ml < 0:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"myocardium volume must be >= 0, got {myocardium_volume_ml}",
        )
    if not (density_g_ml > 0):
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"density must be > 0, got {density_g_ml}")
    return Quantity(myocardium_volume_ml * density_g_ml, Unit.G)


def stroke_volume_index(sv_ml: float, bsa_m2: float) -> Quantity:
    if sv_ml < 0:
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"SV must be >= 0, got {sv_ml}")
    if bsa_m2 <= 0:
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"BSA must be > 0, got {bsa_m2}")
    return Quantity(index_by_bsa(sv_ml, bsa_m2), Unit.ML_PER_M2)


def end_diastolic_volume_index(edv_ml: float, bsa_m2: float) -> Quantity:
    if edv_ml < 0:
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"EDV must be >= 0, got {edv_ml}")
    if bsa_m2 <= 0:
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"BSA must be > 0, got {bsa_m2}")
    return Quantity(index_by_bsa(edv_ml, bsa_m2), Unit.ML_PER_M2)


def end_systolic_volume_index(esv_ml: float, bsa_m2: float) -> Quantity:
    if esv_ml < 0:
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"ESV must be >= 0, got {esv_ml}")
    if bsa_m2 <= 0:
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"BSA must be > 0, got {bsa_m2}")
    return Quantity(index_by_bsa(esv_ml, bsa_m2), Unit.ML_PER_M2)


def myocardial_mass_index(mass_g: float, bsa_m2: float) -> Quantity:
    if mass_g < 0:
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"mass must be >= 0, got {mass_g}")
    if bsa_m2 <= 0:
        raise CMRError(QUANTIFICATION_INVALID_INPUT, f"BSA must be > 0, got {bsa_m2}")
    return Quantity(mass_g / bsa_m2, Unit.G_PER_M2)


def _validate(edv_ml: float, esv_ml: float) -> None:
    if edv_ml < 0 or esv_ml < 0:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"volumes must be >= 0, got EDV={edv_ml}, ESV={esv_ml}",
        )
    if esv_ml > edv_ml:
        raise CMRError(
            QUANTIFICATION_INVALID_INPUT,
            f"ESV ({esv_ml}) cannot exceed EDV ({edv_ml})",
        )
