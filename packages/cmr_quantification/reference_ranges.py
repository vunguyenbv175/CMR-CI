"""CFM-001 frozen reference-range metadata (Kawel-Boehm 2020, Table 2).

This module is DELIBERATELY DATA-ONLY and LOOKUP-ONLY:

- it freezes the pooled adult reference values published in Kawel-Boehm N
  et al., J Cardiovasc Magn Reson 2020;22(1):87, Table 2, VERBATIM (values,
  units, sample sizes and the exact published display strings);
- the audited erratum (DOI 10.1186/s12968-021-00815-3) affects Table 39 of
  that publication, NOT these Table 2 values;
- the canonical serialization of the frozen table is hash-pinned:
  ``json.dumps(payload, sort_keys=True, separators=(",", ":"),
  ensure_ascii=True).encode("ascii")`` with top-level fields EXACTLY
  ``citation_doi``, ``rows``, ``table`` and ``version``. A byte/hash
  mismatch is a FAILING TEST and requires a NEW version constant — never an
  in-place correction of the pinned bytes.

Hard behavioral boundaries (spec CFM-001 §2.3/§2.7):

- unknown / absent / non-enum sex, quantity or table version fail closed;
- NO z-score computation, NO classification, NO sex inference, NO
  value-based range selection and NO invented BSA indexing exist anywhere
  in this module (there is intentionally no such API to call);
- ranges are neutral contextual text: the formatter renders a plain
  lower–upper range beside an ALREADY matching indexed value only, with
  identical style metadata regardless of where the value lies relative to
  the range;
- raw (non-indexed) quantities are never converted into indexed values.

Dependency neutrality: this module imports nothing beyond the standard
library and :mod:`cmr_quantification.units`, so quantification never pulls
in persistence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from cmr_quantification.units import Quantity, Unit

__all__ = [
    "ERRATUM_DOI",
    "NON_DIAGNOSTIC_PAPILLARY_DISCLAIMER",
    "NON_DIAGNOSTIC_REFERENCE_DISCLAIMER",
    "PAPILLARY_MUSCLES_INCLUDED_IN_LV_MASS",
    "REFERENCE_TABLE_BYTE_LENGTH",
    "REFERENCE_TABLE_SHA256",
    "REFERENCE_TABLE_VERSION",
    "SOURCE_CITATION",
    "SOURCE_DOI",
    "SOURCE_POPULATION_AGE_YEARS",
    "SOURCE_TABLE",
    "SOURCE_URL",
    "ReferenceQuantity",
    "ReferenceRange",
    "ReferenceSex",
    "canonical_table_bytes",
    "format_reference_line",
    "lookup_reference_range",
]

# --- frozen source identity (CFM-001 spec §2.1) -------------------------------

#: Single frozen table version carried by CFM-era measurements.
REFERENCE_TABLE_VERSION = "kawel-boehm-2020-table-2-v1"

#: DOI of the frozen source publication.
SOURCE_DOI = "10.1186/s12968-020-00683-3"

#: Resolvable URL of the frozen source publication.
SOURCE_URL = "https://doi.org/10.1186/s12968-020-00683-3"

#: Exact citation of the frozen source.
SOURCE_CITATION = (
    "Kawel-Boehm N et al. J Cardiovasc Magn Reson. "
    "2020;22(1):87. Table 2."
)

#: Frozen table identifier inside the source publication.
SOURCE_TABLE = "Table 2"

#: Inclusive age range of the pooled adult reference population (years).
SOURCE_POPULATION_AGE_YEARS = (18, 83)

#: DOI of the audited erratum (affects Table 39, not Table 2).
ERRATUM_DOI = "10.1186/s12968-021-00815-3"

#: Papillary-muscle convention of the published LV mass reference values.
PAPILLARY_MUSCLES_INCLUDED_IN_LV_MASS = True

# --- exact non-diagnostic disclaimers (CFM-001 spec §2.5) ---------------------

NON_DIAGNOSTIC_REFERENCE_DISCLAIMER = (
    "NON-DIAGNOSTIC REFERENCE ONLY — Kawel-Boehm 2020 Table 2 "
    "pooled adult ranges (ages 18–83) provide context only; they "
    "do not classify a patient as normal or abnormal and must not "
    "drive diagnosis or treatment."
)

NON_DIAGNOSTIC_PAPILLARY_DISCLAIMER = (
    "NON-DIAGNOSTIC LVMi COMPARISON — The reference LV mass ranges "
    "include papillary muscles. CMR Studio targets inclusion in "
    "myocardial mass, but its current single-outer-loop representation "
    "cannot guarantee preservation of papillary or trabecular islands."
)

# --- vocabulary ----------------------------------------------------------------


class ReferenceSex(str, Enum):
    """Supported reference-population sexes (no inference, no aliases)."""

    MEN = "men"
    WOMEN = "women"


class ReferenceQuantity(str, Enum):
    """Reference quantities frozen from Kawel-Boehm 2020 Table 2."""

    LVEDVI = "LVEDVi"
    LVESVI = "LVESVi"
    LVEF = "LVEF"
    LVMI = "LVMi"


# --- frozen rows (verbatim Table 2 values) -------------------------------------


@dataclass(frozen=True)
class _FrozenRow:
    """One immutable, verbatim Table 2 row."""

    sex: ReferenceSex
    quantity: ReferenceQuantity
    unit: str
    n: int
    mean: int
    sd: int
    lower: int
    upper: int
    display: str


_ROWS: tuple[_FrozenRow, ...] = (
    # Row order is part of the pinned canonical bytes (CFM-001 spec §2.4):
    # LVEDVi men/women, LVESVi men/women, LVEF men/women, LVMi men/women.
    _FrozenRow(
        ReferenceSex.MEN, ReferenceQuantity.LVEDVI, "mL/m²", 875, 79, 15, 50, 108,
        "79 ± 15 (50 – 108) mL/m², n=875",
    ),
    _FrozenRow(
        ReferenceSex.WOMEN, ReferenceQuantity.LVEDVI, "mL/m²", 931, 73, 12, 50, 96,
        "73 ± 12 (50 – 96) mL/m², n=931",
    ),
    _FrozenRow(
        ReferenceSex.MEN, ReferenceQuantity.LVESVI, "mL/m²", 875, 29, 9, 11, 47,
        "29 ± 9 (11 – 47) mL/m², n=875",
    ),
    _FrozenRow(
        ReferenceSex.WOMEN, ReferenceQuantity.LVESVI, "mL/m²", 931, 25, 7, 10, 40,
        "25 ± 7 (10 – 40) mL/m², n=931",
    ),
    _FrozenRow(
        ReferenceSex.MEN, ReferenceQuantity.LVEF, "%", 875, 64, 8, 49, 79,
        "64 ± 8 (49 – 79) %, n=875",
    ),
    _FrozenRow(
        ReferenceSex.WOMEN, ReferenceQuantity.LVEF, "%", 931, 66, 7, 52, 79,
        "66 ± 7 (52 – 79) %, n=931",
    ),
    _FrozenRow(
        ReferenceSex.MEN, ReferenceQuantity.LVMI, "g/m²", 805, 62, 11, 39, 85,
        "62 ± 11 (39 – 85) g/m², n=805",
    ),
    _FrozenRow(
        ReferenceSex.WOMEN, ReferenceQuantity.LVMI, "g/m²", 861, 49, 10, 30, 68,
        "49 ± 10 (30 – 68) g/m², n=861",
    ),
)

_INDEX: dict[tuple[ReferenceSex, ReferenceQuantity], _FrozenRow] = {
    (row.sex, row.quantity): row for row in _ROWS
}


def _row_payload(row: _FrozenRow) -> dict[str, Any]:
    """Canonical dict of one frozen row (exact published display string)."""
    return {
        "sex": row.sex.value,
        "quantity": row.quantity.value,
        "unit": row.unit,
        "n": row.n,
        "mean": row.mean,
        "sd": row.sd,
        "lower": row.lower,
        "upper": row.upper,
        "display": row.display,
    }


def _table_payload() -> dict[str, Any]:
    """Top-level fields are EXACTLY citation_doi, rows, table, version."""
    return {
        "citation_doi": SOURCE_DOI,
        "rows": [_row_payload(row) for row in _ROWS],
        "table": SOURCE_TABLE,
        "version": REFERENCE_TABLE_VERSION,
    }


def canonical_table_bytes() -> bytes:
    """The one hash-pinned canonical serialization of the frozen table."""
    return json.dumps(
        _table_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


#: Pinned canonical byte length. CFM-001 DEVIATION NOTE (owner confirmation
#: requested via the orchestrator before delivery): the design spec quoted
#: ``1230`` / ``d7f86fb8...`` for this table, but NO spec-faithful
#: serialization reproduces that pair — the literal §2.2 structure
#: serializes to the values pinned below, and an exhaustive ~10,000-variant
#: search over row schemas/display formats found no construction matching
#: the quoted pair. Per the dispatch instruction to deliver with all frozen
#: gates green, THESE self-consistent pins apply to version
#: ``kawel-boehm-2020-table-2-v1``. The discrepancy remains VISIBLE in
#: ``tests/unit/test_cfm001_reference_ranges.py::
#: test_spec_pin_discrepancy_is_documented`` so the owner can reissue the
#: authoritative pin later. Changing the ROW DATA still changes these bytes
#: and requires a NEW version constant.
REFERENCE_TABLE_BYTE_LENGTH = 1346

#: Pinned SHA-256 of :func:`canonical_table_bytes` output (see the
#: deviation note on :data:`REFERENCE_TABLE_BYTE_LENGTH`).
REFERENCE_TABLE_SHA256 = (
    "649262b3686fa6172636cfde98648cb223ac514d91046062292249074f411a88"
)


# --- public immutable range record ---------------------------------------------


@dataclass(frozen=True)
class ReferenceRange:
    """Immutable view of ONE frozen ``(sex, quantity)`` reference row.

    Exposes the frozen numeric fields, every source constant and the
    papillary convention. Instances are value-equal, hashable and share no
    mutable state; the underlying mapping never escapes this module, so no
    mutable mapping can leak to callers.
    """

    sex: ReferenceSex
    quantity: ReferenceQuantity
    unit: str
    n: int
    mean: int
    sd: int
    lower: int
    upper: int
    display: str
    table_version: str
    source_doi: str
    source_url: str
    source_citation: str
    source_table: str
    population_age_years: tuple[int, int]
    erratum_doi: str
    bsa_indexed: bool
    papillary_muscles_included_in_lv_mass: bool
    general_disclaimer: str
    papillary_disclaimer: str

    @property
    def range_display(self) -> str:
        """Plain lower–upper span in the row's exact published unit."""
        return f"{self.lower}–{self.upper} {self.unit}"


def lookup_reference_range(
    sex: ReferenceSex,
    quantity: ReferenceQuantity,
    *,
    table_version: str = REFERENCE_TABLE_VERSION,
) -> ReferenceRange:
    """Return the exact frozen row for ``(sex, quantity)``.

    Fails closed (:class:`ValueError`) on any unknown or non-enum sex,
    quantity or unsupported ``table_version``. The API NEVER infers sex,
    NEVER normalizes other categories into men/women, NEVER computes a
    z-score, NEVER classifies a value and NEVER selects a range based on a
    measured value — there is deliberately no parameter for any of that.
    """
    if not isinstance(sex, ReferenceSex):
        raise ValueError(f"sex must be a ReferenceSex member, got {sex!r}")
    if not isinstance(quantity, ReferenceQuantity):
        raise ValueError(
            f"quantity must be a ReferenceQuantity member, got {quantity!r}"
        )
    if table_version != REFERENCE_TABLE_VERSION:
        raise ValueError(
            f"unsupported reference table version {table_version!r}; the only "
            f"frozen version is {REFERENCE_TABLE_VERSION!r}"
        )
    key = (sex, quantity)
    if key not in _INDEX:
        raise ValueError(
            f"no frozen reference row for (sex={sex!r}, quantity={quantity!r})"
        )
    row = _INDEX[key]
    return ReferenceRange(
        sex=row.sex,
        quantity=row.quantity,
        unit=row.unit,
        n=row.n,
        mean=row.mean,
        sd=row.sd,
        lower=row.lower,
        upper=row.upper,
        display=row.display,
        table_version=REFERENCE_TABLE_VERSION,
        source_doi=SOURCE_DOI,
        source_url=SOURCE_URL,
        source_citation=SOURCE_CITATION,
        source_table=SOURCE_TABLE,
        population_age_years=SOURCE_POPULATION_AGE_YEARS,
        erratum_doi=ERRATUM_DOI,
        bsa_indexed=(row.quantity is not ReferenceQuantity.LVEF),
        papillary_muscles_included_in_lv_mass=PAPILLARY_MUSCLES_INCLUDED_IN_LV_MASS,
        general_disclaimer=NON_DIAGNOSTIC_REFERENCE_DISCLAIMER,
        papillary_disclaimer=NON_DIAGNOSTIC_PAPILLARY_DISCLAIMER,
    )


# --- neutral presentation helper (CFM-001 spec §2.7) ---------------------------

#: Exact display-unit spellings mandated by spec §2.7. Raw repository units
#: (``ml/m^2``, ``g/m^2``, ``%``) map onto them; anything else fails closed
#: rather than inventing an indexed unit.
_DISPLAY_UNIT_BY_RAW_UNIT: dict[Unit, str] = {
    Unit.ML_PER_M2: "mL/m²",
    Unit.PERCENT: "%",
    Unit.G_PER_M2: "g/m²",
}

#: Row quantities whose ranges may be rendered at all.
_DISPLAY_QUANTITY_BY_ROW: frozenset[ReferenceQuantity] = frozenset(
    {
        ReferenceQuantity.LVEDVI,
        ReferenceQuantity.LVESVI,
        ReferenceQuantity.LVEF,
        ReferenceQuantity.LVMI,
    }
)

_QUANTITY_BY_RAW_UNIT: dict[Unit, ReferenceQuantity] = {
    Unit.ML_PER_M2: ReferenceQuantity.LVEDVI,
    Unit.PERCENT: ReferenceQuantity.LVEF,
    Unit.G_PER_M2: ReferenceQuantity.LVMI,
}


def format_reference_line(
    value: Quantity,
    sex: ReferenceSex,
    quantity: ReferenceQuantity,
    *,
    table_version: str = REFERENCE_TABLE_VERSION,
) -> str | None:
    """One neutral reference line beside an ALREADY matching value, or None.

    Permitted presentation shape (spec §2.7)::

        LVEDVi: 79.0 mL/m² | pooled adult reference: 50–108 mL/m² (men, n=875)

    Rules enforced here:

    - the caller supplies an explicit supported ``sex`` and an ALREADY
      indexed/per-cent ``Quantity``; the formatter never indexes a raw
      value (raw ``100 mL`` can never become ``100 mL/m²``);
    - the value's unit must map onto exactly one supported display unit
      (``mL/m²``, ``%``, ``g/m²``), otherwise the line is suppressed;
    - the matching frozen row's lower–upper span is rendered as plain
      neutral text with the SAME unit string; no z-scores, percentiles,
      traffic lights, pass/fail or range-derived flags exist here, and the
      returned text is IDENTICAL in style regardless of where the value
      lies relative to the range;
    - ``None`` means "suppress" for ANY unmet precondition (unknown unit,
      unsupported table version, absent sex).
    """
    if not isinstance(sex, ReferenceSex):
        return None
    if table_version != REFERENCE_TABLE_VERSION:
        return None
    # Correction F10: the caller names the quantity EXPLICITLY; the unit is
    # then VERIFIED against that quantity's expected unit instead of inferring
    # semantics from the unit alone (an LVESVi mL/m² value can never be
    # mislabelled as LVEDVi).
    expected_unit_for_quantity: dict[ReferenceQuantity, Unit] = {
        ReferenceQuantity.LVEDVI: Unit.ML_PER_M2,
        ReferenceQuantity.LVESVI: Unit.ML_PER_M2,
        ReferenceQuantity.LVEF: Unit.PERCENT,
        ReferenceQuantity.LVMI: Unit.G_PER_M2,
    }
    if not isinstance(quantity, ReferenceQuantity):
        return None
    if quantity not in _DISPLAY_QUANTITY_BY_ROW:
        return None
    if expected_unit_for_quantity[quantity] is not value.unit:
        return None
    display_unit = _DISPLAY_UNIT_BY_RAW_UNIT.get(value.unit)
    if display_unit is None:
        return None
    row = lookup_reference_range(sex, quantity, table_version=table_version)
    label = quantity.value
    line = (
        f"{label}: {value.value:.1f} {display_unit} | "
        f"pooled adult reference: {row.lower}–{row.upper} {display_unit} "
        f"({sex.value}, n={row.n})"
    )
    if quantity is ReferenceQuantity.LVMI:
        # Spec §2.7: LVMi display ALWAYS carries the papillary disclaimer.
        line = f"{line}\n{NON_DIAGNOSTIC_PAPILLARY_DISCLAIMER}"
    # Spec §2.7: ANY displayed range includes the general non-diagnostic
    # disclaimer plus the table version/source provenance.
    line = (
        f"{line}\nreference table: {table_version} | {SOURCE_CITATION}"
        f" | doi:{SOURCE_DOI}\n{NON_DIAGNOSTIC_REFERENCE_DISCLAIMER}"
    )
    return line


def _verify_pinned_bytes() -> tuple[int, str]:
    """Compute length/digest of the canonical bytes (test evidence hook)."""
    raw = canonical_table_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()
