"""Rule-based conservative series classification (Cine SAX MVP + LGE Phase 1).

Task STAGE1_CINE_SAX_CLASSIFIER_001 (cine) and LGE Phase 1 (late
enhancement). Implements fail-closed, conservative classification to identify
likely cine short-axis series, and (additively, Phase 1) delayed-enhancement
("lge") series, without guessing. Ambiguous, contradictory, or unsupported
series remain "unknown".

Two classification passes, each with its OWN inclusion and exclusion rules:

- CINE pass (pre-existing, unchanged): positive Cine evidence AND SAX
  evidence with ZERO CINE-exclusionary tokens -> "cine_sax". Its exclusion
  list deliberately contains LGE/DE tokens (psir, lge, delayed, scar,
  viability ...) because those tokens disqualify a series from being CINE.
- LGE pass (LGE Phase 1, additive): positive delayed-enhancement evidence
  with ZERO LGE-exclusionary tokens -> "lge". Its exclusion list is SEPARATE
  and does NOT reuse the CINE exclusions: long-axis/chamber-view tokens
  (2ch/4ch/lax), inversion-recovery/anatomical tokens (ir, stir, tse, aorta)
  and T1/T2 weighting tokens are NOT LGE exclusions — LGE acquisitions carry
  them routinely. View is deliberately NOT part of the Phase-1 class, so SAX
  and LAX LGE series BOTH classify as "lge".

Rules (both passes):
- A bare "e_" prefix is NOT positive evidence (fail-closed): the DE token must
  be explicit ("de_", "lge", "late enhancement", "delayed", "viability",
  "scar", "psir").
- MAG/PSIR pairs belong to the same DE acquisition: both classify "lge" when
  another positive token is present. The MAG/PSIR distinction is metadata for
  later phases, not a Phase-1 gate.
- `psir` alone IS positive evidence (it is in _LGE_INDICATORS): `psir_sax`
  classifies "lge" without any other DE token. MAG alone proves nothing:
  `mag_sax` stays "unknown"; the `de_` token carries the magnitude stack
  (`de_mag_sax` -> "lge" via `de_`, not via MAG).
- Contradictory metadata (e.g. cine AND DE tokens) stays "unknown".
- Single weak tokens remain "unknown".
- Non-MR modality or fatal geometry warnings force "unknown".
- Deterministic, metadata-only, zero dependencies outside cmr_core/standard library.
- "unknown" remains the default for anything unproven.

The CINE pass is evaluated FIRST and is byte-for-byte the pre-Phase-1
behavior: no existing cine_sax / unknown verdict can change. "lge" is only
reached when the CINE pass did not classify the series.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cmr_core.study import CMRSeries

# --- Exclusion patterns: disqualify a series from being cine_sax -------------

_EXCLUSIONS: tuple[re.Pattern, ...] = (
    # Long axis and chamber-specific views
    re.compile(r"\b2\s*ch(?:amber)?\b"),
    re.compile(r"\b3\s*ch(?:amber)?\b"),
    re.compile(r"\b4\s*ch(?:amber)?\b"),
    re.compile(r"\b5\s*ch(?:amber)?\b"),
    re.compile(r"\b(?:two|three|four|five)\s*chamber\b"),
    re.compile(r"\blong\s*axis\b"),
    re.compile(r"\blax\b"),
    re.compile(r"\blongaxis\b"),
    re.compile(r"\bhla\b"),
    re.compile(r"\bvla\b"),
    re.compile(r"\blvot\b"),
    re.compile(r"\brvot\b"),
    re.compile(r"\binfundibulum\b"),
    # Non-cine: LGE, viability, scar, delayed enhancement
    re.compile(r"\blge\b"),
    re.compile(r"\bpsir\b"),
    re.compile(r"\bdelayed\b"),
    re.compile(r"\blate\s*(?:enhancement|gadolinium|e)\b"),
    re.compile(r"\bscar\b"),
    re.compile(r"\bviability\b"),
    re.compile(r"\blook\s*locker\b"),
    re.compile(r"\bti\s*scout\b"),
    re.compile(r"\bmagprep\b"),
    # Non-cine: T1/T2 weighting and parametric mapping
    re.compile(r"\bt1\b"),
    re.compile(r"\bt2\b"),
    re.compile(r"\bt1w\b"),
    re.compile(r"\bt2w\b"),
    re.compile(r"\bt1\s*w\b"),
    re.compile(r"\bt2\s*w\b"),
    re.compile(r"\bt1\s*weighted\b"),
    re.compile(r"\bt2\s*weighted\b"),
    re.compile(r"\bt1\s*weighting\b"),
    re.compile(r"\bt2\s*weighting\b"),
    re.compile(r"\bt1\s*(?:map|mapping)\b"),
    re.compile(r"\bt2\s*(?:map|mapping)\b"),
    re.compile(r"\bt2\s*star\b"),
    re.compile(r"\bt2star\b"),
    re.compile(r"\bmolli\b"),
    re.compile(r"\bshmolli\b"),
    re.compile(r"\bsasha\b"),
    # Non-cine: Flow / Phase Contrast / Velocity
    re.compile(r"\bphase\s*contrast\b"),
    re.compile(r"\bpc\b"),
    re.compile(r"\bq\s*flow\b"),
    re.compile(r"\bqflow\b"),
    re.compile(r"\bvenc\b"),
    re.compile(r"\bvelocity\b"),
    re.compile(r"\bflow\b"),
    # Non-cine: Perfusion / stress
    re.compile(r"\bperfusion\b"),
    re.compile(r"\bperf\b"),
    re.compile(r"\bstress\b"),
    re.compile(r"\badenosine\b"),
    re.compile(r"\bregadenoson\b"),
    re.compile(r"\bfirst\s*pass\b"),
    # Non-cine: Localizers, scouts, surveys, calibration, non-image
    re.compile(r"\blocaliz(?:er|ation|e)?\b"),
    re.compile(r"\blocalis(?:er|ation|e)?\b"),
    re.compile(r"\bscout\b"),
    re.compile(r"\bsurvey\b"),
    re.compile(r"\bcal(?:ibration)?\b"),
    re.compile(r"\blocator\b"),
    re.compile(r"\btopogram\b"),
    re.compile(r"\bscreen\s*save\b"),
    re.compile(r"\bscreenshot\b"),
    re.compile(r"\breport\b"),
    re.compile(r"\bphoenix\b"),
    re.compile(r"\bnoise\b"),
    re.compile(r"\bdistortion\b"),
    # Non-cine: Static / anatomical / spin echo / inversion recovery
    re.compile(r"\bblack\s*blood\b"),
    re.compile(r"\bdark\s*blood\b"),
    re.compile(r"\bhaste\b"),
    re.compile(r"\bhaaste\b"),
    re.compile(r"\btse\b"),
    re.compile(r"\bfse\b"),
    re.compile(r"\bturbo\s*spin\s*echo\b"),
    re.compile(r"\bspin\s*echo\b"),
    re.compile(r"\bstir\b"),
    re.compile(r"\bir\b"),
    re.compile(r"\binversion\b"),
    re.compile(r"\binversion\s*recovery\b"),
    re.compile(r"\baorta\b"),
    re.compile(r"\baortic\b"),
    re.compile(r"\bpulmonary\b"),
    re.compile(r"\bcoronary\b"),
    re.compile(r"\bvalve\b"),
    re.compile(r"\bvalvular\b"),
)

# --- Positive Cine patterns --------------------------------------------------

_CINE_INDICATORS: tuple[re.Pattern, ...] = (
    re.compile(r"\bcinesax\b"),
    re.compile(r"\bsaxcine\b"),
    re.compile(r"\bcine\b"),
    re.compile(r"\bcines\b"),
    re.compile(r"\bcin[eé]\b"),
    re.compile(r"\bretro\s*cine\b"),
    re.compile(r"\bcine\s*retro\b"),
    re.compile(r"\bretrocine\b"),
    # Vendor cine sequence names
    re.compile(r"\btrufi\b"),
    re.compile(r"\btrue\s*fisp\b"),
    re.compile(r"\btfisp\b"),
    re.compile(r"\bfiesta\b"),
    re.compile(r"\bfastcard\b"),
    re.compile(r"\bfast\s*card\b"),
    re.compile(r"\bb\s*ffe\b"),
    re.compile(r"\bbalanced\s*ffe\b"),
    re.compile(r"\bb\s*ssfp\b"),
    re.compile(r"\bbssfp\b"),
    re.compile(r"\bssfp\b"),
    re.compile(r"\bbalanced\s*ssfp\b"),
    re.compile(r"\bfisp\b"),
)

# --- Positive SAX patterns ---------------------------------------------------

_SAX_INDICATORS: tuple[re.Pattern, ...] = (
    re.compile(r"\bcinesax\b"),
    re.compile(r"\bsaxcine\b"),
    re.compile(r"\bsax\b"),
    re.compile(r"\bsa\s*stack\b"),
    re.compile(r"\bsax\s*stack\b"),
    re.compile(r"\bshort\s*axis\b"),
    re.compile(r"\bshortaxis\b"),
    re.compile(r"\bshort\s*ax\b"),
    re.compile(r"\bsa\b"),
    re.compile(r"\b(?:mid|basal|apical)\s*sax\b"),
    re.compile(r"\b(?:mid|basal|apical)\s*sa\b"),
)

# --- LGE Phase 1: POSITIVE delayed-enhancement patterns ----------------------
#
# These are the ONLY tokens that can carry a series to "lge". Each is an
# explicit delayed-enhancement (DE) marker; a bare "e_" / "e " prefix carries
# NO evidence (fail-closed) and is deliberately absent from this list.
#
# IMPORTANT: this list is separate from the CINE exclusion list above. The CINE
# exclusions ban DE tokens because a DE series is not a cine series; the LGE
# pass needs its own (much smaller) exclusion list because tokens that
# disqualify CINE (long-axis views, inversion recovery, anatomical names,
# T1/T2 weighting) are ROUTINE on real DE acquisitions.

_LGE_INDICATORS: tuple[re.Pattern, ...] = (
    # Explicit delayed-enhancement prefixes / words
    re.compile(r"\bde\b"),            # "de_" / "de-" / "de " -> "de trufi ..."
    re.compile(r"\bde\s*\d+\b"),      # "de_500", "de 300" (TI-labelled DE stacks)
    re.compile(r"\blge\b"),
    re.compile(r"\blate\s*enhancement\b"),
    re.compile(r"\blate\s*gadolinium\b"),
    re.compile(r"\bdelayed\b"),
    re.compile(r"\bdelayed\s*enhancement\b"),
    re.compile(r"\bviability\b"),
    re.compile(r"\bscar\b"),
    re.compile(r"\bpsir\b"),
)

# --- LGE Phase 1: EXCLUSION patterns (LGE-specific; NOT the cine list) -------
#
# These hard-exclude a series from "lge" even when a DE token is present.
# View tokens (sax/lax/2ch/4ch) are intentionally NOT here: LGE is
# view-agnostic in Phase 1.

# Cine tokens: a cine acquisition is never a DE acquisition. A series carrying
# BOTH one of these and a DE token is CONTRADICTORY -> "unknown" (fail-closed),
# never cine_sax and never lge. Kept as its own tuple so the contradiction rule
# can be applied before either pass.
_LGE_CINE_TOKENS: tuple[re.Pattern, ...] = (
    re.compile(r"\bcinesax\b"),
    re.compile(r"\bsaxcine\b"),
    re.compile(r"\bcine\b"),
    re.compile(r"\bcines\b"),
    re.compile(r"\bcin[eé]\b"),
    re.compile(r"\bretro\s*cine\b"),
    re.compile(r"\bcine\s*retro\b"),
    re.compile(r"\bretrocine\b"),
    re.compile(r"\bretro\b"),
)

_LGE_EXCLUSIONS: tuple[re.Pattern, ...] = (
    # Cine tokens (same rule as above; also excludes the LGE verdict).
    *_LGE_CINE_TOKENS,
    # Perfusion / stress
    re.compile(r"\bperfusion\b"),
    re.compile(r"\bperf\b"),
    re.compile(r"\bstress\b"),
    re.compile(r"\badenosine\b"),
    re.compile(r"\bregadenoson\b"),
    re.compile(r"\bfirst\s*pass\b"),
    # Flow / phase contrast / velocity
    re.compile(r"\bq\s*flow\b"),
    re.compile(r"\bqflow\b"),
    re.compile(r"\bvenc\b"),
    re.compile(r"\bvelocity\b"),
    re.compile(r"\bphase\s*contrast\b"),
    re.compile(r"\bflow\b"),
    re.compile(r"\bpc\b"),
    # Parametric mapping
    re.compile(r"\bmolli\b"),
    re.compile(r"\bshmolli\b"),
    re.compile(r"\bsasha\b"),
    re.compile(r"\bt1\s*(?:map|mapping)\b"),
    re.compile(r"\bt2\s*(?:map|mapping)\b"),
    re.compile(r"\bt1\s*map\b"),
    re.compile(r"\bt2\s*map\b"),
    re.compile(r"\bt2\s*star\b"),
    re.compile(r"\bt2star\b"),
    # TI scouting (the TI-look-locker localizer, not a DE acquisition)
    re.compile(r"\bti\s*scout\b"),
    re.compile(r"\btiscout\b"),
    re.compile(r"\blook\s*locker\b"),
    # Localizers / scouts / surveys / calibration / non-image
    re.compile(r"\blocaliz(?:er|ation|e)?\b"),
    re.compile(r"\blocalis(?:er|ation|e)?\b"),
    re.compile(r"\blocator\b"),
    re.compile(r"\bscout\b"),
    re.compile(r"\bsurvey\b"),
    re.compile(r"\btopogram\b"),
    re.compile(r"\bscreen\s*save\b"),
    re.compile(r"\bscreenshot\b"),
    re.compile(r"\bphoenix\b"),
)


def _clean_text(raw: str) -> str:
    """Normalize textual metadata for robust token boundary matching."""
    lower = raw.lower()
    # Replace common punctuation / delimiter characters with spaces
    spaced = re.sub(r"[_\-/\.,:;()\[\]{}&+#\\]", " ", lower)
    return " ".join(spaced.split())


def classify_series(
    series: CMRSeries | None = None,
    *,
    series_description: str | None = None,
    protocol_name: str | None = None,
    modality: str | None = None,
) -> str:
    """Classify a CMR series into a canonical sequence class (SEQUENCE_CLASSES).

    Returns "cine_sax" only when positive Cine and SAX evidence are both
    present and consistent, with zero exclusionary tokens.

    Returns "lge" (LGE Phase 1, additive) only when the CINE pass did not
    classify the series AND positive delayed-enhancement evidence is present
    with zero LGE-exclusionary tokens. View-agnostic: SAX and LAX DE series
    both return "lge"; MAG/PSIR is not a gate.

    Returns "unknown" for ambiguous, contradictory (e.g. cine + DE tokens),
    non-cine, non-SAX, non-DE, or unsupported series.

    Args:
        series: optional CMRSeries domain object to inspect.
        series_description: optional override for SeriesDescription.
        protocol_name: optional override for ProtocolName.
        modality: optional override for Modality (default MR).

    Returns:
        One of the canonical SEQUENCE_CLASSES strings ("cine_sax", "lge",
        "unknown"). The CINE result is never changed by the LGE pass.
    """
    desc = (
        series_description
        if series_description is not None
        else (series.series_description if series else "")
    )
    prot = (
        protocol_name
        if protocol_name is not None
        else (series.protocol_name if series else "")
    )
    mod = modality if modality is not None else (series.modality if series is not None else None)

    # Modality gate: CMR requires AFFIRMATIVE MR evidence. Anything that is not
    # a resolved "MR" — omitted, explicit None, empty, whitespace-only, or any
    # other modality — is fail-closed "unknown" (Bundle L F4, R1 P1: the
    # previous narrow form let omitted/None through as lge, which the contract
    # does not permit).
    #
    # A bare keyword call carries no DICOM Modality tag at all, so it cannot
    # affirm MR. Callers that mean "this is MR" must say so. Every production
    # call site passes a domain object whose construction binds MR
    # (study.py `or "MR"`, pipeline.py CMRSeries default "MR"), so no production
    # verdict changes; only test callers that relied on the implicit default
    # were migrated to bind modality="MR" explicitly.
    if mod is None or not str(mod).strip() or str(mod).strip().upper() != "MR":
        return "unknown"

    # Fatal geometry warnings gate
    if series is not None and any(w.code == "geometry.invalid" for w in series.warnings):
        return "unknown"

    combined_raw = f"{desc or ''} {prot or ''}".strip()
    if not combined_raw:
        return "unknown"

    # Check for raw special tokens (like t2*) before punctuation stripping
    raw_lower = combined_raw.lower()
    if "t2*" in raw_lower:
        return "unknown"

    cleaned = _clean_text(combined_raw)
    if not cleaned:
        return "unknown"

    # --- LGE Phase 1 / C1 (F1): DE-positive metadata NEVER reaches the cine pass
    #
    # Vendor cine indicators (trufi / true fisp / ssfp / fiesta / ...) also
    # occur INSIDE delayed-enhancement series names -- `de_trufi_sax_MAG`,
    # `de_true_fisp_sax_MAG` -- and the cine exclusion list has no `de_`
    # pattern, so the cine pass used to claim those as canonical `cine_sax`,
    # making a late-enhancement stack eligible for the cine auto-select
    # preference. DE-positive metadata is therefore evaluated against the LGE
    # rule ONLY:
    #
    #   * DE-positive AND no LGE hard exclusion               -> "lge"
    #   * DE-positive AND an LGE hard exclusion (cine tokens, perfusion, flow,
    #     mapping, ti scout, localizer, ...)                  -> "unknown"
    #     (contradictory -- fail-closed; never cine_sax, never lge)
    #
    # The cine pass below is consequently NOT run for DE-positive metadata at
    # all. Metadata with no DE evidence is unaffected: the cine pass keeps its
    # pre-Phase-1 behavior byte-for-byte.
    has_de_token = any(ind.search(cleaned) is not None for ind in _LGE_INDICATORS)
    if has_de_token:
        return _classify_lge(cleaned) or "unknown"

    # --- CINE pass (pre-Phase-1 behavior, unchanged) -------------------------
    # A cine exclusion disqualifies the series from cine_sax ONLY. It must not
    # fall through to "unknown" for the whole classifier: several cine
    # exclusions ARE the LGE positive evidence (psir, lge, delayed, scar,
    # viability) and others are routine on DE acquisitions (lax/2ch, ir, tse),
    # so the LGE pass below must still get its chance.
    has_cine_exclusion = any(excl.search(cleaned) is not None for excl in _EXCLUSIONS)

    if not has_cine_exclusion:
        has_cine = any(ind.search(cleaned) is not None for ind in _CINE_INDICATORS)
        has_sax = any(ind.search(cleaned) is not None for ind in _SAX_INDICATORS)
        if has_cine and has_sax:
            return "cine_sax"

    # --- LGE Phase 1 pass (additive) -----------------------------------------
    # Reached when the CINE pass did not classify the series. Uses its OWN
    # exclusion list, NOT the cine one: view tokens (lax/2ch/4ch),
    # inversion-recovery/anatomical tokens (ir, stir, tse, aorta) and T1/T2
    # weighting are NOT LGE exclusions.
    lge = _classify_lge(cleaned)
    if lge is not None:
        return lge

    return "unknown"


def _classify_lge(cleaned: str) -> str | None:
    """Second-pass LGE classification on already-cleaned text.

    Returns "lge" when positive delayed-enhancement evidence is present and no
    LGE-exclusionary token is; otherwise ``None`` (caller falls back to
    "unknown"). View-agnostic: sax AND lax DE series both classify as "lge".

    MAG/PSIR: `psir` alone IS positive evidence (it is in _LGE_INDICATORS),
    while MAG alone proves nothing and never excludes. A bare `mag_sax` stays
    unknown; `de_..._MAG` classifies "lge" via the `de_` token, and
    `e_..._psir_..._PSIR` via the `psir` token.

    Deterministic and metadata-only: no volume/pixel inspection, no ordering
    side effects.
    """
    # Exclusions first (fail-closed): a cine/perf/flow/mapping/ti-scout/
    # localizer token kills the LGE verdict even alongside a DE token.
    for excl in _LGE_EXCLUSIONS:
        if excl.search(cleaned):
            return None

    has_de = any(ind.search(cleaned) is not None for ind in _LGE_INDICATORS)
    if not has_de:
        return None

    return "lge"
