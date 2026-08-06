"""Lightweight CSV column signal classifier for Powerwave.

Classifies data columns by signal type using name-based heuristics, synonym/fuzzy
matching, and optional value-profile hints. Returns ColumnClassification with
confidence score and a requires_user_confirmation flag for low-confidence results.

Priority order (Phase D4.3 — first match wins):
  1. Previously confirmed mapping rule  (applied by IntelligenceManager on top)
  2. Manifest mapping                   (applied by IntelligenceManager on top)
  3. Exact known keyword                (_EXACT table)
  4. Fuzzy/synonym keyword match        (_KEYWORD + synonym library)
  5. Value profile                      (_value_classify)
  6. Operator review required           (confidence=0.0, inferred_from="unknown")

Confidence threshold: columns below 0.80 are flagged for user confirmation.
Classification is metadata — it does not rename or reinterpret source data.

NO SILENT LOSS: every numeric column must receive an explicit disposition via
  numeric_column_disposition(). The return type distinguishes:
    "analog"                 — classified as analog waveform data
    "digital"                — classified as digital (binary) channel
    "unknown_requires_review" — confidence too low; operator review required
    "ignored_with_reason"    — explicitly ignored with a non-empty reason string
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.data.channel_name_matching import has_status_qualifier, has_token_phrase

# Columns below this threshold are flagged requires_user_confirmation=True.
CONFIRMATION_THRESHOLD = 0.80

# Disposition literals returned by numeric_column_disposition()
DISPOSITION_ANALOG = "analog"
DISPOSITION_DIGITAL = "digital"
DISPOSITION_REVIEW = "unknown_requires_review"
DISPOSITION_IGNORED = "ignored_with_reason"


@dataclass(frozen=True, slots=True)
class ColumnClassification:
    """Signal classification result for a single CSV data column."""

    column_name: str
    signal_type: str | None        # e.g. "frequency", "active_power", "reactive_power"
    unit: str | None               # e.g. "Hz", "MW", "MVAr", "kV", "A"
    display_group: str             # e.g. "frequency", "power", "voltage_rms", "other"
    confidence: float              # 0.0 – 1.0
    inferred_from: str             # "name_exact", "name_keyword", "value_profile", "unknown"
    requires_user_confirmation: bool
    notes: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Name-based rules
# ─────────────────────────────────────────────────────────────────────────────

# Exact match on normalised (lowercased, stripped) column name.
# Maps normalised_name → (signal_type, unit, display_group, confidence)
_EXACT: dict[str, tuple[str, str, str, float]] = {
    "frequency":  ("frequency",      "Hz",   "frequency",    0.95),
    "freq":       ("frequency",      "Hz",   "frequency",    0.90),
    "mw":         ("active_power",   "MW",   "power",        0.95),
    "mvar":       ("reactive_power", "MVAr", "power",        0.95),
    "mvarh":      ("reactive_power", "MVAr", "power",        0.72),
    "rocof":      ("rocof",          "Hz/s", "rocof",        0.95),
    "dfdt":       ("rocof",          "Hz/s", "rocof",        0.92),
    "df/dt":      ("rocof",          "Hz/s", "rocof",        0.92),
    "df_dt":      ("rocof",          "Hz/s", "rocof",        0.90),
    "hz/s":       ("rocof",          "Hz/s", "rocof",        0.92),
    "hz_s":       ("rocof",          "Hz/s", "rocof",        0.88),
    "kv":         ("voltage_rms",    "kV",   "voltage_rms",  0.90),
    "vpu":        ("voltage_rms",    "pu",   "voltage_rms",  0.88),
    # Single-letter names — low confidence, always require confirmation
    "p":          ("active_power",   "MW",   "power",        0.72),
    "q":          ("reactive_power", "MVAr", "power",        0.72),
    "f":          ("frequency",      "Hz",   "frequency",    0.72),
    "a":          ("current_rms",    "A",    "current_rms",  0.60),
    # Relay-style phase voltage/current — unambiguous power-system naming
    # convention (IEC 61850 / protection-relay style). Name-based only; never
    # inferred from value magnitude.
    "va":         ("voltage_rms",    "V",    "voltage_rms",  0.85),
    "vb":         ("voltage_rms",    "V",    "voltage_rms",  0.85),
    "vc":         ("voltage_rms",    "V",    "voltage_rms",  0.85),
    "vab":        ("voltage_rms",    "V",    "voltage_rms",  0.85),
    "vbc":        ("voltage_rms",    "V",    "voltage_rms",  0.85),
    "vca":        ("voltage_rms",    "V",    "voltage_rms",  0.85),
    "ia":         ("current_rms",    "A",    "current_rms",  0.85),
    "ib":         ("current_rms",    "A",    "current_rms",  0.85),
    "ic":         ("current_rms",    "A",    "current_rms",  0.85),
    "in":         ("current_rms",    "A",    "current_rms",  0.82),
    # Symmetrical-component notation (zero/positive/negative sequence)
    "v0":         ("voltage_rms",    "V",    "voltage_rms",  0.82),
    "v1":         ("voltage_rms",    "V",    "voltage_rms",  0.82),
    "v2":         ("voltage_rms",    "V",    "voltage_rms",  0.82),
    "i0":         ("current_rms",    "A",    "current_rms",  0.82),
    "i1":         ("current_rms",    "A",    "current_rms",  0.82),
    "i2":         ("current_rms",    "A",    "current_rms",  0.82),
}

# Ordered keyword rules: first match wins.
# Each entry: (phrases_to_search, signal_type, unit, display_group, confidence)
# Matching is boundary-aware (see app.data.channel_name_matching) -- a phrase
# must appear as its own token or contiguous run of tokens, not as an
# arbitrary substring, so e.g. "curr" no longer matches inside "Occurrence"
# and "voltage" no longer matches inside "Voltage Status" (the latter is
# additionally blocked by the qualifier check in _name_classify below,
# regardless of phrase boundaries, since "status" indicates the name
# describes a status/control signal rather than the measurement itself).
_KEYWORD: list[tuple[tuple[str, ...], str, str, str, float]] = [
    # Most specific patterns first
    (("system frequency",),
     "frequency",      "Hz",   "frequency",    0.95),
    (("rocof", "df/dt", "dfdt", "df_dt", "hz/s"),
     "rocof",          "Hz/s", "rocof",        0.90),
    (("frequency", "freq"),
     "frequency",      "Hz",   "frequency",    0.90),
    (("reactive power",),
     "reactive_power", "MVAr", "power",        0.90),
    (("active power", "real power"),
     "active_power",   "MW",   "power",        0.90),
    # "system demand" / "total demand" are unambiguous power-system terms
    (("system demand", "total demand", "total load"),
     "active_power",   "MW",   "power",        0.85),
    (("mvar",),
     "reactive_power", "MVAr", "power",        0.88),
    # Standalone "demand", "load", etc. — above threshold but still general
    (("demand", "load", "generation", "output"),
     "active_power",   "MW",   "power",        0.78),
    # "P Total" / "Q Total" — common relay/meter total-power designators
    (("p total", "p_total", "ptotal"),
     "active_power",   "MW",   "power",        0.75),
    (("q total", "q_total", "qtotal"),
     "reactive_power", "MVAr", "power",        0.75),
    # Tie-line / interchange — likely MW but ambiguous enough for confirmation
    (("tie-line", "tieline", "tie line", "interchange", "export", "import"),
     "active_power",   "MW",   "power",        0.70),
    (("bus voltage", "bus volt", "pu voltage"),
     "voltage_rms",    "pu",   "voltage_rms",  0.88),
    (("voltage",),
     "voltage_rms",    "kV",   "voltage_rms",  0.80),
    (("current", "amps"),
     "current_rms",    "A",    "current_rms",  0.80),
]


def _normalize(name: str) -> str:
    return name.strip().lower()


def _name_classify(
    name: str,
) -> tuple[str | None, str | None, str, float, str] | None:
    """Return (signal_type, unit, display_group, confidence, inferred_from) or None.

    Exact matches (_EXACT) are checked first and are unaffected by qualifier
    suppression below: they require the whole normalised name to equal a
    known relay/unit token exactly (e.g. "va", "mw"), so a status/control
    qualifier word can never be part of an exact match in the first place.

    Keyword matches (_KEYWORD) are boundary-aware phrase matches, gated by
    a status/control qualifier check: if the name contains a qualifier word
    (see has_status_qualifier), no keyword phrase is considered a match --
    a name like "Voltage Status" or "Current State" describes a
    status/control signal *about* the measurement, not the measurement
    itself, and must not be assigned that measurement's type from name
    evidence alone. This does not affect learned/persistent mapping rules
    (applied by IntelligenceManager on top of this function's result, per
    the module docstring's priority order) -- an operator-confirmed rule
    still overrides whatever this function returns, including None.
    """
    norm = _normalize(name)

    if norm in _EXACT:
        st, unit, dg, conf = _EXACT[norm]
        return st, unit, dg, conf, "name_exact"

    if has_status_qualifier(name):
        return None

    for keywords, st, unit, dg, conf in _KEYWORD:
        for kw in keywords:
            if has_token_phrase(name, kw):
                return st, unit, dg, conf, "name_keyword"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Value-profile inference (secondary, always below confirmation threshold)
# ─────────────────────────────────────────────────────────────────────────────


def _value_classify(
    values: Sequence[float],
) -> tuple[str | None, str | None, str, float, str] | None:
    """Return (signal_type, unit, display_group, confidence, inferred_from) or None.

    Numeric magnitude/statistics alone must not assign an electrical parameter
    type: voltage, current, active power, reactive power, ROCOF, and per-unit
    ranges all overlap in plausible magnitude with each other and with
    arbitrary non-electrical process data, so no band exists here for those
    types -- name-based evidence (_name_classify, tried first by
    classify_csv_column) or explicit unit metadata is required instead.

    The one retained exception is a narrow mains-frequency band (50/60 Hz
    nominal, tight tolerance): frequency occupies a well-separated numeric
    range that does not plausibly collide with voltage/current/power
    magnitudes, and this exact band is exercised by existing tests
    (TestValueProfileInference.test_near_50hz_detected,
    TestValueProfileInference.test_near_60hz_detected). It mirrors the same
    narrow exception retained in the Import Wizard's own
    app.import_wizard.column_detector._classify_by_values.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 5:
        return None

    mean_v = float(np.mean(finite))
    std_v = float(np.std(finite))

    # Frequency-like: mean near 50 or 60 Hz, tight range (the one safe
    # magnitude exception -- see docstring).
    for nominal in (50.0, 60.0):
        if abs(mean_v - nominal) < 2.0 and std_v < 2.0:
            return "frequency", "Hz", "frequency", 0.70, "value_profile"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def classify_csv_column(
    column_name: str,
    values: Sequence[float] | None = None,
) -> ColumnClassification:
    """Classify a CSV column by signal type.

    Name-based rules take precedence (higher confidence). Value-profile hints
    are used only when no name rule matches.

    Columns with confidence < CONFIRMATION_THRESHOLD (0.80) receive
    requires_user_confirmation=True regardless of inferred_from.
    """
    result = _name_classify(column_name)
    if result is None and values is not None:
        result = _value_classify(values)

    if result is not None:
        st, unit, dg, conf, inferred = result
        return ColumnClassification(
            column_name=column_name,
            signal_type=st,
            unit=unit,
            display_group=dg,
            confidence=conf,
            inferred_from=inferred,
            requires_user_confirmation=conf < CONFIRMATION_THRESHOLD,
        )

    return ColumnClassification(
        column_name=column_name,
        signal_type=None,
        unit=None,
        display_group="other",
        confidence=0.0,
        inferred_from="unknown",
        requires_user_confirmation=True,
    )


def classify_csv_columns(
    dataframe,  # pd.DataFrame — avoid hard import at module level
    timestamp_column: str | None = None,
) -> dict[str, ColumnClassification]:
    """Classify all non-timestamp columns in a DataFrame.

    Args:
        dataframe:        A pandas DataFrame with data columns.
        timestamp_column: Name of the timestamp column to skip.

    Returns:
        Mapping of column_name → ColumnClassification.
    """
    result: dict[str, ColumnClassification] = {}
    for col in dataframe.columns:
        if col == timestamp_column:
            continue
        try:
            vals: list[float] = dataframe[col].dropna().astype(float).tolist()
        except (ValueError, TypeError):
            vals = []
        result[col] = classify_csv_column(col, vals if vals else None)
    return result


def numeric_column_disposition(
    classification: ColumnClassification,
    is_digital: bool = False,
    ignore_reason: str | None = None,
) -> tuple[str, str | None]:
    """Map a ColumnClassification to an explicit disposition literal.

    This function implements the NO-SILENT-LOSS guarantee: every numeric column
    must be classified into exactly one of four states.

    Args:
        classification:  Result from classify_csv_column / classify_csv_columns.
        is_digital:      True when the column was identified as a digital (0/1) channel.
        ignore_reason:   Non-empty string when the caller explicitly wants to exclude
                         this column from the record (e.g. duplicate timestamp artifact).

    Returns:
        (disposition, reason) where disposition is one of:
            DISPOSITION_ANALOG    — column is an analog waveform channel
            DISPOSITION_DIGITAL   — column is a digital (binary) status channel
            DISPOSITION_REVIEW    — classification uncertain; operator review required
            DISPOSITION_IGNORED   — explicitly excluded; reason explains why
        reason is None except when disposition is DISPOSITION_IGNORED.
    """
    if ignore_reason:
        return DISPOSITION_IGNORED, ignore_reason
    if is_digital:
        return DISPOSITION_DIGITAL, None
    if classification.signal_type is None and classification.confidence == 0.0:
        return DISPOSITION_REVIEW, None
    if classification.requires_user_confirmation:
        # confidence > 0 but below threshold — still useful as analog, just flagged
        return DISPOSITION_ANALOG, None
    return DISPOSITION_ANALOG, None
