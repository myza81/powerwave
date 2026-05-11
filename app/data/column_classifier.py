"""Lightweight CSV column signal classifier for Powerwave.

Classifies data columns by signal type using name-based heuristics and optional
value-profile hints. Returns ColumnClassification with confidence score and a
requires_user_confirmation flag for low-confidence results.

Confidence threshold: columns below 0.80 are flagged for user confirmation.
Classification is metadata — it does not rename or reinterpret source data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Columns below this threshold are flagged requires_user_confirmation=True.
CONFIRMATION_THRESHOLD = 0.80


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
}

# Ordered keyword rules: first match wins.
# Each entry: (substrings_to_search, signal_type, unit, display_group, confidence)
# All substring searches are case-insensitive (normalised name is already lowercased).
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
    """Return (signal_type, unit, display_group, confidence, inferred_from) or None."""
    norm = _normalize(name)

    if norm in _EXACT:
        st, unit, dg, conf = _EXACT[norm]
        return st, unit, dg, conf, "name_exact"

    for keywords, st, unit, dg, conf in _KEYWORD:
        for kw in keywords:
            if kw in norm:
                return st, unit, dg, conf, "name_keyword"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Value-profile inference (secondary, always below confirmation threshold)
# ─────────────────────────────────────────────────────────────────────────────


def _value_classify(
    values: Sequence[float],
) -> tuple[str | None, str | None, str, float, str] | None:
    """Return (signal_type, unit, display_group, confidence, inferred_from) or None."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 5:
        return None

    mean_v = float(np.mean(finite))
    std_v = float(np.std(finite))

    # Frequency-like: mean near 50 or 60 Hz, tight range
    for nominal in (50.0, 60.0):
        if abs(mean_v - nominal) < 2.0 and std_v < 2.0:
            return "frequency", "Hz", "frequency", 0.70, "value_profile"

    # Per-unit voltage-like: mean near 1.0, very tight range
    if 0.85 <= mean_v <= 1.15 and std_v < 0.15:
        return "voltage_rms", "pu", "voltage_rms", 0.55, "value_profile"

    # Power-like: large absolute values with meaningful spread
    if abs(mean_v) > 50 and std_v > 5:
        return "active_power", "MW", "power", 0.50, "value_profile"

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
