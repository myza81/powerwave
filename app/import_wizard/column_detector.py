"""Column classification — map raw column names to ParameterType + unit.

Stateless functions only.  No Qt, pandas, or numpy.

Classification consults the shared, UI-independent semantic classifier in
app.data.column_classifier (the canonical name-based classifier for CSV/Excel
analog quantities, e.g. "System Demand" -> active_power/MW) whenever the
Wizard's own name rules (_NAME_RULES, below) don't recognise a column. The
Wizard's own rules are tried first because they cover things the shared
classifier does not (digital/status and timestamp keywords) and are already
well-tested for the cases they handle; the shared classifier fills in
terminology the Wizard's own rules miss (demand, tie-line, real power, etc.)
without duplicating that keyword knowledge here. See _SHARED_TO_PARAMETER_TYPE
for the small boundary mapping between the two taxonomies.
"""
from __future__ import annotations

import re
from typing import Any

from app.data.column_classifier import classify_csv_column
from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.models import ColumnMappingCandidate

# Boundary mapping: app.data.column_classifier's signal_type vocabulary ->
# this module's ParameterType enum. Deliberately small and explicit per
# integration-boundary policy -- not a repository-wide taxonomy migration.
_SHARED_TO_PARAMETER_TYPE: dict[str, ParameterType] = {
    "active_power":   ParameterType.MW,
    "reactive_power": ParameterType.MVAR,
    "voltage_rms":    ParameterType.VOLTAGE,
    "current_rms":    ParameterType.CURRENT,
    "frequency":      ParameterType.FREQUENCY,
    "rocof":          ParameterType.ROCOF,
}

# ---------------------------------------------------------------------------
# Name-fragment → (ParameterType, unit) lookup
# ---------------------------------------------------------------------------

_NAME_RULES: list[tuple[re.Pattern[str], ParameterType, str | None]] = [
    # Voltage
    (re.compile(r"\bv\b|\bv[abc]\b|volt|voltage|vln|vll|kv|pu_v", re.IGNORECASE), ParameterType.VOLTAGE, "V"),
    # Current
    (re.compile(r"\bi\b|curr|\bamps?\b|\bampere\b|\bi[abc]\b", re.IGNORECASE), ParameterType.CURRENT, "A"),
    # Reactive power — must come before active power to avoid "active" matching in "reactive"
    (re.compile(r"\bmvar|reactive.?power|q_mvar|\bkvar\b", re.IGNORECASE), ParameterType.MVAR, "Mvar"),
    # Active power
    (re.compile(r"\bmw|\bactive.?power|p_mw|\bkw\b|megawatt", re.IGNORECASE), ParameterType.MW, "MW"),
    # Frequency
    (re.compile(r"\bfreq\b|frequency|\bhz|f_hz", re.IGNORECASE), ParameterType.FREQUENCY, "Hz"),
    # ROCOF
    (re.compile(r"rocof|df.?dt|rate.of.change.of.freq", re.IGNORECASE), ParameterType.ROCOF, "Hz/s"),
    # Digital
    (re.compile(r"\bdig\b|digital|status|flag|bool|binary|trip|alarm|cb\b", re.IGNORECASE), ParameterType.DIGITAL, None),
    # Timestamp
    (re.compile(r"\btime\b|timestamp|datetime|date\b|ts\b|t_stamp", re.IGNORECASE), ParameterType.TIMESTAMP, None),
]

# Unit string → ParameterType
_UNIT_RULES: dict[str, tuple[ParameterType, str]] = {
    "v": (ParameterType.VOLTAGE, "V"),
    "kv": (ParameterType.VOLTAGE, "kV"),
    "pu": (ParameterType.VOLTAGE, "pu"),
    "a": (ParameterType.CURRENT, "A"),
    "ka": (ParameterType.CURRENT, "kA"),
    "mw": (ParameterType.MW, "MW"),
    "kw": (ParameterType.MW, "kW"),
    "mvar": (ParameterType.MVAR, "Mvar"),
    "kvar": (ParameterType.MVAR, "kvar"),
    "hz": (ParameterType.FREQUENCY, "Hz"),
    "hz/s": (ParameterType.ROCOF, "Hz/s"),
}

# Canonical output name fragments per type
_CANONICAL_PREFIXES: dict[ParameterType, str] = {
    ParameterType.VOLTAGE: "voltage",
    ParameterType.CURRENT: "current",
    ParameterType.MW: "mw",
    ParameterType.MVAR: "mvar",
    ParameterType.FREQUENCY: "frequency",
    ParameterType.ROCOF: "rocof",
    ParameterType.DIGITAL: "digital",
    ParameterType.TIMESTAMP: "timestamp",
    ParameterType.UNKNOWN: "channel",
}


def _clean_name(name: str) -> str:
    """Strip units in parentheses / brackets from a column name."""
    return re.sub(r"[\(\[\{][^\)\]\}]*[\)\]\}]", "", name).strip()


def classify_by_name(col_name: str) -> tuple[ParameterType, str | None, float, str]:
    """Classify a column by its name alone.

    Returns
    -------
    (parameter_type, unit, confidence, reason)
    """
    clean = _clean_name(col_name)
    for pattern, ptype, unit in _NAME_RULES:
        if pattern.search(clean):
            return ptype, unit, 0.75, f"name matches pattern for {ptype.value}"
    return ParameterType.UNKNOWN, None, 0.10, "no name pattern matched"


def _classify_by_values(
    samples: list[Any],
) -> tuple[ParameterType | None, float]:
    """Heuristic value-based classification.

    Numeric magnitude alone must not authoritatively assign an engineering
    semantic type (voltage, current, MW, MVAr, or ROCOF): a demand channel
    reading in the tens of thousands, a current channel reading in the low
    hundreds, and an actual kV-range voltage channel are numerically
    indistinguishable, so no magnitude band exists here for those types --
    name-based evidence (this module's _NAME_RULES, or the shared semantic
    classifier consulted by detect_column_mappings) is required instead.

    Two exceptions remain, both materially different from a generic magnitude
    guess:
      * Binary/status values ({0, 1}) are a structurally distinct pattern,
        not a magnitude coincidence -- kept as DIGITAL evidence.
      * A tight 45-65 Hz band is kept as FREQUENCY evidence: mains frequency
        occupies a narrow, well-separated numeric range that does not
        plausibly collide with voltage/current/power magnitudes, and this
        exact band is exercised by existing tests
        (TestClassifyByValues.test_frequency_range,
        TestDetectColumnMappings.test_frequency_detected_by_value).

    Returns (type_hint, confidence_boost) or (None, 0.0) when inconclusive.
    """
    non_empty = [v for v in samples if v is not None and str(v).strip() != ""]
    if len(non_empty) < 3:
        return None, 0.0

    # Count binary (0/1) values → DIGITAL
    binary = sum(1 for v in non_empty if str(v).strip() in ("0", "1", "0.0", "1.0"))
    if binary / len(non_empty) >= 0.90:
        return ParameterType.DIGITAL, 0.20

    # Try to parse as float
    floats: list[float] = []
    for v in non_empty:
        try:
            floats.append(float(str(v).strip()))
        except (ValueError, TypeError):
            pass

    if len(floats) < 3:
        return None, 0.0

    # Frequency: 45–65 Hz (see docstring — the one safe magnitude exception)
    if all(45 <= f <= 65 for f in floats):
        return ParameterType.FREQUENCY, 0.20

    return None, 0.0


def _suggested_name(
    source_name: str,
    ptype: ParameterType,
    existing_names: set[str],
) -> str:
    """Generate a unique canonical output name."""
    prefix = _CANONICAL_PREFIXES.get(ptype, "channel")
    # Use the cleaned source name with underscores
    clean = re.sub(r"[^\w]", "_", _clean_name(source_name)).strip("_").lower()
    if clean and clean != prefix:
        candidate = f"{prefix}_{clean}"
    else:
        candidate = prefix

    # Ensure uniqueness
    base = candidate
    counter = 1
    while candidate in existing_names:
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def detect_column_mappings(
    column_names: list[str],
    preview_rows: list[list[Any]],
    *,
    timestamp_column_names: set[str] | None = None,
    max_sample: int = 20,
) -> list[ColumnMappingCandidate]:
    """Classify every column and return ColumnMappingCandidate objects.

    Parameters
    ----------
    column_names:           Header names from the profiler.
    preview_rows:           Data rows (list of lists).
    timestamp_column_names: Names already claimed as timestamp columns (excluded
                            from the output mapping as TIMESTAMP type).
    max_sample:             Maximum rows to use for value-based classification.
    """
    ts_names: set[str] = timestamp_column_names or set()
    mappings: list[ColumnMappingCandidate] = []
    used_names: set[str] = set()

    # Build column samples
    col_samples: list[list[Any]] = [[] for _ in range(len(column_names))]
    for row in preview_rows[:max_sample]:
        for ci in range(len(column_names)):
            val = row[ci] if ci < len(row) else ""
            col_samples[ci].append(val)

    for ci, col_name in enumerate(column_names):
        # Columns claimed by timestamp detection
        if col_name in ts_names:
            sname = _suggested_name(col_name, ParameterType.TIMESTAMP, used_names)
            used_names.add(sname)
            mappings.append(
                ColumnMappingCandidate(
                    source_name=col_name,
                    source_index=ci,
                    suggested_name=sname,
                    parameter_type=ParameterType.TIMESTAMP,
                    unit=None,
                    confidence=0.90,
                    classification_reason="claimed by timestamp detector",
                )
            )
            continue

        ptype, unit, confidence, reason = classify_by_name(col_name)

        # Shared semantic classifier: consulted when the Wizard's own name
        # rules don't recognise the column (e.g. demand/tie-line/real-power
        # terminology). A low-confidence shared result is still adopted here
        # (it replaces an otherwise-UNKNOWN type with a semantically safer
        # guess) but keeps its own sub-threshold confidence so it is not
        # treated as authoritative — confidence stays visible to the caller
        # and the user can still override it.
        if ptype == ParameterType.UNKNOWN:
            shared = classify_csv_column(col_name)
            mapped_type = _SHARED_TO_PARAMETER_TYPE.get(shared.signal_type or "")
            if mapped_type is not None:
                ptype = mapped_type
                unit = shared.unit or unit
                confidence = shared.confidence
                reason = f"shared classifier: {shared.inferred_from}"

        # Value boost — digital/status and a narrow frequency band only
        # (see _classify_by_values docstring); never independently assigns
        # voltage, current, MW, MVAr, or ROCOF from magnitude alone.
        type_hint, boost = _classify_by_values(col_samples[ci])
        if type_hint is not None:
            if ptype == ParameterType.UNKNOWN:
                ptype = type_hint
                reason = "classified by value range analysis"
            confidence = min(1.0, confidence + boost)

        sname = _suggested_name(col_name, ptype, used_names)
        used_names.add(sname)

        # Infer unit from name parenthetical if present (e.g. "Voltage (kV)")
        unit_in_name = re.search(r"[\(\[]([\w/]+)[\)\]]", col_name)
        if unit_in_name:
            candidate_unit = unit_in_name.group(1).strip().lower()
            if candidate_unit in _UNIT_RULES:
                inferred_type, inferred_unit = _UNIT_RULES[candidate_unit]
                unit = inferred_unit
                if ptype == ParameterType.UNKNOWN:
                    ptype = inferred_type

        mappings.append(
            ColumnMappingCandidate(
                source_name=col_name,
                source_index=ci,
                suggested_name=sname,
                parameter_type=ptype,
                unit=unit,
                confidence=confidence,
                classification_reason=reason,
            )
        )

    return mappings
