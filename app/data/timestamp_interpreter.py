"""app/data/timestamp_interpreter.py

Timestamp Interpretation Matrix for Powerwave — Phase D4.3.

Generates and ranks ALL plausible interpretations of ambiguous date/datetime
strings found in CSV/Excel operational files.

Design constraints:
  - Pure Python / NumPy — zero Qt dependency
  - Non-destructive — never mutates caller data
  - Stateless — all scoring context is passed explicitly
  - Safe for testing without any provider or Qt context

Classes:
    TimestampInterpretation  — one candidate parse of an ambiguous value
    TimestampInterpretationMatrix — ranked list for a set of raw values

Supported formats:
  - Ambiguous d/m/y, m/d/y, y/m/d (2-digit or 4-digit year)
  - Date-only and datetime (with/without milliseconds)
  - Excel serial date (float > 1000 treated as days since 1900-01-01)
  - Numeric seconds from epoch (large int/float)
  - ISO-8601 (unambiguous — returned as single top-ranked candidate)

Scoring factors:
  - Overlap with COMTRADE event timestamp
  - Overlap with manifest event timestamp
  - Monotonic increasing series
  - Realistic date range (2000–2060)
  - Parse success rate across sample values
  - Interval consistency (regularity of time steps)
  - Prior confirmed operator rule
  - Timezone consistency if available
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Realistic date range for operational power-system data
_YEAR_MIN = 2000
_YEAR_MAX = 2060

# Excel serial date base (1900-01-01, accounting for Lotus 1-2-3 leap-year bug)
_EXCEL_EPOCH = datetime(1899, 12, 30)

# Synthetic base used only to evaluate monotonicity/interval consistency for
# relative seconds columns; it is not an absolute timestamp interpretation.
_NUMERIC_SECONDS_BASE = datetime(2000, 1, 1)

# Large numeric threshold: values > this are treated as epoch seconds (Unix)
_EPOCH_SECONDS_THRESHOLD = 1_000_000_000  # ≥ 2001-09-09

# Medium numeric threshold: values in 1000–50000 treated as Excel serial dates
_EXCEL_SERIAL_MIN = 1000
_EXCEL_SERIAL_MAX = 73050  # ≈ 2100-01-01

# Confidence score weights — tuned to match D4.3 specification priority
_W_COMTRADE_OVERLAP = 0.30
_W_MANIFEST_OVERLAP = 0.20
_W_MONOTONIC = 0.15
_W_REALISTIC_RANGE = 0.15
_W_PARSE_RATE = 0.10
_W_INTERVAL_CONSISTENCY = 0.05
_W_CONFIRMED_RULE = 0.05   # bonus when confirmed_by_operator


# ─────────────────────────────────────────────────────────────────────────────
# Format catalogue
# ─────────────────────────────────────────────────────────────────────────────

# (format_string, label, ambiguous)
# Ambiguous = True means the same string could be parsed by multiple formats.
_DATE_FORMATS: list[tuple[str, str, bool]] = [
    # ── Unambiguous ISO / long-year ──────────────────────────────────────────
    ("%Y-%m-%d",               "ISO date (YYYY-MM-DD)",               False),
    ("%Y-%m-%dT%H:%M:%S",      "ISO datetime (YYYY-MM-DDTHH:MM:SS)",  False),
    ("%Y-%m-%dT%H:%M:%S.%f",   "ISO datetime with ms",                False),
    ("%Y-%m-%d %H:%M:%S",      "ISO datetime space-sep",              False),
    ("%Y-%m-%d %H:%M:%S.%f",   "ISO datetime space-sep with ms",      False),
    # ── Common ambiguous: M/D/YYYY vs D/M/YYYY ──────────────────────────────
    ("%m/%d/%Y",               "M/D/YYYY",                            True),
    ("%d/%m/%Y",               "D/M/YYYY",                            True),
    ("%m/%d/%Y %H:%M",         "M/D/YYYY HH:MM",                      True),
    ("%d/%m/%Y %H:%M",         "D/M/YYYY HH:MM",                      True),
    ("%m/%d/%Y %H:%M:%S",      "M/D/YYYY HH:MM:SS",                  True),
    ("%d/%m/%Y %H:%M:%S",      "D/M/YYYY HH:MM:SS",                  True),
    ("%m/%d/%Y %H:%M:%S.%f",   "M/D/YYYY HH:MM:SS.ms",               True),
    ("%d/%m/%Y %H:%M:%S.%f",   "D/M/YYYY HH:MM:SS.ms",               True),
    # ── 2-digit year variants ────────────────────────────────────────────────
    ("%m/%d/%y",               "M/D/YY",                              True),
    ("%d/%m/%y",               "D/M/YY",                              True),
    ("%y/%m/%d",               "YY/MM/DD",                            True),
    ("%m/%d/%y %H:%M",         "M/D/YY HH:MM",                       True),
    ("%d/%m/%y %H:%M",         "D/M/YY HH:MM",                       True),
    ("%m/%d/%y %H:%M:%S",      "M/D/YY HH:MM:SS",                    True),
    ("%d/%m/%y %H:%M:%S",      "D/M/YY HH:MM:SS",                    True),
    ("%m/%d/%y %H:%M:%S.%f",   "M/D/YY HH:MM:SS.ms",                 True),
    ("%d/%m/%y %H:%M:%S.%f",   "D/M/YY HH:MM:SS.ms",                 True),
    # ── Dash separators ──────────────────────────────────────────────────────
    ("%m-%d-%Y",               "M-D-YYYY",                            True),
    ("%d-%m-%Y",               "D-M-YYYY",                            True),
    # ── Space-separated (some SCADA exports) ────────────────────────────────
    ("%d %m %Y",               "D M YYYY",                            True),
    ("%m %d %Y",               "M D YYYY",                            True),
]


# ─────────────────────────────────────────────────────────────────────────────
# TimestampInterpretation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TimestampInterpretation:
    """One candidate interpretation of an ambiguous timestamp column.

    Attributes:
        format_string:    Python strptime/strftime format code.
        format_label:     Human-readable name shown in the UI.
        is_ambiguous:     True when other formats could also parse the same text.
        parse_success_rate: Fraction of sample values that parse successfully (0–1).
        parsed_samples:   Up to 5 successfully parsed datetime objects (for UI preview).
        confidence:       Overall ranked score (0–1). Higher = more likely correct.
        reason_codes:     Short reason tags used by the review dialog.
        source_type:      \"string\", \"excel_serial\", \"epoch_seconds\", \"numeric_seconds\"
    """

    format_string: str
    format_label: str
    is_ambiguous: bool
    parse_success_rate: float
    parsed_samples: list[datetime] = field(default_factory=list)
    confidence: float = 0.0
    reason_codes: list[str] = field(default_factory=list)
    source_type: str = "string"  # "string" | "excel_serial" | "epoch_seconds" | "numeric_seconds"


# ─────────────────────────────────────────────────────────────────────────────
# TimestampInterpretationMatrix
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class TimestampInterpretationMatrix:
    """Ranked list of all plausible interpretations for an ambiguous column.

    Attributes:
        column_name:          Source column name.
        sample_values:        Raw string values used for analysis.
        interpretations:      Ranked list (highest confidence first).
        is_ambiguous:         True when ≥2 interpretations have confidence > 0.3.
        recommended:          Top-ranked interpretation (or None if no parse succeeded).
        confirmed_by_rule:    If a persistent operator rule matched, it is stored here.
    """

    column_name: str
    sample_values: list[str]
    interpretations: list[TimestampInterpretation]
    is_ambiguous: bool
    recommended: TimestampInterpretation | None
    confirmed_by_rule: TimestampInterpretation | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal parse helpers
# ─────────────────────────────────────────────────────────────────────────────

def _try_parse(value: str, fmt: str) -> datetime | None:
    """Attempt to parse *value* using *fmt*; return None on failure."""
    try:
        return datetime.strptime(value.strip(), fmt)
    except (ValueError, TypeError):
        return None


def _year_in_range(dt: datetime) -> bool:
    return _YEAR_MIN <= dt.year <= _YEAR_MAX


def _parse_excel_serial(value: str) -> datetime | None:
    """Attempt to interpret *value* as an Excel serial date (days since 1899-12-30)."""
    try:
        v = float(value.strip())
    except (ValueError, TypeError):
        return None
    if not (_EXCEL_SERIAL_MIN <= v <= _EXCEL_SERIAL_MAX):
        return None
    try:
        return _EXCEL_EPOCH + timedelta(days=v)
    except OverflowError:
        return None


def _parse_epoch_seconds(value: str) -> datetime | None:
    """Attempt to interpret *value* as a Unix epoch timestamp (seconds since 1970)."""
    try:
        v = float(value.strip())
    except (ValueError, TypeError):
        return None
    if v < _EPOCH_SECONDS_THRESHOLD:
        return None
    try:
        return datetime.fromtimestamp(v, UTC).replace(tzinfo=None)
    except (OSError, OverflowError, ValueError):
        return None


def _parse_numeric_seconds(value: str) -> float | None:
    """Return value as numeric seconds-since-start (for time-only columns)."""
    try:
        v = float(value.strip())
        if 0.0 <= v < 1e8:
            return v
    except (ValueError, TypeError):
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_monotonic(datetimes: list[datetime]) -> bool:
    if len(datetimes) < 2:
        return True
    return all(datetimes[i] <= datetimes[i + 1] for i in range(len(datetimes) - 1))


def _interval_consistency(datetimes: list[datetime]) -> float:
    """Return a score 0–1 reflecting how regular the time steps are.

    1.0 = perfectly uniform; 0.0 = all over the place.
    """
    if len(datetimes) < 3:
        return 1.0
    intervals = [
        (datetimes[i + 1] - datetimes[i]).total_seconds()
        for i in range(len(datetimes) - 1)
    ]
    arr = np.array(intervals, dtype=np.float64)
    mean = float(np.mean(arr))
    if mean <= 0:
        return 0.0
    std = float(np.std(arr))
    cv = std / mean  # coefficient of variation
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def _overlaps_event(
    datetimes: list[datetime],
    event_start: datetime | None,
    event_end: datetime | None,
    window_seconds: float = 7200.0,
) -> bool:
    """Return True if any parsed datetime falls within *window_seconds* of the event."""
    if not datetimes or event_start is None:
        return False
    end = event_end or event_start
    lo = event_start - timedelta(seconds=window_seconds)
    hi = end + timedelta(seconds=window_seconds)
    return any(lo <= dt <= hi for dt in datetimes)


def _score_interpretation(
    datetimes: list[datetime],
    parse_rate: float,
    is_ambiguous: bool,
    comtrade_start: datetime | None = None,
    comtrade_end: datetime | None = None,
    manifest_start: datetime | None = None,
    manifest_end: datetime | None = None,
    confirmed_rule: bool = False,
) -> tuple[float, list[str]]:
    """Compute a confidence score and reason codes for one interpretation."""
    score = 0.0
    reasons: list[str] = []

    # Parse rate contribution
    score += parse_rate * _W_PARSE_RATE
    if parse_rate >= 0.9:
        reasons.append("high_parse_rate")
    elif parse_rate < 0.5:
        reasons.append("low_parse_rate")

    # Realistic date range
    valid_dts = [dt for dt in datetimes if _year_in_range(dt)]
    realistic_frac = len(valid_dts) / len(datetimes) if datetimes else 0.0
    score += realistic_frac * _W_REALISTIC_RANGE
    if realistic_frac == 1.0:
        reasons.append("realistic_date_range")
    elif realistic_frac < 0.5:
        reasons.append("unrealistic_dates")

    # Monotonic check
    if _is_monotonic(datetimes):
        score += _W_MONOTONIC
        reasons.append("monotonic")

    # Interval consistency
    ic = _interval_consistency(datetimes)
    score += ic * _W_INTERVAL_CONSISTENCY
    if ic > 0.9:
        reasons.append("uniform_intervals")

    # COMTRADE overlap
    if _overlaps_event(datetimes, comtrade_start, comtrade_end):
        score += _W_COMTRADE_OVERLAP
        reasons.append("overlaps_comtrade_event")

    # Manifest overlap
    if _overlaps_event(datetimes, manifest_start, manifest_end):
        score += _W_MANIFEST_OVERLAP
        reasons.append("overlaps_manifest_event")

    # Confirmed operator rule bonus
    if confirmed_rule:
        score += _W_CONFIRMED_RULE
        reasons.append("confirmed_by_operator")

    return float(np.clip(score, 0.0, 1.0)), reasons


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_interpretation_matrix(
    column_name: str,
    raw_values: Sequence[str],
    *,
    comtrade_start: datetime | None = None,
    comtrade_end: datetime | None = None,
    manifest_start: datetime | None = None,
    manifest_end: datetime | None = None,
    confirmed_format: str | None = None,   # operator-confirmed strptime format string
    max_samples: int = 20,
) -> TimestampInterpretationMatrix:
    """Generate and rank all plausible timestamp interpretations for *raw_values*.

    Args:
        column_name:      Column being analysed (for the matrix label).
        raw_values:       Raw string values from the CSV/Excel column.
        comtrade_start:   Optional COMTRADE event start time for overlap scoring.
        comtrade_end:     Optional COMTRADE event end time (defaults to start).
        manifest_start:   Optional manifest event start time for overlap scoring.
        manifest_end:     Optional manifest event end time.
        confirmed_format: Operator-confirmed strptime format from a prior rule.
        max_samples:      Maximum number of raw values to analyse (for performance).

    Returns:
        TimestampInterpretationMatrix with ranked interpretations and recommended choice.
    """
    samples = [str(v).strip() for v in raw_values if str(v).strip()][:max_samples]
    if not samples:
        return TimestampInterpretationMatrix(
            column_name=column_name,
            sample_values=[],
            interpretations=[],
            is_ambiguous=False,
            recommended=None,
        )

    # ── Special numeric checks ────────────────────────────────────────────────
    all_interpretations: list[TimestampInterpretation] = []

    # Excel serial date
    excel_parsed = [dt for s in samples if (dt := _parse_excel_serial(s)) is not None]
    if excel_parsed:
        rate = len(excel_parsed) / len(samples)
        score, reasons = _score_interpretation(
            excel_parsed, rate, False,
            comtrade_start, comtrade_end, manifest_start, manifest_end,
        )
        all_interpretations.append(TimestampInterpretation(
            format_string="excel_serial",
            format_label="Excel serial date (days since 1900)",
            is_ambiguous=False,
            parse_success_rate=rate,
            parsed_samples=excel_parsed[:5],
            confidence=score,
            reason_codes=reasons,
            source_type="excel_serial",
        ))

    # Unix epoch seconds
    epoch_parsed = [dt for s in samples if (dt := _parse_epoch_seconds(s)) is not None]
    if epoch_parsed:
        rate = len(epoch_parsed) / len(samples)
        score, reasons = _score_interpretation(
            epoch_parsed, rate, False,
            comtrade_start, comtrade_end, manifest_start, manifest_end,
        )
        all_interpretations.append(TimestampInterpretation(
            format_string="epoch_seconds",
            format_label="Unix epoch timestamp (seconds since 1970)",
            is_ambiguous=False,
            parse_success_rate=rate,
            parsed_samples=epoch_parsed[:5],
            confidence=score,
            reason_codes=reasons,
            source_type="epoch_seconds",
        ))

    # ── String format attempts ────────────────────────────────────────────────
    # Numeric seconds since record start. This is a relative time basis, so it
    # never earns absolute-date overlap points, but it can still be ranked by
    # parse rate, monotonicity, and interval consistency.
    numeric_seconds = [
        seconds for s in samples if (seconds := _parse_numeric_seconds(s)) is not None
    ]
    if numeric_seconds:
        rate = len(numeric_seconds) / len(samples)
        relative_datetimes = [
            _NUMERIC_SECONDS_BASE + timedelta(seconds=float(seconds))
            for seconds in numeric_seconds
        ]
        score = rate * _W_PARSE_RATE
        reasons: list[str] = []
        if rate >= 0.9:
            reasons.append("high_parse_rate")
        elif rate < 0.5:
            reasons.append("low_parse_rate")
        if _is_monotonic(relative_datetimes):
            score += _W_MONOTONIC
            reasons.append("monotonic")
        interval_score = _interval_consistency(relative_datetimes)
        score += interval_score * _W_INTERVAL_CONSISTENCY
        if interval_score > 0.9:
            reasons.append("uniform_intervals")
        reasons.append("relative_numeric_seconds")
        all_interpretations.append(TimestampInterpretation(
            format_string="numeric_seconds",
            format_label="Numeric seconds since start",
            is_ambiguous=False,
            parse_success_rate=rate,
            parsed_samples=relative_datetimes[:5],
            confidence=float(np.clip(score, 0.0, 1.0)),
            reason_codes=reasons,
            source_type="numeric_seconds",
        ))

    for fmt, label, is_ambiguous in _DATE_FORMATS:
        parsed = [dt for s in samples if (dt := _try_parse(s, fmt)) is not None]
        if not parsed:
            continue
        rate = len(parsed) / len(samples)
        is_confirmed_rule = (confirmed_format is not None and fmt == confirmed_format)
        score, reasons = _score_interpretation(
            parsed, rate, is_ambiguous,
            comtrade_start, comtrade_end, manifest_start, manifest_end,
            confirmed_rule=is_confirmed_rule,
        )
        all_interpretations.append(TimestampInterpretation(
            format_string=fmt,
            format_label=label,
            is_ambiguous=is_ambiguous,
            parse_success_rate=rate,
            parsed_samples=parsed[:5],
            confidence=score,
            reason_codes=reasons,
            source_type="string",
        ))

    # ── Rank by confidence (desc) then parse_success_rate (desc) ─────────────
    ranked = sorted(
        all_interpretations,
        key=lambda i: (i.confidence, i.parse_success_rate),
        reverse=True,
    )

    # ── Ambiguity detection ───────────────────────────────────────────────────
    # Ambiguous when ≥2 interpretations (with is_ambiguous=True) have conf > 0.30
    ambiguous_candidates = [i for i in ranked if i.is_ambiguous and i.confidence > 0.30]
    is_ambiguous = len(ambiguous_candidates) >= 2

    recommended = ranked[0] if ranked else None

    # ── Confirmed-rule interpretation ─────────────────────────────────────────
    confirmed_interp: TimestampInterpretation | None = None
    if confirmed_format is not None:
        confirmed_interp = next(
            (i for i in ranked if i.format_string == confirmed_format), None
        )

    return TimestampInterpretationMatrix(
        column_name=column_name,
        sample_values=samples,
        interpretations=ranked,
        is_ambiguous=is_ambiguous,
        recommended=recommended,
        confirmed_by_rule=confirmed_interp,
    )


def find_timestamp_candidates(
    column_names: list[str],
    raw_df,   # pd.DataFrame — avoid hard import
    *,
    comtrade_start: datetime | None = None,
    comtrade_end: datetime | None = None,
    manifest_start: datetime | None = None,
    manifest_end: datetime | None = None,
    confirmed_formats: dict[str, str] | None = None,
    max_samples: int = 20,
) -> dict[str, TimestampInterpretationMatrix]:
    """Evaluate all *column_names* as timestamp candidates and return a matrix per column.

    Args:
        column_names:       Columns to evaluate (typically all time-like columns).
        raw_df:             Raw pandas DataFrame.
        comtrade_start:     Optional COMTRADE start for overlap scoring.
        comtrade_end:       Optional COMTRADE end.
        manifest_start:     Optional manifest start for overlap scoring.
        manifest_end:       Optional manifest end.
        confirmed_formats:  Map of {column_name: confirmed strptime format}.
        max_samples:        Max values to analyse per column.

    Returns:
        Dict mapping column_name → TimestampInterpretationMatrix.
    """
    confirmed = confirmed_formats or {}
    result: dict[str, TimestampInterpretationMatrix] = {}
    for col in column_names:
        if col not in raw_df.columns:
            continue
        raw_vals = [str(v) for v in raw_df[col].dropna().tolist()]
        matrix = build_interpretation_matrix(
            column_name=col,
            raw_values=raw_vals,
            comtrade_start=comtrade_start,
            comtrade_end=comtrade_end,
            manifest_start=manifest_start,
            manifest_end=manifest_end,
            confirmed_format=confirmed.get(col),
            max_samples=max_samples,
        )
        result[col] = matrix
    return result


def select_best_timestamp_column(
    matrices: dict[str, TimestampInterpretationMatrix],
) -> tuple[str, TimestampInterpretationMatrix] | None:
    """From multiple timestamp candidate matrices, select the best single column.

    Criteria (in order):
      1. Confirmed-by-rule column
      2. Highest recommended confidence
      3. Highest parse success rate

    Returns:
        (column_name, matrix) tuple, or None if no candidates exist.
    """
    if not matrices:
        return None

    # Confirmed-by-rule takes absolute priority
    for col, matrix in matrices.items():
        if matrix.confirmed_by_rule is not None:
            return col, matrix

    # Rank by recommended confidence then parse rate
    scored = [
        (col, matrix)
        for col, matrix in matrices.items()
        if matrix.recommended is not None
    ]
    if not scored:
        return None

    scored.sort(
        key=lambda x: (
            x[1].recommended.confidence if x[1].recommended else 0.0,
            x[1].recommended.parse_success_rate if x[1].recommended else 0.0,
        ),
        reverse=True,
    )
    return scored[0]
