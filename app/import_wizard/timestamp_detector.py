"""Timestamp column detection and format inference.

Stateless functions only.  No Qt, pandas, or numpy.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.import_wizard.models import TimestampCandidate

# ---------------------------------------------------------------------------
# strptime format probes — tried in order; first match wins
# ---------------------------------------------------------------------------

_STRPTIME_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%H:%M:%S.%f",
    "%H:%M:%S",
)

# Epoch ranges
_EPOCH_SECONDS_MIN = 1e9   # ~2001-09-09
_EPOCH_SECONDS_MAX = 2e10  # ~2603-02-07
_EPOCH_MS_MIN = 1e12
_EPOCH_MS_MAX = 2e13
# Excel serial date range
_EXCEL_SERIAL_MIN = 25_000  # ~1968
_EXCEL_SERIAL_MAX = 55_000  # ~2050

# Name fragments that suggest a timestamp column
_TIMESTAMP_NAME_FRAGMENTS: tuple[str, ...] = (
    "time", "timestamp", "date", "datetime", "ts", "t_stamp", "zeit", "fecha",
)

_ELAPSED_SECONDS_LABEL = "elapsed_seconds"
_ELAPSED_MS_LABEL = "elapsed_milliseconds"
_ELAPSED_MINUTES_LABEL = "elapsed_minutes"

_ELAPSED_TIME_NAME_FRAGMENTS: tuple[str, ...] = (
    "time", "elapsed", "duration", "second", "seconds", "sec", "minute",
    "minutes", "min", "millisecond", "milliseconds", "msec", "ms",
)


def _try_strptime(value: str) -> str | None:
    """Return the first matching strptime format string, or None."""
    value = value.strip()
    for fmt in _STRPTIME_FORMATS:
        try:
            datetime.strptime(value, fmt)
            return fmt
        except ValueError:
            continue
    return None


def _detect_special_format(value: str) -> str | None:
    """Detect epoch-seconds, epoch-milliseconds, or Excel serial date."""
    stripped = value.strip()
    try:
        fval = float(stripped)
    except ValueError:
        return None
    if _EPOCH_SECONDS_MIN <= fval <= _EPOCH_SECONDS_MAX:
        return "epoch_seconds"
    if _EPOCH_MS_MIN <= fval <= _EPOCH_MS_MAX:
        return "epoch_milliseconds"
    if _EXCEL_SERIAL_MIN <= fval <= _EXCEL_SERIAL_MAX and "." not in stripped:
        return "excel_serial"
    return None


def _parse_for_duplicate_key(value: str, fmt: str | None) -> object | None:
    """Return a normalized comparable value for duplicate timestamp checks."""
    stripped = str(value).strip()
    if not stripped:
        return None
    if fmt is None:
        return stripped
    if fmt in {
        _ELAPSED_SECONDS_LABEL,
        _ELAPSED_MS_LABEL,
        _ELAPSED_MINUTES_LABEL,
        "epoch_seconds",
        "epoch_milliseconds",
        "excel_serial",
    }:
        try:
            return float(stripped)
        except ValueError:
            return None
    try:
        return datetime.strptime(stripped, fmt)
    except ValueError:
        return None


def _count_duplicate_parsed_values(sample_values: list[str], fmt: str | None) -> int:
    """Count duplicate parsed timestamps in *sample_values*.

    The first occurrence is not counted, matching pandas ``duplicated`` behavior.
    Invalid or blank values are ignored because they are already tracked by
    invalid_sample_count.
    """
    seen: set[object] = set()
    duplicates = 0
    for value in sample_values:
        key = _parse_for_duplicate_key(value, fmt)
        if key is None:
            continue
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def infer_timestamp_format(sample_values: list[str]) -> tuple[str | None, int]:
    """Infer the strptime format (or special label) from a list of sample values.

    Returns
    -------
    (format_string, invalid_count)
        format_string  — strptime format, "epoch_seconds", "epoch_milliseconds",
                         "excel_serial", or None when no format matched.
        invalid_count  — number of sample values that did NOT match the winner.
    """
    non_empty = [v for v in sample_values if v.strip()]
    if not non_empty:
        return None, len(sample_values)

    # Vote: count how many samples match each format
    format_votes: dict[str, int] = {}
    for val in non_empty:
        fmt = _try_strptime(val)
        if fmt is None:
            fmt = _detect_special_format(val)
        if fmt is not None:
            format_votes[fmt] = format_votes.get(fmt, 0) + 1

    if not format_votes:
        return None, len(sample_values)

    best_fmt = max(format_votes, key=lambda k: format_votes[k])
    matched = format_votes[best_fmt]
    invalid = len(sample_values) - matched
    return best_fmt, max(0, invalid)


def _name_suggests_timestamp(col_name: str) -> bool:
    lower = col_name.lower()
    return any(frag in lower for frag in _TIMESTAMP_NAME_FRAGMENTS)


def _name_suggests_elapsed_time(col_name: str) -> bool:
    """Return True when a column name suggests a relative duration axis."""
    lower = col_name.lower().strip()
    if lower in {"t", "time", "sec", "secs", "s", "ms", "msec"}:
        return True
    return any(frag in lower for frag in _ELAPSED_TIME_NAME_FRAGMENTS)


def _elapsed_format_from_name(col_name: str) -> str:
    """Infer elapsed-time unit from a column name; default to seconds."""
    lower = col_name.lower().strip()
    if lower in {"ms", "msec"} or lower.endswith(("_ms", " ms", "-ms")) or any(
        frag in lower for frag in ("millisecond", "milliseconds", "msec")
    ):
        return _ELAPSED_MS_LABEL
    if lower in {"min", "mins"} or any(
        frag in lower for frag in ("minute", "minutes")
    ):
        return _ELAPSED_MINUTES_LABEL
    return _ELAPSED_SECONDS_LABEL


def _detect_elapsed_format(
    sample_values: list[str],
    col_name: str,
) -> tuple[str | None, int]:
    """Detect numeric relative elapsed-time columns.

    Elapsed-time detection is intentionally name-gated because ordinary analog
    channels may also be numeric and monotonic.  Values may start before zero
    for pre-trigger records.
    """
    if not _name_suggests_elapsed_time(col_name):
        return None, len(sample_values)

    non_empty = [v for v in sample_values if str(v).strip()]
    if len(non_empty) < 2:
        return None, len(sample_values)

    numeric: list[float] = []
    invalid = 0
    for val in non_empty:
        try:
            numeric.append(float(str(val).strip()))
        except ValueError:
            invalid += 1

    if len(numeric) < 2:
        return None, len(sample_values)

    parse_rate = len(numeric) / len(non_empty)
    if parse_rate < 0.8:
        return None, len(sample_values) - len(numeric)

    monotonic = all(b > a for a, b in zip(numeric, numeric[1:]))
    if not monotonic:
        return None, len(sample_values) - len(numeric)

    # Exclude absolute numeric date encodings handled elsewhere.
    abs_values = [abs(v) for v in numeric]
    if any(_EPOCH_SECONDS_MIN <= v <= _EPOCH_SECONDS_MAX for v in abs_values):
        return None, len(sample_values) - len(numeric)
    if any(_EPOCH_MS_MIN <= v <= _EPOCH_MS_MAX for v in abs_values):
        return None, len(sample_values) - len(numeric)
    if all(_EXCEL_SERIAL_MIN <= v <= _EXCEL_SERIAL_MAX for v in numeric):
        return None, len(sample_values) - len(numeric)

    return _elapsed_format_from_name(col_name), len(sample_values) - len(numeric)


def _extract_timezone(values: list[str]) -> str | None:
    """Heuristically extract an IANA-like or offset timezone string from sample values."""
    tz_pattern = re.compile(
        r"([+-]\d{2}:?\d{2}|Z|UTC|GMT|[A-Z]{3,5})$"
    )
    for val in values:
        m = tz_pattern.search(val.strip())
        if m:
            return m.group(1)
    return None


def _is_monotonic_increasing(values: list[Any]) -> bool:
    """Return True when a list of comparable values is strictly monotonic increasing."""
    clean = [v for v in values if v is not None and v != ""]
    if len(clean) < 2:
        return False
    for a, b in zip(clean, clean[1:]):
        try:
            if b <= a:
                return False
        except TypeError:
            return False
    return True


def detect_timestamp_candidates(
    column_names: list[str],
    preview_rows: list[list[Any]],
    *,
    max_sample: int = 20,
) -> list[TimestampCandidate]:
    """Analyse column data and return ranked TimestampCandidate objects.

    Confidence formula
    ------------------
    base  = parse_rate (fraction of non-empty samples that matched a format)
    +0.15 if column name contains a timestamp fragment
    +0.10 if parsed values are strictly monotonic increasing (as strings/numbers)
    -0.10 if another column has the same format (duplicate penalty)

    Results are sorted descending by confidence.  Columns with base parse_rate < 0.4
    are excluded unless they have a high name score.
    """
    if not column_names or not preview_rows:
        return []

    num_cols = len(column_names)
    # Collect per-column sample values (up to max_sample rows)
    col_samples: list[list[str]] = [[] for _ in range(num_cols)]
    col_raw: list[list[Any]] = [[] for _ in range(num_cols)]

    for row in preview_rows[:max_sample]:
        for ci in range(num_cols):
            val = row[ci] if ci < len(row) else ""
            col_raw[ci].append(val)
            col_samples[ci].append(str(val) if val is not None else "")

    candidates: list[TimestampCandidate] = []
    format_counts: dict[str, int] = {}

    # First pass — determine format and base parse rate
    intermediate: list[tuple[int, str, str | None, int, float]] = []
    for ci, col_name in enumerate(column_names):
        samples = col_samples[ci]
        non_empty = [v for v in samples if v.strip()]
        if not non_empty:
            continue

        fmt, invalid_count = infer_timestamp_format(samples)
        if fmt is None:
            fmt, invalid_count = _detect_elapsed_format(samples, col_name)
        if fmt is None and not _name_suggests_timestamp(col_name):
            continue

        parse_rate = (len(non_empty) - invalid_count) / len(non_empty) if non_empty else 0.0

        if parse_rate < 0.4 and not _name_suggests_timestamp(col_name):
            continue

        if fmt is not None:
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        intermediate.append((ci, col_name, fmt, invalid_count, parse_rate))

    # Second pass — apply confidence adjustments
    for ci, col_name, fmt, invalid_count, parse_rate in intermediate:
        confidence = parse_rate

        if _name_suggests_timestamp(col_name):
            confidence += 0.15

        # Monotonic check on raw values
        if _is_monotonic_increasing(col_raw[ci]):
            confidence += 0.10

        # Duplicate format penalty
        if fmt is not None and format_counts.get(fmt, 0) > 1:
            confidence -= 0.10

        confidence = max(0.0, min(1.0, confidence))

        example_vals = [v for v in col_samples[ci] if v.strip()][:5]
        tz = _extract_timezone(example_vals)
        duplicate_count = _count_duplicate_parsed_values(col_samples[ci], fmt)

        candidates.append(
            TimestampCandidate(
                column_name=col_name,
                column_index=ci,
                confidence=confidence,
                detected_format=fmt,
                example_values=example_vals,
                invalid_sample_count=invalid_count,
                duplicate_sample_count=duplicate_count,
                timezone_detected=tz,
            )
        )

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates
