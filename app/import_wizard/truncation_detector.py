"""Detect low-resolution (truncated) timestamp columns and infer sample rate.

When a data logger captures samples faster than its timestamp precision, many
consecutive rows share the same truncated timestamp value.  This module detects
that pattern and estimates the true sample rate so the wizard can offer
sub-interval reconstruction.

Algorithm
---------
1. Parse the example values into datetime objects.
2. Walk the parsed series and identify consecutive runs of identical values.
3. The *anchor resolution* is the smallest positive step between distinct
   anchor timestamps (e.g. 0.1 s for a column stored to one decimal second).
4. The *dominant group size* is the mode of the run lengths.
5. Inferred sample rate = dominant_group_size / anchor_resolution_seconds.

Only the example_values slice (≤ 50 rows) is used — the full file is never
loaded here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TruncationAnalysis:
    """Results of low-resolution timestamp detection.

    Fields
    ------
    is_truncated            True when consecutive identical timestamps detected.
    anchor_resolution_s     Smallest positive gap between distinct anchor values (s).
    inferred_sample_rate_hz Estimated true sample rate (dominant_group_size /
                            anchor_resolution_s).  None when detection failed.
    inferred_interval_s     1 / inferred_sample_rate_hz.  None when rate unknown.
    dominant_group_size     Most common run length (samples per anchor window).
    group_sizes             All run lengths observed in the example slice.
    group_count             Number of distinct anchor groups detected.
    sample_count            Total example values analysed.
    notes                   Human-readable summary suitable for display.
    """

    is_truncated: bool = False
    anchor_resolution_s: float | None = None
    inferred_sample_rate_hz: float | None = None
    inferred_interval_s: float | None = None
    dominant_group_size: int = 1
    group_sizes: list[int] = field(default_factory=list)
    group_count: int = 0
    sample_count: int = 0
    notes: str = ""


def detect_timestamp_truncation(
    example_values: list[str],
    fmt: str | None = None,
) -> TruncationAnalysis:
    """Analyse a small sample of raw timestamp strings for low-resolution truncation.

    Parameters
    ----------
    example_values:  Raw string values from the timestamp column (≤ 50 rows).
    fmt:             strptime format string for parsing, or None for auto.

    Returns a TruncationAnalysis with detection results.
    """
    result = TruncationAnalysis(sample_count=len(example_values))
    if len(example_values) < 3:
        result.notes = "Too few samples for truncation analysis."
        return result

    series = pd.Series(example_values, dtype=str)
    if fmt:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
    else:
        parsed = pd.to_datetime(series, format="mixed", errors="coerce")

    valid = parsed.dropna()
    if len(valid) < 3:
        result.notes = "Could not parse enough values for truncation analysis."
        return result

    # Walk the series and collect consecutive run lengths
    run_lengths: list[int] = []
    anchors: list[pd.Timestamp] = []
    current_val = valid.iloc[0]
    current_run = 1
    for val in valid.iloc[1:]:
        if val == current_val:
            current_run += 1
        else:
            run_lengths.append(current_run)
            anchors.append(current_val)
            current_val = val
            current_run = 1
    run_lengths.append(current_run)
    anchors.append(current_val)

    result.group_sizes = run_lengths
    result.group_count = len(run_lengths)

    max_run = max(run_lengths)
    if max_run < 2:
        result.notes = "No consecutive identical timestamps detected — timestamps appear full-resolution."
        return result

    result.is_truncated = True

    # Dominant group size = mode of run lengths
    from collections import Counter
    counts = Counter(run_lengths)
    result.dominant_group_size = counts.most_common(1)[0][0]

    # Anchor resolution: smallest positive gap between distinct anchor datetimes
    if len(anchors) >= 2:
        anchor_series = pd.Series(anchors)
        diffs_s = anchor_series.diff().dt.total_seconds().dropna()
        positive_diffs = diffs_s[diffs_s > 0]
        if len(positive_diffs) > 0:
            result.anchor_resolution_s = float(positive_diffs.min())

    if result.anchor_resolution_s and result.anchor_resolution_s > 0:
        hz = result.dominant_group_size / result.anchor_resolution_s
        result.inferred_sample_rate_hz = round(hz, 6)
        result.inferred_interval_s = 1.0 / hz

    # Build a human-readable summary
    if result.inferred_sample_rate_hz is not None:
        res_ms = (result.anchor_resolution_s or 0) * 1000
        dt_ms = (result.inferred_interval_s or 0) * 1000
        result.notes = (
            f"Low-resolution timestamps detected — anchor precision ≈ {res_ms:.0f} ms, "
            f"inferred sample rate ≈ {result.inferred_sample_rate_hz:.1f} Hz "
            f"({dt_ms:.2f} ms per sample)."
        )
    else:
        result.notes = (
            f"Consecutive identical timestamps detected (max run = {max_run}) "
            "but sample rate could not be inferred — specify it manually."
        )

    return result
