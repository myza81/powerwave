"""Interval inference and regularity analysis for timestamp series.

All functions accept pandas Series of datetime64 values and return plain
Python / dataclass results — no Qt, no plotting, no side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class IntervalAnalysis:
    """Results of analysing the temporal structure of a timestamp series.

    Fields
    ------
    dominant_interval_seconds  Mode (most-common) interval in seconds.
    median_interval_seconds    Median inter-sample interval.
    mean_interval_seconds      Mean inter-sample interval.
    jitter_seconds             Median absolute deviation of intervals (MAD).
    jitter_fraction            jitter_seconds / dominant_interval_seconds, or None.
    is_irregular               True when jitter_fraction > 0.05.
    missing_count              Estimated number of missing samples (gaps > 1.5× dominant).
    duplicate_count            Number of adjacent equal-timestamp pairs.
    non_monotonic_count        Number of intervals ≤ 0 (backward or flat steps).
    is_monotonic               True when no non-monotonic steps exist.
    total_samples              Length of the input series (including NaTs).
    valid_samples              Number of non-NaT timestamps.
    nat_count                  Number of NaT / null timestamps.
    """

    dominant_interval_seconds: float | None = None
    median_interval_seconds: float | None = None
    mean_interval_seconds: float | None = None
    jitter_seconds: float | None = None
    jitter_fraction: float | None = None
    is_irregular: bool = False
    missing_count: int = 0
    duplicate_count: int = 0
    non_monotonic_count: int = 0
    is_monotonic: bool = True
    total_samples: int = 0
    valid_samples: int = 0
    nat_count: int = 0


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    """Coerce a Series to datetime64; non-parseable values become NaT."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", utc=False)


def infer_interval(timestamps: pd.Series) -> IntervalAnalysis:
    """Analyse the interval structure of a timestamp Series.

    Tolerates NaT values, non-monotonic data, and duplicates.  The dominant
    interval is estimated as the mode of rounded intervals (1 ms bins).

    Parameters
    ----------
    timestamps:  pd.Series of datetime64 or anything coercible to it.
    """
    result = IntervalAnalysis(total_samples=len(timestamps))

    dt = _to_datetime_safe(timestamps)
    nat_mask = dt.isna()
    result.nat_count = int(nat_mask.sum())
    result.valid_samples = len(dt) - result.nat_count

    valid_ordered = dt.dropna()  # preserves original row order
    if len(valid_ordered) < 2:
        return result

    # Order-sensitive diffs — non-monotonic and duplicate detection.
    # Use .diff().dt.total_seconds() to avoid datetime64 unit issues
    # (pandas 3 defaults to datetime64[us]; raw int64 would be µs not ns).
    ordered_s_full = valid_ordered.diff().dt.total_seconds().iloc[1:].values  # type: ignore[union-attr]
    # Strictly backward steps only (< 0); duplicates counted separately via sorted diffs
    non_mono = int(np.sum(ordered_s_full < 0))
    result.non_monotonic_count = non_mono

    # Sort for interval statistics AND duplicate detection.
    # After sorting, equal timestamps become adjacent → diff == 0.
    valid = valid_ordered.sort_values()
    diffs_s = valid.diff().dt.total_seconds().iloc[1:].values  # type: ignore[union-attr]
    result.duplicate_count = int(np.sum(diffs_s == 0))

    # Monotonic only when no backward steps and no equal-timestamp pairs
    result.is_monotonic = (non_mono == 0 and result.duplicate_count == 0)

    # Only use positive diffs for interval statistics
    pos_diffs = diffs_s[diffs_s > 0]
    if len(pos_diffs) == 0:
        return result

    result.median_interval_seconds = float(np.median(pos_diffs))
    result.mean_interval_seconds = float(np.mean(pos_diffs))

    # Dominant interval: mode of intervals rounded to nearest microsecond
    # (µs resolution supports sub-ms intervals such as 50 µs PMU data)
    rounded_us = np.round(pos_diffs * 1e6).astype(np.int64)
    if len(rounded_us) > 0:
        values, counts = np.unique(rounded_us, return_counts=True)
        dominant_us = float(values[np.argmax(counts)])
        result.dominant_interval_seconds = dominant_us / 1e6

    # Jitter: MAD of positive intervals
    median = result.median_interval_seconds
    result.jitter_seconds = float(np.median(np.abs(pos_diffs - median)))

    if result.dominant_interval_seconds and result.dominant_interval_seconds > 0:
        result.jitter_fraction = result.jitter_seconds / result.dominant_interval_seconds

    result.is_irregular = (
        result.jitter_fraction is not None and result.jitter_fraction > 0.05
    )

    # Estimate missing samples: gaps significantly larger than dominant interval
    if result.dominant_interval_seconds and result.dominant_interval_seconds > 0:
        threshold = result.dominant_interval_seconds * 1.5
        large_gaps = pos_diffs[pos_diffs > threshold]
        missing = sum(
            int(round(g / result.dominant_interval_seconds)) - 1 for g in large_gaps
        )
        result.missing_count = max(0, missing)

    return result


def detect_duplicates(timestamps: pd.Series) -> pd.Series:
    """Return a boolean mask that is True for duplicate timestamp positions.

    The first occurrence of a duplicated value is NOT marked; only subsequent
    occurrences are flagged (consistent with pandas `duplicated` behaviour).
    """
    dt = _to_datetime_safe(timestamps)
    return dt.duplicated(keep="first")


def detect_non_monotonic(timestamps: pd.Series) -> pd.Series:
    """Return a boolean mask that is True for rows where the timestamp
    does not strictly increase compared to the previous valid timestamp.

    NaT rows are never marked as non-monotonic.
    """
    dt = _to_datetime_safe(timestamps)
    mask = pd.Series(False, index=dt.index)

    valid_idx = dt.dropna().index
    if len(valid_idx) < 2:
        return mask

    valid_vals = dt.loc[valid_idx]
    shifted = valid_vals.shift(1)
    non_mono = valid_idx[1:][valid_vals.iloc[1:].values <= shifted.iloc[1:].values]
    mask.loc[non_mono] = True
    return mask
