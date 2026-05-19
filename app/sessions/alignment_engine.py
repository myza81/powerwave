"""Alignment engine — pure stateless functions for Phase 9A.

Design constraints (from PHASE_9_SPEC.md):
- Never mutate input arrays.
- Do NOT assume uniformly spaced time arrays (PMU readiness).
- Use actual time values, not sample index arithmetic.
- Use median sampling interval for grid density — tolerates dropout gaps.
- NaN outside interpolation coverage; do not extrapolate.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import interp1d

from app.sessions.session_models import AlignmentResult, SourceQualityMetrics


# ---------------------------------------------------------------------------
# Core transform primitives
# ---------------------------------------------------------------------------


def apply_time_offset(time_array: np.ndarray, offset_s: float) -> np.ndarray:
    """Return a new time array shifted by offset_s seconds.

    Positive offset shifts the waveform to the right (later in time).
    Input array is never mutated.
    """
    return time_array + offset_s


def resample_to_grid(
    time_src: np.ndarray,
    values_src: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate (time_src, values_src) onto time_grid.

    - time_src is NOT assumed uniformly spaced (PMU / jitter safe).
    - Grid points outside [min(time_src), max(time_src)] are filled with NaN.
      No extrapolation is performed.
    - Returns float64 array of the same length as time_grid.
    """
    if len(time_src) < 2:
        return np.full(len(time_grid), np.nan, dtype=np.float64)

    src_min = float(np.min(time_src))
    src_max = float(np.max(time_src))

    # Build interpolant over the covered range only
    f = interp1d(
        time_src.astype(np.float64),
        values_src.astype(np.float64),
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    result = f(time_grid.astype(np.float64))

    # Explicitly NaN anything outside the source coverage (belt-and-suspenders)
    out_of_range = (time_grid < src_min) | (time_grid > src_max)
    result[out_of_range] = np.nan
    return result.astype(np.float64)


def build_common_time_grid(
    time_arrays: list[np.ndarray],
    t_start: float,
    t_end: float,
    max_points: int,
) -> np.ndarray:
    """Build a uniform output grid that covers [t_start, t_end].

    Resolution is the median inter-sample interval of the finest-rate
    source (not mean — tolerates dropout-induced gaps in PMU streams).
    Capped at max_points evenly spaced points.

    Returns a 1-D float64 array.
    """
    if not time_arrays or t_end <= t_start:
        return np.linspace(t_start, t_end, min(max_points, 2), dtype=np.float64)

    finest_dt = float("inf")
    for t in time_arrays:
        if len(t) < 2:
            continue
        diffs = np.diff(t.astype(np.float64))
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs) == 0:
            continue
        median_dt = float(np.median(positive_diffs))
        if median_dt < finest_dt:
            finest_dt = median_dt

    if finest_dt == float("inf") or finest_dt <= 0:
        return np.linspace(t_start, t_end, min(max_points, 2), dtype=np.float64)

    span = t_end - t_start
    natural_points = math.ceil(span / finest_dt) + 1
    n_points = min(natural_points, max_points)
    n_points = max(n_points, 2)
    return np.linspace(t_start, t_end, n_points, dtype=np.float64)


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------


def detect_trigger_time(
    time_array: np.ndarray,
    values_array: np.ndarray,
    threshold_factor: float = 3.0,
) -> float | None:
    """Return the time of the first significant event in the signal.

    A significant event is the first sample where
    |value| > threshold_factor × RMS(first 20% of samples).

    - time_array may be non-uniform; actual time values are used.
    - Returns None if no trigger is detected or the signal is too short.
    """
    if len(time_array) < 10 or len(values_array) < 10:
        return None

    values = values_array.astype(np.float64)
    baseline_end = max(2, len(values) // 5)
    baseline = values[:baseline_end]

    rms_baseline = float(np.sqrt(np.nanmean(baseline**2)))
    if rms_baseline < 1e-12:
        # Baseline is near-zero (quiet pre-event window); fall back to global RMS
        rms_global = float(np.sqrt(np.nanmean(values**2)))
        if rms_global < 1e-12:
            return None
        rms_baseline = rms_global

    threshold = threshold_factor * rms_baseline
    exceeding = np.where(np.abs(values) > threshold)[0]
    if len(exceeding) == 0:
        return None

    return float(time_array[exceeding[0]])


# ---------------------------------------------------------------------------
# Auto-alignment
# ---------------------------------------------------------------------------


def _pick_best_channel_for_trigger(
    record,  # DisturbanceRecord
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (time, values) for the analog channel with the largest RMS.

    Chooses the most prominent channel as the trigger candidate.
    Returns None if no usable analog channels exist.
    """
    time = record.waveform_data.get("time")
    if time is None or len(time) < 10:
        return None

    best_rms = -1.0
    best_values = None
    for ch in record.analog_channels:
        values = record.waveform_data.get(ch.name)
        if values is None:
            continue
        rms = float(np.sqrt(np.nanmean(values.astype(np.float64) ** 2)))
        if rms > best_rms:
            best_rms = rms
            best_values = values

    if best_values is None:
        return None
    return time, best_values


def _trigger_confidence(
    values_array: np.ndarray,
    trigger_idx: int,
    threshold_factor: float,
) -> float:
    """Normalised confidence score for a detected trigger.

    Score = clamp(peak_ratio / (threshold_factor * 2), 0.0, 1.0)
    where peak_ratio = |value at trigger| / RMS(baseline).
    """
    baseline_end = max(2, len(values_array) // 5)
    baseline = values_array[:baseline_end].astype(np.float64)
    rms_baseline = float(np.sqrt(np.nanmean(baseline**2)))
    if rms_baseline < 1e-12:
        return 0.0
    peak = float(abs(values_array[trigger_idx]))
    ratio = peak / rms_baseline
    return float(np.clip(ratio / (threshold_factor * 4.0), 0.0, 1.0))


def suggest_alignment_offsets(
    sources: list,  # list[SessionSource]
    threshold_factor: float = 3.0,
) -> list[AlignmentResult]:
    """Suggest per-source time offsets that align all triggers to t=0.

    For each active source:
    - Runs detect_trigger_time on the highest-RMS analog channel.
    - Computes suggested_offset_s = -trigger_time (so the trigger lands at t=0).
    - Computes alignment_confidence from peak-to-baseline ratio.

    Sources with no detectable trigger get offset=0.0 and confidence=0.0.
    Returns one AlignmentResult per source (active and inactive alike).
    """
    results: list[AlignmentResult] = []

    for source in sources:
        record = source.record
        pair = _pick_best_channel_for_trigger(record)

        if pair is None:
            results.append(AlignmentResult(
                source_id=source.source_id,
                suggested_offset_s=0.0,
                alignment_method="auto_trigger",
                alignment_confidence=0.0,
                reference_time=None,
                notes="No usable analog channels found — offset set to 0.0",
            ))
            continue

        time_arr, values_arr = pair
        trigger_time = detect_trigger_time(time_arr, values_arr, threshold_factor)

        if trigger_time is None:
            results.append(AlignmentResult(
                source_id=source.source_id,
                suggested_offset_s=0.0,
                alignment_method="auto_trigger",
                alignment_confidence=0.0,
                reference_time=None,
                notes="No trigger event detected — offset set to 0.0",
            ))
            continue

        # Find the index of the trigger sample for confidence calculation
        trigger_idx = int(np.searchsorted(time_arr, trigger_time))
        trigger_idx = min(trigger_idx, len(values_arr) - 1)
        confidence = _trigger_confidence(values_arr, trigger_idx, threshold_factor)

        results.append(AlignmentResult(
            source_id=source.source_id,
            suggested_offset_s=-trigger_time,
            alignment_method="auto_trigger",
            alignment_confidence=confidence,
            reference_time=trigger_time,
            notes=(
                f"Trigger detected at t={trigger_time:.4f} s "
                f"(confidence {confidence:.2f}). "
                f"Suggested offset: {-trigger_time:+.4f} s."
            ),
        ))

    return results


# ---------------------------------------------------------------------------
# Uniformity assessment
# ---------------------------------------------------------------------------


def assess_time_uniformity(time_array: np.ndarray) -> tuple[bool, float]:
    """Assess whether a time array is uniformly spaced.

    Returns (is_uniform, stability_score) where:
    - is_uniform: True if CoV(diff) < 0.01 (< 1% relative jitter)
    - stability_score: 1.0 − CoV, clipped to [0.0, 1.0]
                       1.0 = perfectly uniform; 0.0 = highly jittered
    """
    if len(time_array) < 3:
        return True, 1.0

    diffs = np.diff(time_array.astype(np.float64))
    positive_diffs = diffs[diffs > 0]

    if len(positive_diffs) == 0:
        return False, 0.0

    mean_dt = float(np.mean(positive_diffs))
    if mean_dt < 1e-15:
        return False, 0.0

    std_dt = float(np.std(positive_diffs))
    cov = std_dt / mean_dt

    is_uniform = cov < 0.01
    stability_score = float(np.clip(1.0 - cov, 0.0, 1.0))
    return is_uniform, stability_score


# ---------------------------------------------------------------------------
# Source quality computation
# ---------------------------------------------------------------------------


def compute_source_quality(
    source_id: str,
    time_array: np.ndarray,
    values_array: np.ndarray,
) -> SourceQualityMetrics:
    """Compute SourceQualityMetrics from raw time and values arrays.

    values_array should be a representative analog channel (e.g. first analog).
    All percentages are in [0.0, 100.0].
    """
    n = len(time_array)

    if n < 2:
        return SourceQualityMetrics(
            source_id=source_id,
            sample_count=n,
            inferred_sample_rate_hz=0.0,
            sample_rate_stability=0.0,
            missing_data_pct=100.0,
            duplicate_timestamp_pct=0.0,
            interpolated_pct=0.0,
            resampling_ratio=1.0,
            time_is_uniform=False,
        )

    time_f64 = time_array.astype(np.float64)
    diffs = np.diff(time_f64)
    positive_diffs = diffs[diffs > 0]

    if len(positive_diffs) > 0:
        median_dt = float(np.median(positive_diffs))
        inferred_rate = 1.0 / median_dt if median_dt > 0 else 0.0
    else:
        median_dt = 0.0
        inferred_rate = 0.0

    is_uniform, stability = assess_time_uniformity(time_array)

    # Duplicate timestamps: diffs == 0
    zero_diffs = int(np.sum(diffs == 0))
    duplicate_pct = 100.0 * zero_diffs / n if n > 0 else 0.0

    # Missing data: NaN values in the values array
    nan_count = int(np.sum(np.isnan(values_array.astype(np.float64))))
    missing_pct = 100.0 * nan_count / n if n > 0 else 0.0

    return SourceQualityMetrics(
        source_id=source_id,
        sample_count=n,
        inferred_sample_rate_hz=inferred_rate,
        sample_rate_stability=stability,
        missing_data_pct=missing_pct,
        duplicate_timestamp_pct=duplicate_pct,
        interpolated_pct=0.0,   # populated by build_aligned_data after resampling
        resampling_ratio=1.0,   # populated by build_aligned_data after resampling
        time_is_uniform=is_uniform,
    )
