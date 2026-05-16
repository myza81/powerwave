"""Vectorized sliding RMS with engineering-parameter helpers.

Wraps the O(N) cumulative-sum primitive from basic_conversions and adds:
  - NaN-safe pre-processing  (NaN → 0 before squaring)
  - right-aligned time array computation (causal convention)
  - window-size helper from physical engineering parameters
"""
from __future__ import annotations

import numpy as np

from app.analytics.basic_conversions import sliding_rms as _primitive_sliding_rms
from app.analytics.rms.rms_models import RMSConfig, RMSWindowMode


def compute_window_samples(
    sample_rate_hz: float,
    nominal_frequency_hz: float = 50.0,
    cycles_per_window: int = 1,
) -> int:
    """Return the integer number of samples per RMS window.

    Args:
        sample_rate_hz:       Waveform sampling rate in Hz.
        nominal_frequency_hz: Power system nominal frequency (50 or 60 Hz).
        cycles_per_window:    Number of full cycles to include in each window.

    Returns:
        Integer sample count, always >= 1.

    Examples:
        5 kHz COMTRADE at 50 Hz → 100 samples/cycle
        3200 Hz COMTRADE at 50 Hz → 64 samples/cycle
        6400 Hz at 60 Hz → 107 samples/cycle
    """
    if sample_rate_hz <= 0 or nominal_frequency_hz <= 0 or cycles_per_window < 1:
        return 1
    window = round(sample_rate_hz / nominal_frequency_hz * cycles_per_window)
    return max(1, window)


def compute_rms_window_samples(
    sample_rate_hz: float,
    config: RMSConfig | None = None,
) -> int:
    """Return RMS window samples from an explicit engineering RMS config."""
    cfg = config or RMSConfig()
    if cfg.window_mode == RMSWindowMode.CUSTOM_SAMPLES:
        return max(1, int(cfg.custom_window_samples or 1))
    if cfg.window_mode == RMSWindowMode.HALF_CYCLE:
        cycles = 0.5
    elif cfg.window_mode == RMSWindowMode.TWO_CYCLE:
        cycles = 2.0
    elif cfg.window_mode == RMSWindowMode.ONE_CYCLE:
        cycles = 1.0
    else:
        cycles = float(cfg.cycles_per_window)
    return _compute_window_samples_float(
        sample_rate_hz,
        cfg.nominal_frequency_hz,
        cycles,
    )


def compute_rms_overlay(
    time: np.ndarray,
    values: np.ndarray,
    sample_rate_hz: float,
    nominal_frequency_hz: float = 50.0,
    cycles_per_window: int = 1,
    config: RMSConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a sliding RMS envelope and return aligned (rms_time, rms_values).

    Uses right-aligned (causal) windowing: output[i] = RMS of
    values[i .. i + window - 1].  The time value assigned to output[i] is
    time[i + window - 1] — the right edge of the window.

    This matches the convention used in digital relays where the RMS value
    becomes available at the end of each measurement window.

    NaN samples are replaced with 0 before squaring so that isolated bad
    samples contribute zero energy rather than propagating NaN across the
    entire output.

    Args:
        time:                  1-D float64 time array (seconds), length N.
        values:                1-D float64 waveform array, length N.
        sample_rate_hz:        Waveform sampling rate in Hz.
        nominal_frequency_hz:  Power system nominal frequency (50 or 60 Hz).
        cycles_per_window:     Full power cycles per RMS window (default 1).

    Returns:
        (rms_time, rms_values): Both float64 arrays, length = N - window + 1.

    Raises:
        ValueError: Forwarded from the underlying primitive when inputs are
                    degenerate (not 1-D, window > length, etc.).
    """
    time_arr = np.asarray(time, dtype=np.float64)
    val_arr = np.asarray(values, dtype=np.float64)

    if config is not None:
        window = compute_rms_window_samples(sample_rate_hz, config)
    else:
        window = compute_window_samples(
            sample_rate_hz,
            nominal_frequency_hz,
            cycles_per_window,
        )

    # Replace NaN/Inf with 0.0 so they contribute zero energy to the window sum
    bad_mask = ~np.isfinite(val_arr)
    if bad_mask.any():
        val_arr = val_arr.copy()
        val_arr[bad_mask] = 0.0

    rms_values = _primitive_sliding_rms(val_arr, window)

    # Right-aligned: the RMS value at output index i corresponds to time[i + window - 1]
    rms_time = time_arr[window - 1:]

    return rms_time, rms_values


def _compute_window_samples_float(
    sample_rate_hz: float,
    nominal_frequency_hz: float,
    cycles_per_window: float,
) -> int:
    if sample_rate_hz <= 0 or nominal_frequency_hz <= 0 or cycles_per_window <= 0:
        return 1
    window = round(sample_rate_hz / nominal_frequency_hz * cycles_per_window)
    return max(1, window)
