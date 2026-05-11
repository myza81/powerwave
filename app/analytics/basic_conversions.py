"""Basic waveform conversion utilities — vectorized NumPy implementations.

These are analytical building blocks, not full analytics:
  sliding_rms   — causal sliding window RMS
  to_per_unit   — per-unit normalisation

Full phasor and impedance calculations are deferred to Phase 5 analytics.
"""
from __future__ import annotations

import numpy as np


def sliding_rms(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Sliding window RMS using cumulative-sum (O(N), no Python loop).

    Returns an array of length ``len(values) - window_samples + 1``.
    Each output[i] is the RMS of values[i : i + window_samples].

    Args:
        values:         1-D numeric array.
        window_samples: Number of samples per RMS window (>= 1).

    Raises:
        ValueError: if values is not 1-D, window_samples < 1, or
                    window_samples > len(values).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {arr.shape}")
    if window_samples < 1:
        raise ValueError(f"window_samples must be >= 1, got {window_samples}")
    if window_samples > len(arr):
        raise ValueError(
            f"window_samples ({window_samples}) exceeds input length ({len(arr)})"
        )
    sq = arr ** 2
    cumsum = np.empty(len(sq) + 1, dtype=np.float64)
    cumsum[0] = 0.0
    np.cumsum(sq, out=cumsum[1:])
    window_sums = cumsum[window_samples:] - cumsum[:-window_samples]
    return np.sqrt(window_sums / window_samples)


def to_per_unit(values: np.ndarray, base_value: float) -> np.ndarray:
    """Convert values to per-unit by dividing by base_value.

    Args:
        values:     Numeric array of any shape.
        base_value: Non-zero base (e.g. nominal voltage in kV).

    Raises:
        ValueError: if base_value is exactly 0.0.
    """
    if base_value == 0.0:
        raise ValueError("base_value must be non-zero for per-unit conversion")
    return np.asarray(values, dtype=np.float64) / base_value
