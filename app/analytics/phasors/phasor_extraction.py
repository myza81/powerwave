"""Vectorized sliding-window phasor extraction using DFT at the fundamental.

Engineering conventions:
  - Output magnitudes are RMS (peak / sqrt(2)).
  - Output angles are in degrees, range [-180, 180].
  - Right-aligned time convention: output[i] corresponds to time[i + window - 1].
    This matches the causal (relay) convention used in sliding_rms.py.
  - NaN/Inf samples are replaced with 0 before the DFT so that isolated bad
    samples contribute zero energy rather than propagating NaN.

Performance:
  - Uses numpy stride tricks for O(N · W) computation where W is the window
    width in samples. At 50 Hz / 5 kHz, W = 100, giving 10M ops for 100k-sample
    files — fast enough for interactive use.
  - No Python loops over individual samples.
"""
from __future__ import annotations

import numpy as np

from app.analytics.phasors.phasor_models import PhasorConfig, PhasorWindowMode


# ─────────────────────────────────────────────────────────────────────────────
# Window-size helpers
# ─────────────────────────────────────────────────────────────────────────────


def compute_phasor_window_samples(
    sample_rate_hz: float,
    config: PhasorConfig | None = None,
) -> int:
    """Return integer sample count for the phasor DFT window.

    Args:
        sample_rate_hz: Waveform sampling rate in Hz.
        config:         PhasorConfig; defaults constructed if None.

    Returns:
        Integer sample count >= 1.

    Examples:
        5000 Hz at 50 Hz, ONE_CYCLE → 100 samples
        3200 Hz at 50 Hz, ONE_CYCLE → 64 samples
        5000 Hz at 50 Hz, HALF_CYCLE → 50 samples
        5000 Hz at 60 Hz, ONE_CYCLE → 83 samples
    """
    cfg = config or PhasorConfig()
    if sample_rate_hz <= 0 or cfg.nominal_hz <= 0:
        return 1

    if cfg.window_mode == PhasorWindowMode.HALF_CYCLE:
        cycles = 0.5
    elif cfg.window_mode == PhasorWindowMode.TWO_CYCLE:
        cycles = 2.0
    else:
        cycles = 1.0

    window = round(sample_rate_hz / cfg.nominal_hz * cycles)
    return max(1, window)


# ─────────────────────────────────────────────────────────────────────────────
# Core DFT phasor extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_phasor(
    time: np.ndarray,
    values: np.ndarray,
    sample_rate_hz: float,
    config: PhasorConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sliding-window DFT phasor from a waveform.

    Uses a vectorized stride-trick approach: the input array is viewed as a
    (N - W + 1, W) matrix of overlapping windows, then dot-multiplied with
    the DFT kernel in one NumPy operation.

    Args:
        time:           1-D float64 time array (seconds), length N.
        values:         1-D float64 waveform array, length N.
        sample_rate_hz: Waveform sampling rate in Hz.
        config:         PhasorConfig; ONE_CYCLE at nominal_hz if None.

    Returns:
        (phasor_time, magnitude_rms, angle_deg, complex_phasor)
        All arrays have length N - window + 1.
        - phasor_time:   right-aligned time stamps (seconds).
        - magnitude_rms: RMS magnitude (same unit as input values).
        - angle_deg:     phase angle in degrees [-180, 180].
        - complex_phasor: complex128 array of peak phasors.

    Raises:
        ValueError: if inputs are not 1-D or window > signal length.
    """
    time_arr = np.asarray(time, dtype=np.float64)
    val_arr = np.asarray(values, dtype=np.float64)

    if time_arr.ndim != 1 or val_arr.ndim != 1:
        raise ValueError("time and values must be 1-D arrays")
    if len(time_arr) != len(val_arr):
        raise ValueError("time and values must have equal length")

    window = compute_phasor_window_samples(sample_rate_hz, config)
    n = len(val_arr)

    if n < window:
        raise ValueError(
            f"Signal length {n} is shorter than phasor window {window}"
        )

    # Replace NaN/Inf with 0 — isolated bad samples contribute zero energy
    bad_mask = ~np.isfinite(val_arr)
    if bad_mask.any():
        val_arr = val_arr.copy()
        val_arr[bad_mask] = 0.0

    # DFT kernel for the fundamental frequency component.
    # The exponent uses n_cycle (samples per one full cycle at nominal Hz),
    # NOT window.  This ensures the kernel always targets f₀ regardless of
    # whether window is a half-cycle, one-cycle or two-cycle window.
    #   half-cycle window : DFT at f₀ with W=N/2 samples → correct
    #   one-cycle window  : DFT at f₀ with W=N   samples → correct (same as before)
    #   two-cycle window  : DFT at f₀ with W=2N  samples → correct
    cfg_used = config or PhasorConfig()
    n_cycle = max(1, round(sample_rate_hz / cfg_used.nominal_hz))
    k = np.arange(window, dtype=np.float64)
    kernel = (2.0 / window) * np.exp(-1j * 2.0 * np.pi * k / n_cycle)

    # Stride trick: build (M, W) view without copying
    m = n - window + 1
    strides = (val_arr.strides[0], val_arr.strides[0])
    windows_view = np.lib.stride_tricks.as_strided(
        val_arr, shape=(m, window), strides=strides
    )

    # Matrix-vector multiply: each row (window) dotted with kernel
    complex_phasor = windows_view @ kernel  # complex128, length M

    # RMS magnitude = |peak phasor| / sqrt(2)
    magnitude_rms = np.abs(complex_phasor) / np.sqrt(2.0)

    # Phase angle in degrees
    angle_deg = np.degrees(np.angle(complex_phasor))

    # Right-aligned time: output[i] ↔ window ending at time[i + window - 1]
    phasor_time = time_arr[window - 1:]

    return phasor_time, magnitude_rms, angle_deg, complex_phasor


def extract_phasor_from_record(
    waveform_data,
    channel_name: str,
    sample_rate_hz: float,
    config: PhasorConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: extract phasor from a waveform DataFrame column.

    Args:
        waveform_data:  pandas DataFrame with 'time' column and channel columns.
        channel_name:   Name of the analog channel column to process.
        sample_rate_hz: Sampling rate in Hz.
        config:         PhasorConfig; defaults used if None.

    Returns:
        (phasor_time, magnitude_rms, angle_deg, complex_phasor)
    """
    time = waveform_data["time"].to_numpy(dtype=np.float64)
    values = waveform_data[channel_name].to_numpy(dtype=np.float64)
    return extract_phasor(time, values, sample_rate_hz, config)
