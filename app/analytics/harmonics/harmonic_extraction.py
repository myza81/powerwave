"""Vectorized harmonic extraction engine — Phase 7.

Approach: sliding-window FFT using NumPy stride tricks.

All windows are computed at once using np.lib.stride_tricks.as_strided
to build a (n_windows, window_samples) view of the input, then a single
np.fft.rfft call processes all windows simultaneously. Per-harmonic RMS
magnitudes are read from the exact frequency bins corresponding to each
harmonic order.

Magnitude normalisation:
  For a window function w with sum S = sum(w):
    peak_amplitude  = 2 * |FFT[bin]| / S
    rms_amplitude   = peak_amplitude / sqrt(2)  =  sqrt(2) * |FFT[bin]| / S

  This is exact when the target frequency falls on a bin centre (which is
  guaranteed when window_samples = n_cycles * round(fs / f0)).  The Hann
  window (default) suppresses leakage from off-bin components.

Time axis convention:
  Right-aligned — harmonic_time[i] = time[window_start_i + window_samples - 1].
  Consistent with the phasor extract_phasor() convention in phasor_extraction.py.
"""
from __future__ import annotations

import numpy as np

from app.analytics.harmonics.harmonic_models import (
    HarmonicConfig,
    HarmonicResult,
    HarmonicWindowMode,
)

# Multiplier for window mode → number of cycles
_WINDOW_MODE_CYCLES: dict[HarmonicWindowMode, int] = {
    HarmonicWindowMode.ONE_CYCLE: 1,
    HarmonicWindowMode.TWO_CYCLE: 2,
    HarmonicWindowMode.FOUR_CYCLE: 4,
}

_SQRT2 = float(np.sqrt(2.0))


# ─────────────────────────────────────────────────────────────────────────────
# Window-size helper
# ─────────────────────────────────────────────────────────────────────────────


def compute_harmonic_window_samples(
    sample_rate_hz: float,
    config: HarmonicConfig | None = None,
) -> int:
    """Return the FFT window size in samples for a given config and sample rate.

    The window is chosen to contain an exact integer number of cycles at the
    nominal frequency, so harmonic components fall precisely on FFT bin centres.

    Args:
        sample_rate_hz: Waveform sampling rate in Hz.
        config:         Harmonic configuration; defaults to HarmonicConfig().

    Returns:
        Window size in samples (>= 1).
    """
    cfg = config or HarmonicConfig()
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        return 1
    nominal_hz = float(cfg.nominal_hz)
    if not np.isfinite(nominal_hz) or nominal_hz <= 0.0:
        return 1
    n_cycles = _WINDOW_MODE_CYCLES.get(cfg.window_mode, 2)
    samples_per_cycle = sample_rate_hz / nominal_hz
    return max(1, int(round(n_cycles * samples_per_cycle)))


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────────────────


def extract_harmonics(
    signal: np.ndarray,
    sample_rate_hz: float,
    config: HarmonicConfig | None = None,
    *,
    time: np.ndarray | None = None,
) -> HarmonicResult:
    """Extract sliding-window per-harmonic RMS magnitudes from a waveform.

    Args:
        signal:         1-D (or flattenable) waveform array.
        sample_rate_hz: Sampling rate in Hz.
        config:         Harmonic configuration; defaults to HarmonicConfig().
        time:           Optional time axis matching ``signal``. If None,
                        a uniform axis starting at 0 is constructed.

    Returns:
        HarmonicResult with per-harmonic RMS magnitude arrays and a
        right-aligned time axis.  Returns HarmonicResult.empty() when the
        signal is shorter than one window.
    """
    cfg = config or HarmonicConfig()

    # ── Sanitise input ────────────────────────────────────────────────────
    sig = np.asarray(signal, dtype=np.float64).ravel()
    N = len(sig)

    bad_mask = ~np.isfinite(sig)
    if bad_mask.any():
        sig = sig.copy()
        sig[bad_mask] = 0.0

    # ── Build time axis ───────────────────────────────────────────────────
    if time is not None:
        t = np.asarray(time, dtype=np.float64).ravel()
        if len(t) != N:
            t = np.arange(N, dtype=np.float64) / max(sample_rate_hz, 1.0)
    else:
        t = np.arange(N, dtype=np.float64) / max(sample_rate_hz, 1.0)

    # ── Window parameters ─────────────────────────────────────────────────
    window_samples = compute_harmonic_window_samples(sample_rate_hz, cfg)
    if N < window_samples:
        return HarmonicResult.empty()

    overlap_clamped = max(0.0, min(cfg.overlap, 0.999))
    hop_samples = max(1, int(round(window_samples * (1.0 - overlap_clamped))))

    # ── Spectral window function ──────────────────────────────────────────
    win_fn = cfg.window_function.strip().lower()
    if win_fn == "rectangular":
        win = np.ones(window_samples, dtype=np.float64)
    else:  # default: hann
        win = np.hanning(window_samples).astype(np.float64)
    win_sum = float(win.sum())
    if win_sum < 1e-12:
        win_sum = 1.0  # degenerate guard

    # ── Stride-trick sliding windows ─────────────────────────────────────
    n_windows = (N - window_samples) // hop_samples + 1

    strides = (sig.strides[0] * hop_samples, sig.strides[0])
    shape = (n_windows, window_samples)
    windows_2d = np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides)

    # ── Apply window and batch FFT ────────────────────────────────────────
    # (n_windows, window_samples) * (window_samples,) → broadcast
    # Remove per-window DC offset before applying the spectral window so slow
    # offsets do not leak into the fundamental/harmonic bins.
    windows_zero_mean = windows_2d - windows_2d.mean(axis=1, keepdims=True)
    windowed = windows_zero_mean * win
    # rfft returns (n_windows, window_samples//2 + 1) complex array
    spectra = np.fft.rfft(windowed, n=window_samples, axis=1)
    amplitudes = np.abs(spectra)  # (n_windows, n_bins)
    max_bin = amplitudes.shape[1] - 1

    # ── Right-aligned time axis ───────────────────────────────────────────
    window_end_indices = (
        np.arange(n_windows, dtype=np.intp) * hop_samples + window_samples - 1
    )
    harmonic_time = t[window_end_indices]

    # ── Extract per-harmonic magnitudes ───────────────────────────────────
    # Frequency resolution: df = sample_rate_hz / window_samples
    # Bin for harmonic order n: bin_n = round(n * nominal_hz / df)
    #                                 = round(n * nominal_hz * window_samples / sample_rate_hz)
    #
    # RMS normalisation (valid for all non-DC, non-Nyquist bins):
    #   rms = sqrt(2) * |FFT[bin]| / win_sum
    sr = float(sample_rate_hz) if np.isfinite(sample_rate_hz) and sample_rate_hz > 0.0 else 1.0
    nominal_hz = float(cfg.nominal_hz)
    if not np.isfinite(nominal_hz) or nominal_hz <= 0.0:
        nominal_hz = 50.0
    max_order = max(0, int(cfg.max_order))
    magnitudes: dict[int, np.ndarray] = {}

    for order in range(1, max_order + 1):
        target_freq = order * nominal_hz
        if target_freq > sr * 0.5:
            magnitudes[order] = np.zeros(n_windows, dtype=np.float64)
            continue
        bin_idx = int(round(target_freq * window_samples / sr))
        if bin_idx < 1 or bin_idx > max_bin:
            magnitudes[order] = np.zeros(n_windows, dtype=np.float64)
            continue

        mag_rms = amplitudes[:, bin_idx] * (_SQRT2 / win_sum)
        magnitudes[order] = mag_rms.copy()

    return HarmonicResult(
        harmonic_time=harmonic_time,
        magnitudes=magnitudes,
        sample_rate_hz=sample_rate_hz,
        nominal_hz=nominal_hz,
        window_samples=window_samples,
        hop_samples=hop_samples,
    )
