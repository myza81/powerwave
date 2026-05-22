"""Cross-source waveform correlation — pure numpy/scipy, no Qt.

Determines whether two sources captured the same event by computing
the normalized FFT cross-correlation of their waveform signatures and
finding the peak lag within a configurable search window.

Usage:
    result = correlate_source_pair(
        "source_A", time_a, data_a,
        "source_B", time_b, data_b,
    )
    if result.same_event_likely:
        print(f"Align B by {result.suggested_offset_s:.4f} s")

Algorithm:
    1. Resample both signals to a common uniform grid.
    2. Zero-mean normalize each signal.
    3. Compute the FFT cross-correlation (O(N log N)).
    4. Search for the peak lag within ±max_lag_s.
    5. Peak prominence (peak / RMS of tails) yields the confidence score.

Returns None (gracefully skipped) when signals do not overlap, have
insufficient common duration, or contain only NaN/flat data.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

try:
    from scipy.signal import correlate as scipy_correlate, correlation_lags
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Result model
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class CrossCorrelationResult:
    """Pairwise cross-correlation result for two waveform sources."""

    source_a_id: str
    source_b_id: str
    channel_a: str            # channel name used from source A
    channel_b: str            # channel name used from source B
    correlation_coeff: float  # peak normalised cross-correlation [-1, 1]
    lag_s: float              # lag in seconds (positive → B lags A)
    suggested_offset_s: float # apply to B's time axis to align with A
    confidence: float         # [0, 1] — peak prominence
    same_event_likely: bool   # True when correlation_coeff > _SAME_EVENT_THRESHOLD

    @property
    def lag_ms(self) -> float:
        return self.lag_s * 1000.0

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.75:
            return "HIGH"
        if self.confidence >= 0.40:
            return "MEDIUM"
        return "LOW"

    @property
    def confidence_color(self) -> str:
        return {"HIGH": "#55CC55", "MEDIUM": "#FFAA33", "LOW": "#FF5555"}[
            self.confidence_label
        ]


_SAME_EVENT_THRESHOLD = 0.60   # min cross-correlation to call it the same event
_MIN_OVERLAP_S = 0.02           # need at least 20 ms of overlapping signal
_MAX_GRID_POINTS = 50_000       # cap resampled array length for performance


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _common_dt(time_a: np.ndarray, time_b: np.ndarray) -> float:
    """Return a sensible shared sampling interval (the coarser of the two)."""
    def _median_dt(t: np.ndarray) -> float:
        if len(t) < 2:
            return 1.0
        d = np.diff(t)
        d = d[d > 0]
        return float(np.median(d)) if len(d) else 1.0

    return max(_median_dt(time_a), _median_dt(time_b))


def _resample_to_grid(
    time_src: np.ndarray,
    data_src: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Linear interpolation onto grid; NaN outside coverage."""
    clean_data = np.where(np.isfinite(data_src), data_src, 0.0)
    return np.interp(grid, time_src, clean_data,
                     left=np.nan, right=np.nan)


def _fft_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized full cross-correlation via FFT (pure numpy)."""
    a_n = (a - np.nanmean(a))
    b_n = (b - np.nanmean(b))
    std_a = np.nanstd(a_n)
    std_b = np.nanstd(b_n)
    if std_a < 1e-12 or std_b < 1e-12:
        return np.zeros(len(a) + len(b) - 1)
    a_n /= std_a
    b_n /= std_b

    # Pad to next power of 2 for speed
    n = len(a) + len(b) - 1
    fft_len = 1 << math.ceil(math.log2(n)) if n > 1 else 1

    fa = np.fft.rfft(a_n, n=fft_len)
    fb = np.fft.rfft(b_n, n=fft_len)
    xcorr = np.fft.irfft(fa * np.conj(fb), n=fft_len)[:n]
    return xcorr / max(len(a), len(b))


def _scipy_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = (a - np.nanmean(a))
    b_n = (b - np.nanmean(b))
    std_a = np.nanstd(a_n)
    std_b = np.nanstd(b_n)
    if std_a < 1e-12 or std_b < 1e-12:
        return np.zeros(len(a) + len(b) - 1)
    a_n /= std_a
    b_n /= std_b
    xcorr = scipy_correlate(a_n, b_n, mode="full", method="fft")
    return xcorr / max(len(a), len(b))


def _xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _scipy_xcorr(a, b) if _SCIPY_AVAILABLE else _fft_xcorr(a, b)


def _peak_confidence(xcorr: np.ndarray, peak_idx: int) -> float:
    """Confidence = peak / (RMS of the tail quartiles), clamped to [0, 1]."""
    n = len(xcorr)
    tail_len = max(1, n // 4)
    tail = np.concatenate([xcorr[:tail_len], xcorr[-tail_len:]])
    tail_rms = float(np.sqrt(np.mean(tail ** 2))) if len(tail) else 1e-12
    if tail_rms < 1e-12:
        return 0.0
    prominence = float(abs(xcorr[peak_idx])) / tail_rms
    return float(np.clip(prominence / 10.0, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def correlate_source_pair(
    source_a_id: str,
    time_a: np.ndarray,
    data_a: np.ndarray,
    channel_a: str,
    source_b_id: str,
    time_b: np.ndarray,
    data_b: np.ndarray,
    channel_b: str,
    *,
    max_lag_s: float = 5.0,
) -> CrossCorrelationResult | None:
    """Cross-correlate two waveform channels from different sources.

    Args:
        source_a_id / source_b_id: Identifiers for display / tracking.
        time_a / time_b:           Monotonic time arrays (seconds).
        data_a / data_b:           Corresponding sample arrays (same units preferred).
        channel_a / channel_b:     Channel names (for display only).
        max_lag_s:                 Search window half-width in seconds.

    Returns:
        CrossCorrelationResult or None if the analysis cannot proceed
        (no temporal overlap, flat signals, or insufficient data).
    """
    time_a = np.asarray(time_a, dtype=float)
    time_b = np.asarray(time_b, dtype=float)
    data_a = np.asarray(data_a, dtype=float)
    data_b = np.asarray(data_b, dtype=float)

    if len(time_a) < 4 or len(time_b) < 4:
        return None

    # ── Find temporal overlap ─────────────────────────────────────────────────
    t_start = max(float(time_a[0]), float(time_b[0]))
    t_end   = min(float(time_a[-1]), float(time_b[-1]))
    if t_end - t_start < _MIN_OVERLAP_S:
        return None  # no meaningful overlap

    # ── Common uniform grid over overlap ─────────────────────────────────────
    dt = _common_dt(time_a, time_b)
    n_points = min(_MAX_GRID_POINTS, max(4, int((t_end - t_start) / dt) + 1))
    grid = np.linspace(t_start, t_end, n_points)

    sig_a = _resample_to_grid(time_a, data_a, grid)
    sig_b = _resample_to_grid(time_b, data_b, grid)

    # Drop NaN regions at edges
    valid = np.isfinite(sig_a) & np.isfinite(sig_b)
    if np.sum(valid) < 4:
        return None
    sig_a = sig_a[valid]
    sig_b = sig_b[valid]

    if np.std(sig_a) < 1e-12 or np.std(sig_b) < 1e-12:
        return None  # flat / constant signal — correlation undefined

    # ── FFT cross-correlation ─────────────────────────────────────────────────
    xcorr = _xcorr(sig_a, sig_b)
    n_a = len(sig_a)
    n_b = len(sig_b)

    # Lag array in samples, then convert to seconds
    lags_samples = np.arange(-(n_a - 1), n_b)
    lags_s = lags_samples * dt

    # Restrict search to ±max_lag_s
    search_mask = np.abs(lags_s) <= max_lag_s
    if not np.any(search_mask):
        search_mask = np.ones(len(lags_s), dtype=bool)

    search_xcorr = xcorr.copy()
    search_xcorr[~search_mask] = 0.0

    peak_idx = int(np.argmax(np.abs(search_xcorr)))
    corr_coeff = float(xcorr[peak_idx])
    lag_s = float(lags_s[peak_idx]) if peak_idx < len(lags_s) else 0.0

    confidence = _peak_confidence(xcorr, peak_idx)

    return CrossCorrelationResult(
        source_a_id=source_a_id,
        source_b_id=source_b_id,
        channel_a=channel_a,
        channel_b=channel_b,
        correlation_coeff=corr_coeff,
        lag_s=lag_s,
        suggested_offset_s=-lag_s,   # shift B to align with A
        confidence=confidence,
        same_event_likely=(abs(corr_coeff) >= _SAME_EVENT_THRESHOLD),
    )


def correlate_all_pairs(
    sources_data: list[tuple[str, np.ndarray, np.ndarray, str]],
    *,
    max_lag_s: float = 5.0,
) -> list[CrossCorrelationResult]:
    """Correlate all unique pairs in a list of sources.

    Args:
        sources_data: list of (source_id, time, data, channel_name) tuples.
        max_lag_s:    Maximum lag to search in seconds.

    Returns:
        List of CrossCorrelationResult, one per unique pair.
        Pairs that cannot be correlated are silently skipped.
    """
    results: list[CrossCorrelationResult] = []
    n = len(sources_data)
    for i in range(n):
        for j in range(i + 1, n):
            a_id, t_a, d_a, ch_a = sources_data[i]
            b_id, t_b, d_b, ch_b = sources_data[j]
            result = correlate_source_pair(
                a_id, t_a, d_a, ch_a,
                b_id, t_b, d_b, ch_b,
                max_lag_s=max_lag_s,
            )
            if result is not None:
                results.append(result)
    return results
