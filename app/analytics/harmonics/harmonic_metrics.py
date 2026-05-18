"""Harmonic distortion metrics — Phase 7.

Standard engineering definitions:

  THD (Total Harmonic Distortion):
    THD = sqrt(sum_{n=2}^{N} H_n^2) / H_1

  where H_n is the RMS magnitude of the n-th harmonic.

  THD is returned as a dimensionless fraction, not a percentage.
  Callers display as 100*THD % or use directly for computation.

  TDD (Total Demand Distortion) is a placeholder for future work.

Safety:
  - Near-zero fundamental (< safe_threshold): returns 0.0 instead of raising.
  - Harmonic dict with no entries above H1: returns 0.0.
  - Negative or non-finite magnitudes are treated as 0.
"""
from __future__ import annotations

import numpy as np

from app.analytics.harmonics.harmonic_models import HarmonicResult

_DEFAULT_SAFE_THRESHOLD = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Scalar THD (single time instant)
# ─────────────────────────────────────────────────────────────────────────────


def compute_thd(
    harmonic_magnitudes: dict[int, float],
    *,
    max_order: int | None = None,
    safe_threshold: float = _DEFAULT_SAFE_THRESHOLD,
) -> float:
    """Compute THD from a mapping of {harmonic_order: rms_magnitude}.

    Args:
        harmonic_magnitudes: Dict mapping integer harmonic order to RMS
                             magnitude value.  Order 1 = fundamental.
        max_order:           Highest order to include in the distortion sum.
                             None = include all orders present in the dict.
        safe_threshold:      Fundamental magnitude below this → return 0.0
                             (avoids divide-by-zero on zero-signal windows).

    Returns:
        THD as a dimensionless fraction (e.g. 0.05 for 5% THD).
    """
    fundamental = float(harmonic_magnitudes.get(1, 0.0))
    if not np.isfinite(fundamental) or fundamental < safe_threshold:
        return 0.0

    orders = sorted(harmonic_magnitudes.keys())
    distortion_sq = 0.0
    for order in orders:
        if order <= 1:
            continue
        if max_order is not None and order > max_order:
            break
        mag = float(harmonic_magnitudes[order])
        if np.isfinite(mag) and mag > 0.0:
            distortion_sq += mag * mag

    return float(np.sqrt(distortion_sq)) / fundamental


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised time-varying THD from magnitude arrays
# ─────────────────────────────────────────────────────────────────────────────


def compute_thd_array(
    magnitudes: dict[int, np.ndarray],
    *,
    max_order: int | None = None,
    safe_threshold: float = _DEFAULT_SAFE_THRESHOLD,
) -> np.ndarray:
    """Compute time-varying THD from a dict of per-order magnitude arrays.

    Args:
        magnitudes:      {order: rms_array} where each array has shape
                         ``(n_windows,)``.  All arrays must have the same length.
        max_order:       Highest harmonic order to include in the distortion sum.
        safe_threshold:  Windows where fundamental < threshold → THD = 0.

    Returns:
        Float64 array of shape ``(n_windows,)`` containing THD at each window.
        Returns a length-0 array if the fundamental is absent or empty.
    """
    fundamental = magnitudes.get(1)
    if fundamental is None or len(fundamental) == 0:
        return np.empty(0, dtype=np.float64)

    h1 = np.asarray(fundamental, dtype=np.float64)
    n = len(h1)

    distortion_sq = np.zeros(n, dtype=np.float64)
    for order, arr in magnitudes.items():
        if order <= 1:
            continue
        if max_order is not None and order > max_order:
            continue
        a = np.asarray(arr, dtype=np.float64)
        if len(a) != n:
            continue
        valid = np.isfinite(a) & (a > 0.0)
        distortion_sq[valid] += a[valid] ** 2

    thd = np.zeros(n, dtype=np.float64)
    safe_mask = np.isfinite(h1) & (h1 >= safe_threshold)
    thd[safe_mask] = np.sqrt(distortion_sq[safe_mask]) / h1[safe_mask]
    return thd


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: THD from a HarmonicResult
# ─────────────────────────────────────────────────────────────────────────────


def compute_thd_from_result(
    result: HarmonicResult,
    *,
    max_order: int | None = None,
    safe_threshold: float = _DEFAULT_SAFE_THRESHOLD,
) -> np.ndarray:
    """Compute time-varying THD from a HarmonicResult object.

    Args:
        result:        Harmonic extraction result containing per-order arrays.
        max_order:     Highest harmonic order to include (None = all).
        safe_threshold: Fundamental threshold for safe division.

    Returns:
        Float64 array of shape ``(n_windows,)``.
    """
    return compute_thd_array(
        result.magnitudes,
        max_order=max_order,
        safe_threshold=safe_threshold,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Individual harmonic distortion ratio
# ─────────────────────────────────────────────────────────────────────────────


def individual_harmonic_distortion(
    harmonic_magnitudes: dict[int, float],
    order: int,
    *,
    safe_threshold: float = _DEFAULT_SAFE_THRESHOLD,
) -> float:
    """Return H_n / H_1 for a single harmonic order.

    Args:
        harmonic_magnitudes: {order: rms_magnitude} dict.
        order:               Harmonic order to evaluate (>= 2).
        safe_threshold:      Fundamental threshold for safe division.

    Returns:
        Dimensionless ratio.  0.0 when fundamental is near-zero.
    """
    fundamental = float(harmonic_magnitudes.get(1, 0.0))
    if not np.isfinite(fundamental) or fundamental < safe_threshold:
        return 0.0
    hn = float(harmonic_magnitudes.get(order, 0.0))
    if not np.isfinite(hn) or hn <= 0.0:
        return 0.0
    return hn / fundamental
