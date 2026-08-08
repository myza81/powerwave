"""Two-cursor measurement engine — pure computation, no Qt dependencies.

Given two time positions (t_a, t_b) and a set of curves, computes the
standard power-system measurements an engineer needs between two markers:
delta-time, delta-Y, frequency from period, RMS, mean, peak, and energy.

Per-curve time bases (Sprint 1A)
---------------------------------
Each curve supplies its OWN time array. A panel routinely contains curves
from different sample rates -- a 6400 sps protection relay channel, a 50 sps
PMU channel, a full-resolution calculated signal -- and none of them are
required to share a common time array or sample count. Every curve is
evaluated independently against the same pair of cursor times; nothing is
resampled onto a shared grid and nothing is silently dropped because its
length differs from another curve's.

A curve that genuinely cannot be evaluated (no data, no overlap with the
cursor range, or no finite samples in range) is still reported, with
``available=False`` and a human-readable ``unavailable_reason`` -- never
omitted from ``MeasurementResult.channels``. Powerwave must never silently
omit engineering evidence.

The per-curve value lookup itself (nearest-sample via ``np.searchsorted``,
then clamped to the curve's own array bounds) is unchanged from the
previous shared-time-array implementation -- this module does not invent a
new interpolation policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class CurveSample:
    """One curve's own (time, values) arrays, submitted independently to
    compute_measurements(). No two curves need share a length, a sample
    rate, or a time base -- each is evaluated entirely on its own array.

    name/unit are display-only (already resolved by the caller, e.g. from
    curve metadata); this module does not know or care whether a curve
    came from a real source channel or a calculated signal.
    """

    name: str
    unit: str
    time: np.ndarray
    values: np.ndarray


@dataclass
class ChannelMeasurement:
    """Per-channel measurements between cursor A and cursor B.

    available=False means this curve could not produce a measurement at
    the current cursor positions (empty data, no time-range overlap, or no
    finite samples in range) -- unavailable_reason explains why. The row
    is still present; nothing is ever silently omitted.
    """

    name: str
    unit: str

    available: bool = True
    unavailable_reason: str | None = None

    # Amplitude at cursor positions (nearest-sample lookup, clamped to
    # this curve's own array bounds -- unchanged policy)
    y_at_a: float | None = None
    y_at_b: float | None = None
    delta_y: float | None = None  # y_at_b − y_at_a

    # Statistics over this curve's own samples inside [t_a, t_b]
    rms: float | None = None
    mean: float | None = None
    peak: float | None = None        # max(|y|) in segment
    peak_to_peak: float | None = None

    # Sample count in segment (useful for sanity checks)
    sample_count: int = 0


@dataclass
class MeasurementResult:
    """Full measurement result for all curves between two time cursors."""

    # Cursor positions in seconds (same time base as waveform arrays)
    t_a: float
    t_b: float

    # Time-domain summary
    delta_t_s: float = 0.0        # |t_b − t_a| in seconds
    delta_t_ms: float = 0.0       # |t_b − t_a| in milliseconds
    delta_t_cycles: float | None = None   # Δt × f_nominal (None if f_nominal unknown)
    frequency_hz: float | None = None     # 1 / Δt (meaningful only when Δt ≈ one period)

    # Per-channel measurements -- always one entry per submitted curve,
    # available or not.
    channels: list[ChannelMeasurement] = field(default_factory=list)

    # Optional energy (W·s) computed when a V/I pair is identified AND both
    # curves share an identical time base (see compute_measurements' energy
    # section below for why this is required).
    energy_ws: float | None = None
    energy_pair: tuple[str, str] | None = None  # (voltage_ch, current_ch)

    @property
    def valid(self) -> bool:
        """True when t_a ≠ t_b and at least one curve was submitted."""
        return self.delta_t_s > 1e-12 and len(self.channels) > 0


def _measure_one_curve(
    curve: CurveSample, lo: float, hi: float, t_a: float, t_b: float
) -> ChannelMeasurement:
    """Evaluate one curve independently against cursor positions t_a/t_b.

    Availability rules (never silently omitted -- always returns a
    ChannelMeasurement, available or not):
      - empty time/values, or a malformed (mismatched-length) curve -> unavailable
      - curve's own time span does not overlap [lo, hi] at all -> unavailable
        ("outside time span" / "no overlap")
      - curve overlaps but every candidate value (both cursor lookups and
        every in-segment sample) is non-finite -> unavailable ("invalid values")
      - otherwise -> available, using the existing nearest-sample-with-clamp
        lookup policy and segment statistics, computed only over this
        curve's own overlap with [lo, hi] -- never resampled onto another
        curve's grid.
    """
    ch = ChannelMeasurement(name=curve.name, unit=curve.unit)
    time = curve.time
    values = curve.values

    if len(time) == 0 or len(values) == 0:
        ch.available = False
        ch.unavailable_reason = "No data"
        return ch
    if len(time) != len(values):
        ch.available = False
        ch.unavailable_reason = "Malformed curve (time/values length mismatch)"
        return ch

    curve_min = float(time[0])
    curve_max = float(time[-1])
    if curve_max < lo or curve_min > hi:
        ch.available = False
        ch.unavailable_reason = "Outside time span (no overlap with cursor range)"
        return ch

    # Nearest-sample index for each cursor position, clamped to this
    # curve's own bounds -- identical policy to the previous implementation.
    idx_a = int(np.searchsorted(time, t_a, side="left"))
    idx_a = max(0, min(idx_a, len(time) - 1))
    idx_b = int(np.searchsorted(time, t_b, side="left"))
    idx_b = max(0, min(idx_b, len(time) - 1))

    y_a = float(values[idx_a]) if np.isfinite(values[idx_a]) else None
    y_b = float(values[idx_b]) if np.isfinite(values[idx_b]) else None
    ch.y_at_a = y_a
    ch.y_at_b = y_b
    ch.delta_y = (y_b - y_a) if (y_a is not None and y_b is not None) else None

    # Segment mask: inclusive [lo, hi], evaluated on this curve's own time array.
    mask = (time >= lo) & (time <= hi)
    seg_indices = np.flatnonzero(mask)
    if len(seg_indices) > 0:
        seg = values[seg_indices]
        finite_seg = seg[np.isfinite(seg)]
        n = len(finite_seg)
        ch.sample_count = n
        if n > 0:
            ch.rms = float(np.sqrt(np.mean(finite_seg ** 2)))
            ch.mean = float(np.mean(finite_seg))
            abs_seg = np.abs(finite_seg)
            ch.peak = float(np.max(abs_seg))
            ch.peak_to_peak = float(np.max(finite_seg) - np.min(finite_seg))
    else:
        # No sample of this curve's own grid falls strictly inside [lo, hi]
        # -- e.g. cursors coincide, or (mixed sample rates) a low-rate
        # curve simply has no sample between two closely-spaced cursors.
        # The nearest-sample cursor values above remain valid; only the
        # segment statistics are unavailable.
        ch.sample_count = 1

    if ch.y_at_a is None and ch.y_at_b is None and ch.rms is None:
        # The curve overlaps in time, but nothing finite could be read at
        # either cursor and no finite sample exists in the segment either.
        ch.available = False
        ch.unavailable_reason = "No valid (finite) samples in range"

    return ch


def compute_measurements(
    t_a: float,
    t_b: float,
    curves: "Iterable[CurveSample]",
    *,
    nominal_hz: float | None = None,
) -> MeasurementResult:
    """Compute two-cursor measurements for a set of independently-timed curves.

    Args:
        t_a:         First cursor time (seconds).
        t_b:         Second cursor time (seconds).
        curves:      Curves to measure, each with its own (time, values).
                     No two curves are required to share a length, sample
                     rate, or time base.
        nominal_hz:  System nominal frequency (e.g. 50.0 or 60.0). Used to
                     compute cycle count. None → cycles not reported.

    Returns:
        MeasurementResult with one ChannelMeasurement per curve, in the
        order submitted -- always present, available or not.
    """
    lo = min(t_a, t_b)
    hi = max(t_a, t_b)

    delta_t_s = hi - lo
    delta_t_ms = delta_t_s * 1_000.0
    delta_t_cycles = delta_t_s * nominal_hz if nominal_hz is not None else None
    frequency_hz = (1.0 / delta_t_s) if delta_t_s > 1e-12 else None

    result = MeasurementResult(
        t_a=t_a,
        t_b=t_b,
        delta_t_s=delta_t_s,
        delta_t_ms=delta_t_ms,
        delta_t_cycles=delta_t_cycles,
        frequency_hz=frequency_hz,
    )

    curve_list = list(curves)
    for curve in curve_list:
        result.channels.append(_measure_one_curve(curve, lo, hi, t_a, t_b))

    # Energy: try to pair one voltage + one current *available* channel by
    # phase/name. Trapezoidal power integration (v(t) * i(t) dt) requires
    # both signals sampled at the same instants -- with independent
    # per-curve time bases that is no longer guaranteed, so energy is only
    # computed when the paired curves' time arrays are exactly identical
    # (the common case for a V/I pair from the same source at the same
    # rate). This is a conservative compatibility check, not a new
    # interpolation policy: mixed-rate V/I pairs simply report no energy
    # rather than a resampled/approximated one.
    voltage_curves: list[CurveSample] = []
    current_curves: list[CurveSample] = []
    available_by_name = {ch.name: ch.available for ch in result.channels}
    for curve in curve_list:
        if not available_by_name.get(curve.name, False):
            continue
        unit_lower = curve.unit.lower() if curve.unit else ""
        name_lower = curve.name.lower()
        if unit_lower in ("kv", "v", "pu") or any(
            x in name_lower for x in ("va", "vb", "vc", "vr", "vy", "voltage")
        ):
            voltage_curves.append(curve)
        elif unit_lower in ("ka", "a") or any(
            x in name_lower for x in ("ia", "ib", "ic", "ir", "iy", "current")
        ):
            current_curves.append(curve)

    if voltage_curves and current_curves:
        v_curve = _best_energy_pair_curve(voltage_curves)
        i_curve = _best_energy_pair_curve(current_curves)
        if (
            v_curve is not None
            and i_curve is not None
            and v_curve.time.shape == i_curve.time.shape
            and np.array_equal(v_curve.time, i_curve.time)
        ):
            time = v_curve.time
            mask = (time >= lo) & (time <= hi)
            seg_indices = np.flatnonzero(mask)
            if len(seg_indices) > 1:
                seg_v = v_curve.values[seg_indices]
                seg_i = i_curve.values[seg_indices]
                valid = np.isfinite(seg_v) & np.isfinite(seg_i)
                if np.any(valid):
                    seg_t = time[seg_indices]
                    dt_arr = np.diff(seg_t[valid])
                    if len(dt_arr) > 0:
                        power_arr = seg_v[valid][:-1] * seg_i[valid][:-1]
                        result.energy_ws = float(np.sum(power_arr * dt_arr))
                        result.energy_pair = (v_curve.name, i_curve.name)

    return result


def _best_energy_pair_curve(candidates: list[CurveSample]) -> CurveSample | None:
    """Return the first candidate (prefer phase-A match), else the first one."""
    if not candidates:
        return None
    for curve in candidates:
        name_l = curve.name.lower()
        if any(x in name_l for x in ("_a", "va", "ia", "phase_a")):
            return curve
    return candidates[0]


def find_zero_crossings(time: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Return time positions where *data* crosses zero (sign changes).

    Used for smart cursor snapping to zero-crossing targets. Already
    operates on one curve's own (time, data) pair -- no change needed for
    per-curve time base support.
    """
    if len(data) < 2:
        return np.empty(0, dtype=np.float64)
    finite = np.isfinite(data)
    signs = np.sign(data)
    crossings: list[float] = []
    for i in range(len(signs) - 1):
        if not finite[i] or not finite[i + 1]:
            continue
        if signs[i] != signs[i + 1] and signs[i] != 0:
            # Linear interpolation to the exact zero crossing
            t0, t1 = float(time[i]), float(time[i + 1])
            y0, y1 = float(data[i]), float(data[i + 1])
            if abs(y1 - y0) > 1e-30:
                t_cross = t0 - y0 * (t1 - t0) / (y1 - y0)
            else:
                t_cross = (t0 + t1) / 2.0
            crossings.append(t_cross)
    return np.array(crossings, dtype=np.float64)


def find_local_peaks(
    time: np.ndarray,
    data: np.ndarray,
    *,
    window: int = 3,
) -> np.ndarray:
    """Return time positions of local absolute-value maxima in *data*.

    Used for smart cursor snapping to waveform peaks. Already operates on
    one curve's own (time, data) pair -- no change needed for per-curve
    time base support.

    Args:
        time:   Time array matching *data*.
        data:   Waveform data array.
        window: Number of samples on each side to check (default 3).
    """
    if len(data) < 2 * window + 1:
        return np.empty(0, dtype=np.float64)
    abs_data = np.abs(data)
    peaks: list[float] = []
    for i in range(window, len(abs_data) - window):
        neighbourhood = abs_data[i - window: i + window + 1]
        if not np.all(np.isfinite(neighbourhood)):
            continue
        if abs_data[i] == np.max(neighbourhood):
            peaks.append(float(time[i]))
    return np.array(peaks, dtype=np.float64)


def nearest_snap_target(
    t: float,
    snap_targets: np.ndarray,
    *,
    max_distance_s: float = 0.01,
) -> float | None:
    """Return the nearest snap target within *max_distance_s* of *t*.

    Returns None when no target is close enough.
    """
    if len(snap_targets) == 0:
        return None
    dists = np.abs(snap_targets - t)
    idx = int(np.argmin(dists))
    if dists[idx] <= max_distance_s:
        return float(snap_targets[idx])
    return None
