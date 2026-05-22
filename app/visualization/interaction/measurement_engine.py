"""Two-cursor measurement engine — pure computation, no Qt dependencies.

Given two time positions (t_a, t_b) and waveform data arrays, computes the
standard power-system measurements an engineer needs between two markers:
delta-time, delta-Y, frequency from period, RMS, mean, peak, and energy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ChannelMeasurement:
    """Per-channel measurements between cursor A and cursor B."""

    name: str
    unit: str

    # Amplitude at cursor positions (nearest-sample interpolation)
    y_at_a: float | None = None
    y_at_b: float | None = None
    delta_y: float | None = None  # y_at_b − y_at_a

    # Statistics over the segment [t_a, t_b]
    rms: float | None = None
    mean: float | None = None
    peak: float | None = None        # max(|y|) in segment
    peak_to_peak: float | None = None

    # Sample count in segment (useful for sanity checks)
    sample_count: int = 0


@dataclass
class MeasurementResult:
    """Full measurement result for all channels between two time cursors."""

    # Cursor positions in seconds (same time base as waveform arrays)
    t_a: float
    t_b: float

    # Time-domain summary
    delta_t_s: float = 0.0        # |t_b − t_a| in seconds
    delta_t_ms: float = 0.0       # |t_b − t_a| in milliseconds
    delta_t_cycles: float | None = None   # Δt × f_nominal (None if f_nominal unknown)
    frequency_hz: float | None = None     # 1 / Δt (meaningful only when Δt ≈ one period)

    # Per-channel measurements
    channels: list[ChannelMeasurement] = field(default_factory=list)

    # Optional energy (W·s) computed when a V/I pair is identified
    energy_ws: float | None = None
    energy_pair: tuple[str, str] | None = None  # (voltage_ch, current_ch)

    @property
    def valid(self) -> bool:
        """True when t_a ≠ t_b and at least one channel has data."""
        return self.delta_t_s > 1e-12 and len(self.channels) > 0


def compute_measurements(
    t_a: float,
    t_b: float,
    time: np.ndarray,
    data_by_channel: dict[str, tuple[np.ndarray, str]],
    *,
    nominal_hz: float | None = None,
) -> MeasurementResult:
    """Compute two-cursor measurements for all channels.

    Args:
        t_a:              First cursor time (seconds, same base as *time*).
        t_b:              Second cursor time (seconds).
        time:             1-D float64 time array (monotonic, seconds).
        data_by_channel:  Mapping of channel_name → (data_array, unit_string).
                          Arrays must match *time* in length.
        nominal_hz:       System nominal frequency (e.g. 50.0 or 60.0). Used
                          to compute cycle count. None → cycles not reported.

    Returns:
        MeasurementResult with all computed statistics.
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

    if len(time) == 0:
        return result

    # Segment mask: inclusive [lo, hi]
    mask = (time >= lo) & (time <= hi)
    seg_indices = np.flatnonzero(mask)

    # Nearest-sample index for each cursor position
    idx_a = int(np.searchsorted(time, t_a, side="left"))
    idx_a = max(0, min(idx_a, len(time) - 1))
    idx_b = int(np.searchsorted(time, t_b, side="left"))
    idx_b = max(0, min(idx_b, len(time) - 1))

    voltage_channels: list[tuple[str, np.ndarray]] = []
    current_channels: list[tuple[str, np.ndarray]] = []

    for name, (data, unit) in data_by_channel.items():
        if len(data) != len(time):
            continue

        y_a = float(data[idx_a]) if np.isfinite(data[idx_a]) else None
        y_b = float(data[idx_b]) if np.isfinite(data[idx_b]) else None
        delta_y = (y_b - y_a) if (y_a is not None and y_b is not None) else None

        ch_meas = ChannelMeasurement(name=name, unit=unit)
        ch_meas.y_at_a = y_a
        ch_meas.y_at_b = y_b
        ch_meas.delta_y = delta_y

        if len(seg_indices) > 0:
            seg = data[seg_indices]
            finite_seg = seg[np.isfinite(seg)]
            n = len(finite_seg)
            ch_meas.sample_count = n
            if n > 0:
                ch_meas.rms = float(np.sqrt(np.mean(finite_seg ** 2)))
                ch_meas.mean = float(np.mean(finite_seg))
                abs_seg = np.abs(finite_seg)
                ch_meas.peak = float(np.max(abs_seg))
                ch_meas.peak_to_peak = float(np.max(finite_seg) - np.min(finite_seg))
        else:
            # Cursors coincide — use single point values only
            ch_meas.sample_count = 1

        result.channels.append(ch_meas)

        # Collect candidates for energy computation
        unit_lower = unit.lower() if unit else ""
        name_lower = name.lower()
        if unit_lower in ("kv", "v", "pu") or any(
            x in name_lower for x in ("va", "vb", "vc", "vr", "vy", "voltage")
        ):
            voltage_channels.append((name, data))
        elif unit_lower in ("ka", "a") or any(
            x in name_lower for x in ("ia", "ib", "ic", "ir", "iy", "current")
        ):
            current_channels.append((name, data))

    # Energy: try to pair one voltage + one current channel by phase
    if voltage_channels and current_channels and len(seg_indices) > 1:
        v_name, v_data = _best_energy_pair_channel(voltage_channels, current_channels)
        if v_name is not None:
            i_name, i_data = _best_energy_pair_channel(
                current_channels, [(v_name, v_data)]
            )
            if i_name is not None:
                seg_v = v_data[seg_indices]
                seg_i = i_data[seg_indices]
                valid = np.isfinite(seg_v) & np.isfinite(seg_i)
                if np.any(valid):
                    seg_t = time[seg_indices]
                    if len(seg_t) > 1:
                        dt_arr = np.diff(seg_t[valid])
                        power_arr = seg_v[valid][:-1] * seg_i[valid][:-1]
                        result.energy_ws = float(np.sum(power_arr * dt_arr))
                        result.energy_pair = (v_name, i_name)

    return result


def _best_energy_pair_channel(
    candidates: list[tuple[str, np.ndarray]],
    partner_list: list[tuple[str, np.ndarray]],
) -> tuple[str | None, np.ndarray | None]:
    """Return the first candidate (prefer phase-A match), else the first one."""
    if not candidates:
        return None, None
    for name, data in candidates:
        name_l = name.lower()
        if any(x in name_l for x in ("_a", "va", "ia", "phase_a")):
            return name, data
    return candidates[0]


def find_zero_crossings(time: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Return time positions where *data* crosses zero (sign changes).

    Used for smart cursor snapping to zero-crossing targets.
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

    Used for smart cursor snapping to waveform peaks.

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
