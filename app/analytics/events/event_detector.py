"""Automatic power-system event detection — pure numpy, no Qt dependencies.

Detects four classes of disturbance from analog waveform arrays:

  1. Voltage dip / swell  — amplitude drops below 90% or rises above 110%
                            of the pre-fault RMS baseline for >0.5 cycle.
  2. Overcurrent          — peak current exceeds 120% of pre-fault peak
                            for >0.5 cycle.
  3. Frequency deviation  — frequency channel deviates from nominal by
                            >0.5 Hz for >0.5 cycle.
  4. Zero-sequence        — magnitude of a neutral/earth/3I0 channel exceeds
                            10% of nominal current baseline for >0.5 cycle.

Detection is baseline-relative: the pre-fault window (before trigger) is used
to establish the reference. If no pre-fault window is available (trigger at or
before t=0), the first quarter of the recording serves as the reference.

All algorithms use an O(N) cumulative-sum sliding window RMS — suitable for
recordings up to several million samples without blocking the UI.
"""
from __future__ import annotations

import numpy as np

from app.analytics.events.event_models import (
    DetectedEvent,
    EventSeverity,
    EventType,
)

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_NOMINAL_HZ = 50.0
_DIP_THRESHOLD = 0.90       # below this fraction of baseline → dip
_SWELL_THRESHOLD = 1.10     # above this → swell
_OC_THRESHOLD = 1.20        # above this factor of baseline peak → overcurrent
_FREQ_DEV_HZ = 0.5          # Hz from nominal before flagging
_ZERO_SEQ_THRESHOLD = 0.10  # fraction of baseline current → zero-seq injection
_MIN_DURATION_CYCLES = 0.5  # minimum event duration in cycles
_MERGE_GAP_CYCLES = 1.0     # merge adjacent regions within this many cycles


def detect_events(
    time: np.ndarray,
    data_by_channel: dict[str, np.ndarray],
    channel_roles: dict[str, str],
    *,
    trigger_time_s: float | None = None,
    nominal_hz: float = _DEFAULT_NOMINAL_HZ,
) -> list[DetectedEvent]:
    """Detect disturbance events across all channels.

    Args:
        time:             1-D float64 monotonic time array (seconds).
        data_by_channel:  channel_name → data array (same length as time).
        channel_roles:    channel_name → role string:
                          "voltage" | "current" | "frequency" | "zero_sequence"
        trigger_time_s:   Trigger instant in seconds (same base as time).
                          Used to isolate the pre-fault baseline window.
                          None → first quarter of recording used as baseline.
        nominal_hz:       System nominal frequency (default 50.0).

    Returns:
        List of DetectedEvent, sorted by t_start then channel name.
    """
    if len(time) < 4:
        return []

    sample_rate = _estimate_sample_rate(time)
    if sample_rate <= 0:
        return []

    cycle_samples = int(round(sample_rate / nominal_hz))
    if cycle_samples < 2:
        cycle_samples = 2

    min_dur_samples = max(1, int(_MIN_DURATION_CYCLES * cycle_samples))
    merge_gap_samples = max(1, int(_MERGE_GAP_CYCLES * cycle_samples))

    prefault_mask = _build_prefault_mask(time, trigger_time_s, cycle_samples)

    events: list[DetectedEvent] = []

    for name, role in channel_roles.items():
        data = data_by_channel.get(name)
        if data is None or len(data) != len(time):
            continue
        finite = np.isfinite(data)
        if not np.any(finite):
            continue

        if role == "voltage":
            events.extend(
                _detect_voltage_events(
                    name, time, data, prefault_mask,
                    cycle_samples, min_dur_samples, merge_gap_samples,
                )
            )
        elif role == "current":
            events.extend(
                _detect_overcurrent(
                    name, time, data, prefault_mask,
                    cycle_samples, min_dur_samples, merge_gap_samples,
                )
            )
        elif role == "frequency":
            events.extend(
                _detect_frequency_deviation(
                    name, time, data, prefault_mask,
                    nominal_hz, min_dur_samples, merge_gap_samples,
                )
            )
        elif role == "zero_sequence":
            events.extend(
                _detect_zero_sequence(
                    name, time, data, prefault_mask,
                    cycle_samples, min_dur_samples, merge_gap_samples,
                )
            )

    events.sort(key=lambda e: (e.t_start, e.channel))
    return events


def classify_channel_roles(
    analog_channels: list,
) -> dict[str, str]:
    """Assign a detection role to each analog channel by name and unit heuristics.

    Returns a dict of channel_name → role string.
    Channels that don't match any role are omitted (not included in detection).
    """
    roles: dict[str, str] = {}
    for ch in analog_channels:
        role = _infer_role(ch.name, getattr(ch, "unit", "") or "")
        if role is not None:
            roles[ch.name] = role
    return roles


# ──────────────────────────────────────────────────────────────────────────────
# Role heuristics
# ──────────────────────────────────────────────────────────────────────────────

_VOLTAGE_UNITS = frozenset({"kv", "v", "pu"})
_CURRENT_UNITS = frozenset({"ka", "a"})
_FREQ_UNITS = frozenset({"hz"})

_VOLTAGE_KEYWORDS = frozenset({"va", "vb", "vc", "vr", "vy", "vab", "vbc", "vca",
                                "van", "vbn", "vcn", "volt", "voltage", "ph_v",
                                "u_a", "u_b", "u_c", "ua", "ub", "uc"})
_CURRENT_KEYWORDS = frozenset({"ia", "ib", "ic", "ir", "iy", "current", "ph_i",
                                "i_a", "i_b", "i_c"})
_FREQ_KEYWORDS = frozenset({"freq", "f_", "frequency", "hz_"})
_ZERO_SEQ_KEYWORDS = frozenset({"3i0", "3u0", "in_", "in", "earth", "neutral",
                                 "zero", "ground", "ig", "io", "i0", "v0"})


def _infer_role(name: str, unit: str) -> str | None:
    name_l = name.lower()
    unit_l = unit.lower().strip()

    # Zero-sequence checked first (some zero-seq channels have unit "A")
    if any(kw in name_l for kw in _ZERO_SEQ_KEYWORDS):
        return "zero_sequence"

    if unit_l in _FREQ_UNITS or any(kw in name_l for kw in _FREQ_KEYWORDS):
        return "frequency"

    if unit_l in _VOLTAGE_UNITS or any(kw == name_l or name_l.startswith(kw)
                                        for kw in _VOLTAGE_KEYWORDS):
        return "voltage"

    if unit_l in _CURRENT_UNITS or any(kw == name_l or name_l.startswith(kw)
                                        for kw in _CURRENT_KEYWORDS):
        return "current"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Detection algorithms
# ──────────────────────────────────────────────────────────────────────────────

def _detect_voltage_events(
    name: str,
    time: np.ndarray,
    data: np.ndarray,
    prefault_mask: np.ndarray,
    cycle_samples: int,
    min_dur: int,
    merge_gap: int,
) -> list[DetectedEvent]:
    rms = _sliding_rms(data, cycle_samples)
    baseline = _masked_rms(rms, prefault_mask)
    if baseline < 1e-9:
        return []

    events: list[DetectedEvent] = []

    # Voltage dip
    dip_mask = rms < (_DIP_THRESHOLD * baseline)
    for seg in _find_segments(dip_mask, min_dur, merge_gap):
        t_s, t_e = float(time[seg[0]]), float(time[seg[-1]])
        peak_rms = float(np.min(rms[seg]))
        ratio = peak_rms / baseline
        sev = _voltage_dip_severity(ratio)
        events.append(DetectedEvent(
            event_type=EventType.VOLTAGE_DIP,
            t_start=t_s,
            t_end=t_e,
            channel=name,
            severity=sev,
            magnitude=peak_rms,
            baseline=baseline,
            description=f"{name}: {ratio * 100:.1f}% of nominal ({(1 - ratio) * 100:.1f}% dip)",
        ))

    # Voltage swell
    swell_mask = rms > (_SWELL_THRESHOLD * baseline)
    for seg in _find_segments(swell_mask, min_dur, merge_gap):
        t_s, t_e = float(time[seg[0]]), float(time[seg[-1]])
        peak_rms = float(np.max(rms[seg]))
        ratio = peak_rms / baseline
        events.append(DetectedEvent(
            event_type=EventType.VOLTAGE_SWELL,
            t_start=t_s,
            t_end=t_e,
            channel=name,
            severity=EventSeverity.MEDIUM,
            magnitude=peak_rms,
            baseline=baseline,
            description=f"{name}: {ratio * 100:.1f}% of nominal (+{(ratio - 1) * 100:.1f}% swell)",
        ))

    return events


def _detect_overcurrent(
    name: str,
    time: np.ndarray,
    data: np.ndarray,
    prefault_mask: np.ndarray,
    cycle_samples: int,
    min_dur: int,
    merge_gap: int,
) -> list[DetectedEvent]:
    rms = _sliding_rms(data, cycle_samples)
    baseline = _masked_rms(rms, prefault_mask)
    if baseline < 1e-9:
        return []

    oc_mask = rms > (_OC_THRESHOLD * baseline)
    events: list[DetectedEvent] = []
    for seg in _find_segments(oc_mask, min_dur, merge_gap):
        t_s, t_e = float(time[seg[0]]), float(time[seg[-1]])
        peak_rms = float(np.max(rms[seg]))
        ratio = peak_rms / baseline
        sev = EventSeverity.HIGH if ratio > 2.0 else EventSeverity.MEDIUM
        events.append(DetectedEvent(
            event_type=EventType.OVERCURRENT,
            t_start=t_s,
            t_end=t_e,
            channel=name,
            severity=sev,
            magnitude=peak_rms,
            baseline=baseline,
            description=f"{name}: {ratio:.2f}× nominal ({ratio * 100 - 100:.0f}% overcurrent)",
        ))
    return events


def _detect_frequency_deviation(
    name: str,
    time: np.ndarray,
    data: np.ndarray,
    prefault_mask: np.ndarray,
    nominal_hz: float,
    min_dur: int,
    merge_gap: int,
) -> list[DetectedEvent]:
    finite = np.isfinite(data)
    if not np.any(finite):
        return []

    dev_mask = np.zeros(len(data), dtype=bool)
    dev_mask[finite] = np.abs(data[finite] - nominal_hz) > _FREQ_DEV_HZ

    events: list[DetectedEvent] = []
    for seg in _find_segments(dev_mask, min_dur, merge_gap):
        t_s, t_e = float(time[seg[0]]), float(time[seg[-1]])
        seg_data = data[seg]
        finite_seg = seg_data[np.isfinite(seg_data)]
        if len(finite_seg) == 0:
            continue
        peak_dev = float(np.max(np.abs(finite_seg - nominal_hz)))
        worst_f = float(finite_seg[np.argmax(np.abs(finite_seg - nominal_hz))])
        sev = EventSeverity.HIGH if peak_dev > 2.0 else EventSeverity.MEDIUM
        events.append(DetectedEvent(
            event_type=EventType.FREQUENCY_DEVIATION,
            t_start=t_s,
            t_end=t_e,
            channel=name,
            severity=sev,
            magnitude=worst_f,
            baseline=nominal_hz,
            description=f"{name}: {worst_f:.3f} Hz (Δ{peak_dev:+.3f} Hz from {nominal_hz:.0f} Hz nominal)",
        ))
    return events


def _detect_zero_sequence(
    name: str,
    time: np.ndarray,
    data: np.ndarray,
    prefault_mask: np.ndarray,
    cycle_samples: int,
    min_dur: int,
    merge_gap: int,
) -> list[DetectedEvent]:
    rms = _sliding_rms(data, cycle_samples)

    # Zero-sequence baseline: use prefault if available, else 5th percentile
    if np.any(prefault_mask):
        baseline_ref = float(np.median(rms[prefault_mask]))
    else:
        baseline_ref = float(np.percentile(rms[np.isfinite(rms)], 5)) if np.any(np.isfinite(rms)) else 0.0

    # For zero-sequence, threshold is absolute: detect where RMS > threshold
    # Use a threshold that is larger of: 10% of max RMS, or a small absolute value
    abs_threshold = max(baseline_ref * 3.0, float(np.max(rms[np.isfinite(rms)])) * _ZERO_SEQ_THRESHOLD)
    if abs_threshold < 1e-9:
        return []

    mask = rms > abs_threshold
    events: list[DetectedEvent] = []
    for seg in _find_segments(mask, min_dur, merge_gap):
        t_s, t_e = float(time[seg[0]]), float(time[seg[-1]])
        peak_rms = float(np.max(rms[seg]))
        events.append(DetectedEvent(
            event_type=EventType.ZERO_SEQUENCE,
            t_start=t_s,
            t_end=t_e,
            channel=name,
            severity=EventSeverity.MEDIUM,
            magnitude=peak_rms,
            baseline=baseline_ref,
            description=f"{name}: zero-sequence injection {peak_rms:.3g} (peak RMS)",
        ))
    return events


# ──────────────────────────────────────────────────────────────────────────────
# Numpy helpers
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_sample_rate(time: np.ndarray) -> float:
    """Estimate sample rate from median inter-sample interval."""
    if len(time) < 2:
        return 0.0
    diffs = np.diff(time[:min(200, len(time))])
    valid = diffs[(diffs > 0) & np.isfinite(diffs)]
    if len(valid) == 0:
        return 0.0
    return 1.0 / float(np.median(valid))


def _build_prefault_mask(
    time: np.ndarray,
    trigger_time_s: float | None,
    cycle_samples: int,
) -> np.ndarray:
    """Return a boolean mask True for samples in the pre-fault baseline window.

    Pre-fault: at least 2 cycles before trigger. If the window would be too
    short (<2 cycles), fall back to the first 25% of the recording.
    """
    n = len(time)
    mask = np.zeros(n, dtype=bool)

    if trigger_time_s is not None:
        # 2 cycles safety margin before trigger
        safety = 2 * cycle_samples / max(1.0, _estimate_sample_rate(time))
        cutoff = trigger_time_s - safety
        pre = np.flatnonzero(time <= cutoff)
        if len(pre) >= 2 * cycle_samples:
            mask[pre] = True
            return mask

    # Fallback: first 25%
    end = max(2 * cycle_samples, n // 4)
    mask[:end] = True
    return mask


def _sliding_rms(data: np.ndarray, window: int) -> np.ndarray:
    """Compute a causal sliding-window RMS using cumulative sum (O(N)).

    Returns an array of the same length as *data*. Values at the start where
    the window is not full use a shorter growing window.
    """
    w = max(1, window)
    d2 = data.astype(np.float64) ** 2
    # Replace NaN/Inf with zero for the cumsum (they won't be in a real signal)
    d2 = np.where(np.isfinite(d2), d2, 0.0)
    cs = np.cumsum(d2)
    # Prepend zero so cs[i] = sum of d2[0:i]
    cs = np.concatenate(([0.0], cs))

    rms = np.empty(len(data), dtype=np.float64)
    for i in range(len(data)):
        lo = max(0, i - w + 1)
        count = i - lo + 1
        mean_sq = (cs[i + 1] - cs[lo]) / count
        rms[i] = np.sqrt(max(0.0, mean_sq))

    return rms


def _masked_rms(rms: np.ndarray, mask: np.ndarray) -> float:
    """Return the median of rms values where mask is True."""
    if not np.any(mask):
        vals = rms[np.isfinite(rms)]
        return float(np.median(vals)) if len(vals) > 0 else 0.0
    vals = rms[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) > 0 else 0.0


def _find_segments(
    mask: np.ndarray,
    min_dur: int,
    merge_gap: int,
) -> list[np.ndarray]:
    """Find contiguous True runs in *mask*, merge close runs, filter short ones.

    Returns list of index arrays — each array is the indices of one segment.
    """
    if not np.any(mask):
        return []

    # Find run boundaries
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)  # exclusive

    if len(starts) == 0:
        return []

    # Merge segments whose gap is within merge_gap samples
    merged_starts = [starts[0]]
    merged_ends = [ends[0]]
    for i in range(1, len(starts)):
        if starts[i] - merged_ends[-1] <= merge_gap:
            merged_ends[-1] = ends[i]
        else:
            merged_starts.append(starts[i])
            merged_ends.append(ends[i])

    segments: list[np.ndarray] = []
    for s, e in zip(merged_starts, merged_ends):
        if e - s >= min_dur:
            segments.append(np.arange(s, e, dtype=np.int64))

    return segments


def _voltage_dip_severity(ratio: float) -> "EventSeverity":
    if ratio < 0.50:
        return EventSeverity.HIGH      # > 50% dip
    if ratio < 0.80:
        return EventSeverity.MEDIUM    # 20–50% dip
    return EventSeverity.LOW           # 10–20% dip
