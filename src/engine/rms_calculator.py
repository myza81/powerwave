"""
src/engine/rms_calculator.py

Cycle-by-cycle RMS computation for disturbance record analogue channels.

One RMS value is produced per power-system cycle (1/nominal_freq seconds).
The result timestamp is the centre of each cycle window.

Algorithm:
  1. Locate cycle boundary indices via np.searchsorted on the time array.
  2. For each cycle window [i_start, i_end): rms = sqrt(mean(data[i:j]**2)).
  3. Return (t_centres, rms_values) as float64 arrays.

No interpolation or resampling — computes purely from the raw sample grid.
Incomplete final cycle (< 0.5 cycle duration) is discarded.

Architecture: Service layer (engine/) — imports models/ only (LAW 1).
              Never import from ui/ here.
"""

from __future__ import annotations

import numpy as np

from models.channel import AnalogueChannel
from models.disturbance_record import DisturbanceRecord

# ── Module constants ───────────────────────────────────────────────────────────

MIN_SAMPLES_PER_CYCLE: int = 2   # discard windows with fewer samples than this


def compute_cycle_rms(
    time_s: np.ndarray,
    data: np.ndarray,
    nominal_freq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cycle-by-cycle RMS for one channel.

    Args:
        time_s:       1-D float64 array of sample times in seconds from record start.
        data:         1-D float array of raw (scaled) sample values.
        nominal_freq: Nominal power system frequency (50.0 or 60.0 Hz).

    Returns:
        Tuple of (t_centres, rms_values) both float64 1-D arrays.
        t_centres holds the mid-point time (seconds) of each cycle window.
        Returns (empty, empty) if the input is too short for even one cycle.
    """
    if len(time_s) < MIN_SAMPLES_PER_CYCLE or nominal_freq <= 0.0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    cycle_s: float = 1.0 / nominal_freq
    t_start: float = float(time_s[0])
    t_end:   float = float(time_s[-1])

    duration: float = t_end - t_start
    if duration < cycle_s:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Build cycle boundary times: [t_start, t_start+T, t_start+2T, ...]
    n_cycles: int = int(duration / cycle_s)
    boundaries: np.ndarray = t_start + np.arange(n_cycles + 1) * cycle_s

    # Map boundary times → nearest sample indices
    idx_boundaries: np.ndarray = np.searchsorted(time_s, boundaries)
    idx_boundaries = np.clip(idx_boundaries, 0, len(time_s))

    t_centres_list: list[float] = []
    rms_list:       list[float] = []

    for k in range(n_cycles):
        i = int(idx_boundaries[k])
        j = int(idx_boundaries[k + 1])
        if j - i < MIN_SAMPLES_PER_CYCLE:
            continue
        window: np.ndarray = data[i:j].astype(np.float64)
        rms_val: float     = float(np.sqrt(np.mean(window ** 2)))
        t_centre: float    = float((time_s[i] + time_s[j - 1]) / 2.0)
        t_centres_list.append(t_centre)
        rms_list.append(rms_val)

    if not t_centres_list:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    return (
        np.array(t_centres_list, dtype=np.float64),
        np.array(rms_list,       dtype=np.float64),
    )


def compute_rms_for_record(
    record: DisturbanceRecord,
    channel_ids: list[int],
    nominal_freq: float,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Compute cycle-by-cycle RMS for selected channels of one record.

    Args:
        record:       The loaded DisturbanceRecord.
        channel_ids:  List of analogue channel_id values to compute.
        nominal_freq: Nominal frequency override for this file (Hz).

    Returns:
        Dict mapping channel_id → (t_rms_s, rms_values).
        t_rms_s is in seconds from record start (same reference as time_array).
        Channels not found in record.analogue_channels are silently omitted.
    """
    ch_map: dict[int, AnalogueChannel] = {
        ch.channel_id: ch for ch in record.analogue_channels
    }
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    t_raw: np.ndarray = record.time_array

    for ch_id in channel_ids:
        ch = ch_map.get(ch_id)
        if ch is None:
            continue
        n = min(len(t_raw), len(ch.raw_data))
        t_rms, rms_vals = compute_cycle_rms(
            t_raw[:n].astype(np.float64),
            ch.raw_data[:n],
            nominal_freq,
        )
        results[ch_id] = (t_rms, rms_vals)

    return results
