"""
src/engine/decimator.py

Vectorised display decimation (LAW 3 — never render > 4000 pts/channel).

Three strategies:
  decimate_minmax   — min/max envelope via NumPy reshape.  WAVEFORM mode.
                      Preserves AC peaks at all zoom levels.
  decimate_uniform  — uniform stride.  TREND mode (PMU / slow records).
                      Preserves smooth slowly-varying signal shape.
  decimate_digital  — state-change-aware.  Digital channels (all modes).
                      Never loses a transition regardless of decimation ratio.

Public entry point:
  prepare_display_data(record, t_display) — decorates every channel with
      _display_t / _display_d arrays ready for pg.PlotDataItem.setData().
      Must be called on the background thread before record_loaded is emitted.

Architecture: Service layer (engine/) — imports models/ only (LAW 1).
"""

from __future__ import annotations

import numpy as np

from models.channel import AnalogueChannel, DigitalChannel
from models.disturbance_record import DisturbanceRecord

# ── Module constants ──────────────────────────────────────────────────────────

MAX_ANALOGUE_POINTS: int = 2000   # per channel, per render
MAX_DIGITAL_POINTS:  int = 500    # per channel, per render

# Time-unit thresholds (seconds) — mirrors channel_canvas logic exactly
TREND_MINUTES_THRESHOLD: float = 60.0


# ── Core decimation functions ─────────────────────────────────────────────────

def decimate_minmax(
    time_array: np.ndarray,
    data_array: np.ndarray,
    max_points: int = MAX_ANALOGUE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope decimation using NumPy reshape — no Python loops.

    Each bucket contributes two output points (the sample at the minimum and
    the sample at the maximum within that bucket, in chronological order) so
    AC waveform peaks are preserved at all zoom levels.

    Args:
        time_array: 1-D float64 time values.
        data_array: 1-D numeric data values (same length).
        max_points: Maximum output points (must be even; default 2000).

    Returns:
        Tuple ``(t_out, d_out)`` with ``len(t_out) <= max_points``.
    """
    n = len(time_array)
    if n <= max_points:
        return time_array, data_array.astype(np.float64)

    bucket_size = max(1, n // (max_points // 2))
    n_buckets   = n // bucket_size
    n_use       = n_buckets * bucket_size

    # Reshape into 2-D: (n_buckets × bucket_size) — no Python loop
    d_2d = data_array[:n_use].astype(np.float64).reshape(n_buckets, bucket_size)
    t_2d = time_array[:n_use].reshape(n_buckets, bucket_size)

    row_idx = np.arange(n_buckets)
    min_idx = np.argmin(d_2d, axis=1)   # shape (n_buckets,)
    max_idx = np.argmax(d_2d, axis=1)   # shape (n_buckets,)

    # Interleave min/max in chronological order
    min_before_max = min_idx < max_idx

    t_out = np.empty(n_buckets * 2, dtype=np.float64)
    d_out = np.empty(n_buckets * 2, dtype=np.float64)

    t_out[0::2] = np.where(
        min_before_max, t_2d[row_idx, min_idx], t_2d[row_idx, max_idx]
    )
    t_out[1::2] = np.where(
        min_before_max, t_2d[row_idx, max_idx], t_2d[row_idx, min_idx]
    )
    d_out[0::2] = np.where(
        min_before_max, d_2d[row_idx, min_idx], d_2d[row_idx, max_idx]
    )
    d_out[1::2] = np.where(
        min_before_max, d_2d[row_idx, max_idx], d_2d[row_idx, min_idx]
    )

    return t_out, d_out


def decimate_uniform(
    time_array: np.ndarray,
    data_array: np.ndarray,
    max_points: int = MAX_ANALOGUE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform stride decimation for TREND data — no Python loops.

    Preserves the smooth shape of slowly-varying signals (PMU magnitude,
    frequency, MW, MVAr).  Not suitable for AC waveforms — use
    ``decimate_minmax`` there.

    Args:
        time_array: 1-D float64 time values.
        data_array: 1-D numeric data values (same length).
        max_points: Maximum output points (default 2000).

    Returns:
        Tuple ``(t_out, d_out)`` with ``len(t_out) <= max_points``.
    """
    n = len(time_array)
    if n <= max_points:
        return time_array, data_array.astype(np.float64)
    step    = max(1, n // max_points)
    indices = np.arange(0, n, step)
    return time_array[indices], data_array[indices].astype(np.float64)


def decimate_digital(
    time_array: np.ndarray,
    data_array: np.ndarray,
    max_points: int = MAX_DIGITAL_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """State-change-preserving decimation for digital channels.

    Keeps every state transition plus a uniform grid of background samples.
    Never drops a transition regardless of decimation ratio.

    Args:
        time_array: 1-D float64 time values.
        data_array: 1-D integer/bool data values (same length).
        max_points: Maximum output points (default 500).

    Returns:
        Tuple ``(t_out, d_out)`` with all transitions preserved.
    """
    n = len(time_array)
    if n <= max_points:
        return time_array, data_array.astype(np.float64)

    # All indices where the digital state changes
    changes = np.where(np.diff(data_array.astype(np.int8)) != 0)[0] + 1

    # Uniform background grid
    step    = max(1, n // max_points)
    uniform = np.arange(0, n, step)

    # Union — always include first and last sample
    keep = np.union1d(np.union1d(uniform, changes), np.array([0, n - 1]))
    keep = keep[keep < n]

    return time_array[keep], data_array[keep].astype(np.float64)


# ── Display time array helper ─────────────────────────────────────────────────

def _build_t_display(record: DisturbanceRecord) -> tuple[np.ndarray, str]:
    """Compute the time array in display units (ms / s / min).

    WAVEFORM: trigger-centred milliseconds.
    TREND ≤ 60 s: trigger-centred seconds.
    TREND > 60 s: trigger-centred minutes.

    Args:
        record: The DisturbanceRecord to compute for.

    Returns:
        Tuple ``(t_display, label)`` where label is the axis unit string.
    """
    trigger_offset_s = (
        record.trigger_time - record.start_time
    ).total_seconds()

    is_trend = record.display_mode == 'TREND'

    if is_trend:
        t_raw    = record.time_array - trigger_offset_s
        duration = float(t_raw[-1] - t_raw[0]) if len(t_raw) > 1 else 0.0
        if duration > TREND_MINUTES_THRESHOLD:
            return t_raw / 60.0, 'Time (min)'
        return t_raw, 'Time (s)'

    return (record.time_array - trigger_offset_s) * 1000.0, 'Time (ms)'


# ── Public entry point ────────────────────────────────────────────────────────

def prepare_display_data(record: DisturbanceRecord) -> DisturbanceRecord:
    """Pre-compute decimated display arrays for every channel in ``record``.

    Must be called on the background (parse) thread before ``record_loaded``
    is emitted (LAW 2).  The UI thread then only calls ``setData()`` — zero
    computation on the UI thread.

    Sets on each channel:
        ch._display_t (np.ndarray float64) — time in display units
        ch._display_d (np.ndarray float64) — data values

    Also sets on the record:
        record._t_display      (np.ndarray float64) — full time array
        record._time_axis_label (str)                — axis label

    Args:
        record: Freshly parsed DisturbanceRecord with ``raw_data`` populated.

    Returns:
        The same ``record`` (mutated in place), for chaining convenience.
    """
    if not record.time_array.size:
        return record

    t_display, label = _build_t_display(record)
    record._t_display       = t_display       # type: ignore[attr-defined]
    record._time_axis_label = label           # type: ignore[attr-defined]

    is_trend = record.display_mode == 'TREND'

    for ch in record.analogue_channels:
        if not ch.visible or not ch.raw_data.size:
            continue
        n = min(len(t_display), len(ch.raw_data))
        t = t_display[:n]
        d = ch.raw_data[:n]
        if is_trend:
            ch._display_t, ch._display_d = decimate_uniform(t, d, MAX_ANALOGUE_POINTS)   # type: ignore[attr-defined]
        else:
            ch._display_t, ch._display_d = decimate_minmax(t, d, MAX_ANALOGUE_POINTS)    # type: ignore[attr-defined]

    for ch in record.digital_channels:
        if not ch.visible or not ch.data.size:
            continue
        n = min(len(t_display), len(ch.data))
        t = t_display[:n]
        d = ch.data[:n]
        ch._display_t, ch._display_d = decimate_digital(t, d, MAX_DIGITAL_POINTS)        # type: ignore[attr-defined]
        # Flag channels whose state never changes — viewport update can skip them
        ch._display_is_static = bool(len(np.unique(d)) == 1)                             # type: ignore[attr-defined]

    return record
