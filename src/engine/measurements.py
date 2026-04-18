"""
src/engine/measurements.py

Point-in-time measurement utilities for cursor readouts.

Architecture: Service layer (engine/) — imports models/ only.
              Never import from ui/ (LAW 1).
"""

from __future__ import annotations

import numpy as np

from models.channel import AnalogueChannel
from models.disturbance_record import DisturbanceRecord

# ── Module constants ───────────────────────────────────────────────────────────

_MS_IN_S:  float = 1_000.0
_MIN_IN_S: float = 1.0 / 60.0


def display_to_raw_s(record: DisturbanceRecord, t_display: float) -> float:
    """Convert a display-unit time value to raw seconds from record start.

    The display unit is determined by the '_time_axis_label' attribute set
    on the record by ``engine.decimator.prepare_display_data()``.

    Args:
        record:    The loaded DisturbanceRecord.
        t_display: Time value in display units (ms / s / min).

    Returns:
        Time in raw seconds from record start (same units as time_array).
    """
    trigger_offset_s: float = (
        record.trigger_time - record.start_time
    ).total_seconds()
    label: str = getattr(record, '_time_axis_label', 'Time (ms)')

    if 'ms' in label:
        scale = _MS_IN_S
    elif 'min' in label:
        scale = _MIN_IN_S
    else:
        scale = 1.0

    return t_display / scale + trigger_offset_s


def get_value_at_time(
    record: DisturbanceRecord,
    channel: AnalogueChannel,
    time_s: float,
) -> float:
    """Return the channel value at the given raw-seconds time.

    Uses nearest-sample lookup via ``np.searchsorted`` — O(log n), no iteration.
    Clamps to the first or last sample if ``time_s`` is out of range.

    Args:
        record:  The loaded DisturbanceRecord (provides time_array).
        channel: The analogue channel to query (reads raw_data).
        time_s:  Query time in raw seconds from record start.

    Returns:
        Nearest-sample float value, or NaN if the record is empty.
    """
    t = record.time_array
    if len(t) == 0:
        return float('nan')

    idx = int(np.searchsorted(t, time_s))
    idx = int(np.clip(idx, 0, len(t) - 1))
    return float(channel.raw_data[idx])
