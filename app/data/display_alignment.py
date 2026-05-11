"""Display alignment utilities for multi-source disturbance sessions.

All functions are pure (no side effects). Original records are never mutated.
Time alignment is virtual — used to compute display offsets, not to resample data.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from app.data.time_alignment import build_display_time_seconds

if TYPE_CHECKING:
    from app.data.multi_source_session import SourceRecord


def determine_reference_start(
    sources: list[SourceRecord],
) -> datetime | float | None:
    """Find the earliest temporal anchor across all sources.

    Checks each source's original_start_time first. Falls back to
    record.timing_info.start_time when original_start_time is None.

    Returns:
        The earliest datetime if any source has a datetime anchor.
        The minimum float if all anchors are numeric.
        None if no anchor exists for any source.
    """
    times_dt: list[datetime] = []
    times_float: list[float] = []

    for src in sources:
        t = src.original_start_time
        if t is None:
            try:
                t = src.record.timing_info.start_time
            except Exception:  # noqa: BLE001
                pass
        if t is None:
            continue
        if isinstance(t, datetime):
            times_dt.append(t)
        elif isinstance(t, (int, float)):
            times_float.append(float(t))

    if times_dt:
        return min(times_dt)
    if times_float:
        return min(times_float)
    return None


def compute_relative_offsets(
    sources: list[SourceRecord],
    reference_start: datetime | float | None,
) -> dict[str, float]:
    """Compute the time offset (seconds) for each source relative to reference_start.

    Each offset = (source.original_start_time - reference_start).
    Sources without a temporal anchor receive offset 0.0.
    Type mismatches (datetime vs float) receive offset 0.0.

    Returns:
        dict mapping source_id → offset_seconds.
    """
    offsets: dict[str, float] = {}

    for src in sources:
        if reference_start is None or src.original_start_time is None:
            offsets[src.source_id] = 0.0
            continue

        t = src.original_start_time
        if isinstance(t, datetime) and isinstance(reference_start, datetime):
            offsets[src.source_id] = (t - reference_start).total_seconds()
        elif isinstance(t, (int, float)) and isinstance(reference_start, (int, float)):
            offsets[src.source_id] = float(t) - float(reference_start)
        else:
            offsets[src.source_id] = 0.0

    return offsets


def build_aligned_display_time(
    source: SourceRecord,
    reference_start: datetime | float | None,
) -> np.ndarray:
    """Build an aligned display time array for a single source.

    The 'time' column in waveform_data is in seconds from that source's own
    start. This function shifts the array so all sources share a common zero
    point (reference_start).

    Args:
        source:          SourceRecord to compute display time for.
        reference_start: Common time origin (from determine_reference_start).

    Returns:
        float64 array of aligned time values in seconds.
        Empty array if 'time' column is absent or waveform_data is empty.
    """
    offsets = compute_relative_offsets([source], reference_start)
    offset = offsets[source.source_id]

    wf = source.record.waveform_data
    if "time" not in wf.columns or wf.empty:
        return np.empty(0, dtype=np.float64)

    timestamps = wf["time"].to_numpy(dtype=np.float64)
    # Timestamps are already relative to source start; just add cross-source offset
    return build_display_time_seconds(timestamps, reference_start=None, offset_seconds=offset)
