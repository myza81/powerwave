from __future__ import annotations

import numpy as np


def decimate_for_display(
    time: np.ndarray,
    data: np.ndarray,
    t_start: float,
    t_end: float,
    max_points: int = 4_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip waveform to [t_start, t_end] and stride-decimate to at most max_points.

    Returns (time_decimated, data_decimated) as float64 arrays.
    Input arrays must be 1-D and equal length.
    t_start > t_end is silently swapped.
    """
    _EMPTY = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

    if time.ndim != 1 or data.ndim != 1:
        raise ValueError(
            f"Both arrays must be 1-D; got time.ndim={time.ndim}, data.ndim={data.ndim}"
        )
    if len(time) != len(data):
        raise ValueError(
            f"Array length mismatch: time has {len(time)} elements, "
            f"data has {len(data)} elements"
        )
    if len(time) == 0:
        return _EMPTY

    if t_start > t_end:
        t_start, t_end = t_end, t_start

    mask = (time >= t_start) & (time <= t_end)
    t_clip = time[mask]
    d_clip = data[mask]

    if len(t_clip) == 0:
        return _EMPTY

    if len(t_clip) <= max_points:
        return t_clip.astype(np.float64), d_clip.astype(np.float64)

    stride = max(1, (len(t_clip) + max_points - 1) // max_points)
    return t_clip[::stride].astype(np.float64), d_clip[::stride].astype(np.float64)
