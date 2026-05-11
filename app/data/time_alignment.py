"""Time alignment utilities for mixed-source disturbance display.

All functions are pure (no side effects) and return float64 NumPy arrays.
Original timestamps are never mutated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_display_time_seconds(
    timestamps,
    reference_start=None,
    offset_seconds: float = 0.0,
) -> np.ndarray:
    """Convert timestamps to float64 seconds relative to reference_start.

    Args:
        timestamps:      Array-like of numeric seconds, int, or datetime-like objects.
        reference_start: Origin for the output time axis.
                         - Numeric input: must also be numeric (subtracted from values).
                         - Datetime input: any pandas-parseable datetime.
                         - None: uses timestamps[0] as origin.
        offset_seconds:  Additional scalar shift applied to every output value.

    Returns:
        np.ndarray of dtype float64. Empty input returns empty array.

    Notes:
        Input arrays are not mutated. datetime parsing uses pandas for robustness.
    """
    arr = np.asarray(timestamps)

    if arr.size == 0:
        return np.empty(0, dtype=np.float64)

    if np.issubdtype(arr.dtype, np.number):
        float_arr = arr.astype(np.float64).copy()
        if reference_start is not None:
            float_arr -= float(reference_start)
        return float_arr + offset_seconds

    # Datetime-like branch — delegate parsing to pandas
    dt_index = pd.to_datetime(arr)
    ref = dt_index[0] if reference_start is None else pd.Timestamp(reference_start)
    seconds = (dt_index - ref).total_seconds().to_numpy(dtype=np.float64)
    return seconds + offset_seconds
