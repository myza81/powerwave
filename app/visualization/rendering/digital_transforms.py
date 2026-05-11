"""Pure-NumPy digital channel processing utilities for step-function display.

No Qt, no DisturbanceRecord imports. All functions independently testable
without a display. Used by DigitalEventTimeline for viewport-efficient
binary state rendering.
"""
from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Role keyword sets — per CHANNEL_MAPPING_POLICY §2 and §6
# ─────────────────────────────────────────────────────────────────────────────

_ALARM_KEYWORDS = (
    "commfail", "comm fail", "comm_fail", "_fail", "alarm", "warning", "mcb", "vts",
)
_CB_KEYWORDS = (
    "gcb", "fcb", "cb_r", "cb_y", "cb_b", "52b", "52a", "cb open", "cb close",
    "cb_open", "cb_close",
)
_AR_KEYWORDS = ("reclose", "autoreclose", "ar_", "79ar", "auto reclose")
_INTERTRIP_KEYWORDS = (
    "intertrip", "inter-trip", "50bf", "bf_send", "bf_rec", "teleprotect",
    "direct trip", "remote trip",
)
_TRIP_KEYWORDS = ("trip", "operated", "gen trip", "op_")
_PICKUP_KEYWORDS = ("pickup", "pick up", "_fd", "fault det", "element start")


def digital_role_color(name: str) -> str:
    """Return display color for a digital channel name.

    Uses keyword heuristics from CHANNEL_MAPPING_POLICY §2 in priority order.
    Alarm/supervision exception is evaluated first (always resolves to grey).
    """
    n = name.lower()
    if any(k in n for k in _ALARM_KEYWORDS):
        return "#AAAAAA"
    if any(k in n for k in _CB_KEYWORDS):
        return "#FF8800"
    if any(k in n for k in _AR_KEYWORDS):
        return "#4488FF"
    if any(k in n for k in _INTERTRIP_KEYWORDS):
        return "#FF44FF"
    if any(k in n for k in _TRIP_KEYWORDS):
        return "#FF2222"
    if any(k in n for k in _PICKUP_KEYWORDS):
        return "#FFAA00"
    return "#AAAAAA"


def extract_transitions(
    time: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract only state-transition points from binary digital data.

    Reduces N sampled points to M transition points (M << N for typical
    digital channels). Non-zero values are coerced to 1. Output is suitable
    for clip_digital_to_viewport().

    Raises ValueError for non-1-D or mismatched-length arrays.
    Returns float64 arrays.
    """
    _EMPTY = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

    if time.ndim != 1 or data.ndim != 1:
        raise ValueError(
            f"Both arrays must be 1-D; got time.ndim={time.ndim}, data.ndim={data.ndim}"
        )
    if len(time) != len(data):
        raise ValueError(
            f"Array length mismatch: time has {len(time)}, data has {len(data)}"
        )
    if len(time) == 0:
        return _EMPTY

    d_bin = (data != 0).astype(np.int8)
    changes = np.concatenate([[True], d_bin[1:] != d_bin[:-1]])

    t_out = np.concatenate([time[changes], [time[-1]]]).astype(np.float64)
    d_out = np.concatenate([d_bin[changes], [d_bin[-1]]]).astype(np.float64)
    return t_out, d_out


def clip_digital_to_viewport(
    t_trans: np.ndarray,
    d_trans: np.ndarray,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip pre-extracted transition data to the [t_start, t_end] viewport.

    Adds a carry-state point at t_start so the correct binary state appears at
    the left viewport edge. t_start > t_end is silently swapped.

    Input must be output of extract_transitions(). Returns float64 arrays
    suitable for build_step_series().
    """
    _EMPTY = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

    if len(t_trans) == 0:
        return _EMPTY
    if t_start > t_end:
        t_start, t_end = t_end, t_start
    if t_trans[0] > t_end:
        return _EMPTY

    # All data lies before the viewport — carry final state as a flat line
    if t_trans[-1] <= t_start:
        state = d_trans[-1]
        return (
            np.array([t_start, t_end], dtype=np.float64),
            np.array([state, state], dtype=np.float64),
        )

    # State at the left viewport edge
    before_idx = int(np.searchsorted(t_trans, t_start, side="right")) - 1

    parts_t: list[np.ndarray] = []
    parts_d: list[np.ndarray] = []

    if before_idx >= 0:
        parts_t.append(np.array([t_start], dtype=np.float64))
        parts_d.append(np.array([d_trans[before_idx]], dtype=np.float64))

    in_mask = (t_trans > t_start) & (t_trans <= t_end)
    if np.any(in_mask):
        parts_t.append(t_trans[in_mask].astype(np.float64))
        parts_d.append(d_trans[in_mask].astype(np.float64))

    if not parts_t:
        return _EMPTY

    t_out = np.concatenate(parts_t)
    d_out = np.concatenate(parts_d)
    t_out = np.append(t_out, t_end)
    d_out = np.append(d_out, d_out[-1])
    return t_out, d_out


def build_step_series(
    t: np.ndarray,
    d: np.ndarray,
    y_offset: float = 0.0,
    track_height: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand (t, d) transition data into explicit step-function line segments.

    Each state value is held as a horizontal segment until the next transition.
    Applies y_offset and scales d by track_height for multi-track display.

    Returns (t_step, y_step) as float64 arrays ready for curve.setData().
    Setting curve fillLevel=y_offset fills HIGH-state rectangles.
    """
    if len(t) < 2:
        return t.astype(np.float64), (d * track_height + y_offset).astype(np.float64)

    n_segs = len(t) - 1
    t_step = np.empty(2 * n_segs, dtype=np.float64)
    t_step[0::2] = t[:-1]
    t_step[1::2] = t[1:]

    y_seg = d[:-1] * track_height + y_offset
    y_step = np.repeat(y_seg, 2)

    return t_step, y_step.astype(np.float64)
