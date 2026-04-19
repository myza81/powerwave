"""
src/engine/rms_merger.py

Nearest-neighbour time join for RMS data from multiple files.

Each file contributes one or more RmsChannelData objects.  The merger:
  1. Converts per-channel RMS timestamps to absolute epoch seconds
     using the file's start_time + per-file user offset.
  2. Builds a common time grid from the union of all adjusted timestamps,
     collapsing points within the join tolerance into one grid point.
  3. For each channel, snaps its timestamps onto the common grid via
     nearest-neighbour lookup.  Points outside tolerance become NaN.

Architecture: Service layer (engine/) — imports models/ only (LAW 1).
              Never import from ui/ here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# ── Module constants ───────────────────────────────────────────────────────────

DEFAULT_TOLERANCE_S:  float = 0.010   # 10 ms default snap tolerance
_EPOCH: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Minimum plausible wall-clock year — start_times earlier than this are
# treated as "no valid timestamp" (fallback to relative time).
_MIN_VALID_YEAR: int = 2000


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class RmsChannelData:
    """RMS result for one analogue channel from one file.

    Attributes:
        file_id:    Unique string key for the source file (e.g. stem of path).
        channel_id: channel_id from AnalogueChannel.
        col_name:   Display name used as table/export column header (editable).
        t_from_start: 1-D float64 — RMS centre times in seconds from record start.
        rms:          1-D float64 — RMS values aligned with t_from_start.
        start_epoch:  POSIX timestamp of the record's start_time (float seconds).
                      0.0 when the record has no valid wall-clock timestamp.
    """

    file_id:        str
    channel_id:     int
    col_name:       str
    t_from_start:   np.ndarray       # seconds from record start
    rms:            np.ndarray       # float64 rms values
    start_epoch:    float = 0.0      # POSIX epoch seconds of record start


@dataclass
class MergeResult:
    """Output of merge_rms_channels.

    Attributes:
        t_common:   1-D float64 — common time grid in absolute POSIX seconds
                    (or relative seconds from t_common[0] when no timestamps).
        col_names:  Column header strings, one per channel in data_2d.
        data_2d:    2-D float64 (n_times × n_channels) — NaN where no snap found.
        has_timestamps: True when at least one file had a valid wall-clock time.
        nan_cells:  List of (row_idx, col_idx) pairs where data_2d is NaN.
    """

    t_common:       np.ndarray
    col_names:      list[str]
    data_2d:        np.ndarray
    has_timestamps: bool
    nan_cells:      list[tuple[int, int]] = field(default_factory=list)


# ── Public helpers ─────────────────────────────────────────────────────────────

def start_epoch_from_datetime(dt: datetime) -> float:
    """Convert a datetime to POSIX epoch float; return 0.0 for pre-2000 dates.

    Args:
        dt: The record's start_time (may be timezone-aware or naive UTC).

    Returns:
        POSIX float seconds, or 0.0 if the date is implausibly old.
    """
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        epoch = dt.timestamp()
        if dt.year < _MIN_VALID_YEAR:
            return 0.0
        return float(epoch)
    except (OSError, OverflowError, ValueError):
        return 0.0


def merge_rms_channels(
    channels: list[RmsChannelData],
    offsets:  dict[str, float],
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> MergeResult:
    """Nearest-neighbour join of RMS channels onto a common time grid.

    Args:
        channels:    All RmsChannelData objects to merge (any number of files).
        offsets:     Dict mapping file_id → time offset in seconds to add to
                     that file's timestamps before merging.
        tolerance_s: Maximum allowed gap (seconds) between a channel sample
                     and the nearest common grid point.  Farther → NaN.

    Returns:
        MergeResult with common time grid, column names, and 2-D data array.
    """
    if not channels:
        return MergeResult(
            t_common=np.empty(0, dtype=np.float64),
            col_names=[],
            data_2d=np.empty((0, 0), dtype=np.float64),
            has_timestamps=False,
        )

    # ── Step 1: build per-channel absolute timestamps ─────────────────────────
    has_any_timestamp = any(ch.start_epoch > 0.0 for ch in channels)

    # t_abs[i] = channel's RMS timestamps in absolute seconds (POSIX or relative)
    t_abs_list: list[np.ndarray] = []
    for ch in channels:
        offset = offsets.get(ch.file_id, 0.0)
        if has_any_timestamp and ch.start_epoch > 0.0:
            t_abs = ch.t_from_start + ch.start_epoch + offset
        elif has_any_timestamp:
            # File has no timestamp — treat as starting at t=0 + offset
            t_abs = ch.t_from_start + offset
        else:
            # No file has timestamps — use relative seconds from start
            t_abs = ch.t_from_start + offset
        t_abs_list.append(t_abs)

    # ── Step 2: build common time grid ────────────────────────────────────────
    # Union all timestamps, then collapse within tolerance by rounding to grid.
    all_times: np.ndarray = np.concatenate(t_abs_list) if t_abs_list else np.empty(0)
    if len(all_times) == 0:
        return MergeResult(
            t_common=np.empty(0, dtype=np.float64),
            col_names=[ch.col_name for ch in channels],
            data_2d=np.empty((0, len(channels)), dtype=np.float64),
            has_timestamps=has_any_timestamp,
        )

    all_times_sorted: np.ndarray = np.sort(all_times)

    # Greedy merge: keep a point if it is > tolerance away from the last kept point
    common: list[float] = [float(all_times_sorted[0])]
    for t in all_times_sorted[1:]:
        if t - common[-1] > tolerance_s:
            common.append(float(t))
    t_common: np.ndarray = np.array(common, dtype=np.float64)

    # ── Step 3: snap each channel onto the common grid ────────────────────────
    n_times = len(t_common)
    n_ch    = len(channels)
    data_2d = np.full((n_times, n_ch), np.nan, dtype=np.float64)

    for col_idx, (ch, t_abs) in enumerate(zip(channels, t_abs_list)):
        if len(t_abs) == 0:
            continue
        # For each sample in this channel, find the nearest common grid index
        grid_idx = np.searchsorted(t_common, t_abs)
        grid_idx = np.clip(grid_idx, 0, n_times - 1)

        # Also check the index before for possible closer match
        grid_idx_prev = np.maximum(grid_idx - 1, 0)
        dist_cur  = np.abs(t_common[grid_idx]      - t_abs)
        dist_prev = np.abs(t_common[grid_idx_prev] - t_abs)
        best_idx  = np.where(dist_prev < dist_cur, grid_idx_prev, grid_idx)
        best_dist = np.minimum(dist_cur, dist_prev)

        # Only assign where within tolerance; duplicate assignments keep last
        valid_mask = best_dist <= tolerance_s
        data_2d[best_idx[valid_mask], col_idx] = ch.rms[valid_mask]

    # ── Step 4: locate NaN cells ──────────────────────────────────────────────
    nan_rows, nan_cols = np.where(np.isnan(data_2d))
    nan_cells: list[tuple[int, int]] = list(zip(nan_rows.tolist(), nan_cols.tolist()))

    return MergeResult(
        t_common=t_common,
        col_names=[ch.col_name for ch in channels],
        data_2d=data_2d,
        has_timestamps=has_any_timestamp,
        nan_cells=nan_cells,
    )


def format_time_column(
    t_common: np.ndarray,
    has_timestamps: bool,
) -> list[str]:
    """Format the common time array into display strings for the table.

    Args:
        t_common:       Common time grid (POSIX seconds or relative seconds).
        has_timestamps: True → format as ISO datetime; False → format as seconds.

    Returns:
        List of formatted time strings, one per row in t_common.
    """
    if len(t_common) == 0:
        return []

    if has_timestamps:
        return [
            datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            for t in t_common
        ]
    else:
        t_rel = t_common - t_common[0]
        return [f'{t:.4f}' for t in t_rel]
