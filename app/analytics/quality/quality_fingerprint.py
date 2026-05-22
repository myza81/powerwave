"""Data quality fingerprint — silent per-channel health checks on file load.

Checks performed per channel:
  - NaN / missing sample percentage
  - Sample-rate gaps (dt > 2 × median interval)
  - ADC rail clipping (4+ consecutive samples at global min or max)
  - SNR estimate (signal peak vs MAD-based noise floor)
  - DC offset ratio (|mean| / peak)

The overall grade for a channel is the worst of all individual issues.
Record-level grade is the worst channel grade.
"""
from __future__ import annotations

import dataclasses
import math
from enum import Enum

import numpy as np


class QualityGrade(str, Enum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"

    @property
    def color(self) -> str:
        return {"ok": "#55CC55", "warn": "#FFAA33", "error": "#FF5555"}[self.value]

    def worse_than(self, other: "QualityGrade") -> bool:
        _rank = {QualityGrade.OK: 0, QualityGrade.WARN: 1, QualityGrade.ERROR: 2}
        return _rank[self] > _rank[other]


def _max_grade(*grades: QualityGrade) -> QualityGrade:
    result = QualityGrade.OK
    for g in grades:
        if g.worse_than(result):
            result = g
    return result


@dataclasses.dataclass
class ChannelQuality:
    name: str
    grade: QualityGrade
    nan_pct: float          # 0-100
    gap_count: int          # number of timing gaps > 2× median dt
    clip_count: int         # number of clipping runs (4+ consecutive rail samples)
    snr_db: float | None    # None if cannot be estimated
    dc_offset_ratio: float  # |mean| / peak, 0-1
    issues: list[str]       # human-readable issue strings

    @property
    def tooltip(self) -> str:
        if not self.issues:
            return f"{self.name}: OK"
        return f"{self.name}:\n" + "\n".join(f"  • {i}" for i in self.issues)


@dataclasses.dataclass
class RecordQuality:
    overall_grade: QualityGrade
    channels: dict[str, ChannelQuality]
    sample_count: int
    duration_s: float


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_clipping_runs(rail_mask: np.ndarray, min_run: int = 4) -> int:
    """Count contiguous True runs of length >= min_run in a boolean mask."""
    if not np.any(rail_mask):
        return 0
    padded = np.concatenate(([False], rail_mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return int(np.sum((ends - starts) >= min_run))


def _check_channel(
    name: str,
    data: np.ndarray,
    time_gaps: np.ndarray,  # pre-computed boolean array for whole record
) -> ChannelQuality:
    issues: list[str] = []
    grade_parts: list[QualityGrade] = []

    n = len(data)

    # ── NaN percentage ────────────────────────────────────────────────────────
    nan_count = int(np.sum(np.isnan(data)))
    nan_pct = nan_count / n * 100 if n > 0 else 0.0
    if nan_pct > 5.0:
        issues.append(f"{nan_pct:.1f}% missing samples")
        grade_parts.append(QualityGrade.ERROR)
    elif nan_pct > 0.1:
        issues.append(f"{nan_pct:.2f}% missing samples")
        grade_parts.append(QualityGrade.WARN)

    # ── Timing gaps (shared across channels) ─────────────────────────────────
    gap_count = int(np.sum(time_gaps))
    if gap_count > 10:
        issues.append(f"{gap_count} timing gaps")
        grade_parts.append(QualityGrade.ERROR)
    elif gap_count > 0:
        issues.append(f"{gap_count} timing gap{'s' if gap_count > 1 else ''}")
        grade_parts.append(QualityGrade.WARN)

    # ── Clipping ──────────────────────────────────────────────────────────────
    clean = data[~np.isnan(data)]
    clip_count = 0
    if len(clean) >= 4:
        d_min, d_max = float(np.min(clean)), float(np.max(clean))
        if d_max > d_min:  # constant channels are not clipped
            at_rail = (data == d_min) | (data == d_max)
            clip_count = _count_clipping_runs(at_rail, min_run=4)
            if clip_count > 0:
                issues.append(f"clipping detected ({clip_count} run{'s' if clip_count > 1 else ''})")
                grade_parts.append(QualityGrade.WARN)

    # ── SNR estimate ──────────────────────────────────────────────────────────
    snr_db: float | None = None
    if len(clean) >= 4:
        peak = float(np.max(np.abs(clean)))
        diffs = np.abs(np.diff(clean))
        noise_floor = float(np.median(diffs)) * math.sqrt(math.pi / 2)
        if noise_floor > 0 and peak > 0:
            snr_db = 20.0 * math.log10(peak / noise_floor)
            if snr_db < 10.0:
                issues.append(f"low SNR ({snr_db:.1f} dB)")
                grade_parts.append(QualityGrade.WARN)

    # ── DC offset ─────────────────────────────────────────────────────────────
    dc_offset_ratio = 0.0
    if len(clean) >= 2:
        peak = float(np.max(np.abs(clean)))
        if peak > 0:
            dc_offset_ratio = abs(float(np.mean(clean))) / peak
            if dc_offset_ratio > 0.3:
                issues.append(f"high DC offset ({dc_offset_ratio:.2f} pu)")
                grade_parts.append(QualityGrade.WARN)

    grade = _max_grade(*grade_parts) if grade_parts else QualityGrade.OK

    return ChannelQuality(
        name=name,
        grade=grade,
        nan_pct=nan_pct,
        gap_count=gap_count,
        clip_count=clip_count,
        snr_db=snr_db,
        dc_offset_ratio=dc_offset_ratio,
        issues=issues,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_quality_fingerprint(
    time: np.ndarray,
    data_by_channel: dict[str, np.ndarray],
) -> RecordQuality:
    """Compute per-channel quality metrics for a loaded record.

    Args:
        time:             1-D time array (seconds).
        data_by_channel:  Channel name → 1-D sample array (same length as time).

    Returns:
        RecordQuality with per-channel results and overall grade.
    """
    n = len(time)
    duration_s = float(time[-1] - time[0]) if n >= 2 else 0.0

    # Pre-compute timing gaps (shared — same time axis for all channels)
    time_gaps = np.zeros(n - 1, dtype=bool) if n >= 2 else np.zeros(0, dtype=bool)
    if n >= 3:
        dt = np.diff(time)
        median_dt = float(np.median(dt))
        if median_dt > 0:
            time_gaps = dt > 2.0 * median_dt

    channels: dict[str, ChannelQuality] = {}
    for name, data in data_by_channel.items():
        arr = np.asarray(data, dtype=float)
        channels[name] = _check_channel(name, arr, time_gaps)

    overall = _max_grade(*(ch.grade for ch in channels.values())) if channels else QualityGrade.OK

    return RecordQuality(
        overall_grade=overall,
        channels=channels,
        sample_count=n,
        duration_s=duration_s,
    )
