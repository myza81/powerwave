"""Lightweight per-record RMS result cache.

Architecture: raw waveform → cached RMS result → viewport decimation → render.

The cache eliminates redundant RMS computation during viewport pan/zoom.
The cache key is (channel_id, window_samples, sample_rate_hz) so that
changing either the window or the source record triggers a recompute.

Invalidation is intentionally simple: per-channel or full clear.  Fine-grained
partial invalidation (e.g. window-size change only) is deferred to future phases.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class _CacheKey(NamedTuple):
    channel_id: str
    window_samples: int
    sample_rate_hz: float


class RMSCache:
    """In-memory store for pre-computed (rms_time, rms_values) pairs.

    Designed to be held per FlexiblePlotCanvas instance and cleared whenever
    a new DisturbanceRecord is loaded into that canvas.
    """

    def __init__(self) -> None:
        self._store: dict[_CacheKey, tuple[np.ndarray, np.ndarray]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get(
        self,
        channel_id: str,
        window_samples: int,
        sample_rate_hz: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return cached (rms_time, rms_values) or None if not present."""
        return self._store.get(_CacheKey(channel_id, window_samples, sample_rate_hz))

    def put(
        self,
        channel_id: str,
        window_samples: int,
        sample_rate_hz: float,
        rms_time: np.ndarray,
        rms_values: np.ndarray,
    ) -> None:
        """Store a pre-computed RMS result keyed by channel + window + rate."""
        key = _CacheKey(channel_id, window_samples, sample_rate_hz)
        self._store[key] = (rms_time, rms_values)

    def invalidate_channel(self, channel_id: str) -> None:
        """Remove all entries for a given channel identifier."""
        keys_to_drop = [k for k in self._store if k.channel_id == channel_id]
        for k in keys_to_drop:
            del self._store[k]

    def clear(self) -> None:
        """Remove all cached entries (call when a new record is loaded)."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, item: object) -> bool:
        return item in self._store
