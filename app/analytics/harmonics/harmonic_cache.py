"""Lightweight analytics cache for harmonic extraction results — Phase 7.

Architecture mirrors PhasorCache:
  raw waveform → cached harmonic extraction → viewport decimation → render

Cache key: (channel_id, window_samples, hop_samples, nominal_hz, max_order)

This ensures that changing any extraction parameter (window size, overlap,
nominal frequency, or harmonic order limit) triggers a recompute, while
pure display-mode switches (HARMONIC_MAGNITUDE ↔ THD) reuse the same entry.

Invalidation is per-channel or full clear.  Full clear is called whenever
a new DisturbanceRecord is loaded.
"""
from __future__ import annotations

from typing import NamedTuple

from app.analytics.harmonics.harmonic_models import HarmonicResult


class _HarmonicKey(NamedTuple):
    channel_id: str
    window_samples: int
    hop_samples: int
    nominal_hz: float
    max_order: int


class HarmonicCache:
    """In-memory store for pre-computed harmonic extraction results.

    One instance should be held per-session or per-canvas and cleared
    whenever a new DisturbanceRecord is loaded or the HarmonicConfig changes.
    """

    def __init__(self) -> None:
        self._store: dict[_HarmonicKey, HarmonicResult] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Get / Put
    # ─────────────────────────────────────────────────────────────────────────

    def get(
        self,
        channel_id: str,
        window_samples: int,
        hop_samples: int,
        nominal_hz: float,
        max_order: int,
    ) -> HarmonicResult | None:
        """Return a cached HarmonicResult, or None if not present."""
        return self._store.get(
            _HarmonicKey(channel_id, window_samples, hop_samples, nominal_hz, max_order)
        )

    def put(
        self,
        channel_id: str,
        window_samples: int,
        hop_samples: int,
        nominal_hz: float,
        max_order: int,
        result: HarmonicResult,
    ) -> None:
        """Store a pre-computed HarmonicResult."""
        self._store[
            _HarmonicKey(channel_id, window_samples, hop_samples, nominal_hz, max_order)
        ] = result

    # ─────────────────────────────────────────────────────────────────────────
    # Cache management
    # ─────────────────────────────────────────────────────────────────────────

    def invalidate_channel(self, channel_id: str) -> None:
        """Remove all entries for a given channel."""
        keys = [k for k in self._store if k.channel_id == channel_id]
        for k in keys:
            del self._store[k]

    def clear(self) -> None:
        """Remove all cached entries (call when a new record is loaded)."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    @property
    def entry_count(self) -> int:
        return len(self._store)

    def contains(
        self,
        channel_id: str,
        window_samples: int,
        hop_samples: int,
        nominal_hz: float,
        max_order: int,
    ) -> bool:
        """Return True if a result is cached for the given key."""
        return (
            _HarmonicKey(channel_id, window_samples, hop_samples, nominal_hz, max_order)
            in self._store
        )
