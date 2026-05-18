"""Lightweight analytics cache for phasor extraction results.

Architecture:
  raw waveform → cached phasor extraction → cached sequence components
                                          → viewport decimation → render

Cache key: (channel_id, window_samples, nominal_hz)
This ensures that changing the phasor window or loading a new record
triggers a recompute.

Invalidation is intentionally simple: per-channel or full clear.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class _PhasorKey(NamedTuple):
    channel_id: str
    window_samples: int
    nominal_hz: float


class _SequenceKey(NamedTuple):
    channel_a: str
    channel_b: str
    channel_c: str
    window_samples: int
    nominal_hz: float


# Type alias for a phasor result tuple
PhasorResult = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (phasor_time, magnitude_rms, angle_deg, complex_phasor)

# Type alias for sequence result dict
SequenceResult = dict[str, np.ndarray]


class PhasorCache:
    """In-memory store for pre-computed phasor and sequence component results.

    Designed to be held per-session or per-canvas and cleared whenever
    a new DisturbanceRecord is loaded or the phasor config changes.
    """

    def __init__(self) -> None:
        self._phasor_store: dict[_PhasorKey, PhasorResult] = {}
        self._sequence_store: dict[_SequenceKey, SequenceResult] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Phasor cache
    # ─────────────────────────────────────────────────────────────────────────

    def get_phasor(
        self,
        channel_id: str,
        window_samples: int,
        nominal_hz: float,
    ) -> PhasorResult | None:
        """Return cached phasor result or None."""
        return self._phasor_store.get(
            _PhasorKey(channel_id, window_samples, nominal_hz)
        )

    def put_phasor(
        self,
        channel_id: str,
        window_samples: int,
        nominal_hz: float,
        result: PhasorResult,
    ) -> None:
        """Store a pre-computed phasor result."""
        self._phasor_store[_PhasorKey(channel_id, window_samples, nominal_hz)] = result

    # ─────────────────────────────────────────────────────────────────────────
    # Sequence component cache
    # ─────────────────────────────────────────────────────────────────────────

    def get_sequence(
        self,
        channel_a: str,
        channel_b: str,
        channel_c: str,
        window_samples: int,
        nominal_hz: float,
    ) -> SequenceResult | None:
        """Return cached sequence component result or None."""
        return self._sequence_store.get(
            _SequenceKey(channel_a, channel_b, channel_c, window_samples, nominal_hz)
        )

    def put_sequence(
        self,
        channel_a: str,
        channel_b: str,
        channel_c: str,
        window_samples: int,
        nominal_hz: float,
        result: SequenceResult,
    ) -> None:
        """Store a pre-computed sequence component result."""
        self._sequence_store[
            _SequenceKey(channel_a, channel_b, channel_c, window_samples, nominal_hz)
        ] = result

    # ─────────────────────────────────────────────────────────────────────────
    # Cache management
    # ─────────────────────────────────────────────────────────────────────────

    def invalidate_channel(self, channel_id: str) -> None:
        """Remove all phasor and sequence entries involving a given channel."""
        phasor_keys = [
            k for k in self._phasor_store if k.channel_id == channel_id
        ]
        for k in phasor_keys:
            del self._phasor_store[k]

        seq_keys = [
            k for k in self._sequence_store
            if k.channel_a == channel_id
            or k.channel_b == channel_id
            or k.channel_c == channel_id
        ]
        for k in seq_keys:
            del self._sequence_store[k]

    def clear(self) -> None:
        """Remove all cached entries (call when a new record is loaded)."""
        self._phasor_store.clear()
        self._sequence_store.clear()

    def __len__(self) -> int:
        return len(self._phasor_store) + len(self._sequence_store)

    @property
    def phasor_count(self) -> int:
        return len(self._phasor_store)

    @property
    def sequence_count(self) -> int:
        return len(self._sequence_store)
