"""
src/models/disturbance_record.py

DisturbanceRecord — the single source of truth for one loaded file.
Every COMTRADE / CSV / Excel / PMU-CSV parser produces exactly one
DisturbanceRecord (LAW 5).

Architecture: Data layer — imported by parsers/ and engine/ only.
              Never import from ui/ or engine/ here (LAW 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.models.channel import AnalogueChannel, DigitalChannel

# ── Module-level constants ────────────────────────────────────────────────────

WAVEFORM_THRESHOLD: float = 200.0          # Hz — LAW 9 boundary
VALID_NOMINAL_FREQUENCIES: frozenset[float] = frozenset({50.0, 60.0})


# ── Source Format Constants ───────────────────────────────────────────────────

class SourceFormat:
    """String constants for the source_format field of DisturbanceRecord."""
    COMTRADE_1991 = "COMTRADE_1991"
    COMTRADE_1999 = "COMTRADE_1999"
    COMTRADE_2013 = "COMTRADE_2013"
    CSV           = "CSV"
    EXCEL         = "EXCEL"
    PMU_CSV       = "PMU_CSV"

    ALL: frozenset[str] = frozenset({
        COMTRADE_1991, COMTRADE_1999, COMTRADE_2013, CSV, EXCEL, PMU_CSV,
    })


# ── DisturbanceRecord ─────────────────────────────────────────────────────────

@dataclass
class DisturbanceRecord:
    """Complete parsed contents of one disturbance-record file.

    This is the single source of truth (LAW 5).  Every parser in parsers/
    produces exactly one DisturbanceRecord.  PmuRecord is a subtype for PMU
    CSV files; MergedSession wraps multiple DisturbanceRecords.

    Field ordering: required fields first (no default), then optional fields
    (with defaults), then private cache fields (init=False).

    Validation in __post_init__:
      - nominal_frequency must be 50.0 or 60.0 (ValueError if not)
      - sample_rate must be > 0 (ValueError if not)
      - time_array dtype is coerced to float64
      - display_mode is always auto-derived from sample_rate (LAW 9)
      - _rms_cache and _phasor_cache are initialised as empty dicts
    """

    # ── Required identity fields ──────────────────────────────────────────
    station_name: str
    device_id: str
    start_time: datetime
    trigger_time: datetime
    trigger_sample: int
    sample_rate: float           # Hz — must be > 0
    nominal_frequency: float     # 50.0 or 60.0 only
    source_format: str           # one of SourceFormat constants
    file_path: Path

    # ── Optional fields (with defaults) ──────────────────────────────────
    display_mode: str = "WAVEFORM"
    # Auto-overridden in __post_init__ from sample_rate (LAW 9):
    #   sample_rate >= WAVEFORM_THRESHOLD → "WAVEFORM"
    #   sample_rate <  WAVEFORM_THRESHOLD → "TREND"

    analogue_channels: list[AnalogueChannel] = field(default_factory=list)
    digital_channels: list[DigitalChannel]   = field(default_factory=list)

    time_array: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    # float64, seconds from record start.  NON-UNIFORM across NARI multi-rate
    # section boundaries — build section by section in the parser.

    header_text: str = ""

    bay_names: list[str] = field(default_factory=list)
    # Populated from analogue channel parsing first (BEN32 multi-bay),
    # then used to classify digital channels.  Each entry becomes a named
    # group in the channel panel and multi-bay canvas.

    # ── Private computation caches (not in __init__) ──────────────────────
    _rms_cache: dict = field(init=False, repr=False, default_factory=dict)
    _phasor_cache: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields, coerce array dtype, derive display_mode, init caches."""

        # ── Validate nominal_frequency ────────────────────────────────────
        if self.nominal_frequency not in VALID_NOMINAL_FREQUENCIES:
            raise ValueError(
                f"nominal_frequency must be 50.0 or 60.0 Hz, "
                f"got {self.nominal_frequency!r}"
            )

        # ── Validate sample_rate ──────────────────────────────────────────
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be > 0 Hz, got {self.sample_rate!r}"
            )

        # ── Coerce time_array to float64 (LAW 4 requires native rates) ───
        if not isinstance(self.time_array, np.ndarray):
            self.time_array = np.asarray(self.time_array, dtype=np.float64)
        elif self.time_array.dtype != np.float64:
            self.time_array = self.time_array.astype(np.float64)

        # ── Derive display_mode from sample_rate (LAW 9) ──────────────────
        self.display_mode = (
            "WAVEFORM" if self.sample_rate >= WAVEFORM_THRESHOLD else "TREND"
        )

        # ── Initialise private computation caches ─────────────────────────
        self._rms_cache = {}
        self._phasor_cache = {}

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def n_analogue(self) -> int:
        """Number of analogue channels in this record."""
        return len(self.analogue_channels)

    @property
    def n_digital(self) -> int:
        """Number of digital channels in this record."""
        return len(self.digital_channels)

    @property
    def duration(self) -> float:
        """Record duration in seconds.  0.0 if time_array is empty."""
        if len(self.time_array) < 2:
            return 0.0
        return float(self.time_array[-1] - self.time_array[0])

    def get_analogue_channel(self, channel_id: int) -> Optional[AnalogueChannel]:
        """Return the AnalogueChannel with the given channel_id, or None."""
        for ch in self.analogue_channels:
            if ch.channel_id == channel_id:
                return ch
        return None

    def get_digital_channel(self, channel_id: int) -> Optional[DigitalChannel]:
        """Return the DigitalChannel with the given channel_id, or None."""
        for ch in self.digital_channels:
            if ch.channel_id == channel_id:
                return ch
        return None
