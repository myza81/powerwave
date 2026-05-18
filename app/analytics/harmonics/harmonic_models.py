"""Data models for harmonic analysis — Phase 7.

Engineering conventions:
  - Magnitudes are RMS, never peak.
  - THD is expressed as a fraction (not percent): THD=0.05 means 5%.
  - Harmonic orders are positive integers starting at 1 (fundamental).
  - Window sizes are expressed in power-system cycles for fs-independence.
  - Nominal frequency is 50 or 60 Hz; no auto-detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class HarmonicDisplayMode(Enum):
    """Session-level harmonic visualization mode.

    OFF                — no harmonic rendering.
    HARMONIC_MAGNITUDE — per-order RMS magnitude trends alongside waveforms.
    THD                — time-varying THD trend panel.
    SPECTRUM           — single-window FFT spectrum view (future).
    """

    OFF = "off"
    HARMONIC_MAGNITUDE = "harmonic_magnitude"
    THD = "thd"
    SPECTRUM = "spectrum"


class HarmonicWindowMode(Enum):
    """Engineering window size for harmonic FFT analysis.

    Window size is expressed in power-system cycles so the same mode gives
    correct harmonic bin alignment at any sample rate.

    ONE_CYCLE  — 1 cycle (20 ms @ 50 Hz). Fast but lower frequency resolution.
    TWO_CYCLE  — 2 cycles (40 ms @ 50 Hz). Standard for harmonic analysis.
    FOUR_CYCLE — 4 cycles (80 ms @ 50 Hz). Higher resolution, fewer windows.
    """

    ONE_CYCLE = "one_cycle"
    TWO_CYCLE = "two_cycle"
    FOUR_CYCLE = "four_cycle"


class HarmonicChannelRole(Enum):
    """Classification result for a waveform channel in harmonic analytics.

    VOLTAGE_HARMONIC — voltage waveform eligible for harmonic extraction.
    CURRENT_HARMONIC — current waveform eligible for harmonic extraction.
    UNKNOWN          — channel role not determinable or channel is ineligible
                       (RMS, frequency, power, telemetry, etc.).
    """

    VOLTAGE_HARMONIC = "voltage_harmonic"
    CURRENT_HARMONIC = "current_harmonic"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class HarmonicChannelResult:
    """Immutable classification result for one channel.

    Attributes:
        role:            Classified harmonic role (or UNKNOWN if ineligible).
        reason:          Short machine-readable classification reason string.
        auto_classified: True = heuristic; False = operator override.
        display_unit:    Appropriate display unit, or None for UNKNOWN.
    """

    role: HarmonicChannelRole
    reason: str
    auto_classified: bool
    display_unit: str | None = None


@dataclass(frozen=True)
class HarmonicConfig:
    """Configuration for harmonic extraction.

    Attributes:
        nominal_hz:      Power system nominal frequency (50 or 60 Hz).
        max_order:       Highest harmonic order to extract (inclusive).
                         Orders 1 through max_order are always computed.
        window_mode:     Engineering window size selection.
        window_function: Spectral window shape: 'hann' or 'rectangular'.
                         Hann window recommended to reduce spectral leakage.
        overlap:         Fractional window overlap in [0.0, 1.0).
                         0.5 = 50% overlap (default). 0.0 = no overlap.
    """

    nominal_hz: float = 50.0
    max_order: int = 25
    window_mode: HarmonicWindowMode = HarmonicWindowMode.TWO_CYCLE
    window_function: str = "hann"
    overlap: float = 0.5


@dataclass
class HarmonicResult:
    """Sliding-window harmonic extraction result for one waveform channel.

    Attributes:
        harmonic_time:  Right-aligned time array, shape ``(n_windows,)``.
                        harmonic_time[i] is the timestamp of the last sample
                        in window i — consistent with the phasor right-aligned
                        convention used elsewhere in Powerwave.
        magnitudes:     Mapping ``{harmonic_order: rms_array}``.
                        Each value is a float64 ndarray of shape ``(n_windows,)``
                        giving the RMS magnitude of that harmonic component.
        sample_rate_hz: Sample rate at which extraction was performed.
        nominal_hz:     Nominal power frequency used for bin alignment.
        window_samples: FFT window size in samples.
        hop_samples:    Hop size between successive windows (samples).
    """

    harmonic_time: np.ndarray
    magnitudes: dict[int, np.ndarray] = field(default_factory=dict)
    sample_rate_hz: float = 0.0
    nominal_hz: float = 50.0
    window_samples: int = 0
    hop_samples: int = 0

    @property
    def orders(self) -> list[int]:
        """Sorted list of harmonic orders present in this result."""
        return sorted(self.magnitudes.keys())

    @property
    def n_windows(self) -> int:
        """Number of analysis windows."""
        return len(self.harmonic_time)

    def get_magnitude(self, order: int) -> np.ndarray | None:
        """Return the RMS magnitude array for a given harmonic order, or None."""
        return self.magnitudes.get(order)

    @property
    def fundamental(self) -> np.ndarray | None:
        """RMS magnitude array for the fundamental (H1), or None."""
        return self.magnitudes.get(1)

    @classmethod
    def empty(cls) -> HarmonicResult:
        """Return an empty result (no windows, no data)."""
        return cls(
            harmonic_time=np.empty(0, dtype=np.float64),
            magnitudes={},
            sample_rate_hz=0.0,
            nominal_hz=50.0,
            window_samples=0,
            hop_samples=0,
        )
