"""Data models for phasor and sequence component analytics — Phase 6A.

Engineering conventions:
  - Magnitudes are RMS (peak / sqrt(2)), never peak, never dBm.
  - Angles are in degrees, range [-180, 180], relative to DFT window reference.
  - Sequence components use standard Fortescue convention (positive/negative/zero).
  - Display units follow engineering scaling: kV, V, A, kA, pu — never auto-scaled.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PhasorDisplayMode(Enum):
    """Session-level phasor visualization mode.

    OFF                — no phasor or sequence overlays rendered.
    MAGNITUDE          — magnitude envelopes rendered alongside raw waveforms.
    ANGLE              — phase angle traces rendered alongside raw waveforms.
    SEQUENCE_COMPONENTS — positive/negative/zero sequence magnitude panels rendered.
    """

    OFF = "off"
    MAGNITUDE = "magnitude"
    ANGLE = "angle"
    SEQUENCE_COMPONENTS = "sequence_components"


class PhasorWindowMode(Enum):
    """Engineering phasor measurement window selection.

    Mirrors RMSWindowMode philosophy: window is expressed in power cycles,
    not samples, so that the same mode works correctly at any sample rate.
    """

    HALF_CYCLE = "half_cycle"
    ONE_CYCLE = "one_cycle"
    TWO_CYCLE = "two_cycle"


class PhaseLabel(Enum):
    """Identified power system phase for a waveform channel.

    A/B/C: standard ABC (or RYB mapped to A/B/C internally).
    UNKNOWN: heuristic could not determine phase; operator override required.
    """

    A = "A"
    B = "B"
    C = "C"
    UNKNOWN = "unknown"


class PhasorChannelRole(Enum):
    """Classification result for a waveform channel in phasor analytics.

    VOLTAGE_PHASOR  — channel is a voltage waveform eligible for phasor extraction.
    CURRENT_PHASOR  — channel is a current waveform eligible for phasor extraction.
    UNKNOWN         — channel role is not determinable from metadata/heuristics.
    """

    VOLTAGE_PHASOR = "voltage_phasor"
    CURRENT_PHASOR = "current_phasor"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PhasorChannelResult:
    """Immutable classification result for one channel.

    Attributes:
        role:            Classified phasor role.
        phase:           Identified phase label (may be UNKNOWN).
        reason:          Short machine-readable classification reason.
        auto_classified: True = heuristic result; False = operator override.
        display_unit:    Appropriate display unit for this role, or None.
    """

    role: PhasorChannelRole
    phase: PhaseLabel
    reason: str
    auto_classified: bool
    display_unit: str | None = None


@dataclass(frozen=True)
class PhasorConfig:
    """Configuration for phasor extraction.

    Attributes:
        nominal_hz:   Power system nominal frequency (50 or 60 Hz).
        window_mode:  Engineering window for DFT computation.
    """

    nominal_hz: float = 50.0
    window_mode: PhasorWindowMode = PhasorWindowMode.ONE_CYCLE


@dataclass(frozen=True, slots=True)
class ThreePhaseGroup:
    """Three channels identified as a balanced three-phase set.

    phase_a/b/c hold the channel names. The group is eligible for
    symmetrical component computation when all three are non-None.
    """

    phase_a: str | None
    phase_b: str | None
    phase_c: str | None
    signal_type: str  # "voltage" or "current"
    prefix: str = ""  # common name prefix, e.g. "V" for VA/VB/VC

    @property
    def complete(self) -> bool:
        """True when all three phases are identified."""
        return (
            self.phase_a is not None
            and self.phase_b is not None
            and self.phase_c is not None
        )

    def channel_names(self) -> list[str]:
        """Return [A, B, C] channel names; may include None values."""
        return [self.phase_a, self.phase_b, self.phase_c]  # type: ignore[list-item]
