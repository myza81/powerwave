"""Data models for RMS overlay — display modes, configuration, eligibility results."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RMSDisplayMode(Enum):
    """Global RMS overlay rendering mode.

    OFF       — raw waveform only; no RMS computation or display.
    OVERLAY   — raw waveform rendered alongside its RMS envelope.
    RMS_ONLY  — RMS envelope rendered; raw waveform curves hidden.
    """

    OFF = "off"
    OVERLAY = "overlay"
    RMS_ONLY = "rms_only"


class RMSWindowMode(Enum):
    """Engineering RMS measurement window selection."""

    HALF_CYCLE = "half_cycle"
    ONE_CYCLE = "one_cycle"
    TWO_CYCLE = "two_cycle"
    CUSTOM_SAMPLES = "custom_samples"


@dataclass(frozen=True)
class RMSConfig:
    """Configuration parameters for sliding RMS computation.

    Attributes:
        nominal_frequency_hz: Power system nominal frequency (50 or 60 Hz).
        window_mode:          Explicit engineering RMS window mode.
        custom_window_samples: Sample count used only for CUSTOM_SAMPLES mode.
        cycles_per_window:    Backward-compatible cycle count. New code should
                              prefer window_mode; legacy callers passing ``2``
                              still get two-cycle RMS behavior.
    """

    nominal_frequency_hz: float = 50.0
    window_mode: RMSWindowMode = RMSWindowMode.ONE_CYCLE
    custom_window_samples: int | None = None
    cycles_per_window: int = 1


@dataclass(frozen=True)
class RMSEligibilityResult:
    """Result of eligibility classification for a single channel.

    Attributes:
        eligible:        True if the channel is eligible for RMS overlay.
        reason:          Short machine-readable reason string.
        auto_classified: True = automatic heuristic; False = operator override.
    """

    eligible: bool
    reason: str
    auto_classified: bool
