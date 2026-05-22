"""Data models for automatically detected power-system events.

A DetectedEvent represents a single anomaly found in a waveform — voltage dip,
overcurrent, frequency deviation, or zero-sequence injection. Events are produced
by the event_detector and consumed by the visualization layer (marker lines) and
the event list panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EventType(str, Enum):
    VOLTAGE_DIP = "voltage_dip"
    VOLTAGE_SWELL = "voltage_swell"
    OVERCURRENT = "overcurrent"
    FREQUENCY_DEVIATION = "frequency_deviation"
    ZERO_SEQUENCE = "zero_sequence"

    @property
    def label(self) -> str:
        return {
            EventType.VOLTAGE_DIP: "V-Dip",
            EventType.VOLTAGE_SWELL: "V-Swell",
            EventType.OVERCURRENT: "OC",
            EventType.FREQUENCY_DEVIATION: "Freq",
            EventType.ZERO_SEQUENCE: "3I0",
        }[self]

    @property
    def color(self) -> str:
        return {
            EventType.VOLTAGE_DIP: "#FF8800",
            EventType.VOLTAGE_SWELL: "#AAFF00",
            EventType.OVERCURRENT: "#FF3333",
            EventType.FREQUENCY_DEVIATION: "#AA44FF",
            EventType.ZERO_SEQUENCE: "#00CCAA",
        }[self]

    @property
    def description(self) -> str:
        return {
            EventType.VOLTAGE_DIP: "Voltage dip",
            EventType.VOLTAGE_SWELL: "Voltage swell",
            EventType.OVERCURRENT: "Overcurrent",
            EventType.FREQUENCY_DEVIATION: "Frequency deviation",
            EventType.ZERO_SEQUENCE: "Zero-sequence injection",
        }[self]


class EventSeverity(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class DetectedEvent:
    """A single automatically detected power-system anomaly.

    Attributes:
        event_type:   Category of the disturbance.
        t_start:      Start time in seconds (same time base as waveform arrays).
        t_end:        End time in seconds. Equal to t_start for instantaneous events.
        channel:      Name of the channel that triggered this detection.
        severity:     HIGH / MEDIUM / LOW based on deviation magnitude.
        magnitude:    Peak deviation value (e.g. 0.85 pu for a 15% voltage dip).
        baseline:     Reference value (pre-fault RMS or nominal) used for ratio.
        description:  Human-readable summary shown in the event list panel.
    """

    event_type: EventType
    t_start: float
    t_end: float
    channel: str
    severity: EventSeverity
    magnitude: float
    baseline: float
    description: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_end - self.t_start)

    @property
    def duration_ms(self) -> float:
        return self.duration_s * 1_000.0

    @property
    def ratio(self) -> float:
        """magnitude / baseline — < 1.0 for dips, > 1.0 for swells/overcurrent."""
        if abs(self.baseline) < 1e-12:
            return 0.0
        return self.magnitude / self.baseline
