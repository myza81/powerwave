from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SignalMetadata:
    """Per-channel display and source metadata for Phase D1 mixed-source records.

    Carries information that cannot be stored in AnalogChannel (which is a
    locked data contract) and is needed for display grouping, time alignment,
    and future analytics (RMS, per-unit, phasors, sequence components).

    Electrical reference fields (all optional, backward-compatible):
      electrical_type:  "voltage" | "current" | "power" | "frequency" | "rocof"
      phase_reference:  "phase_ground" | "phase_phase" | "sequence"
      nominal_voltage:  line-to-ground nominal voltage in kV (for per-unit base)
    """

    name: str
    unit: str | None = None
    source: str | None = None
    signal_type: str | None = None
    sampling_rate_hz: float | None = None
    time_offset_seconds: float = 0.0
    display_group: str | None = None
    # Electrical reference — populated by providers/generators; consumed by analytics
    electrical_type: str | None = None
    phase_reference: str | None = None
    nominal_voltage: float | None = None
    # Classification provenance — populated by column classifier or manifest loader
    confidence: float | None = None
    inferred_from: str | None = None
    requires_user_confirmation: bool = False
    # Analytics classification — used by RMS eligibility and future analytics layers
    # Values: "instantaneous" | "rms" | "average" | "calculated" | "telemetry" | "unknown"
    measurement_kind: str | None = None
