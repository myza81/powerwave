"""Data models for the engineering scaling layer.

All types are frozen to ensure immutability and prevent accidental mutation
of runtime scaling configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EngineeringScalingMode(str, Enum):
    """Global per-canvas waveform scaling mode."""

    RAW = "raw"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    PER_UNIT = "per_unit"


class VoltageReference(str, Enum):
    """Voltage measurement convention used for per-unit normalization.

    Critical because Vpu = V_LN / (Vbase_LL / sqrt(3)) differs from
    Vpu = V_LL / Vbase_LL by a factor of sqrt(3).
    """

    PHASE_TO_GROUND = "phase_to_ground"
    PHASE_TO_PHASE = "phase_to_phase"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class GlobalScalingConfig:
    """Session-level scaling parameters applied as fallback for all channels.

    Per-signal overrides take priority — see ScalingRegistry.
    Default is unity ratios and no per-unit base (RAW pass-through).
    """

    ct_ratio: float = 1.0
    pt_ratio: float = 1.0
    voltage_base_kv: float | None = None
    current_base_ka: float | None = None
    voltage_reference: VoltageReference = VoltageReference.PHASE_TO_GROUND


@dataclass(frozen=True, slots=True)
class SignalScalingConfig:
    """Per-signal scaling override for operator-specified engineering bases.

    Any field set to None falls back to the GlobalScalingConfig default.
    """

    ct_ratio: float | None = None
    pt_ratio: float | None = None
    voltage_base_kv: float | None = None
    current_base_ka: float | None = None
    voltage_reference: VoltageReference | None = None


@dataclass(frozen=True, slots=True)
class ScalingResult:
    """Resolved scaling factor and display metadata for one channel.

    factor:      Multiply raw array by this to get display values.
    display_unit: String label for the axis (e.g. "kV", "pu").
    description:  Human-readable description for debugging/logging.
    configured:   False when a required base (e.g. voltage_base_kv) is missing.
                  Callers must NOT display per-unit labels for unconfigured channels.
    """

    factor: float
    display_unit: str
    description: str
    configured: bool
