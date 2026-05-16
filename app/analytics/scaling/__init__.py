"""Engineering scaling and per-unit normalization for Powerwave.

Public API:
  EngineeringScalingMode  — RAW / PRIMARY / SECONDARY / PER_UNIT
  VoltageReference        — PHASE_TO_GROUND / PHASE_TO_PHASE / UNKNOWN
  GlobalScalingConfig     — session-level CT/PT ratios + voltage/current bases
  SignalScalingConfig     — per-channel override (any field None → fall back)
  ScalingResult           — resolved factor + display_unit + configured flag
  ScalingRegistry         — priority-chain config lookup and management
  compute_scaling_factor  — pure function (usable in tests and tools directly)
  compute_pu_voltage_factor / compute_pu_current_factor — low-level pu math
"""
from app.analytics.scaling.scaling_models import (
    EngineeringScalingMode,
    GlobalScalingConfig,
    ScalingResult,
    SignalScalingConfig,
    VoltageReference,
)
from app.analytics.scaling.engineering_scaling import compute_scaling_factor
from app.analytics.scaling.per_unit import (
    compute_pu_current_factor,
    compute_pu_voltage_factor,
    pu_voltage_base_kv,
)
from app.analytics.scaling.scaling_registry import ScalingRegistry

__all__ = [
    "EngineeringScalingMode",
    "GlobalScalingConfig",
    "ScalingResult",
    "SignalScalingConfig",
    "VoltageReference",
    "compute_scaling_factor",
    "compute_pu_voltage_factor",
    "compute_pu_current_factor",
    "pu_voltage_base_kv",
    "ScalingRegistry",
]
