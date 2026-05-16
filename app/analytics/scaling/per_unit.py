"""Per-unit voltage and current normalization.

Standard per-unit convention (IEC/IEEE):
  Phase-to-phase voltage:  Vpu = V_LL  / Vbase_LL
  Phase-to-ground voltage: Vpu = V_LN  / (Vbase_LL / sqrt(3))
  Current:                 Ipu = I_kA  / Ibase_kA

Example — 275 kV system, phase-to-ground waveform stored in kV:
  Vbase_LN = 275 / sqrt(3) = 158.771 kV
  factor   = 1 / 158.771  ≈ 0.006301
  158.771 kV × factor    = 1.000 pu  ✓
"""
from __future__ import annotations

import math

from app.analytics.scaling.scaling_models import VoltageReference


def pu_voltage_base_kv(
    nominal_voltage_base_kv: float,
    voltage_reference: VoltageReference,
) -> float:
    """Return the effective voltage base in kV for normalization.

    For a 275 kV system:
      PHASE_TO_PHASE  → base = 275.000 kV  (line-to-line)
      PHASE_TO_GROUND → base = 158.771 kV  (275 / sqrt(3))
      UNKNOWN         → treated as PHASE_TO_GROUND (conservative default)
    """
    if nominal_voltage_base_kv <= 0:
        return 1.0
    if voltage_reference == VoltageReference.PHASE_TO_PHASE:
        return nominal_voltage_base_kv
    return nominal_voltage_base_kv / math.sqrt(3.0)


def compute_pu_voltage_factor(
    nominal_voltage_base_kv: float,
    voltage_reference: VoltageReference,
    raw_unit: str = "kV",
) -> float:
    """Return the multiplicative factor: raw_value * factor = pu_value.

    Args:
        nominal_voltage_base_kv: Nominal system line-to-line voltage in kV.
        voltage_reference:       Whether the raw waveform is phase-to-ground
                                 or phase-to-phase.
        raw_unit:                Engineering unit of the raw waveform ('kV' or 'V').
    """
    base_kv = pu_voltage_base_kv(nominal_voltage_base_kv, voltage_reference)
    if base_kv <= 0:
        return 1.0
    raw_scale_kv = _unit_to_kv_scale(raw_unit)
    return raw_scale_kv / base_kv


def compute_pu_current_factor(
    current_base_ka: float,
    raw_unit: str = "A",
) -> float:
    """Return the multiplicative factor: raw_value * factor = pu_value.

    Args:
        current_base_ka: Rated or base current in kA.
        raw_unit:        Engineering unit of the raw waveform ('A' or 'kA').
    """
    if current_base_ka <= 0:
        return 1.0
    raw_scale_ka = _unit_to_ka_scale(raw_unit)
    return raw_scale_ka / current_base_ka


# ─────────────────────────────────────────────────────────────────────────────
# Internal unit converters
# ─────────────────────────────────────────────────────────────────────────────


def _unit_to_kv_scale(unit: str) -> float:
    """Return multiplier to convert from *unit* to kV."""
    u = (unit or "").strip().lower()
    if u == "v":
        return 1e-3
    return 1.0  # kV or unrecognised → treat as kV


def _unit_to_ka_scale(unit: str) -> float:
    """Return multiplier to convert from *unit* to kA."""
    u = (unit or "").strip().lower()
    if u == "a":
        return 1e-3
    return 1.0  # kA or unrecognised → treat as kA
