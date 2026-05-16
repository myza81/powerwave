"""Compute per-channel engineering scaling factors for visualization.

Applies scaling at the visualization layer only — raw waveform arrays are
never mutated.  Scaling is O(N) multiplication producing a new float64 view.

Supported signal roles: voltage, current.
Power (MW/MVar), frequency, ROCOF, and other roles: pass-through (factor=1.0).
"""
from __future__ import annotations

from app.analytics.scaling.scaling_models import (
    EngineeringScalingMode,
    GlobalScalingConfig,
    ScalingResult,
    SignalScalingConfig,
    VoltageReference,
)
from app.analytics.scaling.per_unit import (
    compute_pu_current_factor,
    compute_pu_voltage_factor,
)
from app.visualization.engineering_display import (
    infer_signal_type,
    normalize_engineering_unit,
)


def compute_scaling_factor(
    channel_name: str,
    unit: str | None,
    mode: EngineeringScalingMode,
    global_config: GlobalScalingConfig,
    per_signal_config: SignalScalingConfig | None = None,
) -> ScalingResult:
    """Return the multiplicative scaling factor and display unit for one channel.

    For non-voltage/current signals the factor is always 1.0 (no scaling).
    For RAW mode the factor is always 1.0 regardless of signal type.

    The ``configured`` flag is False when a required base value (e.g.
    voltage_base_kv for PER_UNIT) is absent.  Callers must not display
    per-unit axis labels for unconfigured channels.
    """
    role = infer_signal_type(channel_name, unit)
    raw_unit = _display_unit(unit, role)

    if mode == EngineeringScalingMode.RAW:
        return ScalingResult(
            factor=1.0,
            display_unit=raw_unit,
            description="raw",
            configured=True,
        )

    if role == "voltage":
        return _voltage_result(mode, unit or "kV", global_config, per_signal_config)
    if role == "current":
        return _current_result(mode, unit or "A", global_config, per_signal_config)

    # MW, MVar, frequency, ROCOF, other — no engineering scaling defined yet
    return ScalingResult(
        factor=1.0,
        display_unit=raw_unit,
        description="no_scaling",
        configured=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Voltage
# ─────────────────────────────────────────────────────────────────────────────


def _voltage_result(
    mode: EngineeringScalingMode,
    raw_unit: str,
    gcfg: GlobalScalingConfig,
    scfg: SignalScalingConfig | None,
) -> ScalingResult:
    pt = _eff_pt(gcfg, scfg)
    vbase = _eff_vbase(gcfg, scfg)
    vref = _eff_vref(gcfg, scfg)
    unit_out = _display_unit(raw_unit, "voltage")

    if mode == EngineeringScalingMode.PRIMARY:
        return ScalingResult(
            factor=pt,
            display_unit=unit_out,
            description=f"primary(PT×{pt:.4g})",
            configured=True,
        )

    if mode == EngineeringScalingMode.SECONDARY:
        factor = 1.0 / pt if pt > 0 else 1.0
        return ScalingResult(
            factor=factor,
            display_unit=unit_out,
            description=f"secondary(÷PT={pt:.4g})",
            configured=True,
        )

    if mode == EngineeringScalingMode.PER_UNIT:
        if vbase is None or vbase <= 0:
            return ScalingResult(
                factor=1.0,
                display_unit=unit_out,
                description="pu_unconfigured",
                configured=False,
            )
        pu_factor = compute_pu_voltage_factor(vbase, vref, raw_unit) * pt
        return ScalingResult(
            factor=pu_factor,
            display_unit="pu",
            description=f"pu(Vbase={vbase:.3g}kV,{vref.value},PT×{pt:.4g})",
            configured=True,
        )

    return ScalingResult(factor=1.0, display_unit=unit_out, description="raw", configured=True)


# ─────────────────────────────────────────────────────────────────────────────
# Current
# ─────────────────────────────────────────────────────────────────────────────


def _current_result(
    mode: EngineeringScalingMode,
    raw_unit: str,
    gcfg: GlobalScalingConfig,
    scfg: SignalScalingConfig | None,
) -> ScalingResult:
    ct = _eff_ct(gcfg, scfg)
    ibase = _eff_ibase(gcfg, scfg)
    unit_out = _display_unit(raw_unit, "current")

    if mode == EngineeringScalingMode.PRIMARY:
        return ScalingResult(
            factor=ct,
            display_unit=unit_out,
            description=f"primary(CT×{ct:.4g})",
            configured=True,
        )

    if mode == EngineeringScalingMode.SECONDARY:
        factor = 1.0 / ct if ct > 0 else 1.0
        return ScalingResult(
            factor=factor,
            display_unit=unit_out,
            description=f"secondary(÷CT={ct:.4g})",
            configured=True,
        )

    if mode == EngineeringScalingMode.PER_UNIT:
        if ibase is None or ibase <= 0:
            return ScalingResult(
                factor=1.0,
                display_unit=unit_out,
                description="pu_unconfigured",
                configured=False,
            )
        pu_factor = compute_pu_current_factor(ibase, raw_unit) * ct
        return ScalingResult(
            factor=pu_factor,
            display_unit="pu",
            description=f"pu(Ibase={ibase:.3g}kA,CT×{ct:.4g})",
            configured=True,
        )

    return ScalingResult(factor=1.0, display_unit=unit_out, description="raw", configured=True)


# ─────────────────────────────────────────────────────────────────────────────
# Config resolution — per-signal overrides global when set
# ─────────────────────────────────────────────────────────────────────────────


def _eff_pt(g: GlobalScalingConfig, s: SignalScalingConfig | None) -> float:
    return float(s.pt_ratio) if s is not None and s.pt_ratio is not None else float(g.pt_ratio)


def _eff_ct(g: GlobalScalingConfig, s: SignalScalingConfig | None) -> float:
    return float(s.ct_ratio) if s is not None and s.ct_ratio is not None else float(g.ct_ratio)


def _eff_vbase(g: GlobalScalingConfig, s: SignalScalingConfig | None) -> float | None:
    if s is not None and s.voltage_base_kv is not None:
        return float(s.voltage_base_kv)
    return float(g.voltage_base_kv) if g.voltage_base_kv is not None else None


def _eff_ibase(g: GlobalScalingConfig, s: SignalScalingConfig | None) -> float | None:
    if s is not None and s.current_base_ka is not None:
        return float(s.current_base_ka)
    return float(g.current_base_ka) if g.current_base_ka is not None else None


def _eff_vref(g: GlobalScalingConfig, s: SignalScalingConfig | None) -> VoltageReference:
    if s is not None and s.voltage_reference is not None:
        return s.voltage_reference
    return g.voltage_reference


def _display_unit(unit: str | None, role: str | None) -> str:
    return normalize_engineering_unit(unit, signal_type=role) or (unit or "?")
