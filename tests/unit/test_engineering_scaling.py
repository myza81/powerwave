"""Unit tests for app/analytics/scaling/engineering_scaling.py

Tests cover:
  - RAW mode: factor=1.0 for all signal types
  - PRIMARY mode: voltage factor=PT, current factor=CT
  - SECONDARY mode: voltage factor=1/PT, current factor=1/CT
  - PER_UNIT mode: voltage and current pu factors
  - PER_UNIT unconfigured: missing base → configured=False, factor=1.0
  - Non-voltage/non-current signals: pass-through in all modes
  - Per-signal override overrides global config
"""
from __future__ import annotations

import math

import pytest

from app.analytics.scaling.engineering_scaling import compute_scaling_factor
from app.analytics.scaling.scaling_models import (
    EngineeringScalingMode,
    GlobalScalingConfig,
    SignalScalingConfig,
    VoltageReference,
)


def _global(
    pt: float = 1.0,
    ct: float = 1.0,
    vbase: float | None = None,
    ibase: float | None = None,
    vref: VoltageReference = VoltageReference.PHASE_TO_GROUND,
) -> GlobalScalingConfig:
    return GlobalScalingConfig(
        pt_ratio=pt,
        ct_ratio=ct,
        voltage_base_kv=vbase,
        current_base_ka=ibase,
        voltage_reference=vref,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAW mode
# ─────────────────────────────────────────────────────────────────────────────


class TestRawMode:
    def test_voltage_raw_factor_1(self) -> None:
        r = compute_scaling_factor("VA", "kV", EngineeringScalingMode.RAW, _global(pt=100.0))
        assert r.factor == 1.0
        assert r.configured is True

    def test_current_raw_factor_1(self) -> None:
        r = compute_scaling_factor("IA", "A", EngineeringScalingMode.RAW, _global(ct=500.0))
        assert r.factor == 1.0

    def test_power_raw_factor_1(self) -> None:
        r = compute_scaling_factor("MW", "MW", EngineeringScalingMode.RAW, _global())
        assert r.factor == 1.0

    def test_raw_description(self) -> None:
        r = compute_scaling_factor("VA", "kV", EngineeringScalingMode.RAW, _global())
        assert r.description == "raw"


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY mode
# ─────────────────────────────────────────────────────────────────────────────


class TestPrimaryMode:
    def test_voltage_factor_is_pt(self) -> None:
        r = compute_scaling_factor("VA", "kV", EngineeringScalingMode.PRIMARY, _global(pt=100.0))
        assert math.isclose(r.factor, 100.0, rel_tol=1e-9)

    def test_current_factor_is_ct(self) -> None:
        r = compute_scaling_factor("IA", "A", EngineeringScalingMode.PRIMARY, _global(ct=500.0))
        assert math.isclose(r.factor, 500.0, rel_tol=1e-9)

    def test_power_factor_1_primary(self) -> None:
        r = compute_scaling_factor("MW", "MW", EngineeringScalingMode.PRIMARY, _global(pt=100.0))
        assert r.factor == 1.0

    def test_frequency_factor_1_primary(self) -> None:
        r = compute_scaling_factor("FREQ", "Hz", EngineeringScalingMode.PRIMARY, _global())
        assert r.factor == 1.0

    def test_configured_true(self) -> None:
        r = compute_scaling_factor("VA", "kV", EngineeringScalingMode.PRIMARY, _global(pt=100.0))
        assert r.configured is True


# ─────────────────────────────────────────────────────────────────────────────
# SECONDARY mode
# ─────────────────────────────────────────────────────────────────────────────


class TestSecondaryMode:
    def test_voltage_factor_is_1_over_pt(self) -> None:
        r = compute_scaling_factor("VA", "kV", EngineeringScalingMode.SECONDARY, _global(pt=100.0))
        assert math.isclose(r.factor, 1.0 / 100.0, rel_tol=1e-9)

    def test_current_factor_is_1_over_ct(self) -> None:
        r = compute_scaling_factor("IA", "A", EngineeringScalingMode.SECONDARY, _global(ct=500.0))
        assert math.isclose(r.factor, 1.0 / 500.0, rel_tol=1e-9)

    def test_pt_ratio_1_secondary_factor_1(self) -> None:
        r = compute_scaling_factor("VA", "kV", EngineeringScalingMode.SECONDARY, _global(pt=1.0))
        assert math.isclose(r.factor, 1.0, rel_tol=1e-9)

    def test_zero_pt_safe_fallback(self) -> None:
        cfg = GlobalScalingConfig(pt_ratio=0.0)
        r = compute_scaling_factor("VA", "kV", EngineeringScalingMode.SECONDARY, cfg)
        assert r.factor == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PER_UNIT mode
# ─────────────────────────────────────────────────────────────────────────────


class TestPerUnitMode:
    def test_voltage_pu_configured(self) -> None:
        r = compute_scaling_factor(
            "VA", "kV", EngineeringScalingMode.PER_UNIT,
            _global(pt=1.0, vbase=275.0),
        )
        assert r.configured is True
        assert r.display_unit == "pu"

    def test_voltage_pu_unconfigured_no_base(self) -> None:
        r = compute_scaling_factor(
            "VA", "kV", EngineeringScalingMode.PER_UNIT,
            _global(vbase=None),
        )
        assert r.configured is False
        assert r.factor == 1.0

    def test_voltage_pu_factor_correct_275kv(self) -> None:
        vbase_ln = 275.0 / math.sqrt(3)
        r = compute_scaling_factor(
            "VA", "kV", EngineeringScalingMode.PER_UNIT,
            _global(pt=1.0, vbase=275.0, vref=VoltageReference.PHASE_TO_GROUND),
        )
        assert math.isclose(r.factor * vbase_ln, 1.0, rel_tol=1e-6)

    def test_current_pu_configured(self) -> None:
        r = compute_scaling_factor(
            "IA", "A", EngineeringScalingMode.PER_UNIT,
            _global(ct=1.0, ibase=1.0),
        )
        assert r.configured is True
        assert r.display_unit == "pu"

    def test_current_pu_unconfigured_no_base(self) -> None:
        r = compute_scaling_factor(
            "IA", "A", EngineeringScalingMode.PER_UNIT,
            _global(ibase=None),
        )
        assert r.configured is False

    def test_current_pu_factor_correct(self) -> None:
        # 1000 A on 1 kA base → factor = 0.001
        r = compute_scaling_factor(
            "IA", "A", EngineeringScalingMode.PER_UNIT,
            _global(ct=1.0, ibase=1.0),
        )
        assert math.isclose(r.factor * 1000.0, 1.0, rel_tol=1e-9)

    def test_power_passthrough_pu_mode(self) -> None:
        r = compute_scaling_factor(
            "MW", "MW", EngineeringScalingMode.PER_UNIT,
            _global(vbase=275.0),
        )
        assert r.factor == 1.0
        assert r.configured is True


# ─────────────────────────────────────────────────────────────────────────────
# Per-signal override
# ─────────────────────────────────────────────────────────────────────────────


class TestPerSignalOverride:
    def test_per_signal_pt_overrides_global(self) -> None:
        gcfg = _global(pt=100.0)
        scfg = SignalScalingConfig(pt_ratio=200.0)
        r = compute_scaling_factor("VA", "kV", EngineeringScalingMode.PRIMARY, gcfg, scfg)
        assert math.isclose(r.factor, 200.0, rel_tol=1e-9)

    def test_per_signal_vbase_overrides_global(self) -> None:
        gcfg = _global(vbase=275.0)
        scfg = SignalScalingConfig(voltage_base_kv=110.0)
        r_per = compute_scaling_factor(
            "VA", "kV", EngineeringScalingMode.PER_UNIT, gcfg, scfg
        )
        r_global = compute_scaling_factor(
            "VA", "kV", EngineeringScalingMode.PER_UNIT, gcfg, None
        )
        # Different base → different factor
        assert not math.isclose(r_per.factor, r_global.factor, rel_tol=1e-6)

    def test_none_per_signal_falls_back_to_global(self) -> None:
        gcfg = _global(pt=100.0)
        r = compute_scaling_factor("VA", "kV", EngineeringScalingMode.PRIMARY, gcfg, None)
        assert math.isclose(r.factor, 100.0, rel_tol=1e-9)
