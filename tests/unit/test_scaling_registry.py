"""Unit tests for app/analytics/scaling/scaling_registry.py

Tests cover:
  - Default registry computes RAW (factor=1.0) for all modes
  - set_global_config: updates base values used in computation
  - set_signal_config / clear_signal_config: per-channel priority chain
  - clear_all_signal_configs: removes all overrides
  - has_signal_override / per_signal_config accessors
"""
from __future__ import annotations

import math

import pytest

from app.analytics.scaling.scaling_registry import ScalingRegistry
from app.analytics.scaling.scaling_models import (
    EngineeringScalingMode,
    GlobalScalingConfig,
    SignalScalingConfig,
    VoltageReference,
)


# ─────────────────────────────────────────────────────────────────────────────
# Default registry
# ─────────────────────────────────────────────────────────────────────────────


class TestDefaultRegistry:
    def test_raw_voltage_factor_1(self) -> None:
        reg = ScalingRegistry()
        r = reg.compute_scaling_result("VA", "kV", EngineeringScalingMode.RAW)
        assert r.factor == 1.0

    def test_primary_default_pt1_factor_1(self) -> None:
        reg = ScalingRegistry()
        r = reg.compute_scaling_result("VA", "kV", EngineeringScalingMode.PRIMARY)
        assert math.isclose(r.factor, 1.0, rel_tol=1e-9)

    def test_pu_no_base_unconfigured(self) -> None:
        reg = ScalingRegistry()
        r = reg.compute_scaling_result("VA", "kV", EngineeringScalingMode.PER_UNIT)
        assert r.configured is False

    def test_has_no_override_by_default(self) -> None:
        reg = ScalingRegistry()
        assert reg.has_signal_override("VA") is False

    def test_per_signal_config_none_by_default(self) -> None:
        reg = ScalingRegistry()
        assert reg.per_signal_config("VA") is None


# ─────────────────────────────────────────────────────────────────────────────
# Global config updates
# ─────────────────────────────────────────────────────────────────────────────


class TestGlobalConfig:
    def test_set_global_pt_affects_primary(self) -> None:
        reg = ScalingRegistry()
        reg.set_global_config(GlobalScalingConfig(pt_ratio=100.0))
        r = reg.compute_scaling_result("VA", "kV", EngineeringScalingMode.PRIMARY)
        assert math.isclose(r.factor, 100.0, rel_tol=1e-9)

    def test_set_global_vbase_enables_pu(self) -> None:
        reg = ScalingRegistry()
        reg.set_global_config(GlobalScalingConfig(voltage_base_kv=275.0))
        r = reg.compute_scaling_result("VA", "kV", EngineeringScalingMode.PER_UNIT)
        assert r.configured is True
        assert r.display_unit == "pu"

    def test_global_config_accessor(self) -> None:
        reg = ScalingRegistry()
        cfg = GlobalScalingConfig(pt_ratio=50.0)
        reg.set_global_config(cfg)
        assert reg.global_config.pt_ratio == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-signal config
# ─────────────────────────────────────────────────────────────────────────────


class TestPerSignalConfig:
    def test_per_signal_overrides_global_pt(self) -> None:
        reg = ScalingRegistry()
        reg.set_global_config(GlobalScalingConfig(pt_ratio=100.0))
        reg.set_signal_config("VA", SignalScalingConfig(pt_ratio=200.0))
        r = reg.compute_scaling_result("VA", "kV", EngineeringScalingMode.PRIMARY)
        assert math.isclose(r.factor, 200.0, rel_tol=1e-9)

    def test_other_channel_uses_global(self) -> None:
        reg = ScalingRegistry()
        reg.set_global_config(GlobalScalingConfig(pt_ratio=100.0))
        reg.set_signal_config("VA", SignalScalingConfig(pt_ratio=200.0))
        r = reg.compute_scaling_result("VB", "kV", EngineeringScalingMode.PRIMARY)
        assert math.isclose(r.factor, 100.0, rel_tol=1e-9)

    def test_clear_signal_config_restores_global(self) -> None:
        reg = ScalingRegistry()
        reg.set_global_config(GlobalScalingConfig(pt_ratio=100.0))
        reg.set_signal_config("VA", SignalScalingConfig(pt_ratio=200.0))
        reg.clear_signal_config("VA")
        r = reg.compute_scaling_result("VA", "kV", EngineeringScalingMode.PRIMARY)
        assert math.isclose(r.factor, 100.0, rel_tol=1e-9)

    def test_clear_nonexistent_signal_is_safe(self) -> None:
        reg = ScalingRegistry()
        reg.clear_signal_config("NONEXISTENT")  # must not raise

    def test_clear_all_removes_all_overrides(self) -> None:
        reg = ScalingRegistry()
        reg.set_global_config(GlobalScalingConfig(pt_ratio=100.0))
        reg.set_signal_config("VA", SignalScalingConfig(pt_ratio=200.0))
        reg.set_signal_config("VB", SignalScalingConfig(pt_ratio=300.0))
        reg.clear_all_signal_configs()
        r_a = reg.compute_scaling_result("VA", "kV", EngineeringScalingMode.PRIMARY)
        r_b = reg.compute_scaling_result("VB", "kV", EngineeringScalingMode.PRIMARY)
        assert math.isclose(r_a.factor, 100.0, rel_tol=1e-9)
        assert math.isclose(r_b.factor, 100.0, rel_tol=1e-9)

    def test_has_signal_override_true_after_set(self) -> None:
        reg = ScalingRegistry()
        reg.set_signal_config("VA", SignalScalingConfig(pt_ratio=200.0))
        assert reg.has_signal_override("VA") is True

    def test_has_signal_override_false_after_clear(self) -> None:
        reg = ScalingRegistry()
        reg.set_signal_config("VA", SignalScalingConfig(pt_ratio=200.0))
        reg.clear_signal_config("VA")
        assert reg.has_signal_override("VA") is False

    def test_per_signal_config_accessor_returns_config(self) -> None:
        reg = ScalingRegistry()
        scfg = SignalScalingConfig(pt_ratio=200.0)
        reg.set_signal_config("VA", scfg)
        assert reg.per_signal_config("VA") == scfg
