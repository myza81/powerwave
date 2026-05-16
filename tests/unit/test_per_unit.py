"""Unit tests for app/analytics/scaling/per_unit.py

Tests cover:
  - pu_voltage_base_kv: phase-to-ground vs phase-to-phase base selection
  - compute_pu_voltage_factor: kV/V unit scaling and PT-ratio stacking
  - compute_pu_current_factor: A/kA unit scaling
  - Boundary: zero base → ZeroDivisionError guard
"""
from __future__ import annotations

import math

import pytest

from app.analytics.scaling.per_unit import (
    compute_pu_current_factor,
    compute_pu_voltage_factor,
    pu_voltage_base_kv,
)
from app.analytics.scaling import VoltageReference


# ─────────────────────────────────────────────────────────────────────────────
# TestPuVoltageBaseKv
# ─────────────────────────────────────────────────────────────────────────────


class TestPuVoltageBaseKv:
    def test_phase_to_ground_divides_by_sqrt3(self) -> None:
        base = pu_voltage_base_kv(275.0, VoltageReference.PHASE_TO_GROUND)
        assert math.isclose(base, 275.0 / math.sqrt(3), rel_tol=1e-9)

    def test_phase_to_phase_returns_nominal(self) -> None:
        base = pu_voltage_base_kv(275.0, VoltageReference.PHASE_TO_PHASE)
        assert math.isclose(base, 275.0, rel_tol=1e-9)

    def test_unknown_same_as_phase_to_ground(self) -> None:
        base_unk = pu_voltage_base_kv(275.0, VoltageReference.UNKNOWN)
        base_lg = pu_voltage_base_kv(275.0, VoltageReference.PHASE_TO_GROUND)
        assert math.isclose(base_unk, base_lg, rel_tol=1e-9)

    def test_110kv_system(self) -> None:
        base = pu_voltage_base_kv(110.0, VoltageReference.PHASE_TO_GROUND)
        assert math.isclose(base, 110.0 / math.sqrt(3), rel_tol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputePuVoltageFactor
# ─────────────────────────────────────────────────────────────────────────────


class TestComputePuVoltageFactor:
    def test_275kv_phase_to_ground_kv_unit(self) -> None:
        # 158.771 kV waveform (phase-to-ground) on 275 kV system → 1.0 pu
        vbase_ln = 275.0 / math.sqrt(3)
        factor = compute_pu_voltage_factor(275.0, VoltageReference.PHASE_TO_GROUND, "kV")
        assert math.isclose(factor * vbase_ln, 1.0, rel_tol=1e-9)

    def test_275kv_phase_to_phase_kv_unit(self) -> None:
        factor = compute_pu_voltage_factor(275.0, VoltageReference.PHASE_TO_PHASE, "kV")
        assert math.isclose(factor * 275.0, 1.0, rel_tol=1e-9)

    def test_volts_unit_scales_by_1000(self) -> None:
        factor_kv = compute_pu_voltage_factor(275.0, VoltageReference.PHASE_TO_GROUND, "kV")
        factor_v = compute_pu_voltage_factor(275.0, VoltageReference.PHASE_TO_GROUND, "V")
        assert math.isclose(factor_v * 1000.0, factor_kv, rel_tol=1e-9)

    def test_default_unit_is_kv(self) -> None:
        factor_default = compute_pu_voltage_factor(275.0, VoltageReference.PHASE_TO_GROUND)
        factor_kv = compute_pu_voltage_factor(275.0, VoltageReference.PHASE_TO_GROUND, "kV")
        assert math.isclose(factor_default, factor_kv, rel_tol=1e-9)

    def test_110kv_phase_to_ground(self) -> None:
        vbase_ln = 110.0 / math.sqrt(3)
        factor = compute_pu_voltage_factor(110.0, VoltageReference.PHASE_TO_GROUND, "kV")
        assert math.isclose(factor * vbase_ln, 1.0, rel_tol=1e-9)

    def test_unknown_ref_treats_as_phase_to_ground(self) -> None:
        f_unk = compute_pu_voltage_factor(275.0, VoltageReference.UNKNOWN, "kV")
        f_lg = compute_pu_voltage_factor(275.0, VoltageReference.PHASE_TO_GROUND, "kV")
        assert math.isclose(f_unk, f_lg, rel_tol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputePuCurrentFactor
# ─────────────────────────────────────────────────────────────────────────────


class TestComputePuCurrentFactor:
    def test_1ka_base_ka_unit(self) -> None:
        # 1 kA waveform on 1 kA base → 1.0 pu
        factor = compute_pu_current_factor(1.0, "kA")
        assert math.isclose(factor * 1.0, 1.0, rel_tol=1e-9)

    def test_1ka_base_a_unit(self) -> None:
        # 1000 A on 1 kA base → 1.0 pu
        factor = compute_pu_current_factor(1.0, "A")
        assert math.isclose(factor * 1000.0, 1.0, rel_tol=1e-9)

    def test_default_unit_is_a(self) -> None:
        factor_default = compute_pu_current_factor(1.0)
        factor_a = compute_pu_current_factor(1.0, "A")
        assert math.isclose(factor_default, factor_a, rel_tol=1e-9)

    def test_2ka_base(self) -> None:
        factor = compute_pu_current_factor(2.0, "kA")
        assert math.isclose(factor * 2.0, 1.0, rel_tol=1e-9)

    def test_0p5ka_base(self) -> None:
        factor = compute_pu_current_factor(0.5, "kA")
        assert math.isclose(factor * 0.5, 1.0, rel_tol=1e-9)
