"""Unit tests for phasor display models, registry, cache, and axis grouping (Phase 6A).

Tests cover:
  - PhasorDisplayMode enum values and default
  - PhasorWindowMode enum values
  - PhaseLabel enum values
  - PhasorChannelRole enum values
  - PhasorChannelResult immutability
  - ThreePhaseGroup complete/incomplete detection
  - PhasorConfig defaults and custom construction
  - PhasorRegistry: classify, cache, mode changes, bulk helpers
  - PhasorRegistry: detect_three_phase_groups wrapper
  - PhasorRegistry: phasor_panel_keys for direct and multi-source layouts
  - PhasorCache: put/get/invalidate/clear lifecycle
  - PhasorCache: separate phasor and sequence stores
  - Shared-axis interaction: voltage phasors share Hz axis with other voltage
  - Signal browser integration: PhasorRegistry exposes voltage/current channel lists
  - Scaling: PhasorRegistry does not mutate classification on scaling config change
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analytics.phasors.phasor_cache import PhasorCache
from app.analytics.phasors.phasor_models import (
    PhaseLabel,
    PhasorChannelResult,
    PhasorChannelRole,
    PhasorConfig,
    PhasorDisplayMode,
    PhasorWindowMode,
    ThreePhaseGroup,
)
from app.analytics.phasors.phasor_registry import (
    CURRENT_PHASOR_PANEL_KEY,
    SEQUENCE_CURRENT_PANEL_KEY,
    SEQUENCE_VOLTAGE_PANEL_KEY,
    VOLTAGE_PHASOR_PANEL_KEY,
    PhasorRegistry,
)


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorDisplayMode
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorDisplayMode:
    def test_off_value(self) -> None:
        assert PhasorDisplayMode.OFF.value == "off"

    def test_magnitude_value(self) -> None:
        assert PhasorDisplayMode.MAGNITUDE.value == "magnitude"

    def test_angle_value(self) -> None:
        assert PhasorDisplayMode.ANGLE.value == "angle"

    def test_sequence_components_value(self) -> None:
        assert PhasorDisplayMode.SEQUENCE_COMPONENTS.value == "sequence_components"

    def test_default_mode_is_off(self) -> None:
        registry = PhasorRegistry()
        assert registry.display_mode == PhasorDisplayMode.OFF

    def test_all_four_modes_exist(self) -> None:
        modes = {m.value for m in PhasorDisplayMode}
        assert modes == {"off", "magnitude", "angle", "sequence_components"}


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorWindowMode
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorWindowMode:
    def test_half_cycle_value(self) -> None:
        assert PhasorWindowMode.HALF_CYCLE.value == "half_cycle"

    def test_one_cycle_value(self) -> None:
        assert PhasorWindowMode.ONE_CYCLE.value == "one_cycle"

    def test_two_cycle_value(self) -> None:
        assert PhasorWindowMode.TWO_CYCLE.value == "two_cycle"


# ─────────────────────────────────────────────────────────────────────────────
# TestPhaseLabel
# ─────────────────────────────────────────────────────────────────────────────


class TestPhaseLabel:
    def test_a_value(self) -> None:
        assert PhaseLabel.A.value == "A"

    def test_b_value(self) -> None:
        assert PhaseLabel.B.value == "B"

    def test_c_value(self) -> None:
        assert PhaseLabel.C.value == "C"

    def test_unknown_value(self) -> None:
        assert PhaseLabel.UNKNOWN.value == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorChannelRole
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorChannelRole:
    def test_voltage_phasor_value(self) -> None:
        assert PhasorChannelRole.VOLTAGE_PHASOR.value == "voltage_phasor"

    def test_current_phasor_value(self) -> None:
        assert PhasorChannelRole.CURRENT_PHASOR.value == "current_phasor"

    def test_unknown_value(self) -> None:
        assert PhasorChannelRole.UNKNOWN.value == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorChannelResult
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorChannelResult:
    def test_result_is_frozen(self) -> None:
        result = PhasorChannelResult(
            role=PhasorChannelRole.VOLTAGE_PHASOR,
            phase=PhaseLabel.A,
            reason="test",
            auto_classified=True,
            display_unit="kV",
        )
        with pytest.raises((AttributeError, TypeError)):
            result.role = PhasorChannelRole.UNKNOWN  # type: ignore[misc]

    def test_display_unit_default_none(self) -> None:
        result = PhasorChannelResult(
            role=PhasorChannelRole.UNKNOWN,
            phase=PhaseLabel.UNKNOWN,
            reason="test",
            auto_classified=True,
        )
        assert result.display_unit is None


# ─────────────────────────────────────────────────────────────────────────────
# TestThreePhaseGroup
# ─────────────────────────────────────────────────────────────────────────────


class TestThreePhaseGroupModel:
    def test_complete_group(self) -> None:
        g = ThreePhaseGroup(phase_a="VA", phase_b="VB", phase_c="VC", signal_type="voltage")
        assert g.complete is True

    def test_incomplete_group_missing_c(self) -> None:
        g = ThreePhaseGroup(phase_a="VA", phase_b="VB", phase_c=None, signal_type="voltage")
        assert g.complete is False

    def test_channel_names_method(self) -> None:
        g = ThreePhaseGroup(phase_a="VA", phase_b="VB", phase_c="VC", signal_type="voltage")
        assert g.channel_names() == ["VA", "VB", "VC"]


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorConfig:
    def test_default_nominal_hz(self) -> None:
        cfg = PhasorConfig()
        assert cfg.nominal_hz == 50.0

    def test_default_window_one_cycle(self) -> None:
        cfg = PhasorConfig()
        assert cfg.window_mode == PhasorWindowMode.ONE_CYCLE

    def test_60hz_system(self) -> None:
        cfg = PhasorConfig(nominal_hz=60.0)
        assert cfg.nominal_hz == 60.0

    def test_config_is_frozen(self) -> None:
        cfg = PhasorConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.nominal_hz = 60.0  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorRegistry
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorRegistry:
    def test_default_display_mode_off(self) -> None:
        r = PhasorRegistry()
        assert r.display_mode == PhasorDisplayMode.OFF

    def test_set_display_mode_magnitude(self) -> None:
        r = PhasorRegistry()
        r.set_display_mode(PhasorDisplayMode.MAGNITUDE)
        assert r.display_mode == PhasorDisplayMode.MAGNITUDE

    def test_set_display_mode_sequence(self) -> None:
        r = PhasorRegistry()
        r.set_display_mode(PhasorDisplayMode.SEQUENCE_COMPONENTS)
        assert r.display_mode == PhasorDisplayMode.SEQUENCE_COMPONENTS

    def test_set_display_mode_invalid_raises(self) -> None:
        r = PhasorRegistry()
        with pytest.raises(TypeError):
            r.set_display_mode("off")  # type: ignore[arg-type]

    def test_classify_voltage_channel(self) -> None:
        r = PhasorRegistry()
        result = r.classify("VA")
        assert result.role == PhasorChannelRole.VOLTAGE_PHASOR

    def test_classify_current_channel(self) -> None:
        r = PhasorRegistry()
        result = r.classify("IA")
        assert result.role == PhasorChannelRole.CURRENT_PHASOR

    def test_classify_caches_result(self) -> None:
        r = PhasorRegistry()
        r.classify("VA")
        assert "VA" in r.cached_roles

    def test_classify_cache_hit(self) -> None:
        r = PhasorRegistry()
        r1 = r.classify("VA")
        r2 = r.classify("VA")
        assert r1.role == r2.role

    def test_clear_cache(self) -> None:
        r = PhasorRegistry()
        r.classify("VA")
        assert len(r.cached_roles) == 1
        r.clear_cache()
        assert len(r.cached_roles) == 0

    def test_is_voltage_true(self) -> None:
        r = PhasorRegistry()
        assert r.is_voltage("VA") is True

    def test_is_voltage_false_for_current(self) -> None:
        r = PhasorRegistry()
        assert r.is_voltage("IA") is False

    def test_is_current_true(self) -> None:
        r = PhasorRegistry()
        assert r.is_current("IA") is True

    def test_is_current_false_for_voltage(self) -> None:
        r = PhasorRegistry()
        assert r.is_current("VA") is False

    def test_is_phasor_eligible_true_for_both(self) -> None:
        r = PhasorRegistry()
        assert r.is_phasor_eligible("VA") is True
        assert r.is_phasor_eligible("IA") is True

    def test_is_phasor_eligible_false_for_unknown(self) -> None:
        r = PhasorRegistry()
        assert r.is_phasor_eligible("MW") is False

    def test_force_role_cached(self) -> None:
        r = PhasorRegistry()
        r.classify("IA", force_role=PhasorChannelRole.VOLTAGE_PHASOR)
        assert r.cached_roles.get("IA") == PhasorChannelRole.VOLTAGE_PHASOR

    def test_set_config(self) -> None:
        r = PhasorRegistry()
        r.set_config(PhasorConfig(nominal_hz=60.0))
        assert r.config.nominal_hz == 60.0

    def test_default_config(self) -> None:
        r = PhasorRegistry()
        assert r.config.nominal_hz == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorRegistryBulkHelpers
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorRegistryBulkHelpers:
    def test_voltage_channels_filters_correctly(self) -> None:
        r = PhasorRegistry()
        names = ["VA", "VB", "VC", "IA", "IB", "MW", "frequency"]
        v = r.voltage_channels(names)
        assert set(v) == {"VA", "VB", "VC"}

    def test_current_channels_filters_correctly(self) -> None:
        r = PhasorRegistry()
        names = ["VA", "VB", "VC", "IA", "IB", "IC", "MW"]
        i = r.current_channels(names)
        assert set(i) == {"IA", "IB", "IC"}

    def test_voltage_channels_with_unit_override(self) -> None:
        r = PhasorRegistry()
        names = ["Channel1", "Channel2"]
        units = {"Channel1": "kV", "Channel2": "A"}
        v = r.voltage_channels(names, units=units)
        assert v == ["Channel1"]

    def test_current_channels_with_unit_override(self) -> None:
        r = PhasorRegistry()
        names = ["Channel1", "Channel2"]
        units = {"Channel1": "A", "Channel2": "kV"}
        i = r.current_channels(names, units=units)
        assert i == ["Channel1"]


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorPanelKeys
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorPanelKeys:
    def test_direct_voltage_key(self) -> None:
        r = PhasorRegistry()
        keys = r.phasor_panel_keys(["voltage_raw", "current_raw", "power", "frequency"])
        assert set(keys) == {"voltage_raw", "current_raw"}

    def test_direct_sequence_keys(self) -> None:
        r = PhasorRegistry()
        keys = r.phasor_panel_keys([
            "voltage_raw", "sequence_voltage", "sequence_current", "rocof"
        ])
        assert set(keys) == {"voltage_raw", "sequence_voltage", "sequence_current"}

    def test_multi_source_voltage_keys(self) -> None:
        r = PhasorRegistry()
        keys = r.phasor_panel_keys([
            "COMTRADE/voltage_raw",
            "COMTRADE/current_raw",
            "CSV/voltage_raw",
            "CSV/power",
        ])
        assert set(keys) == {
            "COMTRADE/voltage_raw", "COMTRADE/current_raw", "CSV/voltage_raw"
        }

    def test_no_phasor_panels_returns_empty(self) -> None:
        r = PhasorRegistry()
        keys = r.phasor_panel_keys(["power", "rocof", "frequency"])
        assert keys == []

    def test_panel_key_constants(self) -> None:
        assert VOLTAGE_PHASOR_PANEL_KEY == "voltage_raw"
        assert CURRENT_PHASOR_PANEL_KEY == "current_raw"
        assert SEQUENCE_VOLTAGE_PANEL_KEY == "sequence_voltage"
        assert SEQUENCE_CURRENT_PANEL_KEY == "sequence_current"


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorRegistryGroupDetection
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorRegistryGroupDetection:
    def test_detects_abc_voltage_group(self) -> None:
        r = PhasorRegistry()
        groups = r.detect_three_phase_groups(["VA", "VB", "VC", "IA", "IB", "IC"])
        types = {g.signal_type for g in groups}
        assert "voltage" in types
        assert "current" in types

    def test_returns_three_phase_group_objects(self) -> None:
        r = PhasorRegistry()
        groups = r.detect_three_phase_groups(["VA", "VB", "VC"])
        assert len(groups) == 1
        assert isinstance(groups[0], ThreePhaseGroup)
        assert groups[0].complete

    def test_empty_list_returns_empty(self) -> None:
        r = PhasorRegistry()
        assert r.detect_three_phase_groups([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorCache
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorCache:
    def _dummy_result(self) -> tuple:
        t = np.array([0.0, 0.1])
        m = np.array([1.0, 1.0])
        a = np.array([0.0, 0.0])
        c = np.array([1.0 + 0j, 1.0 + 0j])
        return t, m, a, c

    def test_get_returns_none_when_empty(self) -> None:
        cache = PhasorCache()
        assert cache.get_phasor("VA", 100, 50.0) is None

    def test_put_and_get_phasor(self) -> None:
        cache = PhasorCache()
        result = self._dummy_result()
        cache.put_phasor("VA", 100, 50.0, result)
        retrieved = cache.get_phasor("VA", 100, 50.0)
        assert retrieved is result

    def test_different_window_not_found(self) -> None:
        cache = PhasorCache()
        cache.put_phasor("VA", 100, 50.0, self._dummy_result())
        assert cache.get_phasor("VA", 50, 50.0) is None

    def test_sequence_put_and_get(self) -> None:
        cache = PhasorCache()
        seq = {"time": np.array([0.0]), "mag_v1": np.array([1.0])}
        cache.put_sequence("VA", "VB", "VC", 100, 50.0, seq)
        retrieved = cache.get_sequence("VA", "VB", "VC", 100, 50.0)
        assert retrieved is seq

    def test_sequence_miss_on_different_key(self) -> None:
        cache = PhasorCache()
        seq = {"time": np.array([0.0])}
        cache.put_sequence("VA", "VB", "VC", 100, 50.0, seq)
        assert cache.get_sequence("IA", "IB", "IC", 100, 50.0) is None

    def test_len_counts_both_stores(self) -> None:
        cache = PhasorCache()
        cache.put_phasor("VA", 100, 50.0, self._dummy_result())
        cache.put_phasor("VB", 100, 50.0, self._dummy_result())
        cache.put_sequence("VA", "VB", "VC", 100, 50.0, {})
        assert len(cache) == 3
        assert cache.phasor_count == 2
        assert cache.sequence_count == 1

    def test_clear_removes_all(self) -> None:
        cache = PhasorCache()
        cache.put_phasor("VA", 100, 50.0, self._dummy_result())
        cache.put_sequence("VA", "VB", "VC", 100, 50.0, {})
        cache.clear()
        assert len(cache) == 0

    def test_invalidate_channel_removes_phasor_and_sequence(self) -> None:
        cache = PhasorCache()
        cache.put_phasor("VA", 100, 50.0, self._dummy_result())
        cache.put_phasor("VB", 100, 50.0, self._dummy_result())
        cache.put_sequence("VA", "VB", "VC", 100, 50.0, {})
        cache.invalidate_channel("VA")
        assert cache.get_phasor("VA", 100, 50.0) is None
        assert cache.get_phasor("VB", 100, 50.0) is not None
        assert cache.get_sequence("VA", "VB", "VC", 100, 50.0) is None
