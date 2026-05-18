"""Unit tests for harmonic registry and cache — Phase 7.

Tests cover:
  HarmonicRegistry:
    - Default state (mode=OFF, config=default)
    - classify() caches role on first call
    - Repeated classify() returns cached role
    - Cache invalidation via clear_cache()
    - set_display_mode() accepts HarmonicDisplayMode only
    - set_config() updates config without clearing cache
    - Bulk helpers: harmonic_eligible_channels, voltage/current filters
    - is_harmonic_eligible, is_voltage, is_current

  HarmonicCache:
    - get() on empty returns None
    - put() + get() returns the same result
    - Different keys produce distinct entries
    - invalidate_channel() removes only matching entries
    - clear() empties everything
    - len() and entry_count report correctly
    - contains() predicate works correctly

  HarmonicDisplayMode enum:
    - All four values exist
    - Default registry mode is OFF

  HarmonicConfig defaults:
    - nominal_hz = 50.0
    - max_order = 25
    - window_mode = TWO_CYCLE
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analytics.harmonics.harmonic_cache import HarmonicCache
from app.analytics.harmonics.harmonic_models import (
    HarmonicChannelRole,
    HarmonicConfig,
    HarmonicDisplayMode,
    HarmonicResult,
    HarmonicWindowMode,
)
from app.analytics.harmonics.harmonic_registry import HarmonicRegistry
from app.data.signal_metadata import SignalMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _meta(**kwargs) -> SignalMetadata:
    return SignalMetadata(name="test", **kwargs)


def _make_result(n: int = 10) -> HarmonicResult:
    t = np.linspace(0.0, 1.0, n)
    return HarmonicResult(
        harmonic_time=t,
        magnitudes={1: np.ones(n) * 100.0, 3: np.ones(n) * 5.0},
        sample_rate_hz=5000.0,
        nominal_hz=50.0,
        window_samples=200,
        hop_samples=100,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestHarmonicDisplayMode
# ─────────────────────────────────────────────────────────────────────────────


class TestHarmonicDisplayMode:
    def test_off_value(self) -> None:
        assert HarmonicDisplayMode.OFF.value == "off"

    def test_harmonic_magnitude_value(self) -> None:
        assert HarmonicDisplayMode.HARMONIC_MAGNITUDE.value == "harmonic_magnitude"

    def test_thd_value(self) -> None:
        assert HarmonicDisplayMode.THD.value == "thd"

    def test_spectrum_value(self) -> None:
        assert HarmonicDisplayMode.SPECTRUM.value == "spectrum"

    def test_all_four_modes_exist(self) -> None:
        values = {m.value for m in HarmonicDisplayMode}
        assert values == {"off", "harmonic_magnitude", "thd", "spectrum"}


# ─────────────────────────────────────────────────────────────────────────────
# TestHarmonicConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestHarmonicConfig:
    def test_default_nominal_hz(self) -> None:
        assert HarmonicConfig().nominal_hz == 50.0

    def test_default_max_order(self) -> None:
        assert HarmonicConfig().max_order == 25

    def test_default_window_mode(self) -> None:
        assert HarmonicConfig().window_mode == HarmonicWindowMode.TWO_CYCLE

    def test_default_window_function(self) -> None:
        assert HarmonicConfig().window_function == "hann"

    def test_default_overlap(self) -> None:
        assert HarmonicConfig().overlap == 0.5

    def test_custom_config(self) -> None:
        cfg = HarmonicConfig(
            nominal_hz=60.0,
            max_order=13,
            window_mode=HarmonicWindowMode.FOUR_CYCLE,
            overlap=0.0,
        )
        assert cfg.nominal_hz == 60.0
        assert cfg.max_order == 13
        assert cfg.window_mode == HarmonicWindowMode.FOUR_CYCLE
        assert cfg.overlap == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TestHarmonicRegistryDefaultState
# ─────────────────────────────────────────────────────────────────────────────


class TestHarmonicRegistryDefaultState:
    def test_default_display_mode_is_off(self) -> None:
        registry = HarmonicRegistry()
        assert registry.display_mode == HarmonicDisplayMode.OFF

    def test_default_config_is_harmonic_config(self) -> None:
        registry = HarmonicRegistry()
        assert isinstance(registry.config, HarmonicConfig)

    def test_cached_roles_initially_empty(self) -> None:
        registry = HarmonicRegistry()
        assert registry.cached_roles == {}

    def test_custom_initial_mode(self) -> None:
        registry = HarmonicRegistry(display_mode=HarmonicDisplayMode.THD)
        assert registry.display_mode == HarmonicDisplayMode.THD

    def test_custom_initial_config(self) -> None:
        cfg = HarmonicConfig(nominal_hz=60.0)
        registry = HarmonicRegistry(config=cfg)
        assert registry.config.nominal_hz == 60.0


# ─────────────────────────────────────────────────────────────────────────────
# TestHarmonicRegistryClassify
# ─────────────────────────────────────────────────────────────────────────────


class TestHarmonicRegistryClassify:
    def test_classify_voltage_channel(self) -> None:
        registry = HarmonicRegistry()
        result = registry.classify("VA")
        assert result.role == HarmonicChannelRole.VOLTAGE_HARMONIC

    def test_classify_current_channel(self) -> None:
        registry = HarmonicRegistry()
        result = registry.classify("IA")
        assert result.role == HarmonicChannelRole.CURRENT_HARMONIC

    def test_classify_ineligible_channel(self) -> None:
        registry = HarmonicRegistry()
        result = registry.classify("VA_RMS")
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_classify_caches_role(self) -> None:
        registry = HarmonicRegistry()
        registry.classify("VA")
        assert "VA" in registry.cached_roles
        assert registry.cached_roles["VA"] == HarmonicChannelRole.VOLTAGE_HARMONIC

    def test_repeated_classify_returns_same_result(self) -> None:
        registry = HarmonicRegistry()
        r1 = registry.classify("VA")
        r2 = registry.classify("VA")
        assert r1.role == r2.role

    def test_clear_cache_removes_entries(self) -> None:
        registry = HarmonicRegistry()
        registry.classify("VA")
        registry.classify("IA")
        registry.clear_cache()
        assert registry.cached_roles == {}

    def test_force_role_stored_in_cache(self) -> None:
        registry = HarmonicRegistry()
        registry.classify("X", force_role=HarmonicChannelRole.VOLTAGE_HARMONIC)
        assert registry.cached_roles["X"] == HarmonicChannelRole.VOLTAGE_HARMONIC

    def test_is_harmonic_eligible_voltage(self) -> None:
        registry = HarmonicRegistry()
        assert registry.is_harmonic_eligible("VA") is True

    def test_is_harmonic_eligible_current(self) -> None:
        registry = HarmonicRegistry()
        assert registry.is_harmonic_eligible("IA") is True

    def test_is_harmonic_eligible_rms_false(self) -> None:
        registry = HarmonicRegistry()
        assert registry.is_harmonic_eligible("VA_RMS") is False

    def test_is_voltage(self) -> None:
        registry = HarmonicRegistry()
        assert registry.is_voltage("VA") is True
        assert registry.is_voltage("IA") is False

    def test_is_current(self) -> None:
        registry = HarmonicRegistry()
        assert registry.is_current("IA") is True
        assert registry.is_current("VA") is False


# ─────────────────────────────────────────────────────────────────────────────
# TestHarmonicRegistryBulkHelpers
# ─────────────────────────────────────────────────────────────────────────────


class TestHarmonicRegistryBulkHelpers:
    def test_harmonic_eligible_channels_filters_correctly(self) -> None:
        registry = HarmonicRegistry()
        names = ["VA", "VB", "VC", "IA", "IB", "IC", "VA_RMS", "Freq", "P_MW"]
        eligible = registry.harmonic_eligible_channels(names)
        assert set(eligible) == {"VA", "VB", "VC", "IA", "IB", "IC"}

    def test_harmonic_eligible_channels_preserves_order(self) -> None:
        registry = HarmonicRegistry()
        names = ["VC", "VB", "VA"]
        eligible = registry.harmonic_eligible_channels(names)
        assert eligible == ["VC", "VB", "VA"]

    def test_voltage_harmonic_channels(self) -> None:
        registry = HarmonicRegistry()
        names = ["VA", "VB", "IA", "IB"]
        assert registry.voltage_harmonic_channels(names) == ["VA", "VB"]

    def test_current_harmonic_channels(self) -> None:
        registry = HarmonicRegistry()
        names = ["VA", "VB", "IA", "IB"]
        assert registry.current_harmonic_channels(names) == ["IA", "IB"]

    def test_units_dict_used_for_classification(self) -> None:
        registry = HarmonicRegistry()
        names = ["Ch1", "Ch2"]
        units = {"Ch1": "kV", "Ch2": "A"}
        eligible = registry.harmonic_eligible_channels(names, units=units)
        assert set(eligible) == {"Ch1", "Ch2"}

    def test_signal_metadata_used_for_classification(self) -> None:
        registry = HarmonicRegistry()
        names = ["X"]
        meta = {"X": _meta(electrical_type="current")}
        eligible = registry.harmonic_eligible_channels(names, signal_metadata=meta)
        assert eligible == ["X"]

    def test_empty_names_returns_empty(self) -> None:
        registry = HarmonicRegistry()
        assert registry.harmonic_eligible_channels([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# TestHarmonicRegistryModeAndConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestHarmonicRegistryModeAndConfig:
    def test_set_display_mode_updates_mode(self) -> None:
        registry = HarmonicRegistry()
        registry.set_display_mode(HarmonicDisplayMode.THD)
        assert registry.display_mode == HarmonicDisplayMode.THD

    def test_set_display_mode_rejects_wrong_type(self) -> None:
        registry = HarmonicRegistry()
        with pytest.raises(TypeError):
            registry.set_display_mode("thd")  # type: ignore[arg-type]

    def test_set_config_updates_config(self) -> None:
        registry = HarmonicRegistry()
        cfg = HarmonicConfig(nominal_hz=60.0, max_order=13)
        registry.set_config(cfg)
        assert registry.config.nominal_hz == 60.0
        assert registry.config.max_order == 13

    def test_set_config_does_not_clear_cache(self) -> None:
        registry = HarmonicRegistry()
        registry.classify("VA")
        registry.set_config(HarmonicConfig(nominal_hz=60.0))
        assert "VA" in registry.cached_roles

    def test_cached_roles_snapshot_is_copy(self) -> None:
        registry = HarmonicRegistry()
        registry.classify("VA")
        snapshot = registry.cached_roles
        snapshot["fake"] = HarmonicChannelRole.UNKNOWN
        assert "fake" not in registry.cached_roles


# ─────────────────────────────────────────────────────────────────────────────
# TestHarmonicCache
# ─────────────────────────────────────────────────────────────────────────────


class TestHarmonicCache:
    def test_get_on_empty_returns_none(self) -> None:
        cache = HarmonicCache()
        assert cache.get("VA", 200, 100, 50.0, 25) is None

    def test_put_then_get_returns_result(self) -> None:
        cache = HarmonicCache()
        result = _make_result()
        cache.put("VA", 200, 100, 50.0, 25, result)
        retrieved = cache.get("VA", 200, 100, 50.0, 25)
        assert retrieved is result

    def test_different_channel_id_gives_different_entry(self) -> None:
        cache = HarmonicCache()
        r1, r2 = _make_result(), _make_result()
        cache.put("VA", 200, 100, 50.0, 25, r1)
        cache.put("IA", 200, 100, 50.0, 25, r2)
        assert cache.get("VA", 200, 100, 50.0, 25) is r1
        assert cache.get("IA", 200, 100, 50.0, 25) is r2

    def test_different_window_samples_gives_different_entry(self) -> None:
        cache = HarmonicCache()
        r1, r2 = _make_result(), _make_result()
        cache.put("VA", 200, 100, 50.0, 25, r1)
        cache.put("VA", 100, 50, 50.0, 25, r2)
        assert cache.get("VA", 200, 100, 50.0, 25) is r1
        assert cache.get("VA", 100, 50, 50.0, 25) is r2

    def test_different_nominal_hz_gives_different_entry(self) -> None:
        cache = HarmonicCache()
        r1, r2 = _make_result(), _make_result()
        cache.put("VA", 200, 100, 50.0, 25, r1)
        cache.put("VA", 200, 100, 60.0, 25, r2)
        assert cache.get("VA", 200, 100, 50.0, 25) is r1
        assert cache.get("VA", 200, 100, 60.0, 25) is r2

    def test_different_max_order_gives_different_entry(self) -> None:
        cache = HarmonicCache()
        r1, r2 = _make_result(), _make_result()
        cache.put("VA", 200, 100, 50.0, 25, r1)
        cache.put("VA", 200, 100, 50.0, 13, r2)
        assert cache.get("VA", 200, 100, 50.0, 25) is r1
        assert cache.get("VA", 200, 100, 50.0, 13) is r2

    def test_invalidate_channel_removes_entries(self) -> None:
        cache = HarmonicCache()
        result = _make_result()
        cache.put("VA", 200, 100, 50.0, 25, result)
        cache.put("IA", 200, 100, 50.0, 25, _make_result())
        cache.invalidate_channel("VA")
        assert cache.get("VA", 200, 100, 50.0, 25) is None
        assert cache.get("IA", 200, 100, 50.0, 25) is not None

    def test_invalidate_unknown_channel_is_noop(self) -> None:
        cache = HarmonicCache()
        cache.invalidate_channel("nonexistent")  # should not raise

    def test_clear_empties_all_entries(self) -> None:
        cache = HarmonicCache()
        cache.put("VA", 200, 100, 50.0, 25, _make_result())
        cache.put("IA", 200, 100, 50.0, 25, _make_result())
        cache.clear()
        assert len(cache) == 0
        assert cache.get("VA", 200, 100, 50.0, 25) is None

    def test_len_reports_entry_count(self) -> None:
        cache = HarmonicCache()
        assert len(cache) == 0
        cache.put("VA", 200, 100, 50.0, 25, _make_result())
        assert len(cache) == 1
        cache.put("IA", 200, 100, 50.0, 25, _make_result())
        assert len(cache) == 2

    def test_entry_count_property(self) -> None:
        cache = HarmonicCache()
        cache.put("VA", 200, 100, 50.0, 25, _make_result())
        assert cache.entry_count == 1

    def test_contains_returns_true_after_put(self) -> None:
        cache = HarmonicCache()
        cache.put("VA", 200, 100, 50.0, 25, _make_result())
        assert cache.contains("VA", 200, 100, 50.0, 25) is True

    def test_contains_returns_false_for_missing(self) -> None:
        cache = HarmonicCache()
        assert cache.contains("VA", 200, 100, 50.0, 25) is False

    def test_mode_switch_reuses_same_cache_entry(self) -> None:
        """HARMONIC_MAGNITUDE ↔ THD switch doesn't change the cache key."""
        cache = HarmonicCache()
        result = _make_result()
        cache.put("VA", 200, 100, 50.0, 25, result)
        # The cache key doesn't include mode — same entry returned regardless
        assert cache.get("VA", 200, 100, 50.0, 25) is result
