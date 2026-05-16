"""Unit tests for frequency/ROCOF display models, registry, axis grouping,
and engineering unit conventions.

Tests cover:
  - FrequencyDisplayMode enum values
  - FrequencyChannelRole enum values
  - FrequencyChannelResult fields
  - FrequencyRegistry: classify, cache, mode changes, bulk helpers
  - Shared-axis behavior: frequency channels share Hz axis, ROCOF shares Hz/s
  - Frequency and ROCOF never share the same axis
  - Engineering unit display: Hz, Hz/s — never kHz or scientific notation
  - Panel title formatting for frequency/rocof groups
  - ROCOF display label helpers
  - FrequencyRegistry.frequency_panel_keys for direct and multi-source layouts
"""
from __future__ import annotations

import pytest

from app.analytics.frequency.frequency_models import (
    FrequencyChannelResult,
    FrequencyChannelRole,
    FrequencyConfig,
    FrequencyDisplayMode,
)
from app.analytics.frequency.frequency_registry import (
    FREQUENCY_PANEL_KEY,
    ROCOF_PANEL_KEY,
    FrequencyRegistry,
)
from app.analytics.frequency.rocof_overlay import (
    FREQUENCY_DISPLAY_UNIT,
    ROCOF_DISPLAY_UNIT,
    frequency_axis_label,
    frequency_display_label,
    rocof_axis_label,
    rocof_display_label,
)
from app.visualization.axis_management import AxisDisplayMode, axis_group_for_signal
from app.visualization.engineering_display import (
    format_panel_title,
    normalize_engineering_unit,
)


# ─────────────────────────────────────────────────────────────────────────────
# TestFrequencyDisplayMode
# ─────────────────────────────────────────────────────────────────────────────


class TestFrequencyDisplayMode:
    def test_off_value(self) -> None:
        assert FrequencyDisplayMode.OFF.value == "off"

    def test_overlay_value(self) -> None:
        assert FrequencyDisplayMode.OVERLAY.value == "overlay"

    def test_panel_only_value(self) -> None:
        assert FrequencyDisplayMode.PANEL_ONLY.value == "panel_only"

    def test_default_mode_is_panel_only(self) -> None:
        registry = FrequencyRegistry()
        assert registry.display_mode == FrequencyDisplayMode.PANEL_ONLY

    def test_all_three_modes_exist(self) -> None:
        modes = {m.value for m in FrequencyDisplayMode}
        assert modes == {"off", "overlay", "panel_only"}


# ─────────────────────────────────────────────────────────────────────────────
# TestFrequencyChannelRole
# ─────────────────────────────────────────────────────────────────────────────


class TestFrequencyChannelRole:
    def test_frequency_value(self) -> None:
        assert FrequencyChannelRole.FREQUENCY.value == "frequency"

    def test_rocof_value(self) -> None:
        assert FrequencyChannelRole.ROCOF.value == "rocof"

    def test_unknown_value(self) -> None:
        assert FrequencyChannelRole.UNKNOWN.value == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# TestFrequencyChannelResult
# ─────────────────────────────────────────────────────────────────────────────


class TestFrequencyChannelResult:
    def test_result_is_frozen(self) -> None:
        result = FrequencyChannelResult(
            role=FrequencyChannelRole.FREQUENCY,
            reason="test",
            auto_classified=True,
            display_unit="Hz",
        )
        with pytest.raises((AttributeError, TypeError)):
            result.role = FrequencyChannelRole.ROCOF  # type: ignore[misc]

    def test_display_unit_default_none(self) -> None:
        result = FrequencyChannelResult(
            role=FrequencyChannelRole.UNKNOWN,
            reason="test",
            auto_classified=True,
        )
        assert result.display_unit is None


# ─────────────────────────────────────────────────────────────────────────────
# TestFrequencyRegistry
# ─────────────────────────────────────────────────────────────────────────────


class TestFrequencyRegistry:
    def test_default_display_mode(self) -> None:
        r = FrequencyRegistry()
        assert r.display_mode == FrequencyDisplayMode.PANEL_ONLY

    def test_set_display_mode_off(self) -> None:
        r = FrequencyRegistry()
        r.set_display_mode(FrequencyDisplayMode.OFF)
        assert r.display_mode == FrequencyDisplayMode.OFF

    def test_set_display_mode_overlay(self) -> None:
        r = FrequencyRegistry()
        r.set_display_mode(FrequencyDisplayMode.OVERLAY)
        assert r.display_mode == FrequencyDisplayMode.OVERLAY

    def test_set_display_mode_invalid_raises(self) -> None:
        r = FrequencyRegistry()
        with pytest.raises(TypeError):
            r.set_display_mode("off")  # type: ignore[arg-type]

    def test_classify_returns_result(self) -> None:
        r = FrequencyRegistry()
        result = r.classify("SysFreq")
        assert isinstance(result, FrequencyChannelResult)
        assert result.role == FrequencyChannelRole.FREQUENCY

    def test_classify_rocof(self) -> None:
        r = FrequencyRegistry()
        result = r.classify("ROCOF")
        assert result.role == FrequencyChannelRole.ROCOF

    def test_classify_caches_result(self) -> None:
        r = FrequencyRegistry()
        r.classify("SysFreq")
        assert "SysFreq" in r.cached_roles

    def test_classify_cache_hit(self) -> None:
        r = FrequencyRegistry()
        first = r.classify("SysFreq")
        second = r.classify("SysFreq")
        assert first is second  # same object from cache

    def test_clear_cache(self) -> None:
        r = FrequencyRegistry()
        r.classify("SysFreq")
        assert len(r.cached_roles) == 1
        r.clear_cache()
        assert len(r.cached_roles) == 0

    def test_is_frequency_true(self) -> None:
        r = FrequencyRegistry()
        assert r.is_frequency("Frequency") is True

    def test_is_frequency_false_for_rocof(self) -> None:
        r = FrequencyRegistry()
        assert r.is_frequency("ROCOF") is False

    def test_is_rocof_true(self) -> None:
        r = FrequencyRegistry()
        assert r.is_rocof("dfdt") is True

    def test_is_rocof_false_for_frequency(self) -> None:
        r = FrequencyRegistry()
        assert r.is_rocof("Frequency") is False

    def test_is_frequency_or_rocof_true_for_both(self) -> None:
        r = FrequencyRegistry()
        assert r.is_frequency_or_rocof("Frequency") is True
        assert r.is_frequency_or_rocof("ROCOF") is True

    def test_is_frequency_or_rocof_false_for_waveform(self) -> None:
        r = FrequencyRegistry()
        assert r.is_frequency_or_rocof("VA") is False

    def test_force_role_is_cached(self) -> None:
        r = FrequencyRegistry()
        r.classify("VA", force_role=FrequencyChannelRole.FREQUENCY)
        assert r.cached_roles.get("VA") == FrequencyChannelRole.FREQUENCY


# ─────────────────────────────────────────────────────────────────────────────
# TestFrequencyRegistryBulkHelpers
# ─────────────────────────────────────────────────────────────────────────────


class TestFrequencyRegistryBulkHelpers:
    def test_frequency_channels_filters_correctly(self) -> None:
        r = FrequencyRegistry()
        names = ["SysFreq", "ROCOF", "VA", "Frequency", "dfdt"]
        freq = r.frequency_channels(names)
        assert set(freq) == {"SysFreq", "Frequency"}

    def test_rocof_channels_filters_correctly(self) -> None:
        r = FrequencyRegistry()
        names = ["SysFreq", "ROCOF", "VA", "Frequency", "dfdt"]
        rocof = r.rocof_channels(names)
        assert set(rocof) == {"ROCOF", "dfdt"}

    def test_frequency_channels_with_unit_override(self) -> None:
        r = FrequencyRegistry()
        names = ["Channel1", "Channel2"]
        units = {"Channel1": "Hz", "Channel2": "kV"}
        freq = r.frequency_channels(names, units=units)
        assert freq == ["Channel1"]

    def test_rocof_channels_with_unit_override(self) -> None:
        r = FrequencyRegistry()
        names = ["Channel1", "Channel2"]
        units = {"Channel1": "Hz/s", "Channel2": "Hz"}
        rocof = r.rocof_channels(names, units=units)
        assert rocof == ["Channel1"]


# ─────────────────────────────────────────────────────────────────────────────
# TestFrequencyPanelKeys
# ─────────────────────────────────────────────────────────────────────────────


class TestFrequencyPanelKeys:
    def test_direct_frequency_key(self) -> None:
        r = FrequencyRegistry()
        keys = r.frequency_panel_keys([
            "voltage_raw", "current_raw", "power", "frequency", "rocof", "other"
        ])
        assert set(keys) == {"frequency", "rocof"}

    def test_multi_source_frequency_keys(self) -> None:
        r = FrequencyRegistry()
        keys = r.frequency_panel_keys([
            "COMTRADE/voltage_raw",
            "COMTRADE/frequency",
            "CSV/frequency",
            "CSV/rocof",
            "CSV/power",
        ])
        assert set(keys) == {"COMTRADE/frequency", "CSV/frequency", "CSV/rocof"}

    def test_no_frequency_panels_returns_empty(self) -> None:
        r = FrequencyRegistry()
        keys = r.frequency_panel_keys(["voltage_raw", "power", "other"])
        assert keys == []

    def test_frequency_panel_key_constants(self) -> None:
        assert FREQUENCY_PANEL_KEY == "frequency"
        assert ROCOF_PANEL_KEY == "rocof"


# ─────────────────────────────────────────────────────────────────────────────
# TestSharedAxisGrouping
# ─────────────────────────────────────────────────────────────────────────────


class TestSharedAxisGrouping:
    def test_frequency_channels_share_hz_axis(self) -> None:
        g1 = axis_group_for_signal("SysFreq", "Hz", mode=AxisDisplayMode.SHARED)
        g2 = axis_group_for_signal("PMUFreq", "Hz", mode=AxisDisplayMode.SHARED)
        assert g1.key == g2.key, "Frequency channels should share the same axis"

    def test_rocof_channels_share_hzs_axis(self) -> None:
        g1 = axis_group_for_signal("ROCOF1", "Hz/s", mode=AxisDisplayMode.SHARED)
        g2 = axis_group_for_signal("dfdt", "Hz/s", mode=AxisDisplayMode.SHARED)
        assert g1.key == g2.key, "ROCOF channels should share the same axis"

    def test_frequency_and_rocof_do_not_share_axis(self) -> None:
        g_freq = axis_group_for_signal("SysFreq", "Hz", mode=AxisDisplayMode.SHARED)
        g_rocof = axis_group_for_signal("ROCOF", "Hz/s", mode=AxisDisplayMode.SHARED)
        assert g_freq.key != g_rocof.key, (
            "Frequency and ROCOF must NOT share the same axis"
        )

    def test_frequency_does_not_share_axis_with_voltage(self) -> None:
        g_freq = axis_group_for_signal("SysFreq", "Hz", mode=AxisDisplayMode.SHARED)
        g_volt = axis_group_for_signal("VA", "kV", mode=AxisDisplayMode.SHARED)
        assert g_freq.key != g_volt.key

    def test_frequency_does_not_share_axis_with_current(self) -> None:
        g_freq = axis_group_for_signal("SysFreq", "Hz", mode=AxisDisplayMode.SHARED)
        g_curr = axis_group_for_signal("Ia", "A", mode=AxisDisplayMode.SHARED)
        assert g_freq.key != g_curr.key

    def test_rocof_does_not_share_axis_with_power(self) -> None:
        g_rocof = axis_group_for_signal("ROCOF", "Hz/s", mode=AxisDisplayMode.SHARED)
        g_power = axis_group_for_signal("MW", "MW", mode=AxisDisplayMode.SHARED)
        assert g_rocof.key != g_power.key

    def test_frequency_axis_label_unit_is_hz(self) -> None:
        g = axis_group_for_signal("SysFreq", "Hz", mode=AxisDisplayMode.SHARED)
        assert g.label.unit == "Hz"

    def test_rocof_axis_label_unit_is_hzs(self) -> None:
        g = axis_group_for_signal("ROCOF", "Hz/s", mode=AxisDisplayMode.SHARED)
        assert g.label.unit == "Hz/s"


# ─────────────────────────────────────────────────────────────────────────────
# TestEngineeringUnitDisplay
# ─────────────────────────────────────────────────────────────────────────────


class TestEngineeringUnitDisplay:
    def test_hz_unit_normalized_to_hz(self) -> None:
        result = normalize_engineering_unit("Hz")
        assert result == "Hz"

    def test_hz_case_insensitive(self) -> None:
        result = normalize_engineering_unit("hz")
        assert result == "Hz"

    def test_hzs_normalized_to_hzs(self) -> None:
        result = normalize_engineering_unit("Hz/s")
        assert result == "Hz/s"

    def test_frequency_unit_never_khz(self) -> None:
        result = normalize_engineering_unit("Hz", signal_type="frequency")
        assert result != "kHz"
        assert result != "MHz"

    def test_rocof_unit_never_scaled(self) -> None:
        result = normalize_engineering_unit("Hz/s", signal_type="rocof")
        assert result == "Hz/s"

    def test_display_unit_constants(self) -> None:
        assert FREQUENCY_DISPLAY_UNIT == "Hz"
        assert ROCOF_DISPLAY_UNIT == "Hz/s"


# ─────────────────────────────────────────────────────────────────────────────
# TestDisplayLabelHelpers
# ─────────────────────────────────────────────────────────────────────────────


class TestDisplayLabelHelpers:
    def test_rocof_display_label_includes_unit(self) -> None:
        label = rocof_display_label("ROCOF_PULU")
        assert "Hz/s" in label
        assert "ROCOF_PULU" in label

    def test_rocof_display_label_empty_name_fallback(self) -> None:
        label = rocof_display_label("")
        assert "Hz/s" in label
        assert label  # non-empty

    def test_rocof_axis_label(self) -> None:
        label = rocof_axis_label()
        assert "Hz/s" in label
        assert "ROCOF" in label

    def test_frequency_display_label_includes_unit(self) -> None:
        label = frequency_display_label("SysFreq")
        assert "Hz" in label
        assert "SysFreq" in label

    def test_frequency_axis_label(self) -> None:
        label = frequency_axis_label()
        assert "Hz" in label

    def test_frequency_axis_label_does_not_contain_hzs(self) -> None:
        label = frequency_axis_label()
        assert "Hz/s" not in label


# ─────────────────────────────────────────────────────────────────────────────
# TestPanelTitleFormatting
# ─────────────────────────────────────────────────────────────────────────────


class TestPanelTitleFormatting:
    def test_frequency_group_title(self) -> None:
        title = format_panel_title("frequency")
        assert title  # non-empty
        assert "Frequency" in title or "frequency" in title.lower()

    def test_rocof_group_title(self) -> None:
        title = format_panel_title("rocof")
        assert title
        assert "ROCOF" in title or "rocof" in title.lower()

    def test_frequency_title_with_source_id(self) -> None:
        title = format_panel_title("frequency", source_id="CSV")
        assert "CSV" in title
