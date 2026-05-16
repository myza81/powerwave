"""Tests for domain-aware engineering display label policy."""
from __future__ import annotations

from app.visualization.engineering_display import (
    EngineeringDisplayPreferences,
    format_axis_label,
    format_panel_title,
    format_rms_curve_label,
    normalize_engineering_unit,
)


def test_active_power_display_stays_mw_without_si_prefix() -> None:
    assert normalize_engineering_unit("MW", signal_type="active_power") == "MW"
    assert normalize_engineering_unit("kMW", signal_type="active_power") == "MW"
    assert normalize_engineering_unit("mMW", signal_type="active_power") == "MW"


def test_frequency_display_stays_hz() -> None:
    assert normalize_engineering_unit("Hz", signal_type="frequency") == "Hz"
    assert normalize_engineering_unit("kHz", signal_type="frequency") == "Hz"


def test_reactive_power_display_uses_mvar() -> None:
    assert normalize_engineering_unit("MVAr", signal_type="reactive_power") == "MVar"
    assert normalize_engineering_unit("MVAR") == "MVar"


def test_voltage_current_and_rocof_units() -> None:
    assert normalize_engineering_unit("kV", signal_type="voltage") == "kV"
    assert normalize_engineering_unit("V", signal_type="voltage") == "V"
    assert normalize_engineering_unit("A", signal_type="current") == "A"
    assert normalize_engineering_unit("kA", signal_type="current") == "kA"
    assert normalize_engineering_unit("Hz/sec", signal_type="rocof") == "Hz/s"


def test_axis_label_formats_name_and_unit() -> None:
    label = format_axis_label("System Demand", "kMW")
    assert label.text == "System Demand"
    assert label.unit == "MW"


def test_rms_curve_label_is_explicit() -> None:
    assert format_rms_curve_label("VA", "kV") == "VA RMS (kV)"


def test_panel_titles_are_consistent() -> None:
    assert format_panel_title("power", 2) == "Power"
    assert format_panel_title("frequency", 1) == "Frequency"
    assert format_panel_title("other", 42) == "Other Analog Channels (42)"
    assert format_panel_title("power", 2, "csv_ops") == "csv_ops - Power"


def test_preferences_are_future_hook_not_scaling_engine() -> None:
    prefs = EngineeringDisplayPreferences(active_power_unit="MW")
    assert normalize_engineering_unit("kMW", signal_type="active_power", preferences=prefs) == "MW"
