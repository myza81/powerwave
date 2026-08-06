"""Tests for domain-aware engineering display label policy."""
from __future__ import annotations

from app.visualization.engineering_display import (
    EngineeringDisplayPreferences,
    format_axis_label,
    format_panel_title,
    format_rms_curve_label,
    infer_signal_type,
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


# ─────────────────────────────────────────────────────────────────────────────
# infer_signal_type — boundary-aware name matching (Phase A safety hardening)
# ─────────────────────────────────────────────────────────────────────────────


class TestInferSignalTypeCollisionHardening:
    """Ordinary words that happen to contain a role fragment ("pu" in
    "Output", a leading "i"/"v") must not receive an electrical role from
    the name alone.
    """

    def test_ordinary_words_get_no_role_without_a_unit(self) -> None:
        for name in [
            "Output", "Input", "Pump", "Impulse", "Index", "Interval", "Info",
            "InputFile", "OutputFile", "PumpRunning", "ImpulseCounter",
        ]:
            assert infer_signal_type(name, None) is None, f"{name!r} should have no role"

    def test_ordinary_words_get_no_role_with_unrecognized_unit(self) -> None:
        for name in ["Output", "Interval", "Index"]:
            assert infer_signal_type(name, "unknown") is None


class TestInferSignalTypeValidRoles:
    def test_explicit_units_determine_role(self) -> None:
        assert infer_signal_type("Chan", "kV") == "voltage"
        assert infer_signal_type("Chan", "A") == "current"
        assert infer_signal_type("Chan", "MW") == "active_power"
        assert infer_signal_type("Chan", "MVAr") == "reactive_power"
        assert infer_signal_type("Chan", "Hz") == "frequency"
        assert infer_signal_type("Chan", "Hz/s") == "rocof"
        assert infer_signal_type("Chan", "pu") == "per_unit"

    def test_exact_relay_names_identify_voltage_and_current(self) -> None:
        assert infer_signal_type("Va", None) == "voltage"
        assert infer_signal_type("Ia", None) == "current"

    def test_clear_tokenized_electrical_terms_remain_supported(self) -> None:
        assert infer_signal_type("Bus Voltage", None) == "voltage"
        assert infer_signal_type("Phase Current", None) == "current"
        assert infer_signal_type("Per Unit Voltage", None) == "voltage"

    def test_per_unit_still_resolves_from_an_explicit_pu_token(self) -> None:
        # A standalone "pu" token in the name is legitimate evidence (unlike
        # "pu" appearing mid-word in "Output"/"Pump") and is checked ahead of
        # the voltage branch, matching the pre-hardening precedence order.
        assert infer_signal_type("Voltage pu", None) == "per_unit"
        assert infer_signal_type("Chan", "pu") == "per_unit"
