from __future__ import annotations

from app.visualization.axis_management import AxisDisplayMode, axis_group_for_signal


def test_shared_axis_groups_voltage_by_role_and_unit() -> None:
    vr = axis_group_for_signal("KPDN1 VR", "kV")
    vy = axis_group_for_signal("KPDN1 VY", "kV")

    assert vr.key == vy.key
    assert vr.label.text == "Voltage"
    assert vr.label.unit == "kV"


def test_shared_axis_groups_power_without_generic_si_prefixing() -> None:
    demand = axis_group_for_signal("System Demand", "MW")
    tie = axis_group_for_signal("Tie-Line", "MW")

    assert demand.key == tie.key
    assert demand.label.text == "Power"
    assert demand.label.unit == "MW"
    assert "kMW" not in demand.key


def test_shared_axis_keeps_incompatible_quantities_separate() -> None:
    power = axis_group_for_signal("System Demand", "MW")
    frequency = axis_group_for_signal("Frequency", "Hz")

    assert power.key != frequency.key
    assert frequency.label.text == "Frequency"
    assert frequency.label.unit == "Hz"


def test_dedicated_axis_mode_preserves_per_signal_axes() -> None:
    first = axis_group_for_signal("System Demand", "MW", mode=AxisDisplayMode.DEDICATED)
    second = axis_group_for_signal("Tie-Line", "MW", mode=AxisDisplayMode.DEDICATED)

    assert first.key != second.key
    assert first.label.unit == "MW"
    assert second.label.unit == "MW"
