"""Downstream verification for the semantic-classification consolidation.

Confirms that session default panel placement (app.sessions.event_session)
and axis grouping (app.visualization.axis_management) already recognise both
taxonomies -- app.data.column_classifier's signal_type strings used by direct
providers (e.g. "active_power") and the Import Wizard's ParameterType.value
strings (e.g. "mw") -- via their existing alias/fallback logic, so newly
correct parameter types (e.g. System Demand no longer tagged "voltage") land
in the right panel/axis without any change to these modules.

No production code in app/sessions or app/visualization was modified for
this task; these tests exist to prove that was safe, not to change behaviour.
"""
from __future__ import annotations

from app.sessions.event_session import _infer_panel_for_channel
from app.visualization.axis_management import (
    AxisDisplayMode,
    axis_group_for_signal,
    normalize_signal_type_hint,
)


class TestSessionPanelPlacement:
    def test_direct_provider_active_power_lands_in_power_panel(self) -> None:
        panel_id, _ = _infer_panel_for_channel("System Demand", unit="MW", param_type="active_power")
        assert panel_id == "power"

    def test_wizard_mw_lands_in_power_panel(self) -> None:
        panel_id, _ = _infer_panel_for_channel("mw_system_demand", unit="MW", param_type="mw")
        assert panel_id == "power"

    def test_both_paths_agree_on_panel_for_the_same_quantity(self) -> None:
        direct_panel, _ = _infer_panel_for_channel("Tie-Line", unit="MW", param_type="active_power")
        wizard_panel, _ = _infer_panel_for_channel("mw_tie_line", unit="MW", param_type="mw")
        assert direct_panel == wizard_panel == "power"

    def test_reactive_power_lands_in_power_panel_not_voltage(self) -> None:
        panel_id, _ = _infer_panel_for_channel("Reactive Power", unit="MVAr", param_type="reactive_power")
        assert panel_id == "power"

    def test_relay_voltage_lands_in_voltage_panel(self) -> None:
        panel_id, _ = _infer_panel_for_channel("Va", unit="V", param_type="voltage_rms")
        assert panel_id == "voltage"

    def test_missing_parameter_type_falls_back_to_unit(self) -> None:
        # Low-confidence classifications intentionally leave parameter_type
        # None; the panel must still be inferred safely from unit/name.
        panel_id, _ = _infer_panel_for_channel("Tie-Line", unit="MW", param_type=None)
        assert panel_id == "power"


class TestAxisGrouping:
    def test_active_power_alias_resolves_to_same_role_as_mw(self) -> None:
        assert normalize_signal_type_hint("active_power") == normalize_signal_type_hint("mw") == "active_power"

    def test_reactive_power_alias_resolves_to_same_role_as_mvar(self) -> None:
        assert normalize_signal_type_hint("reactive_power") == normalize_signal_type_hint("mvar") == "reactive_power"

    def test_direct_and_wizard_power_channels_share_one_axis_group(self) -> None:
        direct_group = axis_group_for_signal(
            "System Demand", "MW", mode=AxisDisplayMode.SHARED, signal_type_hint="active_power"
        )
        wizard_group = axis_group_for_signal(
            "mw_system_demand", "MW", mode=AxisDisplayMode.SHARED, signal_type_hint="mw"
        )
        assert direct_group.key == wizard_group.key

    def test_axis_label_does_not_display_kv_for_power_channel(self) -> None:
        group = axis_group_for_signal(
            "System Demand", "MW", mode=AxisDisplayMode.SHARED, signal_type_hint="active_power"
        )
        assert "kV" not in group.label.unit
        assert "kv" not in group.label.unit.lower()

    def test_voltage_axis_group_still_correct(self) -> None:
        group = axis_group_for_signal(
            "Va", "V", mode=AxisDisplayMode.SHARED, signal_type_hint="voltage_rms"
        )
        assert group.key.startswith("voltage")
