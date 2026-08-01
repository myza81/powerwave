"""Downstream verification for the semantic-classification consolidation.

Confirms that session default panel placement (app.sessions.event_session)
and axis grouping (app.visualization.axis_management) already recognise both
taxonomies -- app.data.column_classifier's signal_type strings used by direct
providers (e.g. "active_power") and the Import Wizard's ParameterType.value
strings (e.g. "mw") -- via their existing alias/fallback logic, so newly
correct parameter types (e.g. System Demand no longer tagged "voltage") land
in the right panel/axis without any change to these modules.

No production code in app/sessions or app/visualization was modified for
this task, or for the follow-up task that removed the shared classifier's
unsafe magnitude-only electrical fallbacks and gated provider unit
assignment on confirmation status -- these tests exist to prove both changes
were safe from the downstream panel/axis perspective, not to change
behaviour there.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

from app.providers.csv.csv_provider import CsvProvider
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


class TestUnconfirmedMagnitudeNoLongerRoutesToElectricalPanel:
    """End-to-end regression test: a neutral-header column whose values
    previously triggered a magnitude-only electrical guess (removed in this
    task) must no longer receive an electrical unit, and must therefore no
    longer be routed to the Power or Voltage panel via the unit fallback in
    _infer_panel_for_channel. _infer_panel_for_channel itself is unchanged;
    only the provider-level input it receives has changed.
    """

    def _load_channel(self, tmp_path: Path, header: str, values: list[float]):
        p = tmp_path / "t.csv"
        lines = [f"time,{header}"] + [f"{i}.0,{v}" for i, v in enumerate(values)]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = CsvProvider().load(p)
        return next(c for c in rec.analog_channels if c.name == header)

    def test_neutral_near_1pu_no_longer_enters_voltage_panel(self, tmp_path: Path) -> None:
        values = [0.98, 1.01, 0.99, 1.02, 1.00, 1.01, 0.99]
        ch = self._load_channel(tmp_path, "Column 1", values)
        assert ch.unit != "pu"
        assert ch.unit == "unknown"
        panel_id, _ = _infer_panel_for_channel("Column 1", unit=ch.unit, param_type=ch.parameter_type)
        assert panel_id != "voltage"
        assert panel_id == "other"

    def test_neutral_noisy_large_magnitude_no_longer_enters_power_panel(self, tmp_path: Path) -> None:
        values = [18700.0, 18712.0, 18705.0, 18730.0, 18711.0, 18698.0, 18720.0]
        ch = self._load_channel(tmp_path, "Column 1", values)
        assert ch.unit != "MW"
        assert ch.unit == "unknown"
        panel_id, _ = _infer_panel_for_channel("Column 1", unit=ch.unit, param_type=ch.parameter_type)
        assert panel_id != "power"
        assert panel_id == "other"

    def test_neutral_negative_mw_like_no_longer_enters_power_panel(self, tmp_path: Path) -> None:
        values = [-120.0, -140.0, -118.0, -145.0, -121.0, -110.0, -133.0]
        ch = self._load_channel(tmp_path, "Column 1", values)
        assert ch.unit == "unknown"
        panel_id, _ = _infer_panel_for_channel("Column 1", unit=ch.unit, param_type=ch.parameter_type)
        assert panel_id == "other"

    def test_named_electrical_channel_still_routes_correctly(self, tmp_path: Path) -> None:
        # Contrast case: confirms this test class isn't just observing
        # everything landing in "other" regardless of input.
        values = [18700.0, 18712.0, 18705.0, 18730.0, 18711.0]
        ch = self._load_channel(tmp_path, "System Demand", values)
        assert ch.unit == "MW"
        panel_id, _ = _infer_panel_for_channel("System Demand", unit=ch.unit, param_type=ch.parameter_type)
        assert panel_id == "power"
