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

A later Phase A safety-fix task DID modify app/sessions/event_session.py
and app/visualization/engineering_display.py, to fix two problems a
follow-up architecture assessment found:

  1. _infer_panel_for_type's alias table only recognised the Import
     Wizard's ParameterType vocabulary (mw, mvar, voltage, current), not
     the shared classifier vocabulary CSV/Excel providers actually set on
     AnalogChannel.parameter_type (active_power, reactive_power,
     voltage_rms, current_rms) -- so a confidently-classified direct
     provider channel's parameter_type was silently ignored by panel
     routing, which fell through to the unit/name fallback instead.
  2. The panel name-fallback and infer_signal_type's role inference both
     used raw substring/prefix matching (e.g. "in" in "Input", "pu" in
     "Output", name.startswith("i")), which could misroute or mis-role
     ordinary non-electrical names.

TestPanelTypeAliases, TestPanelNameFallbackHardening, and
TestNumericalSafetyAfterHardening below cover that fix. See also
tests/unit/test_engineering_display.py for the infer_signal_type-specific
coverage.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

from app.analytics.scaling.engineering_scaling import compute_scaling_factor
from app.analytics.scaling.scaling_models import EngineeringScalingMode, GlobalScalingConfig
from app.providers.csv.csv_provider import CsvProvider
from app.sessions.event_session import _infer_panel_for_channel, _infer_panel_for_type
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


class TestPanelTypeAliases:
    """_TYPE_TO_PANEL must recognise both the shared classifier's
    parameter_type vocabulary (used by direct CSV/Excel providers) and the
    Import Wizard's ParameterType vocabulary, mapping both to the same panel
    for the same physical quantity.
    """

    def test_shared_classifier_vocabulary_resolves_directly(self) -> None:
        assert _infer_panel_for_type("active_power") == "power"
        assert _infer_panel_for_type("reactive_power") == "power"
        assert _infer_panel_for_type("voltage_rms") == "voltage"
        assert _infer_panel_for_type("current_rms") == "current"
        assert _infer_panel_for_type("frequency") == "frequency"
        assert _infer_panel_for_type("rocof") == "frequency"

    def test_wizard_vocabulary_still_resolves(self) -> None:
        assert _infer_panel_for_type("mw") == "power"
        assert _infer_panel_for_type("mvar") == "power"
        assert _infer_panel_for_type("voltage") == "voltage"
        assert _infer_panel_for_type("current") == "current"
        assert _infer_panel_for_type("frequency") == "frequency"
        assert _infer_panel_for_type("rocof") == "frequency"

    def test_both_vocabularies_agree_on_panel_via_type_alone(self) -> None:
        # unit intentionally omitted so the panel is decided by parameter_type,
        # not by falling back to the unit or name.
        for shared_type, wizard_type, expected_panel in [
            ("active_power", "mw", "power"),
            ("reactive_power", "mvar", "power"),
            ("voltage_rms", "voltage", "voltage"),
            ("current_rms", "current", "current"),
        ]:
            shared_panel, _ = _infer_panel_for_channel("Chan", unit=None, param_type=shared_type)
            wizard_panel, _ = _infer_panel_for_channel("Chan", unit=None, param_type=wizard_type)
            assert shared_panel == wizard_panel == expected_panel

    def test_direct_provider_channel_routes_via_type_not_name(self) -> None:
        # "Chan1" carries no name-based electrical evidence at all; only a
        # confident parameter_type should route it.
        panel_id, _ = _infer_panel_for_channel("Chan1", unit="unknown", param_type="active_power")
        assert panel_id == "power"
        panel_id, _ = _infer_panel_for_channel("Chan2", unit="unknown", param_type="voltage_rms")
        assert panel_id == "voltage"
        panel_id, _ = _infer_panel_for_channel("Chan3", unit="unknown", param_type="current_rms")
        assert panel_id == "current"
        panel_id, _ = _infer_panel_for_channel("Chan4", unit="unknown", param_type="reactive_power")
        assert panel_id == "power"


class TestPanelNameFallbackHardening:
    """The last-resort name-keyword fallback (type and unit both absent/
    unresolved) must use boundary-aware matching, not raw substring
    containment.
    """

    def test_short_fragments_do_not_match_inside_unrelated_words(self) -> None:
        for name in [
            "Input", "Index", "Interval", "Info", "Variable",
            "IndexValue", "IntervalCount", "InputFile", "Information",
        ]:
            panel_id, _ = _infer_panel_for_channel(name, unit=None, param_type=None)
            assert panel_id == "other", f"{name!r} should not collide, got {panel_id!r}"

    def test_plain_electrical_words_still_match(self) -> None:
        assert _infer_panel_for_channel("Voltage", unit=None, param_type=None)[0] == "voltage"
        assert _infer_panel_for_channel("Bus Voltage", unit=None, param_type=None)[0] == "voltage"
        assert _infer_panel_for_channel("Phase Current", unit=None, param_type=None)[0] == "current"
        assert _infer_panel_for_channel("Frequency", unit=None, param_type=None)[0] == "frequency"
        assert _infer_panel_for_channel("Active Power", unit=None, param_type=None)[0] == "power"
        assert _infer_panel_for_channel("Reactive Power", unit=None, param_type=None)[0] == "power"

    def test_exact_relay_style_short_tokens_still_match(self) -> None:
        assert _infer_panel_for_channel("Va", unit=None, param_type=None)[0] == "voltage"
        assert _infer_panel_for_channel("Ia", unit=None, param_type=None)[0] == "current"
        assert _infer_panel_for_channel("Vab", unit=None, param_type=None)[0] == "voltage"

    def test_symmetrical_component_relay_names_are_not_supported_by_fallback(self) -> None:
        # Documented, pre-existing gap (not broadened or newly introduced by
        # this hardening): the name-keyword fallback never recognised "I0"
        # (unlike the shared classifier's own _EXACT table, which does).
        # Confirmed unchanged before and after the boundary-matching fix.
        panel_id, _ = _infer_panel_for_channel("I0", unit=None, param_type=None)
        assert panel_id == "other"

    def test_type_and_unit_still_outrank_name_fallback(self) -> None:
        # A misleading name must not override confident type/unit evidence.
        panel_id, _ = _infer_panel_for_channel("Interval", unit="MW", param_type="active_power")
        assert panel_id == "power"
        panel_id, _ = _infer_panel_for_channel("Output", unit="kV", param_type=None)
        assert panel_id == "voltage"


class TestNumericalSafetyAfterHardening:
    """Proves the panel/axis hardening actually removes the pathway to an
    unsafe PER_UNIT/engineering scaling activation for a name that used to
    collide -- not just that the label/panel looks right.
    """

    def test_previously_colliding_name_does_not_activate_scaling(self) -> None:
        cfg = GlobalScalingConfig(voltage_base_kv=275.0, current_base_ka=1.0)
        for name in ["Output", "Input", "Pump", "Impulse", "Interval", "Index"]:
            result = compute_scaling_factor(name, None, EngineeringScalingMode.PER_UNIT, cfg)
            assert result.factor == 1.0
            assert result.description == "no_scaling"

    def test_named_electrical_channel_still_activates_scaling(self) -> None:
        # Contrast case: a genuine voltage channel still gets PER_UNIT math.
        cfg = GlobalScalingConfig(voltage_base_kv=275.0)
        result = compute_scaling_factor("Va", "kV", EngineeringScalingMode.PER_UNIT, cfg)
        assert result.configured is True
        assert result.display_unit == "pu"
