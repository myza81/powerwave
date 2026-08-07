"""Unit tests for Calculated Signals session ownership (Phase 2C-1) in
app.sessions.event_session.EventAnalysisSession.

Uses only generic, synthetic session fixtures (Source A / Source B with
made-up analog/digital channel names) -- no filename, station, or event
identity is special-cased anywhere in this file or in production code.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculatedSignalResult,
    CalculationStatus,
    ChannelRef,
)
from app.models.channels import AnalogChannel, DigitalChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.event_session import EventAnalysisSession
from app.sessions.session_models import ChannelEligibility


# ─────────────────────────────────────────────────────────────────────────────
# Generic synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_record(
    analog: dict[str, str] | list[str],
    digital: list[str] | None = None,
    n: int = 10,
    non_numeric_columns: list[str] | None = None,
    declared_but_missing_column: str | None = None,
) -> DisturbanceRecord:
    """Build a minimal, generic DisturbanceRecord.

    analog: either a list of names (unit defaults to "MW") or a
            {name: unit} mapping.
    non_numeric_columns: analog channel names whose DataFrame column is
            deliberately given string dtype (for eligibility tests).
    declared_but_missing_column: an analog channel name declared in
            analog_channels but deliberately NOT added as a DataFrame
            column (for eligibility tests).
    """
    if isinstance(analog, list):
        analog = {name: "MW" for name in analog}
    digital = digital or []
    non_numeric_columns = non_numeric_columns or []

    data: dict[str, object] = {"time": np.linspace(0, 1, n)}
    for name in analog:
        if name == declared_but_missing_column:
            continue
        if name in non_numeric_columns:
            data[name] = [f"x{i}" for i in range(n)]
        else:
            data[name] = np.arange(n, dtype=float)
    for name in digital:
        data[name] = np.zeros(n, dtype=np.int8)

    df = pd.DataFrame(data)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="GenericStation", recorder_name="GenericRecorder",
            source_file="generic.csv", provider_type="csv", nominal_frequency=50.0,
        ),
        waveform_data=df,
        analog_channels=[AnalogChannel(name=n, unit=u, index=i) for i, (n, u) in enumerate(analog.items())],
        digital_channels=[DigitalChannel(name=n, index=i) for i, n in enumerate(digital)],
        sampling_info=SamplingInformation(sampling_rates=[10.0], samples_per_rate=[n]),
        timing_info=TimingInformation(start_time=datetime(2024, 1, 1), trigger_time=datetime(2024, 1, 1)),
    )


def _session_with_two_sources() -> tuple[EventAnalysisSession, str, str]:
    """Source A: analog Va, MW. Source B: analog Ia, Frequency; digital Trip."""
    sess = EventAnalysisSession()
    sid_a = sess.add_source(
        _make_record({"Va": "kV", "MW": "MW"}), "Source A", "csv"
    )
    sid_b = sess.add_source(
        _make_record({"Ia": "A", "Frequency": "Hz"}, digital=["Trip"]), "Source B", "csv"
    )
    return sess, sid_a, sid_b


def _defn(calc_id: str, name: str, expression: str, bindings: dict[str, ChannelRef], reference: str, **kwargs) -> CalculatedSignalDefinition:
    return CalculatedSignalDefinition(
        calc_id=calc_id, name=name, expression=expression,
        variable_bindings=bindings, reference_variable=reference, **kwargs,
    )


def _result(calc_id: str, status: CalculationStatus = CalculationStatus.OK, error_message: str | None = None) -> CalculatedSignalResult:
    return CalculatedSignalResult(
        calc_id=calc_id,
        time=np.array([0.0, 1.0]),
        values=np.array([1.0, 2.0]),
        validity_mask=np.array([True, True]),
        unit="MW",
        status=status,
        error_message=error_message,
        computed_at=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Creation
# ─────────────────────────────────────────────────────────────────────────────


class TestCreation:
    def test_valid_calculated_signal_across_sources(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("c1", "N1", "A / B", {
            "A": ChannelRef(sid_a, "MW"), "B": ChannelRef(sid_b, "Frequency"),
        }, "A")
        calc_id = sess.add_calculated_signal(defn)
        assert calc_id == "c1"
        assert sess.get_calculated_signal_definition("c1") is defn

    def test_structural_validity_does_not_require_unit_compatibility(self) -> None:
        # Per Step 21: session storage only validates dependency
        # eligibility, not expression engineering/unit compatibility.
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("c1", "N1", "A / B", {
            "A": ChannelRef(sid_a, "Va"), "B": ChannelRef(sid_b, "Frequency"),
        }, "A")
        calc_id = sess.add_calculated_signal(defn)  # kV / Hz is numerically nonsensical, structurally fine
        assert calc_id == "c1"

    def test_duplicate_calc_id_rejected(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        defn1 = _defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        defn2 = _defn("c1", "N2", "A", {"A": ChannelRef(sid_a, "MW")}, "A")
        sess.add_calculated_signal(defn1)
        with pytest.raises(ValueError, match="already exists"):
            sess.add_calculated_signal(defn2)

    def test_duplicate_name_rejected(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        defn1 = _defn("c1", "SameName", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        defn2 = _defn("c2", "SameName", "A", {"A": ChannelRef(sid_a, "MW")}, "A")
        sess.add_calculated_signal(defn1)
        with pytest.raises(ValueError, match="already exists"):
            sess.add_calculated_signal(defn2)

    def test_missing_source_rejected(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        defn = _defn("c1", "N1", "A", {"A": ChannelRef("nonexistent-source", "Va")}, "A")
        with pytest.raises(ValueError, match="missing_source"):
            sess.add_calculated_signal(defn)

    def test_inactive_source_rejected(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.set_source_active(sid_a, False)
        defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        with pytest.raises(ValueError, match="inactive_source"):
            sess.add_calculated_signal(defn)

    def test_missing_analog_channel_rejected(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "DoesNotExist")}, "A")
        with pytest.raises(ValueError, match="missing_channel"):
            sess.add_calculated_signal(defn)

    def test_digital_channel_rejected(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid_b, "Trip")}, "A")
        with pytest.raises(ValueError, match="digital_channel"):
            sess.add_calculated_signal(defn)

    def test_non_numeric_analog_column_rejected(self) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(
            _make_record(["Weird"], non_numeric_columns=["Weird"]), "Source", "csv"
        )
        defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid, "Weird")}, "A")
        with pytest.raises(ValueError, match="non_numeric_channel"):
            sess.add_calculated_signal(defn)

    def test_declared_analog_but_missing_dataframe_column_rejected(self) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(
            _make_record(["Ghost"], declared_but_missing_column="Ghost"), "Source", "csv"
        )
        defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid, "Ghost")}, "A")
        with pytest.raises(ValueError, match="data_column_missing"):
            sess.add_calculated_signal(defn)

    def test_analog_digital_name_overlap_rejected(self) -> None:
        # A channel name declared in BOTH analog_channels and digital_channels
        # must be treated as ineligible (digital wins), never silently analog.
        sess = EventAnalysisSession()
        record = _make_record(["Ambiguous"])
        record.digital_channels.append(DigitalChannel(name="Ambiguous", index=99))
        sid = sess.add_source(record, "Source", "csv")
        defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid, "Ambiguous")}, "A")
        with pytest.raises(ValueError, match="digital_channel"):
            sess.add_calculated_signal(defn)

    def test_result_calc_id_mismatch_rejected(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        with pytest.raises(ValueError, match="does not match"):
            sess.add_calculated_signal(defn, result=_result("different-id"))

    def test_result_with_matching_id_accepted(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        sess.add_calculated_signal(defn, result=_result("c1"))
        assert sess.get_calculated_signal_result("c1").status == CalculationStatus.OK

    def test_failed_creation_leaves_session_unchanged(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        good = _defn("c1", "Good", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        sess.add_calculated_signal(good)

        bad = _defn("c2", "Bad", "A", {"A": ChannelRef(sid_b, "Trip")}, "A")
        with pytest.raises(ValueError):
            sess.add_calculated_signal(bad)

        assert [e.definition.calc_id for e in sess.list_calculated_signals()] == ["c1"]
        assert sess.get_calculated_dependents_for_source(sid_b) == ()
        assert sess.get_calculated_dependencies("c2") == ()

    def test_calc_on_calc_dependency_rejected(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        first = _defn("c1", "First", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        sess.add_calculated_signal(first)
        second = _defn("c2", "Second", "X", {"X": ChannelRef("c1", "whatever")}, "X")
        with pytest.raises(ValueError, match="calculated-signal-to-calculated-signal"):
            sess.add_calculated_signal(second)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrieval:
    def test_get_by_id(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        sess.add_calculated_signal(defn)
        entry = sess.get_calculated_signal("c1")
        assert entry is not None
        assert entry.definition is defn
        assert entry.result is None

    def test_get_unknown_id_returns_none(self) -> None:
        sess = EventAnalysisSession()
        assert sess.get_calculated_signal("nope") is None
        assert sess.get_calculated_signal_definition("nope") is None
        assert sess.get_calculated_signal_result("nope") is None

    def test_list_is_deterministic_insertion_order(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "First", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.add_calculated_signal(_defn("c2", "Second", "A", {"A": ChannelRef(sid_a, "MW")}, "A"))
        ids = [e.definition.calc_id for e in sess.list_calculated_signals()]
        assert ids == ["c1", "c2"]
        # Repeated calls are stable.
        assert [e.definition.calc_id for e in sess.list_calculated_signals()] == ids

    def test_internal_dict_not_exposed(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        signals = sess.list_calculated_signals()
        signals.clear()  # mutate the returned list
        assert len(sess.list_calculated_signals()) == 1  # session unaffected


# ─────────────────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────────────────


class TestDependencies:
    def test_dependencies_recorded(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("c1", "N1", "A + B", {
            "A": ChannelRef(sid_a, "Va"), "B": ChannelRef(sid_b, "Ia"),
        }, "A")
        sess.add_calculated_signal(defn)
        deps = sess.get_calculated_dependencies("c1")
        assert set(deps) == {ChannelRef(sid_a, "Va"), ChannelRef(sid_b, "Ia")}

    def test_dependents_by_source(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        assert sess.get_calculated_dependents_for_source(sid_a) == ("c1",)
        assert sess.get_calculated_dependents_for_source(sid_b) == ()

    def test_dependents_by_channel(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        assert sess.get_calculated_dependents_for_channel(ChannelRef(sid_a, "Va")) == ("c1",)
        assert sess.get_calculated_dependents_for_channel(ChannelRef(sid_a, "MW")) == ()

    def test_multiple_calculations_sharing_same_source(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.add_calculated_signal(_defn("c2", "N2", "A", {"A": ChannelRef(sid_a, "MW")}, "A"))
        assert sess.get_calculated_dependents_for_source(sid_a) == ("c1", "c2")

    def test_multiple_calculations_sharing_same_channel(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        ref = ChannelRef(sid_a, "Va")
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ref}, "A"))
        sess.add_calculated_signal(_defn("c2", "N2", "A * 2", {"A": ref}, "A"))
        assert sess.get_calculated_dependents_for_channel(ref) == ("c1", "c2")

    def test_delete_cleans_reverse_indexes(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        ref = ChannelRef(sid_a, "Va")
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ref}, "A"))
        sess.remove_calculated_signal("c1")
        assert sess.get_calculated_dependents_for_source(sid_a) == ()
        assert sess.get_calculated_dependents_for_channel(ref) == ()
        assert sess.get_calculated_dependencies("c1") == ()

    def test_edit_updates_reverse_indexes(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        old_ref = ChannelRef(sid_a, "Va")
        new_ref = ChannelRef(sid_b, "Ia")
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": old_ref}, "A"))
        updated = _defn("c1", "N1", "A", {"A": new_ref}, "A")
        sess.update_calculated_signal_definition("c1", updated)
        assert sess.get_calculated_dependents_for_channel(old_ref) == ()
        assert sess.get_calculated_dependents_for_channel(new_ref) == ("c1",)
        assert sess.get_calculated_dependents_for_source(sid_a) == ()
        assert sess.get_calculated_dependents_for_source(sid_b) == ("c1",)


# ─────────────────────────────────────────────────────────────────────────────
# Editing
# ─────────────────────────────────────────────────────────────────────────────


class TestEditing:
    def _setup(self):
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        sess.add_calculated_signal(defn, result=_result("c1"))
        return sess, sid_a, sid_b

    def test_expression_change_marks_stale(self) -> None:
        sess, sid_a, _ = self._setup()
        new_defn = _defn("c1", "N1", "A * 2", {"A": ChannelRef(sid_a, "Va")}, "A")
        sess.update_calculated_signal_definition("c1", new_defn)
        assert sess.get_calculated_signal_result("c1").status == CalculationStatus.STALE

    def test_binding_change_marks_stale(self) -> None:
        sess, sid_a, _ = self._setup()
        new_defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "MW")}, "A")
        sess.update_calculated_signal_definition("c1", new_defn)
        assert sess.get_calculated_signal_result("c1").status == CalculationStatus.STALE

    def test_reference_variable_change_marks_stale(self) -> None:
        sess, sid_a, _ = self._setup()
        new_defn = _defn("c1", "N1", "A + B", {
            "A": ChannelRef(sid_a, "Va"), "B": ChannelRef(sid_a, "MW"),
        }, "B")
        sess.update_calculated_signal_definition("c1", new_defn)
        assert sess.get_calculated_signal_result("c1").status == CalculationStatus.STALE

    def test_output_unit_change_marks_stale(self) -> None:
        sess, sid_a, _ = self._setup()
        new_defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A", output_unit="kV")
        sess.update_calculated_signal_definition("c1", new_defn)
        assert sess.get_calculated_signal_result("c1").status == CalculationStatus.STALE

    def test_interpolation_change_marks_stale(self) -> None:
        sess, sid_a, _ = self._setup()
        new_defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A", interpolation="linear")
        # interpolation is already "linear" by default -- force a genuine
        # change is impossible in V1 (only "linear" is a supported value),
        # so instead verify that an *identical* interpolation value does
        # NOT spuriously stale (sanity companion to the meaningful-field set).
        sess.update_calculated_signal_definition("c1", new_defn)
        assert sess.get_calculated_signal_result("c1").status == CalculationStatus.OK

    def test_name_only_change_preserves_result_freshness(self) -> None:
        sess, sid_a, _ = self._setup()
        new_defn = _defn("c1", "Renamed", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        sess.update_calculated_signal_definition("c1", new_defn)
        assert sess.get_calculated_signal_result("c1").status == CalculationStatus.OK
        assert sess.get_calculated_signal_definition("c1").name == "Renamed"

    def test_calc_id_change_rejected(self) -> None:
        sess, sid_a, _ = self._setup()
        new_defn = _defn("different", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        with pytest.raises(ValueError, match="cannot change"):
            sess.update_calculated_signal_definition("c1", new_defn)

    def test_duplicate_name_edit_rejected(self) -> None:
        sess, sid_a, _ = self._setup()
        sess.add_calculated_signal(_defn("c2", "Other", "A", {"A": ChannelRef(sid_a, "MW")}, "A"))
        new_defn = _defn("c1", "Other", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        with pytest.raises(ValueError, match="already exists"):
            sess.update_calculated_signal_definition("c1", new_defn)

    def test_failed_edit_is_atomic(self) -> None:
        sess, sid_a, sid_b = self._setup()
        original_defn = sess.get_calculated_signal_definition("c1")
        original_result = sess.get_calculated_signal_result("c1")
        original_deps = sess.get_calculated_dependencies("c1")

        bad_defn = _defn("c1", "N1", "A", {"A": ChannelRef(sid_b, "Trip")}, "A")
        with pytest.raises(ValueError):
            sess.update_calculated_signal_definition("c1", bad_defn)

        assert sess.get_calculated_signal_definition("c1") is original_defn
        assert sess.get_calculated_signal_result("c1") is original_result
        assert sess.get_calculated_dependencies("c1") == original_deps

    def test_edit_unknown_calc_id_raises(self) -> None:
        sess, sid_a, _ = self._setup()
        new_defn = _defn("nope", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A")
        with pytest.raises(KeyError):
            sess.update_calculated_signal_definition("nope", new_defn)

    def test_edit_with_no_prior_result_stays_none(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        new_defn = _defn("c1", "N1", "A * 2", {"A": ChannelRef(sid_a, "Va")}, "A")
        sess.update_calculated_signal_definition("c1", new_defn)
        assert sess.get_calculated_signal_result("c1") is None


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────


class TestResults:
    def test_set_result(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.set_calculated_signal_result("c1", _result("c1"))
        assert sess.get_calculated_signal_result("c1").status == CalculationStatus.OK

    def test_replace_result(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.set_calculated_signal_result("c1", _result("c1"))
        second = _result("c1", status=CalculationStatus.ERROR, error_message="boom")
        sess.set_calculated_signal_result("c1", second)
        assert sess.get_calculated_signal_result("c1") is second

    def test_set_result_unknown_calc_id_raises(self) -> None:
        sess = EventAnalysisSession()
        with pytest.raises(KeyError):
            sess.set_calculated_signal_result("nope", _result("nope"))

    def test_set_result_mismatched_id_rejected(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        with pytest.raises(ValueError, match="does not match"):
            sess.set_calculated_signal_result("c1", _result("other"))

    def test_mark_stale_keeps_arrays(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        original = _result("c1")
        sess.set_calculated_signal_result("c1", original)
        sess.mark_calculated_signal_stale("c1")
        staled = sess.get_calculated_signal_result("c1")
        assert staled.status == CalculationStatus.STALE
        np.testing.assert_array_equal(staled.time, original.time)
        np.testing.assert_array_equal(staled.values, original.values)
        np.testing.assert_array_equal(staled.validity_mask, original.validity_mask)

    def test_mark_stale_reason_becomes_warning(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.set_calculated_signal_result("c1", _result("c1"))
        sess.mark_calculated_signal_stale("c1", reason="Source offset changed")
        warnings = sess.get_calculated_signal_result("c1").warnings
        assert any("Source offset changed" in w for w in warnings)

    def test_mark_stale_without_reason_no_extra_warning_text(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        original = _result("c1")
        sess.set_calculated_signal_result("c1", original)
        sess.mark_calculated_signal_stale("c1")
        staled = sess.get_calculated_signal_result("c1")
        assert staled.warnings == original.warnings

    def test_mark_stale_with_no_result_is_a_noop(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.mark_calculated_signal_stale("c1", reason="irrelevant")
        assert sess.get_calculated_signal_result("c1") is None

    def test_mark_stale_unknown_calc_id_raises(self) -> None:
        sess = EventAnalysisSession()
        with pytest.raises(KeyError):
            sess.mark_calculated_signal_stale("nope")

    def test_mark_stale_preserves_computed_at(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        original = _result("c1")
        sess.set_calculated_signal_result("c1", original)
        sess.mark_calculated_signal_stale("c1")
        assert sess.get_calculated_signal_result("c1").computed_at == original.computed_at


# ─────────────────────────────────────────────────────────────────────────────
# Removal
# ─────────────────────────────────────────────────────────────────────────────


class TestRemoval:
    def test_removes_entry(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.remove_calculated_signal("c1")
        assert sess.get_calculated_signal("c1") is None
        assert sess.list_calculated_signals() == []

    def test_removes_dependencies(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        ref = ChannelRef(sid_a, "Va")
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ref}, "A"))
        sess.remove_calculated_signal("c1")
        assert sess.get_calculated_dependents_for_channel(ref) == ()
        assert sess.get_calculated_dependents_for_source(sid_a) == ()

    def test_original_source_remains_untouched(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.remove_calculated_signal("c1")
        assert sess.get_source(sid_a) is not None
        assert sess.get_channel(sid_a, "Va") is not None

    def test_missing_calc_id_is_a_noop(self) -> None:
        sess = EventAnalysisSession()
        sess.remove_calculated_signal("nope")  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Analog eligibility
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalogEligibility:
    def test_analog_valid(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        result = sess.check_calculated_input_eligibility(ChannelRef(sid_a, "Va"))
        assert result.is_valid
        assert result.eligibility == ChannelEligibility.VALID

    def test_digital_invalid(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        result = sess.check_calculated_input_eligibility(ChannelRef(sid_b, "Trip"))
        assert not result.is_valid
        assert result.eligibility == ChannelEligibility.DIGITAL_CHANNEL

    def test_same_name_in_analog_and_digital_invalid(self) -> None:
        sess = EventAnalysisSession()
        record = _make_record(["Dup"])
        record.digital_channels.append(DigitalChannel(name="Dup", index=99))
        sid = sess.add_source(record, "Source", "csv")
        result = sess.check_calculated_input_eligibility(ChannelRef(sid, "Dup"))
        assert not result.is_valid
        assert result.eligibility == ChannelEligibility.DIGITAL_CHANNEL

    def test_missing_dataframe_column_invalid(self) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(
            _make_record(["Ghost"], declared_but_missing_column="Ghost"), "Source", "csv"
        )
        result = sess.check_calculated_input_eligibility(ChannelRef(sid, "Ghost"))
        assert not result.is_valid
        assert result.eligibility == ChannelEligibility.DATA_COLUMN_MISSING

    def test_non_numeric_analog_invalid(self) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(
            _make_record(["Weird"], non_numeric_columns=["Weird"]), "Source", "csv"
        )
        result = sess.check_calculated_input_eligibility(ChannelRef(sid, "Weird"))
        assert not result.is_valid
        assert result.eligibility == ChannelEligibility.NON_NUMERIC_CHANNEL

    def test_parameter_type_none_still_valid(self) -> None:
        sess = EventAnalysisSession()
        record = _make_record(["Plain"])
        assert record.analog_channels[0].parameter_type is None  # default, never set
        sid = sess.add_source(record, "Source", "csv")
        result = sess.check_calculated_input_eligibility(ChannelRef(sid, "Plain"))
        assert result.is_valid

    def test_unit_none_still_valid(self) -> None:
        sess = EventAnalysisSession()
        record = _make_record({"Plain": None})
        sid = sess.add_source(record, "Source", "csv")
        result = sess.check_calculated_input_eligibility(ChannelRef(sid, "Plain"))
        assert result.is_valid

    def test_missing_channel_name_invalid(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        result = sess.check_calculated_input_eligibility(ChannelRef(sid_a, "NoSuchChannel"))
        assert not result.is_valid
        assert result.eligibility == ChannelEligibility.MISSING_CHANNEL

    def test_missing_source_invalid(self) -> None:
        sess, _, _ = _session_with_two_sources()
        result = sess.check_calculated_input_eligibility(ChannelRef("nonexistent", "X"))
        assert not result.is_valid
        assert result.eligibility == ChannelEligibility.MISSING_SOURCE

    def test_inactive_source_invalid(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.set_source_active(sid_a, False)
        result = sess.check_calculated_input_eligibility(ChannelRef(sid_a, "Va"))
        assert not result.is_valid
        assert result.eligibility == ChannelEligibility.INACTIVE_SOURCE

    def test_is_valid_calculated_input_wrapper(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        assert sess.is_valid_calculated_input(ChannelRef(sid_a, "Va")) is True
        assert sess.is_valid_calculated_input(ChannelRef(sid_b, "Trip")) is False


# ─────────────────────────────────────────────────────────────────────────────
# Dependency status (source removal / deactivation discoverability)
# ─────────────────────────────────────────────────────────────────────────────


class TestDependencyStatus:
    def test_resolvable_when_all_deps_valid(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        status = sess.get_dependency_status("c1")
        assert status.is_resolvable

    def test_unresolvable_after_source_deactivated(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.set_source_active(sid_a, False)
        status = sess.get_dependency_status("c1")
        assert not status.is_resolvable
        assert status.inactive_sources == (sid_a,)

    def test_unresolvable_after_source_removed(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.remove_source(sid_a)
        status = sess.get_dependency_status("c1")
        assert not status.is_resolvable
        assert status.missing_sources == (sid_a,)
        # The calculated-signal definition itself is NOT deleted by source removal.
        assert sess.get_calculated_signal("c1") is not None

    def test_dependents_still_discoverable_after_source_removed(self) -> None:
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.remove_source(sid_a)
        # Even though the source is gone, the reverse index (keyed by the
        # now-defunct source_id string) still reports the impact.
        assert sess.get_calculated_dependents_for_source(sid_a) == ("c1",)

    def test_unknown_calc_id_returns_trivially_resolvable(self) -> None:
        sess = EventAnalysisSession()
        status = sess.get_dependency_status("nope")
        assert status.is_resolvable
        assert status.missing_sources == status.inactive_sources == status.missing_channels == status.digital_channels == ()


# ─────────────────────────────────────────────────────────────────────────────
# Existing session regression -- pre-existing APIs unaffected
# ─────────────────────────────────────────────────────────────────────────────


class TestExistingSessionRegression:
    def test_add_and_remove_source_unaffected(self) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(_make_record(["Va"]), "Source", "csv")
        assert sess.get_source(sid) is not None
        sess.remove_source(sid)
        assert sess.get_source(sid) is None

    def test_channel_registry_unaffected(self) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(_make_record(["Va"], digital=["Trip"]), "Source", "csv")
        assert len(sess.list_analog_channels()) == 1
        assert len(sess.list_digital_channels()) == 1

    def test_time_offset_unaffected(self) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(_make_record(["Va"]), "Source", "csv")
        sess.set_time_offset(sid, 1.5, method="manual")
        assert sess.get_time_offset(sid) == 1.5

    def test_default_layout_unaffected(self) -> None:
        sess = EventAnalysisSession()
        sess.add_source(_make_record(["Va"], digital=["Trip"]), "Source", "csv")
        sess.default_layout()
        assert len(sess.list_panels()) >= 1

    def test_remove_source_does_not_touch_calculated_signals_dicts_shape(self) -> None:
        # Phase 2C-1 deliberately does not wire calculated-signal cleanup
        # into remove_source() -- confirm it still behaves exactly as before
        # (no exception, no unexpected side effect) when calculated signals
        # exist and depend on the removed source.
        sess, sid_a, _ = _session_with_two_sources()
        sess.add_calculated_signal(_defn("c1", "N1", "A", {"A": ChannelRef(sid_a, "Va")}, "A"))
        sess.remove_source(sid_a)  # must not raise
        assert sess.get_source(sid_a) is None
        assert sess.get_calculated_signal("c1") is not None  # not silently deleted
