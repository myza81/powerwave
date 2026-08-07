"""Unit tests for the Calculated Signal Resolution Service (Phase 2C-3) in
app.calculated_signals.resolver.

Uses only generic, synthetic session fixtures (Source A / Source B with
made-up analog/digital channel names) -- no filename, station, or event
identity is special-cased anywhere in this file or in production code.
"""
from __future__ import annotations

import time as time_module
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.calculated_signals.engine import CalculationEngineConfig
from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculatedSignalResult,
    CalculationStatus,
    ChannelRef,
)
from app.calculated_signals.resolver import (
    CalculatedSignalResolutionError,
    CalculatedSignalResolutionService,
    ResolutionBatchResult,
    ResolutionFailure,
)
import app.calculated_signals.resolver as resolver_module
from app.models.channels import AnalogChannel, DigitalChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.event_session import EventAnalysisSession


# ─────────────────────────────────────────────────────────────────────────────
# Generic synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_record(
    analog: dict[str, str] | list[str],
    digital: list[str] | None = None,
    time: np.ndarray | None = None,
    values: dict[str, np.ndarray] | None = None,
    n: int = 10,
    non_numeric_columns: list[str] | None = None,
    declared_but_missing_column: str | None = None,
) -> DisturbanceRecord:
    """Build a minimal, generic DisturbanceRecord.

    analog: either a list of names (unit defaults to "MW") or a
            {name: unit} mapping.
    time: explicit time array; defaults to np.linspace(0, 1, n).
    values: explicit {channel_name: array} overrides; defaults to
            np.arange(n, dtype=float) per analog channel.
    """
    if isinstance(analog, list):
        analog = {name: "MW" for name in analog}
    digital = digital or []
    non_numeric_columns = non_numeric_columns or []
    values = values or {}

    if time is not None:
        n = len(time)
    else:
        time = np.linspace(0, 1, n)

    data: dict[str, object] = {"time": time}
    for name in analog:
        if name == declared_but_missing_column:
            continue
        if name in non_numeric_columns:
            data[name] = [f"x{i}" for i in range(n)]
        elif name in values:
            data[name] = values[name]
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


def _defn(
    calc_id: str, name: str, expression: str,
    bindings: dict[str, ChannelRef], reference: str, **kwargs
) -> CalculatedSignalDefinition:
    return CalculatedSignalDefinition(
        calc_id=calc_id, name=name, expression=expression,
        variable_bindings=bindings, reference_variable=reference, **kwargs,
    )


def _manual_result(
    calc_id: str, status: CalculationStatus = CalculationStatus.OK,
    error_message: str | None = None,
) -> CalculatedSignalResult:
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
# Input resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveInput:
    def test_offset_is_applied(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        sess.set_time_offset(sid_b, 5.0)
        svc = CalculatedSignalResolutionService(sess)

        resolved = svc._resolve_input("b", ChannelRef(sid_b, "Ia"))
        expected_time = np.linspace(0, 1, 10) + 5.0
        np.testing.assert_array_almost_equal(resolved.time, expected_time)

    def test_zero_offset_leaves_time_unchanged(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        svc = CalculatedSignalResolutionService(sess)

        resolved = svc._resolve_input("a", ChannelRef(sid_a, "Va"))
        np.testing.assert_array_almost_equal(resolved.time, np.linspace(0, 1, 10))

    def test_values_and_unit_preserved(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        svc = CalculatedSignalResolutionService(sess)

        resolved = svc._resolve_input("a", ChannelRef(sid_a, "Va"))
        np.testing.assert_array_almost_equal(resolved.values, np.arange(10, dtype=float))
        assert resolved.unit == "kV"
        assert resolved.variable == "a"
        assert resolved.source_id == sid_a
        assert resolved.channel_name == "Va"

    def test_full_resolution_not_decimated_display_style(self) -> None:
        """Calculated Signals must resolve every sample, unlike
        build_aligned_data() which decimates above max_points (default
        4000). A record with more than max_points samples must still
        resolve at full resolution here.
        """
        n = 10_000
        record = _make_record({"Va": "kV"}, time=np.linspace(0, 10, n))
        sess = EventAnalysisSession()
        sid = sess.add_source(record, "Source A", "csv")
        svc = CalculatedSignalResolutionService(sess)

        resolved = svc._resolve_input("a", ChannelRef(sid, "Va"))
        assert resolved.time.size == n
        assert resolved.values.size == n

        # Contrast with the display path, which does decimate.
        aligned = sess.build_aligned_data(sid, "Va", 0.0, 10.0, max_points=4000)
        assert aligned.time.size < n

    def test_does_not_mutate_source_record(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        sess.set_time_offset(sid_a, 3.0)
        svc = CalculatedSignalResolutionService(sess)

        before = np.asarray(sess.get_source(sid_a).record.waveform_data["time"]).copy()
        svc._resolve_input("a", ChannelRef(sid_a, "Va"))
        after = np.asarray(sess.get_source(sid_a).record.waveform_data["time"])
        np.testing.assert_array_equal(before, after)

    def test_missing_source_raises_resolution_error(self) -> None:
        sess = EventAnalysisSession()
        svc = CalculatedSignalResolutionService(sess)
        with pytest.raises(CalculatedSignalResolutionError, match="missing_source"):
            svc._resolve_input("a", ChannelRef("nonexistent", "Va"))

    def test_digital_channel_raises_resolution_error(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        svc = CalculatedSignalResolutionService(sess)
        with pytest.raises(CalculatedSignalResolutionError, match="digital_channel"):
            svc._resolve_input("b", ChannelRef(sid_b, "Trip"))


# ─────────────────────────────────────────────────────────────────────────────
# resolve_one
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveOne:
    def test_successful_resolution_stores_ok_result(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn(
            "calc-1", "MyCalc", "a + a",
            {"a": ChannelRef(sid_a, "Va")}, "a",
        )
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        result = svc.resolve_one("calc-1")
        assert result.status == CalculationStatus.OK
        stored = sess.get_calculated_signal_result("calc-1")
        assert stored is result

    def test_unknown_calc_id_raises(self) -> None:
        sess = EventAnalysisSession()
        svc = CalculatedSignalResolutionService(sess)
        with pytest.raises(CalculatedSignalResolutionError, match="unknown calc_id"):
            svc.resolve_one("nope")

    def test_does_not_modify_definition(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn(
            "calc-1", "MyCalc", "a + a",
            {"a": ChannelRef(sid_a, "Va")}, "a",
        )
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        svc.resolve_one("calc-1")
        assert sess.get_calculated_signal_definition("calc-1") is defn

    def test_does_not_modify_dependency_indexes(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn(
            "calc-1", "MyCalc", "a + a",
            {"a": ChannelRef(sid_a, "Va")}, "a",
        )
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        before = sess.get_calculated_dependencies("calc-1")
        svc.resolve_one("calc-1")
        after = sess.get_calculated_dependencies("calc-1")
        assert before == after

    def test_unresolvable_dependency_raises_without_fabricating_result(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn(
            "calc-1", "MyCalc", "a + a",
            {"a": ChannelRef(sid_a, "Va")}, "a",
        )
        sess.add_calculated_signal(defn)
        sess.set_source_active(sid_a, False)
        svc = CalculatedSignalResolutionService(sess)

        with pytest.raises(CalculatedSignalResolutionError, match="unresolvable dependencies"):
            svc.resolve_one("calc-1")
        assert sess.get_calculated_signal_result("calc-1") is None

    def test_calculation_engine_failure_wrapped_in_resolution_error(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        # kV and A are incompatible unit families -> UnitCompatibilityError
        defn = _defn(
            "calc-1", "MyCalc", "a + b",
            {"a": ChannelRef(sid_a, "Va"), "b": ChannelRef(sid_b, "Ia")}, "a",
        )
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        with pytest.raises(CalculatedSignalResolutionError, match="failed to calculate"):
            svc.resolve_one("calc-1")

    def test_engine_config_is_forwarded(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn(
            "calc-1", "MyCalc", "a + a",
            {"a": ChannelRef(sid_a, "Va")}, "a",
        )
        sess.add_calculated_signal(defn)
        cfg = CalculationEngineConfig(gap_multiplier=2.0)
        svc = CalculatedSignalResolutionService(sess, engine_config=cfg)

        result = svc.resolve_one("calc-1")
        assert result.status == CalculationStatus.OK


# ─────────────────────────────────────────────────────────────────────────────
# Failure preservation (never destroy last-known-good data)
# ─────────────────────────────────────────────────────────────────────────────


class TestFailurePreservation:
    def test_ok_result_preserved_on_subsequent_failure(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn(
            "calc-1", "MyCalc", "a + a",
            {"a": ChannelRef(sid_a, "Va")}, "a",
        )
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        first = svc.resolve_one("calc-1")
        assert first.status == CalculationStatus.OK

        # Deactivating the source makes the dependency unresolvable; the
        # session's own lifecycle logic (Phase 2C-2) immediately replaces
        # the stored result with a STALE copy -- capture that object here
        # so we can prove the resolver leaves it completely untouched.
        sess.set_source_active(sid_a, False)
        stale = sess.get_calculated_signal_result("calc-1")
        assert stale.status == CalculationStatus.STALE

        with pytest.raises(CalculatedSignalResolutionError):
            svc.resolve_one("calc-1")

        preserved = sess.get_calculated_signal_result("calc-1")
        assert preserved is stale

    def test_error_result_preserved_as_error_on_further_failure(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn(
            "calc-1", "MyCalc", "a + a",
            {"a": ChannelRef(sid_a, "Va")}, "a",
        )
        error_result = _manual_result("calc-1", status=CalculationStatus.ERROR, error_message="prior failure")
        sess.add_calculated_signal(defn, error_result)
        sess.set_source_active(sid_a, False)
        svc = CalculatedSignalResolutionService(sess)

        with pytest.raises(CalculatedSignalResolutionError):
            svc.resolve_one("calc-1")

        preserved = sess.get_calculated_signal_result("calc-1")
        assert preserved is error_result
        assert preserved.status == CalculationStatus.ERROR

    def test_never_calculated_stays_none_on_failure(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn(
            "calc-1", "MyCalc", "a + a",
            {"a": ChannelRef(sid_a, "Va")}, "a",
        )
        sess.add_calculated_signal(defn)
        sess.set_source_active(sid_a, False)
        svc = CalculatedSignalResolutionService(sess)

        with pytest.raises(CalculatedSignalResolutionError):
            svc.resolve_one("calc-1")
        assert sess.get_calculated_signal_result("calc-1") is None

    def test_resolution_failure_not_appended_to_result_warnings(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn(
            "calc-1", "MyCalc", "a + a",
            {"a": ChannelRef(sid_a, "Va")}, "a",
        )
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)
        first = svc.resolve_one("calc-1")
        assert first.warnings == []

        sess.set_source_active(sid_a, False)
        stale = sess.get_calculated_signal_result("calc-1")
        warnings_before = list(stale.warnings)  # includes the session's own "Marked stale: ..." note

        with pytest.raises(CalculatedSignalResolutionError):
            svc.resolve_one("calc-1")

        preserved = sess.get_calculated_signal_result("calc-1")
        assert preserved is stale
        assert preserved.warnings == warnings_before  # resolver appended nothing of its own


# ─────────────────────────────────────────────────────────────────────────────
# Batch APIs
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchAPIs:
    def _two_signals_one_broken(self) -> tuple[EventAnalysisSession, CalculatedSignalResolutionService, str, str]:
        sess, sid_a, sid_b = _session_with_two_sources()
        good = _defn("calc-good", "Good", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        bad = _defn("calc-bad", "Bad", "a + a", {"a": ChannelRef(sid_b, "Ia")}, "a")
        sess.add_calculated_signal(good)
        sess.add_calculated_signal(bad)
        sess.set_source_active(sid_b, False)  # makes calc-bad unresolvable
        svc = CalculatedSignalResolutionService(sess)
        return sess, svc, "calc-good", "calc-bad"

    def test_resolve_all_continues_past_failures(self) -> None:
        sess, svc, good_id, bad_id = self._two_signals_one_broken()
        batch = svc.resolve_all()
        assert isinstance(batch, ResolutionBatchResult)
        assert len(batch.successful) == 1
        assert batch.successful[0].calc_id == good_id
        assert len(batch.failures) == 1
        assert isinstance(batch.failures[0], ResolutionFailure)
        assert batch.failures[0].calc_id == bad_id
        assert batch.failures[0].message  # non-empty, human-readable

    def test_resolve_all_stale_only_touches_stale_signals(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        # Never calculated -> not STALE -> resolve_all_stale ignores it.
        batch = svc.resolve_all_stale()
        assert batch.successful == ()
        assert batch.failures == ()
        assert sess.get_calculated_signal_result("calc-1") is None

        svc.resolve_one("calc-1")
        sess.set_time_offset(sid_a, 1.0)  # marks stale
        assert sess.get_stale_calculated_signal_ids() == ("calc-1",)

        batch = svc.resolve_all_stale()
        assert len(batch.successful) == 1
        assert batch.successful[0].calc_id == "calc-1"
        assert sess.get_calculated_signal_result("calc-1").status == CalculationStatus.OK

    def test_resolve_for_source_uses_reverse_dependency_index(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        depends_on_a = _defn("calc-a", "CA", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        depends_on_b = _defn("calc-b", "CB", "b + b", {"b": ChannelRef(sid_b, "Ia")}, "b")
        sess.add_calculated_signal(depends_on_a)
        sess.add_calculated_signal(depends_on_b)
        svc = CalculatedSignalResolutionService(sess)

        batch = svc.resolve_for_source(sid_a)
        assert len(batch.successful) == 1
        assert batch.successful[0].calc_id == "calc-a"
        assert sess.get_calculated_signal_result("calc-b") is None  # untouched

    def test_resolve_all_deterministic_order_matches_insertion(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        for i in range(5):
            defn = _defn(f"calc-{i}", f"C{i}", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
            sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        batch = svc.resolve_all()
        assert [r.calc_id for r in batch.successful] == [f"calc-{i}" for i in range(5)]


# ─────────────────────────────────────────────────────────────────────────────
# Full lifecycle round-trips
# ─────────────────────────────────────────────────────────────────────────────


class TestLifecycleRoundTrip:
    def test_offset_change_round_trip(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        first = svc.resolve_one("calc-1")
        assert first.status == CalculationStatus.OK
        first_time = first.time.copy()

        sess.set_time_offset(sid_a, 2.0)
        assert sess.get_calculated_signal_result("calc-1").status == CalculationStatus.STALE

        second = svc.resolve_one("calc-1")
        assert second.status == CalculationStatus.OK
        np.testing.assert_array_almost_equal(second.time, first_time + 2.0)

    def test_method_only_change_round_trip(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)
        svc.resolve_one("calc-1")

        sess.set_time_offset(sid_a, 0.0, method="auto_trigger", confidence=0.9)
        assert sess.get_calculated_signal_result("calc-1").status == CalculationStatus.STALE
        result = svc.resolve_one("calc-1")
        assert result.status == CalculationStatus.OK

    def test_deactivation_then_reactivation_round_trip(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)
        svc.resolve_one("calc-1")

        sess.set_source_active(sid_a, False)
        stale = sess.get_calculated_signal_result("calc-1")
        with pytest.raises(CalculatedSignalResolutionError):
            svc.resolve_one("calc-1")
        assert sess.get_calculated_signal_result("calc-1") is stale  # untouched

        sess.set_source_active(sid_a, True)
        second = svc.resolve_one("calc-1")
        assert second.status == CalculationStatus.OK

    def test_removal_leaves_definition_and_fails_resolution(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)
        svc.resolve_one("calc-1")

        sess.remove_source(sid_a)
        assert sess.get_calculated_signal_definition("calc-1") is not None  # retained

        with pytest.raises(CalculatedSignalResolutionError):
            svc.resolve_one("calc-1")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-source alignment correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestCrossSourceAlignment:
    def _build_sources(self, offset_b: float) -> tuple[EventAnalysisSession, str, str]:
        sess = EventAnalysisSession()
        sid_a = sess.add_source(
            _make_record({"A": "pu"}, time=np.array([0.0, 1.0, 2.0]), values={"A": np.array([10.0, 20.0, 30.0])}),
            "Source A", "csv",
        )
        sid_b = sess.add_source(
            _make_record({"B": "pu"}, time=np.array([0.0, 1.0, 2.0]), values={"B": np.array([1.0, 2.0, 3.0])}),
            "Source B", "csv",
        )
        sess.set_time_offset(sid_b, offset_b)
        return sess, sid_a, sid_b

    def test_positive_offset_aligns_correctly(self) -> None:
        sess, sid_a, sid_b = self._build_sources(offset_b=1.0)
        defn = _defn(
            "calc-1", "AminusB", "a - b",
            {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a",
        )
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        result = svc.resolve_one("calc-1")
        np.testing.assert_array_almost_equal(result.time, [1.0, 2.0])
        np.testing.assert_array_almost_equal(result.values, [19.0, 28.0])

    def test_negative_offset_aligns_correctly(self) -> None:
        sess, sid_a, sid_b = self._build_sources(offset_b=-1.0)
        defn = _defn(
            "calc-1", "AminusB", "a - b",
            {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a",
        )
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        result = svc.resolve_one("calc-1")
        np.testing.assert_array_almost_equal(result.time, [0.0, 1.0])
        np.testing.assert_array_almost_equal(result.values, [8.0, 17.0])

    def test_reference_variable_choice_does_not_change_which_source_wins(self) -> None:
        """No source-ordering special-casing: swapping which variable is the
        reference still produces the same overlap window content, just
        expressed on the other input's own sample points."""
        sess, sid_a, sid_b = self._build_sources(offset_b=1.0)
        defn = _defn(
            "calc-1", "BminusA", "b - a",
            {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "b",
        )
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        result = svc.resolve_one("calc-1")
        # Reference is now B's own (shifted) time base: [1.0, 2.0, 3.0],
        # trimmed to the overlap [1.0, 2.0].
        np.testing.assert_array_almost_equal(result.time, [1.0, 2.0])
        np.testing.assert_array_almost_equal(result.values, [-19.0, -28.0])


# ─────────────────────────────────────────────────────────────────────────────
# Source-name / display-name independence (calc_id is the only identity)
# ─────────────────────────────────────────────────────────────────────────────


class TestSourceNameIndependence:
    def test_renaming_source_display_name_does_not_affect_resolution(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        before = svc.resolve_one("calc-1")
        sess.get_source(sid_a).display_name = "Totally Renamed Source"
        after = svc.resolve_one("calc-1")

        np.testing.assert_array_almost_equal(before.values, after.values)
        assert after.status == CalculationStatus.OK

    def test_renaming_channel_display_name_does_not_affect_resolution(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        before = svc.resolve_one("calc-1")
        sess.set_channel_display_name(sid_a, "Va", "Voltage (renamed)")
        after = svc.resolve_one("calc-1")

        np.testing.assert_array_almost_equal(before.values, after.values)

    def test_binding_resolved_by_source_id_and_channel_name_only(self) -> None:
        """ChannelRef carries no display name at all -- resolution is keyed
        purely on (source_id, channel_name), never on any user-facing text."""
        sess, sid_a, sid_b = _session_with_two_sources()
        ref = ChannelRef(sid_a, "Va")
        assert not hasattr(ref, "display_name")


# ─────────────────────────────────────────────────────────────────────────────
# Analog-only enforcement
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalogOnlyEnforcement:
    def test_missing_source_rejected(self) -> None:
        sess = EventAnalysisSession()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef("ghost", "Va")}, "a")

        # add_calculated_signal itself validates dependencies at creation
        # time -- confirm it rejects this before the resolver ever sees it.
        with pytest.raises(ValueError):
            sess.add_calculated_signal(defn)

    def test_inactive_source_rejected_at_resolve_time(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        sess.add_calculated_signal(defn)
        sess.set_source_active(sid_a, False)
        svc = CalculatedSignalResolutionService(sess)

        with pytest.raises(CalculatedSignalResolutionError):
            svc.resolve_one("calc-1")

    def test_digital_channel_rejected_at_definition_time(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "b + b", {"b": ChannelRef(sid_b, "Trip")}, "b")
        with pytest.raises(ValueError, match="digital_channel"):
            sess.add_calculated_signal(defn)

    def test_missing_channel_rejected_at_definition_time(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid_a, "NoSuchChannel")}, "a")
        with pytest.raises(ValueError, match="missing_channel"):
            sess.add_calculated_signal(defn)

    def test_non_numeric_channel_rejected_at_definition_time(self) -> None:
        sess = EventAnalysisSession()
        record = _make_record({"Va": "kV"}, non_numeric_columns=["Va"])
        sid = sess.add_source(record, "Source A", "csv")
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid, "Va")}, "a")
        with pytest.raises(ValueError, match="non_numeric_channel"):
            sess.add_calculated_signal(defn)

    def test_data_column_missing_rejected_at_resolve_time(self) -> None:
        """A channel declared analog but whose waveform_data column later
        disappears (simulated by clearing the DataFrame) fails eligibility
        at resolve time even though it passed at definition time."""
        sess = EventAnalysisSession()
        record = _make_record({"Va": "kV"})
        sid = sess.add_source(record, "Source A", "csv")
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid, "Va")}, "a")
        sess.add_calculated_signal(defn)

        record.waveform_data.drop(columns=["Va"], inplace=True)
        svc = CalculatedSignalResolutionService(sess)
        with pytest.raises(CalculatedSignalResolutionError, match="unresolvable dependencies"):
            svc.resolve_one("calc-1")
        eligibility = sess.check_calculated_input_eligibility(ChannelRef(sid, "Va"))
        assert eligibility.eligibility.value == "data_column_missing"

    def test_calculate_signal_never_called_when_dependency_unresolvable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Structural proof: calculate_signal() is never invoked when the
        dependency-status pre-check already fails."""
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        sess.add_calculated_signal(defn)
        sess.set_source_active(sid_a, False)
        svc = CalculatedSignalResolutionService(sess)

        def _forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("calculate_signal() must not be called when dependencies are unresolvable")

        monkeypatch.setattr(resolver_module, "calculate_signal", _forbidden)

        with pytest.raises(CalculatedSignalResolutionError):
            svc.resolve_one("calc-1")


# ─────────────────────────────────────────────────────────────────────────────
# Calculated-signal-to-calculated-signal chaining remains rejected
# ─────────────────────────────────────────────────────────────────────────────


class TestNoChaining:
    def test_binding_to_a_calculated_signal_is_rejected_at_definition_time(self) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        base = _defn("calc-base", "Base", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        sess.add_calculated_signal(base)

        chained = _defn(
            "calc-chained", "Chained", "x + x",
            {"x": ChannelRef("calc-base", "Va")}, "x",
        )
        with pytest.raises(ValueError, match="calculated-signal-to-calculated-signal"):
            sess.add_calculated_signal(chained)

    def test_resolver_has_no_fallback_lookup_for_calculated_signal_sources(self) -> None:
        """The resolver never treats a calc_id as if it were a source_id --
        it relies entirely on the session's own definition-time rejection,
        with no cycle-detection or chaining logic of its own."""
        sess, sid_a, sid_b = _session_with_two_sources()
        defn = _defn("calc-1", "C1", "a + a", {"a": ChannelRef(sid_a, "Va")}, "a")
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        # No calculated signal can ever legally reference "calc-1" as a
        # source_id (add_calculated_signal always rejects it), so resolving
        # a fabricated ChannelRef pointing at it must fail as MISSING_SOURCE.
        with pytest.raises(CalculatedSignalResolutionError, match="missing_source"):
            svc._resolve_input("x", ChannelRef("calc-1", "Va"))


# ─────────────────────────────────────────────────────────────────────────────
# Performance (full-resolution, large arrays, cross-rate interpolation)
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformance:
    def test_million_sample_resolution_completes_quickly(self) -> None:
        n_a = 1_000_000
        n_b = 250_000  # different rate -> forces real interpolation, not identity passthrough
        time_a = np.linspace(0.0, 100.0, n_a)
        time_b = np.linspace(0.0, 100.0, n_b)

        sess = EventAnalysisSession()
        sid_a = sess.add_source(
            _make_record({"A": "pu"}, time=time_a, values={"A": np.sin(time_a)}),
            "Source A", "csv",
        )
        sid_b = sess.add_source(
            _make_record({"B": "pu"}, time=time_b, values={"B": np.cos(time_b)}),
            "Source B", "csv",
        )
        defn = _defn(
            "calc-1", "SumAB", "a + b",
            {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a",
        )
        sess.add_calculated_signal(defn)
        svc = CalculatedSignalResolutionService(sess)

        started = time_module.perf_counter()
        result = svc.resolve_one("calc-1")
        elapsed = time_module.perf_counter() - started

        assert result.status == CalculationStatus.OK
        assert result.time.size == n_a  # reference is A -> full resolution, no decimation
        assert elapsed < 15.0, f"resolve_one took {elapsed:.2f}s for {n_a}+{n_b} samples"
        print(f"\n[perf] resolve_one with {n_a}+{n_b} samples took {elapsed:.3f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous execution (no worker/thread infrastructure)
# ─────────────────────────────────────────────────────────────────────────────


class TestSynchronousExecution:
    def test_resolver_module_has_no_qt_or_threading_imports(self) -> None:
        import ast
        import inspect

        source = inspect.getsource(resolver_module)
        tree = ast.parse(source)
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        assert not any("PyQt" in m or m in ("threading", "concurrent.futures") for m in imported_modules)
