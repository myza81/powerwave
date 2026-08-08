"""Sprint 1C — calculated-signal recalculation after committed alignment
edits: call-count, lifecycle, dedup, failure-preservation, timing-warning
regression, and end-to-end tests against the real PowerwaveMainWindow.

This is the fix for the EAV-identified fine-nudge gap: fine-left/fine-right
changed a source's offset (staling dependent calculated signals) but never
triggered recalculation, because setValue() never fires editingFinished.

Uses the repository's offscreen Qt conventions (QT_QPA_PLATFORM=offscreen is
set process-wide by tests/conftest.py; a module-scoped `qapp` fixture
provides a single QApplication).
"""
from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from PyQt6.QtWidgets import QApplication

from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculationStatus,
    ChannelRef,
)
from app.calculated_signals.resolver import CalculatedSignalResolutionService
from app.models.channels import AnalogChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.alignment_engine import AlignmentResult
from app.sessions.event_session import EventAnalysisSession
from app.sessions.timing_compatibility import TimingCompatibilityLevel
from app.ui.main_window.main_window import PowerwaveMainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _record(analog: dict[str, str], time: np.ndarray) -> DisturbanceRecord:
    n = len(time)
    data: dict[str, object] = {"time": time}
    for name in analog:
        data[name] = np.sin(time)
    df = pd.DataFrame(data)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="GenericStation", recorder_name="GenericRecorder",
            source_file="generic.csv", provider_type="csv", nominal_frequency=50.0,
        ),
        waveform_data=df,
        analog_channels=[AnalogChannel(name=n, unit=u, index=i) for i, (n, u) in enumerate(analog.items())],
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[10.0], samples_per_rate=[n]),
        timing_info=TimingInformation(start_time=datetime(2024, 1, 1), trigger_time=datetime(2024, 1, 1)),
    )


def _build_window(qapp: QApplication) -> PowerwaveMainWindow:
    win = PowerwaveMainWindow()
    win._active_session = EventAnalysisSession()
    win._session_canvas_action.setEnabled(True)
    panel = win._ensure_session_panel()
    panel.refresh_all(win._active_session)
    win._activate_session_canvas()
    qapp.processEvents()
    return win


def _add_source(win: PowerwaveMainWindow, analog: dict[str, str], name: str, n: int = 200) -> str:
    sess = win._active_session
    time = np.linspace(0, 2, n)
    sid = sess.add_source(_record(analog, time), name, "csv")
    sess.default_layout()
    win._ensure_session_panel().refresh_all(sess)
    return sid


def _add_calc(win: PowerwaveMainWindow, calc_id: str, name: str, expr: str, bindings: dict, ref: str) -> None:
    sess = win._active_session
    defn = CalculatedSignalDefinition(
        calc_id=calc_id, name=name, expression=expr,
        variable_bindings=bindings, reference_variable=ref,
    )
    sess.add_calculated_signal(defn)
    CalculatedSignalResolutionService(sess).resolve_one(calc_id)
    win._sync_calculated_signals_to_canvas()


def _row(win: PowerwaveMainWindow, source_id: str):
    return win._session_panel._source_rows[source_id]


# ─────────────────────────────────────────────────────────────────────────────
# Fine nudge
# ─────────────────────────────────────────────────────────────────────────────


class TestFineNudgeRecalculation:
    def test_fine_right_changes_offset_and_recalculates_once(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
            before = win._active_session.get_calculated_signal_result("c1")
            assert before.status == CalculationStatus.OK
            values_before = before.values.copy()

            row_b = _row(win, sid_b)
            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy:
                row_b._on_fine_right()
                qapp.processEvents()
                assert spy.call_count == 1

            after = win._active_session.get_calculated_signal_result("c1")
            assert after.status == CalculationStatus.OK
            assert not np.array_equal(after.values, values_before)
            step = round(row_b._sample_interval_s, 3)
            assert win._active_session.get_source(sid_b).time_offset_s == pytest.approx(step)
        finally:
            win.close()
            qapp.processEvents()

    def test_fine_left_changes_offset_and_recalculates_once(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            row_b = _row(win, sid_b)
            step = round(row_b._sample_interval_s, 3)
            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy:
                row_b._on_fine_left()
                qapp.processEvents()
                assert spy.call_count == 1
            assert win._active_session.get_source(sid_b).time_offset_s == pytest.approx(-step)
        finally:
            win.close()
            qapp.processEvents()

    def test_stale_marking_precedes_recalculation(self, qapp: QApplication) -> None:
        """The valueChanged path marks the signal STALE synchronously
        before the committed event even fires -- verify by intercepting
        resolve_for_sources and checking the session already reports
        STALE at call time (proving the ordering, not just the outcome)."""
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            observed_status_at_call = []
            original = CalculatedSignalResolutionService.resolve_for_sources

            def _spy(self, source_ids):
                observed_status_at_call.append(
                    self._session.get_calculated_signal_result("c1").status
                )
                return original(self, source_ids)

            row_b = _row(win, sid_b)
            with patch.object(CalculatedSignalResolutionService, "resolve_for_sources", _spy):
                row_b._on_fine_right()
                qapp.processEvents()

            assert observed_status_at_call == [CalculationStatus.STALE]
        finally:
            win.close()
            qapp.processEvents()

    def test_unrelated_calculation_untouched_by_nudge(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            sid_c = _add_source(win, {"C": "MW"}, "Source C")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
            _add_calc(win, "c3", "DoubleC", "c + c", {"c": ChannelRef(sid_c, "C")}, "c")

            c3_before = win._active_session.get_calculated_signal_result("c3")

            row_b = _row(win, sid_b)
            row_b._on_fine_right()
            qapp.processEvents()

            c3_after = win._active_session.get_calculated_signal_result("c3")
            assert c3_after.computed_at == c3_before.computed_at
            assert c3_after.status == CalculationStatus.OK
        finally:
            win.close()
            qapp.processEvents()

    def test_no_duplicate_recalculation_for_single_click(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            row_b = _row(win, sid_b)
            with patch.object(
                CalculatedSignalResolutionService, "resolve_one", autospec=True,
                side_effect=CalculatedSignalResolutionService.resolve_one,
            ) as spy:
                row_b._on_fine_right()
                qapp.processEvents()
                calc1_calls = [c for c in spy.call_args_list if c.args[1] == "c1"]
                assert len(calc1_calls) == 1
        finally:
            win.close()
            qapp.processEvents()


class TestRepeatedFineNudges:
    def test_three_clicks_produce_three_recalculations(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            row_b = _row(win, sid_b)
            step = round(row_b._sample_interval_s, 3)
            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy:
                row_b._on_fine_right()
                qapp.processEvents()
                row_b._on_fine_right()
                qapp.processEvents()
                row_b._on_fine_right()
                qapp.processEvents()
                assert spy.call_count == 3
            assert win._active_session.get_source(sid_b).time_offset_s == pytest.approx(3 * step)
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Typed edit
# ─────────────────────────────────────────────────────────────────────────────


class TestTypedEditRecalculation:
    def test_change_and_editing_finished_recalculates_once(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            row_b = _row(win, sid_b)
            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy:
                row_b._offset_spin.setValue(0.5)
                row_b._on_offset_spin_editing_finished()
                qapp.processEvents()
                assert spy.call_count == 1
        finally:
            win.close()
            qapp.processEvents()

    def test_unchanged_editing_finished_recalculates_zero_times(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            row_b = _row(win, sid_b)
            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy:
                row_b._on_offset_spin_editing_finished()
                qapp.processEvents()
                assert spy.call_count == 0
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────────────────────


class TestResetRecalculation:
    def test_changed_reset_recalculates_once(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            win._active_session.set_time_offset(sid_b, 0.5, method="manual")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy:
                win._on_session_offset_reset(sid_b)
                qapp.processEvents()
                assert spy.call_count == 1
            assert win._active_session.get_source(sid_b).time_offset_s == pytest.approx(0.0)
        finally:
            win.close()
            qapp.processEvents()

    def test_noop_reset_recalculates_zero_times(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")  # offset already 0.0, method "none"
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy:
                win._on_session_offset_reset(sid_b)
                qapp.processEvents()
                assert spy.call_count == 0
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-align
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoAlignRecalculation:
    def test_changed_auto_align_result_recalculates_once(self, qapp: QApplication, monkeypatch) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            def _fake_suggest(sources):
                return [AlignmentResult(
                    source_id=sid_b, suggested_offset_s=1.5, alignment_method="auto_trigger",
                    alignment_confidence=0.9, reference_time=0.0, notes="test",
                )]

            monkeypatch.setattr("app.sessions.alignment_engine.suggest_alignment_offsets", _fake_suggest)
            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy:
                win._on_session_auto_align(sid_b)
                qapp.processEvents()
                assert spy.call_count == 1
            assert win._active_session.get_source(sid_b).time_offset_s == pytest.approx(1.5)
        finally:
            win.close()
            qapp.processEvents()

    def test_unchanged_auto_align_result_recalculates_zero_times(self, qapp: QApplication, monkeypatch) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")  # offset 0.0, method "none"
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            def _fake_suggest_noop(sources):
                # Reports the identical offset/method the source already has.
                return [AlignmentResult(
                    source_id=sid_b, suggested_offset_s=0.0, alignment_method="none",
                    alignment_confidence=0.0, reference_time=None, notes="no change",
                )]

            monkeypatch.setattr("app.sessions.alignment_engine.suggest_alignment_offsets", _fake_suggest_noop)
            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy:
                win._on_session_auto_align(sid_b)
                qapp.processEvents()
                assert spy.call_count == 0
        finally:
            win.close()
            qapp.processEvents()

    def test_no_targets_recalculates_zero_times(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy:
                win._on_session_auto_align("nonexistent-source")
                qapp.processEvents()
                assert spy.call_count == 0
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Set-as-reference
# ─────────────────────────────────────────────────────────────────────────────


class TestSetAsReferenceRecalculation:
    def test_only_affected_calc_recalculated_and_deduplicated(self, qapp: QApplication) -> None:
        """Calc depends on BOTH Source A and Source B; Set-as-Reference
        changes both sources' offsets in one action -- the shared
        calculated signal must be resolved exactly once, not twice."""
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            win._active_session.set_time_offset(sid_a, 1.0, method="manual")
            win._active_session.set_time_offset(sid_b, 2.0, method="manual")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            with patch.object(
                CalculatedSignalResolutionService, "resolve_one", autospec=True,
                side_effect=CalculatedSignalResolutionService.resolve_one,
            ) as spy:
                win._on_session_set_as_reference(sid_a)
                qapp.processEvents()
                calc1_calls = [c for c in spy.call_args_list if c.args[1] == "c1"]
                assert len(calc1_calls) == 1
            assert win._active_session.get_source(sid_a).time_offset_s == pytest.approx(0.0)
            assert win._active_session.get_source(sid_b).time_offset_s == pytest.approx(1.0)
        finally:
            win.close()
            qapp.processEvents()

    def test_unchanged_source_not_recalculated(self, qapp: QApplication) -> None:
        """A third source whose offset happens to already be correct
        relative to the new reference must not trigger recalculation for
        its dependent calculated signal."""
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            sid_c = _add_source(win, {"C": "MW"}, "Source C")
            # A is already the reference (offset 0); C's offset/method are
            # already exactly what Set-as-Reference(A) would (re)apply, so
            # this action changes nothing observable for C.
            win._active_session.set_time_offset(sid_c, 0.0, method="manual")
            _add_calc(win, "c_c", "DoubleC", "c + c", {"c": ChannelRef(sid_c, "C")}, "c")
            c_before = win._active_session.get_calculated_signal_result("c_c")

            win._on_session_set_as_reference(sid_a)
            qapp.processEvents()

            c_after = win._active_session.get_calculated_signal_result("c_c")
            assert c_after.computed_at == c_before.computed_at
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Failure preservation
# ─────────────────────────────────────────────────────────────────────────────


class TestFailurePreservation:
    def test_failed_recalculation_after_nudge_retains_stale_result(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
            ok_result = win._active_session.get_calculated_signal_result("c1")
            assert ok_result.status == CalculationStatus.OK

            # Deactivate B so any recalculation attempt fails.
            win._on_session_source_active(sid_b, False)
            stale_result = win._active_session.get_calculated_signal_result("c1")
            assert stale_result.status == CalculationStatus.STALE
            stale_values = stale_result.values.copy()

            # Nudge A (unrelated to c1's now-broken dependency on B, but c1
            # also depends on A, so this DOES attempt recalculation and
            # must fail cleanly without destroying the stale result).
            row_a = _row(win, sid_a)
            row_a._on_fine_right()
            qapp.processEvents()

            after = win._active_session.get_calculated_signal_result("c1")
            assert after.status == CalculationStatus.STALE
            assert after is not None
            np.testing.assert_array_equal(after.values, stale_values)
        finally:
            win.close()
            qapp.processEvents()

    def test_failure_does_not_delete_result_or_definition(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
            win._on_session_source_active(sid_b, False)

            row_a = _row(win, sid_a)
            row_a._on_fine_right()
            qapp.processEvents()

            assert win._active_session.get_calculated_signal_definition("c1") is not None
            assert win._active_session.get_calculated_signal_result("c1") is not None
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Timing warning regression (Sprint 1B)
# ─────────────────────────────────────────────────────────────────────────────


class TestTimingWarningRegression:
    def test_fine_nudge_does_not_change_timing_compatibility_level(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            win._refresh_timing_assessment()
            level_before = win._session_panel._timing_assessment.level

            row_b = _row(win, sid_b)
            row_b._on_fine_right()
            qapp.processEvents()

            level_after = win._session_panel._timing_assessment.level
            assert level_after == level_before
        finally:
            win.close()
            qapp.processEvents()

    def test_fine_nudge_updates_timing_details_offset(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")

            row_b = _row(win, sid_b)
            step = round(row_b._sample_interval_s, 3)
            row_b._on_fine_right()
            qapp.processEvents()

            assessment = win._session_panel._timing_assessment
            by_id = {p.source_id: p for p in assessment.source_profiles}
            assert by_id[sid_b].time_offset_s == pytest.approx(step)
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end (Step 17)
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndFineAlignmentWorkflow:
    def test_full_nudge_and_reverse_workflow(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            result_initial = win._active_session.get_calculated_signal_result("c1")
            assert result_initial.status == CalculationStatus.OK
            original_values = result_initial.values.copy()

            row_b = _row(win, sid_b)
            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy_forward:
                row_b._on_fine_right()
                qapp.processEvents()
                assert spy_forward.call_count == 1

            result_after_right = win._active_session.get_calculated_signal_result("c1")
            assert result_after_right.status == CalculationStatus.OK
            assert not np.array_equal(result_after_right.values, original_values)

            with patch.object(
                CalculatedSignalResolutionService, "resolve_for_sources",
                autospec=True, side_effect=CalculatedSignalResolutionService.resolve_for_sources,
            ) as spy_back:
                row_b._on_fine_left()
                qapp.processEvents()
                assert spy_back.call_count == 1

            result_final = win._active_session.get_calculated_signal_result("c1")
            assert result_final.status == CalculationStatus.OK
            np.testing.assert_array_almost_equal(result_final.values, original_values)
        finally:
            win.close()
            qapp.processEvents()

    def test_deactivation_makes_nudge_recalc_fail_gracefully(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Source A")
            sid_b = _add_source(win, {"B": "MW"}, "Source B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            win._on_session_source_active(sid_b, False)
            stale = win._active_session.get_calculated_signal_result("c1")
            assert stale.status == CalculationStatus.STALE

            row_a = _row(win, sid_a)
            row_a._on_fine_right()
            qapp.processEvents()

            still_stale = win._active_session.get_calculated_signal_result("c1")
            assert still_stale.status == CalculationStatus.STALE
            assert still_stale is not None
        finally:
            win.close()
            qapp.processEvents()
