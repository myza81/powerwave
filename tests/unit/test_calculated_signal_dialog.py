"""Unit tests for the Calculated Signals creation/preview dialog (Phase 3A)
in app.ui.calculated_signals.

Uses the repository's offscreen Qt conventions (QT_QPA_PLATFORM=offscreen is
set process-wide by tests/conftest.py; a module-scoped `qapp` fixture
provides a single QApplication). AnalogInputSelectorDialog's modal exec()
is monkeypatched with a stub so "Add Input" can be driven without a real
user interaction loop, following the same QMessageBox-monkeypatching
convention already used elsewhere in this test suite (e.g.
test_main_window_embedded_wizard.py).

Uses only generic, synthetic session fixtures -- no filename, station, or
event identity is special-cased anywhere in this file or in production code.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

import app.ui.calculated_signals.calculated_signal_dialog as cs_dialog_module
from app.calculated_signals.models import ChannelRef
from app.calculated_signals.resolver import CalculatedSignalResolutionService
from app.models.channels import AnalogChannel, DigitalChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.event_session import EventAnalysisSession
from app.ui.calculated_signals.analog_input_selector import (
    AnalogInputSelectorDialog,
    SelectedAnalogChannel,
)
from app.ui.calculated_signals.calculated_signal_dialog import CalculatedSignalDialog

# ─────────────────────────────────────────────────────────────────────────────
# QApplication fixture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Generic synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_record(
    analog: dict[str, str | None] | list[str],
    digital: list[str] | None = None,
    time: np.ndarray | None = None,
    values: dict[str, np.ndarray] | None = None,
    n: int = 10,
    non_numeric_columns: list[str] | None = None,
) -> DisturbanceRecord:
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
    """Source A: analog Va[kV], MW[MW]. Source B: analog Ia[A], Unclassified[None]; digital Trip."""
    sess = EventAnalysisSession()
    sid_a = sess.add_source(
        _make_record({"Va": "kV", "MW": "MW"}), "Source A", "csv"
    )
    sid_b = sess.add_source(
        _make_record({"Ia": "A", "Unclassified": None}, digital=["Trip"]), "Source B", "csv"
    )
    return sess, sid_a, sid_b


class _StubSelectorDialog:
    """Stand-in for AnalogInputSelectorDialog's modal exec() loop."""

    def __init__(self, selection: SelectedAnalogChannel | None) -> None:
        self._selection = selection

    def exec(self) -> int:
        return (
            QDialog.DialogCode.Accepted
            if self._selection is not None
            else QDialog.DialogCode.Rejected
        )

    def selected_channel(self) -> SelectedAnalogChannel | None:
        return self._selection


def _add_binding(
    dialog: CalculatedSignalDialog,
    monkeypatch: pytest.MonkeyPatch,
    session: EventAnalysisSession,
    source_id: str,
    channel_name: str,
) -> None:
    """Drive dialog._on_add_input() as if the user picked (source_id,
    channel_name) from the real AnalogInputSelectorDialog."""
    source = session.get_source(source_id)
    unit = next((ch.unit for ch in source.record.analog_channels if ch.name == channel_name), None)
    channel = session.get_channel(source_id, channel_name)
    selection = SelectedAnalogChannel(
        ref=ChannelRef(source_id, channel_name),
        source_display_name=source.display_name,
        channel_display_name=channel.display_name,
        unit=unit,
    )
    monkeypatch.setattr(
        cs_dialog_module,
        "AnalogInputSelectorDialog",
        lambda session, parent=None, _sel=selection: _StubSelectorDialog(_sel),
    )
    dialog._on_add_input()


def _silence_message_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)


# ─────────────────────────────────────────────────────────────────────────────
# Dialog opening
# ─────────────────────────────────────────────────────────────────────────────


class TestDialogOpening:
    def test_opens_with_active_session(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        assert dlg.windowTitle() == "Calculated Signals"

    def test_eligible_analog_channels_appear_in_selector(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        selector = AnalogInputSelectorDialog(sess)
        all_leaf_texts = _collect_leaf_labels(selector)
        assert "Va" in all_leaf_texts
        assert "MW" in all_leaf_texts
        assert "Ia" in all_leaf_texts

    def test_multiple_sources_grouped_correctly(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        selector = AnalogInputSelectorDialog(sess)
        top_level_labels = [
            selector._tree.topLevelItem(i).text(0) for i in range(selector._tree.topLevelItemCount())
        ]
        assert "Source A" in top_level_labels
        assert "Source B" in top_level_labels
        assert selector._tree.topLevelItemCount() == 2

    def test_digital_channels_absent_from_selector(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        selector = AnalogInputSelectorDialog(sess)
        assert "Trip" not in _collect_leaf_labels(selector)

    def test_inactive_source_absent_from_selector(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        sess.set_source_active(sid_b, False)
        selector = AnalogInputSelectorDialog(sess)
        top_level_labels = [
            selector._tree.topLevelItem(i).text(0) for i in range(selector._tree.topLevelItemCount())
        ]
        assert "Source B" not in top_level_labels
        assert "Ia" not in _collect_leaf_labels(selector)


def _collect_leaf_labels(selector: AnalogInputSelectorDialog) -> list[str]:
    labels: list[str] = []
    tree = selector._tree
    for i in range(tree.topLevelItemCount()):
        source_item = tree.topLevelItem(i)
        for j in range(source_item.childCount()):
            labels.append(source_item.child(j).text(0))
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Binding management
# ─────────────────────────────────────────────────────────────────────────────


class TestBindingManagement:
    def test_add_first_binding_assigned_alias_a(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        assert [b.variable for b in dlg._bindings] == ["A"]

    def test_add_second_binding_assigned_alias_b(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_b, "Ia")
        assert [b.variable for b in dlg._bindings] == ["A", "B"]

    def test_remove_binding(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_b, "Ia")
        dlg._bindings_table.selectRow(0)
        dlg._on_remove_input()
        assert [b.variable for b in dlg._bindings] == ["B"]

    def test_reference_selector_updates_on_add(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        assert dlg._reference_combo.count() == 1
        _add_binding(dlg, monkeypatch, sess, sid_b, "Ia")
        assert dlg._reference_combo.count() == 2

    def test_reference_selector_updates_on_remove(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_b, "Ia")
        dlg._bindings_table.selectRow(0)
        dlg._on_remove_input()
        assert dlg._reference_combo.count() == 1
        assert dlg._reference_combo.currentData() == "B"

    def test_duplicate_source_channel_cannot_be_bound_twice(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _silence_message_boxes(monkeypatch)
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        assert len(dlg._bindings) == 1

    def test_duplicate_channel_names_from_different_sources_are_distinguishable(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess = EventAnalysisSession()
        sid_a = sess.add_source(_make_record({"Va": "kV"}), "Source A", "csv")
        sid_b = sess.add_source(_make_record({"Va": "kV"}), "Source B", "csv")
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_b, "Va")
        assert len(dlg._bindings) == 2
        assert dlg._bindings[0].ref != dlg._bindings[1].ref
        assert {b.source_display_name for b in dlg._bindings} == {"Source A", "Source B"}


# ─────────────────────────────────────────────────────────────────────────────
# Expression
# ─────────────────────────────────────────────────────────────────────────────


class TestExpression:
    def _dialog_with_one_binding(self, qapp, monkeypatch, sess, sid_a):
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._name_edit.setText("Test Calc")
        return dlg

    def test_valid_expression_preview_succeeds(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = self._dialog_with_one_binding(qapp, monkeypatch, sess, sid_a)
        dlg._expression_edit.setText("A + A")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        assert not dlg._preview_error_label.isVisible()

    def test_unsupported_operator_shows_clean_error(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = self._dialog_with_one_binding(qapp, monkeypatch, sess, sid_a)
        dlg._expression_edit.setText("A ** 2")
        dlg._on_preview_clicked()
        assert dlg._preview_error_label.text() != ""
        assert "Traceback" not in dlg._preview_error_label.text()
        assert dlg._preview_current is False

    def test_unknown_alias_shows_clean_error(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = self._dialog_with_one_binding(qapp, monkeypatch, sess, sid_a)
        dlg._expression_edit.setText("A + C")
        dlg._on_preview_clicked()
        assert dlg._preview_error_label.text() != ""
        assert dlg._preview_current is False

    def test_abs_function_supported(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = self._dialog_with_one_binding(qapp, monkeypatch, sess, sid_a)
        dlg._expression_edit.setText("abs(A)")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True

    def test_constants_supported(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = self._dialog_with_one_binding(qapp, monkeypatch, sess, sid_a)
        dlg._expression_edit.setText("A + 1")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True

    def test_preview_invalidated_on_expression_change(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = self._dialog_with_one_binding(qapp, monkeypatch, sess, sid_a)
        dlg._expression_edit.setText("A + A")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        dlg._expression_edit.setText("A - A")
        assert dlg._preview_current is False


# ─────────────────────────────────────────────────────────────────────────────
# Reference
# ─────────────────────────────────────────────────────────────────────────────


class TestReferenceSelection:
    def test_defaults_to_first_variable(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_b, "Ia")
        assert dlg._reference_combo.currentData() == "A"

    def test_user_can_change_reference(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_b, "Ia")
        dlg._reference_combo.setCurrentIndex(1)
        assert dlg._reference_combo.currentData() == "B"

    def test_only_current_bindings_listed(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_a, "MW")
        options = [dlg._reference_combo.itemData(i) for i in range(dlg._reference_combo.count())]
        assert set(options) == {"A", "B"}

    def test_changing_reference_invalidates_preview(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_a, "MW")
        dlg._name_edit.setText("Test")
        # A[kV] and B[MW] are different unit families; the expression only
        # references A so both A and B remain valid reference choices.
        dlg._expression_edit.setText("A + 1")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        dlg._reference_combo.setCurrentIndex(1)
        assert dlg._preview_current is False


# ─────────────────────────────────────────────────────────────────────────────
# Preview
# ─────────────────────────────────────────────────────────────────────────────


class TestPreview:
    def test_same_source_calculation(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_a, "MW")
        dlg._name_edit.setText("SameSource")
        # A[kV] and B[MW] are different unit families -- only A is
        # referenced here so this deliberately avoids combining them.
        dlg._expression_edit.setText("A + 1")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        assert dlg._preview_result.status.value == "ok"
        assert "All inputs share one source" in dlg._preview_text.toPlainText()

    def test_cross_source_calculation(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess = EventAnalysisSession()
        sid_a = sess.add_source(
            _make_record({"A": "MW"}, time=np.array([0.0, 1.0, 2.0])), "Source A", "csv"
        )
        sid_b = sess.add_source(
            _make_record({"B": "MW"}, time=np.array([0.0, 1.0, 2.0])), "Source B", "csv"
        )
        sess.set_time_offset(sid_b, 1.0)
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "A")
        _add_binding(dlg, monkeypatch, sess, sid_b, "B")
        dlg._name_edit.setText("CrossSource")
        dlg._expression_edit.setText("A - B")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        text = dlg._preview_text.toPlainText()
        assert "A offset: 0.000 s" in text
        assert "B offset: 1.000 s" in text

    def test_output_unit_reported(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._name_edit.setText("Test")
        dlg._expression_edit.setText("A + A")
        dlg._on_preview_clicked()
        assert "Unit: kV" in dlg._preview_text.toPlainText()

    def test_sample_and_valid_counts_from_backend_result(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._name_edit.setText("Test")
        dlg._expression_edit.setText("A + A")
        dlg._on_preview_clicked()
        result = dlg._preview_result
        total = int(result.time.size)
        valid = int(np.count_nonzero(result.validity_mask))
        text = dlg._preview_text.toPlainText()
        assert f"Samples: {total:,}" in text
        assert f"Valid samples: {valid:,} / {total:,}" in text

    def test_warnings_displayed(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess = EventAnalysisSession()
        sid_a = sess.add_source(
            _make_record({"A": "MW"}, time=np.array([0.0, 1.0, 2.0, 3.0])), "Source A", "csv"
        )
        sid_b = sess.add_source(
            _make_record({"B": "MW"}, time=np.array([0.0, 2.0, 3.0])), "Source B", "csv"
        )
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "A")
        _add_binding(dlg, monkeypatch, sess, sid_b, "B")
        dlg._name_edit.setText("Warn")
        dlg._expression_edit.setText("A - B")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        text = dlg._preview_text.toPlainText()
        assert "Warnings:" in text
        assert "interpolated" in text

    def test_manual_alignment_warning_shown(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        sess.set_time_offset(sid_b, 0.1, method="manual")
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_b, "Ia")
        dlg._name_edit.setText("Manual")
        dlg._expression_edit.setText("A + A")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        text = dlg._preview_text.toPlainText()
        assert "manually aligned source" in text

    def test_unknown_unit_warning_shown(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_b, "Unclassified")
        dlg._name_edit.setText("Unknown")
        dlg._expression_edit.setText("A + 1")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        text = dlg._preview_text.toPlainText()
        assert "no engineering unit" in text

    def test_preview_does_not_add_permanent_session_entry(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._name_edit.setText("Test")
        dlg._expression_edit.setText("A + A")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        assert sess.list_calculated_signals() == []


# ─────────────────────────────────────────────────────────────────────────────
# Creation
# ─────────────────────────────────────────────────────────────────────────────


class TestCreation:
    def _ready_dialog(self, qapp, monkeypatch, sess, sid_a, name="MyCalc", expr="A + A"):
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._name_edit.setText(name)
        dlg._expression_edit.setText(expr)
        return dlg

    def test_create_without_fresh_preview_is_blocked(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = self._ready_dialog(qapp, monkeypatch, sess, sid_a)
        dlg._on_create_clicked()
        assert dlg._preview_error_label.text() != ""
        assert sess.list_calculated_signals() == []
        assert dlg.result() != QDialog.DialogCode.Accepted

    def test_create_after_edit_without_new_preview_is_blocked(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = self._ready_dialog(qapp, monkeypatch, sess, sid_a)
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        dlg._expression_edit.setText("A - A")  # invalidates preview
        dlg._on_create_clicked()
        assert sess.list_calculated_signals() == []

    def test_successful_create_adds_session_entry_with_ok_result(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = self._ready_dialog(qapp, monkeypatch, sess, sid_a, name="NetPower")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        dlg._on_create_clicked()

        assert dlg.result() == QDialog.DialogCode.Accepted
        entries = sess.list_calculated_signals()
        assert len(entries) == 1
        assert entries[0].definition.name == "NetPower"
        assert entries[0].result is not None
        assert entries[0].result.status.value == "ok"

    def test_calc_id_is_a_stable_uuid_not_derived_from_name(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = self._ready_dialog(qapp, monkeypatch, sess, sid_a, name="NetPower")
        dlg._on_preview_clicked()
        dlg._on_create_clicked()

        entry = sess.list_calculated_signals()[0]
        calc_id = entry.definition.calc_id
        assert calc_id != entry.definition.name
        uuid.UUID(calc_id)  # raises ValueError if not a valid UUID string

    def test_display_name_stored_on_definition(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = self._ready_dialog(qapp, monkeypatch, sess, sid_a, name="Net System Power")
        dlg._on_preview_clicked()
        dlg._on_create_clicked()
        assert sess.list_calculated_signals()[0].definition.name == "Net System Power"

    def test_duplicate_name_rejected(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        _silence_message_boxes(monkeypatch)
        sess, sid_a, sid_b = _session_with_two_sources()

        first = self._ready_dialog(qapp, monkeypatch, sess, sid_a, name="Dup")
        first._on_preview_clicked()
        first._on_create_clicked()
        assert len(sess.list_calculated_signals()) == 1

        second = self._ready_dialog(qapp, monkeypatch, sess, sid_a, name="Dup")
        second._on_preview_clicked()
        second._on_create_clicked()

        assert len(sess.list_calculated_signals()) == 1  # second was rejected
        assert second.result() != QDialog.DialogCode.Accepted

    def test_creation_failure_rolls_back_new_entry_and_keeps_dialog_open(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _silence_message_boxes(monkeypatch)
        sess = EventAnalysisSession()
        sid_a = sess.add_source(
            _make_record({"A": "MW"}, time=np.array([0.0, 1.0, 2.0])), "Source A", "csv"
        )
        sid_b = sess.add_source(
            _make_record({"B": "MW"}, time=np.array([0.0, 1.0, 2.0])), "Source B", "csv"
        )
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "A")
        _add_binding(dlg, monkeypatch, sess, sid_b, "B")
        dlg._name_edit.setText("WillFail")
        dlg._expression_edit.setText("A - B")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True

        # Break the cross-source overlap between Preview and Create without
        # changing anything the dialog's own fingerprint tracks -- eligibility
        # (channel presence/analog/numeric) is unaffected by offset, so
        # add_calculated_signal() still succeeds, but resolve_one() must now
        # fail with AlignmentError (no more time overlap).
        sess.set_time_offset(sid_b, 1000.0)

        dlg._on_create_clicked()

        assert sess.list_calculated_signals() == []  # rolled back
        assert dlg.result() != QDialog.DialogCode.Accepted  # dialog stays open


# ─────────────────────────────────────────────────────────────────────────────
# Analog-only
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalogOnly:
    def test_digital_channel_cannot_appear_in_selector(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        selector = AnalogInputSelectorDialog(sess)
        assert "Trip" not in _collect_leaf_labels(selector)

    def test_malformed_analog_data_excluded_from_selector(self, qapp) -> None:
        sess = EventAnalysisSession()
        record = _make_record({"BadCol": "MW"}, non_numeric_columns=["BadCol"])
        sess.add_source(record, "Source A", "csv")
        selector = AnalogInputSelectorDialog(sess)
        assert "BadCol" not in _collect_leaf_labels(selector)

    def test_unknown_unit_analog_channel_can_appear(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        selector = AnalogInputSelectorDialog(sess)
        assert "Unclassified" in _collect_leaf_labels(selector)


# ─────────────────────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────────────────────


class TestIdentity:
    def test_bindings_use_source_id_and_channel_name(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        assert dlg._bindings[0].ref == ChannelRef(sid_a, "Va")

    def test_source_display_rename_does_not_break_preview(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._name_edit.setText("Test")
        dlg._expression_edit.setText("A + A")

        sess.get_source(sid_a).display_name = "Totally Renamed"
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        assert dlg._preview_result.status.value == "ok"

    def test_calculation_display_name_is_not_used_as_identity(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._name_edit.setText("Human Friendly Name")
        dlg._expression_edit.setText("A + A")
        dlg._on_preview_clicked()
        dlg._on_create_clicked()

        entry = sess.list_calculated_signals()[0]
        assert entry.definition.calc_id != "Human Friendly Name"
        assert sess.get_calculated_signal(entry.definition.calc_id) is entry


# ─────────────────────────────────────────────────────────────────────────────
# Expression Builder (UX experiment)
# ─────────────────────────────────────────────────────────────────────────────


def _session_with_three_mw_sources() -> tuple[EventAnalysisSession, str, str, str]:
    """Three sources, each exposing one MW-unit channel -- for testing
    3+-signal arithmetic where unit compatibility must hold."""
    sess = EventAnalysisSession()
    sid_a = sess.add_source(_make_record({"MW_A": "MW"}), "Relay A", "csv")
    sid_b = sess.add_source(_make_record({"MW_B": "MW"}), "SCADA Tie-Line", "csv")
    sid_c = sess.add_source(_make_record({"MW_C": "MW"}), "PMU", "csv")
    return sess, sid_a, sid_b, sid_c


def _session_with_duplicate_channel_names() -> tuple[EventAnalysisSession, str, str, str]:
    """Three sources that all expose a channel literally named 'Ia' -- the
    Signal menu must disambiguate via source display name, not show three
    identical 'Ia' entries."""
    sess = EventAnalysisSession()
    sid_1 = sess.add_source(_make_record({"Ia": "A"}), "Relay 1", "csv")
    sid_2 = sess.add_source(_make_record({"Ia": "A"}), "Relay 2", "csv")
    sid_3 = sess.add_source(_make_record({"Ia": "A"}), "PMU", "csv")
    return sess, sid_1, sid_2, sid_3


class TestExpressionBuilderLayout:
    def test_dialog_minimum_size_unchanged(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        assert dlg.minimumSize().width() == 600
        assert dlg.minimumSize().height() == 700

    def test_dialog_does_not_exceed_prior_footprint(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        dlg.adjustSize()
        # The dialog must render at (at most) its documented minimum
        # footprint -- the builder toolbar must not force the window larger.
        assert dlg.width() <= 600
        assert dlg.height() <= 700

    def test_fits_1366x768(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        dlg.show()
        assert dlg.width() <= 1366
        assert dlg.height() <= 768
        dlg.close()

    def test_fits_1280x720(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        dlg.show()
        assert dlg.width() <= 1280
        assert dlg.height() <= 720
        dlg.close()

    def test_create_and_cancel_remain_within_dialog_bounds(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        dlg.show()
        create_bottom = dlg._create_btn.geometry().y() + dlg._create_btn.geometry().height()
        cancel_bottom = dlg._cancel_btn.geometry().y() + dlg._cancel_btn.geometry().height()
        assert create_bottom <= dlg.height()
        assert cancel_bottom <= dlg.height()
        dlg.close()

    def test_builder_row_present_and_visible(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        dlg.show()
        assert not dlg._signal_menu_button.isHidden()
        assert not dlg._function_menu_button.isHidden()
        dlg.close()


class TestSignalInsertion:
    def test_menu_reflects_current_bindings(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        assert dlg._signal_menu.actions() == []
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        labels = [a.text() for a in dlg._signal_menu.actions()]
        assert labels == ["A — Source A / Va [kV]"]

    def test_menu_disabled_with_no_bindings(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        assert not dlg._signal_menu_button.isEnabled()

    def test_menu_enabled_after_binding(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        assert dlg._signal_menu_button.isEnabled()

    def test_inserts_correct_alias_at_cursor(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_b, "Ia")
        dlg._expression_edit.setText("A+B")
        dlg._expression_edit.setCursorPosition(2)  # A+|B
        dlg._insert_text_at_cursor("A")
        assert dlg._expression_edit.text() == "A+AB"

    def test_inserted_alias_replaces_selection(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_b, "Ia")
        dlg._expression_edit.setText("A+B")
        dlg._expression_edit.selectAll()
        dlg._insert_text_at_cursor("A")
        assert dlg._expression_edit.text() == "A"

    def test_duplicate_channel_names_show_source_context(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_1, sid_2, sid_3 = _session_with_duplicate_channel_names()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_1, "Ia")
        _add_binding(dlg, monkeypatch, sess, sid_2, "Ia")
        _add_binding(dlg, monkeypatch, sess, sid_3, "Ia")
        labels = [a.text() for a in dlg._signal_menu.actions()]
        assert labels == [
            "A — Relay 1 / Ia [A]",
            "B — Relay 2 / Ia [A]",
            "C — PMU / Ia [A]",
        ]
        assert len(set(labels)) == 3  # never three indistinguishable "Ia" entries

    def test_menu_refreshes_after_remove_input(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_b, "Ia")
        assert len(dlg._signal_menu.actions()) == 2
        dlg._bindings_table.selectRow(0)  # select the "A" row
        dlg._on_remove_input()
        labels = [a.text() for a in dlg._signal_menu.actions()]
        assert labels == ["B — Source B / Ia [A]"]


class TestOperatorInsertion:
    @pytest.mark.parametrize(
        "glyph, expected_token",
        [("+", "+"), ("−", "-"), ("×", "*"), ("÷", "/"), ("(", "("), (")", ")")],
    )
    def test_operator_button_maps_to_parser_token(
        self, qapp, glyph: str, expected_token: str
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        token = next(t for g, t, _tip in cs_dialog_module._OPERATOR_BUTTONS if g == glyph)
        assert token == expected_token
        dlg._expression_edit.clear()
        dlg._insert_text_at_cursor(token)
        assert dlg._expression_edit.text() == expected_token

    def test_operators_insert_at_cursor_in_sequence(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        dlg._insert_text_at_cursor("(")
        dlg._insert_text_at_cursor(")")
        assert dlg._expression_edit.text() == "()"
        dlg._expression_edit.setCursorPosition(1)
        dlg._insert_text_at_cursor("+")
        assert dlg._expression_edit.text() == "(+)"


class TestFunctionInsertion:
    def test_only_backend_supported_functions_offered(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        offered = [a.text() for a in dlg._function_menu_button.menu().actions()]
        assert offered == ["abs()"]
        assert not any("sqrt" in f for f in offered)

    def test_abs_inserts_with_cursor_inside_parens_when_nothing_selected(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        dlg._expression_edit.clear()
        dlg._insert_function("abs")
        assert dlg._expression_edit.text() == "abs()"
        assert dlg._expression_edit.cursorPosition() == 4

    def test_abs_wraps_selected_expression(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        dlg._expression_edit.setText("A-B")
        dlg._expression_edit.selectAll()
        dlg._insert_function("abs")
        assert dlg._expression_edit.text() == "abs(A-B)"

    def test_abs_wraps_partial_selection(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        dlg._expression_edit.setText("A+B")
        dlg._expression_edit.setSelection(0, 1)  # select "A"
        dlg._insert_function("abs")
        assert dlg._expression_edit.text() == "abs(A)+B"

    def test_sqrt_not_offered_and_rejected_if_typed_manually(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The builder must reflect real backend capability exactly (no
        capability silently expanded): sqrt() is absent from the Function
        menu, and manually typing it is still rejected by the unmodified
        parser -- this is a regression guard, not a builder feature."""
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        offered = [a.text() for a in dlg._function_menu_button.menu().actions()]
        assert "sqrt()" not in offered

        dlg._name_edit.setText("Test")
        dlg._expression_edit.setText("sqrt(A)")
        dlg._on_preview_clicked()
        assert dlg._preview_current is False
        assert not dlg._preview_error_label.isHidden()
        assert "sqrt" in dlg._preview_error_label.text()


class TestLiveSyntaxFeedback:
    def test_empty_expression_shows_supported_hint(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        assert "abs()" in dlg._expression_edit.toolTip()
        assert dlg._expression_edit.styleSheet() == ""

    def test_incomplete_expression_shows_subtle_feedback_not_a_dialog(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        # If this ever tried to show a modal dialog, exec() would block the
        # test indefinitely -- reaching the assertion below already proves
        # no popup occurred.
        dlg._expression_edit.setText("A+")
        assert "border" in dlg._expression_edit.styleSheet()
        assert dlg._expression_edit.toolTip() != ""

    def test_unknown_variable_reported_inline(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._expression_edit.setText("A+Z")
        assert "Z" in dlg._expression_edit.toolTip()

    def test_valid_expression_clears_feedback(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._expression_edit.setText("A+")
        dlg._expression_edit.setText("A+A")
        assert dlg._expression_edit.toolTip() == "Valid expression"
        assert dlg._expression_edit.styleSheet() == ""

    def test_live_feedback_never_touches_preview_state(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Syntax feedback must be independent of (and never substitute
        for) the numerical Preview -- typing alone must not mark the
        preview current."""
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._expression_edit.setText("A+A")
        assert dlg._expression_edit.toolTip() == "Valid expression"
        assert dlg._preview_current is False
        assert dlg._preview_result is None


class TestManualEditingPreserved:
    def test_field_remains_directly_editable(self, qapp) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        assert not dlg._expression_edit.isReadOnly()

    def test_manually_typed_valid_expression_previews(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        _add_binding(dlg, monkeypatch, sess, sid_b, "Ia")
        dlg._name_edit.setText("Manual")
        dlg._expression_edit.setText("(A+A-A)/2")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        assert dlg._preview_result.status.value == "ok"

    def test_manually_typed_invalid_expression_rejected_by_existing_validation(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._name_edit.setText("Manual")
        dlg._expression_edit.setText("A ** 2")  # exponent not in the grammar
        dlg._on_preview_clicked()
        assert dlg._preview_current is False
        assert not dlg._preview_error_label.isHidden()


class TestComplexFormulas:
    def test_three_signal_sum_scaled_by_constant(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b, sid_c = _session_with_three_mw_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "MW_A")
        _add_binding(dlg, monkeypatch, sess, sid_b, "MW_B")
        _add_binding(dlg, monkeypatch, sess, sid_c, "MW_C")
        dlg._name_edit.setText("Scaled Sum")

        dlg._expression_edit.clear()
        dlg._insert_text_at_cursor("(")
        dlg._insert_text_at_cursor("A")
        dlg._insert_text_at_cursor("+")
        dlg._insert_text_at_cursor("B")
        dlg._insert_text_at_cursor("+")
        dlg._insert_text_at_cursor("C")
        dlg._insert_text_at_cursor(")")
        dlg._insert_text_at_cursor("*")
        dlg._expression_edit.insert("1.732")

        assert dlg._expression_edit.text() == "(A+B+C)*1.732"
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        assert dlg._preview_result.status.value == "ok"
        assert dlg._preview_result.unit == "MW"

    def test_abs_of_half_difference(self, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
        sess, sid_a, sid_b, sid_c = _session_with_three_mw_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "MW_A")
        _add_binding(dlg, monkeypatch, sess, sid_b, "MW_B")
        dlg._name_edit.setText("Abs Half Diff")

        dlg._expression_edit.setText("(A-B)/2")
        dlg._expression_edit.selectAll()
        dlg._insert_function("abs")
        assert dlg._expression_edit.text() == "abs((A-B)/2)"

        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        assert dlg._preview_result.status.value == "ok"

    def test_more_than_two_bound_channels_all_selectable_from_menu(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b, sid_c = _session_with_three_mw_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "MW_A")
        _add_binding(dlg, monkeypatch, sess, sid_b, "MW_B")
        _add_binding(dlg, monkeypatch, sess, sid_c, "MW_C")
        assert len(dlg._signal_menu.actions()) == 3
        assert [b.variable for b in dlg._bindings] == ["A", "B", "C"]


class TestBuilderPreviewAndCreateRegression:
    def test_builder_modification_invalidates_preview_like_manual_edit(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._name_edit.setText("Test")
        dlg._expression_edit.setText("A+A")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True

        dlg._insert_text_at_cursor("-A")  # builder-driven edit, not typing
        assert dlg._preview_current is False

    def test_create_disabled_until_builder_generated_expression_previewed(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._name_edit.setText("Test")
        dlg._insert_text_at_cursor("A")
        dlg._insert_text_at_cursor("+")
        dlg._insert_text_at_cursor("A")
        dlg._on_create_clicked()  # no preview run yet
        assert not dlg._preview_error_label.isHidden()
        assert sess.list_calculated_signals() == []

    def test_create_succeeds_after_builder_generated_expression_previewed(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sess, sid_a, sid_b = _session_with_two_sources()
        dlg = CalculatedSignalDialog(sess)
        _add_binding(dlg, monkeypatch, sess, sid_a, "Va")
        dlg._name_edit.setText("Test")
        dlg._insert_text_at_cursor("A")
        dlg._insert_text_at_cursor("+")
        dlg._insert_text_at_cursor("A")
        dlg._on_preview_clicked()
        assert dlg._preview_current is True
        dlg._on_create_clicked()
        entries = sess.list_calculated_signals()
        assert len(entries) == 1
        assert entries[0].definition.expression == "A+A"
        assert entries[0].result is not None
        assert entries[0].result.status.value == "ok"
