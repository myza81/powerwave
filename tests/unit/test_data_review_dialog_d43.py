"""tests/unit/test_data_review_dialog_d43.py

Phase D4.3 — DataReviewDialog timestamp interpretation panel tests.

Tests the D4.3 additions:
  - Dialog accepts ts_matrices kwarg without breaking existing API
  - Auto-apply: high-confidence unambiguous → no panel shown, format pre-populated
  - Ambiguous: panel shown, radio buttons created, recommendation pre-selected
  - Low confidence: panel shown, no radio pre-selected
  - _harvest_ts_selections correctly reads radio choices
  - selected_timestamp_formats populated on accept
  - ts_matrices=None behaves identically to original dialog
  - Multiple sources × multiple columns all rendered
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PyQt6")  # skip if no Qt

from PyQt6.QtWidgets import QApplication, QRadioButton

from app.data.review_summary import (
    ColumnReviewRow,
    EventReviewSummary,
    SourceReviewSummary,
    TimestampReviewSummary,
)
from app.data.timestamp_interpreter import (
    TimestampInterpretation,
    TimestampInterpretationMatrix,
)
from app.ui.dialogs.data_review_dialog import DataReviewDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _minimal_summary() -> EventReviewSummary:
    ts = TimestampReviewSummary(
        source_id="csv_ops",
        raw_format="M/D/YYYY",
        confirmed_format=None,
        confidence=0.80,
        inferred_from="name_keyword",
        warnings=["Ambiguous date format"],
    )
    row = ColumnReviewRow(
        column_name="Frequency",
        signal_type="frequency",
        unit="Hz",
        display_group="frequency",
        confidence=0.95,
        inferred_from="name_exact",
        requires_user_confirmation=False,
    )
    src = SourceReviewSummary(
        source_id="csv_ops",
        provider_type="csv",
        file_path="mock.csv",
        start_time=None,
        trigger_time=None,
        sample_count=65,
        analog_channel_count=3,
        digital_channel_count=0,
        sampling_rates=[0.016667],
        display_offset_seconds=0.0,
        timestamp_summary=ts,
        column_rows=[row],
    )
    return EventReviewSummary(
        event_id="pulu_20260306",
        reference_start=None,
        manifest_notes=None,
        sources=[src],
    )


def _make_interpretation(
    fmt: str,
    label: str,
    confidence: float,
    ambiguous: bool = True,
    parse_rate: float = 1.0,
) -> TimestampInterpretation:
    from datetime import datetime
    return TimestampInterpretation(
        format_string=fmt,
        format_label=label,
        is_ambiguous=ambiguous,
        parse_success_rate=parse_rate,
        parsed_samples=[datetime(2026, 3, 6, 17, 25, 0)],
        confidence=confidence,
        reason_codes=["monotonic", "high_parse_rate"],
        source_type="string",
    )


def _ambiguous_matrix() -> TimestampInterpretationMatrix:
    """Returns an ambiguous matrix with two plausible interpretations."""
    interp_mdy = _make_interpretation("%m/%d/%Y %H:%M", "M/D/YYYY HH:MM", 0.72)
    interp_dmy = _make_interpretation("%d/%m/%Y %H:%M", "D/M/YYYY HH:MM", 0.65)
    return TimestampInterpretationMatrix(
        column_name="Time",
        sample_values=["3/6/2026 17:25", "3/6/2026 17:26"],
        interpretations=[interp_mdy, interp_dmy],
        is_ambiguous=True,
        recommended=interp_mdy,
    )


def _high_conf_matrix() -> TimestampInterpretationMatrix:
    """High-confidence, unambiguous → should auto-apply."""
    interp = _make_interpretation(
        "%Y-%m-%d %H:%M:%S", "ISO datetime space-sep",
        confidence=0.92, ambiguous=False,
    )
    return TimestampInterpretationMatrix(
        column_name="ts",
        sample_values=["2026-03-06 17:25:00"],
        interpretations=[interp],
        is_ambiguous=False,
        recommended=interp,
    )


def _low_conf_matrix() -> TimestampInterpretationMatrix:
    interp1 = _make_interpretation("%m/%d/%Y", "M/D/YYYY", 0.45)
    interp2 = _make_interpretation("%d/%m/%Y", "D/M/YYYY", 0.40)
    return TimestampInterpretationMatrix(
        column_name="date",
        sample_values=["03/06/25"],
        interpretations=[interp1, interp2],
        is_ambiguous=True,
        recommended=interp1,
    )


class TestDataReviewDialogD43:
    def test_no_ts_matrices_works_as_before(self, qapp):
        """Original API (no ts_matrices) must not break."""
        dlg = DataReviewDialog(_minimal_summary())
        assert dlg.selected_timestamp_formats == {}

    def test_empty_ts_matrices_no_interp_section(self, qapp):
        dlg = DataReviewDialog(_minimal_summary(), ts_matrices={})
        # _build_ts_interpretation_section returns None → no panel added
        assert dlg.selected_timestamp_formats == {}

    def test_high_confidence_unambiguous_auto_applied(self, qapp):
        """ISO format >= 0.85 confidence, not ambiguous → auto-populated, no panel."""
        matrices = {"csv_ops": {"ts": _high_conf_matrix()}}
        dlg = DataReviewDialog(_minimal_summary(), ts_matrices=matrices)
        # The format should be pre-populated during __init__ / _build_ts_interpretation_section
        assert "csv_ops" in dlg.selected_timestamp_formats
        assert dlg.selected_timestamp_formats["csv_ops"]["ts"] == "%Y-%m-%d %H:%M:%S"

    def test_ambiguous_creates_radio_group(self, qapp):
        """Ambiguous column should generate a radio group."""
        matrices = {"csv_ops": {"Time": _ambiguous_matrix()}}
        dlg = DataReviewDialog(_minimal_summary(), ts_matrices=matrices)
        assert ("csv_ops", "Time") in dlg._radio_groups

    def test_ambiguous_recommendation_pre_selected(self, qapp):
        """When recommended confidence >= 0.60, first radio should be pre-checked."""
        matrices = {"csv_ops": {"Time": _ambiguous_matrix()}}
        dlg = DataReviewDialog(_minimal_summary(), ts_matrices=matrices)
        group = dlg._radio_groups[("csv_ops", "Time")]
        checked = group.checkedButton()
        assert checked is not None
        assert checked.property("format_string") == "%m/%d/%Y %H:%M"

    def test_low_confidence_no_preselection(self, qapp):
        """Below _CONFIRM_THRESHOLD, no radio button should be pre-selected."""
        matrices = {"csv_ops": {"date": _low_conf_matrix()}}
        dlg = DataReviewDialog(_minimal_summary(), ts_matrices=matrices)
        group = dlg._radio_groups.get(("csv_ops", "date"))
        if group is not None:
            # Confidence 0.45 < 0.60 — should NOT be pre-checked
            assert group.checkedButton() is None

    def test_harvest_selections_populates_dict(self, qapp):
        """_harvest_ts_selections should read checked radios into selected_timestamp_formats."""
        matrices = {"csv_ops": {"Time": _ambiguous_matrix()}}
        dlg = DataReviewDialog(_minimal_summary(), ts_matrices=matrices)
        # Simulate operator choice: select the second interpretation
        group = dlg._radio_groups[("csv_ops", "Time")]
        buttons = group.buttons()
        assert len(buttons) >= 2
        # Click the second button (alternative)
        buttons[1].setChecked(True)
        dlg._harvest_ts_selections()
        result = dlg.selected_timestamp_formats.get("csv_ops", {}).get("Time")
        assert result == buttons[1].property("format_string")

    def test_multiple_sources_and_columns(self, qapp):
        """Multiple sources × multiple columns all get radio groups."""
        matrices = {
            "csv_ops": {
                "Time": _ambiguous_matrix(),
                "date": _low_conf_matrix(),
            },
        }
        dlg = DataReviewDialog(_minimal_summary(), ts_matrices=matrices)
        # Both columns should have radio groups (both need operator interaction)
        assert ("csv_ops", "Time") in dlg._radio_groups
        assert ("csv_ops", "date") in dlg._radio_groups

    def test_radio_count_limited_to_top5(self, qapp):
        """Only up to 5 interpretations should be shown as radio buttons."""
        interps = [
            _make_interpretation(f"%m/%d/%Y_{i}", f"Format {i}", 0.65 - i * 0.05)
            for i in range(8)
        ]
        matrix = TimestampInterpretationMatrix(
            column_name="Time",
            sample_values=["03/06/2026"],
            interpretations=interps,
            is_ambiguous=True,
            recommended=interps[0],
        )
        matrices = {"csv_ops": {"Time": matrix}}
        dlg = DataReviewDialog(_minimal_summary(), ts_matrices=matrices)
        group = dlg._radio_groups.get(("csv_ops", "Time"))
        if group is not None:
            assert len(group.buttons()) <= 5
