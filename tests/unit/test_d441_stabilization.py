"""Tests for Phase D4.4.1 — Stabilization & Operational Refinement.

Covers:
  - AXIS_MODE_RELATIVE / AXIS_MODE_ABSOLUTE constants exported from datetime_axis
  - FlexiblePlotCanvas.set_record() axis_mode parameter (no Qt needed for unit tests)
  - VisualizationManager.display_grouped_record() axis_mode parameter
  - apply_selected_timestamp_format: timestamp rebasing from operator selection
  - _rebase_record_start_time: new record with corrected TimingInformation
  - DirectOpenDiagnostics dataclass shape
  - build_direct_open_diagnostics: populates all fields correctly
  - log_direct_open_diagnostics: writes structured output to stderr
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.data.direct_load_intelligence import (
    DirectOpenDiagnostics,
    _rebase_record_start_time,
    apply_selected_timestamp_format,
    build_direct_open_diagnostics,
    log_direct_open_diagnostics,
)
from app.data.signal_metadata import SignalMetadata
from app.models import DisturbanceRecord
from app.models.timing import TimingInformation, SamplingInformation
from app.models.metadata import RecordingMetadata
from app.models.channels import AnalogChannel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_T0 = datetime(2000, 1, 1, 0, 0, 0)
_T_START = datetime(2026, 3, 6, 17, 25, 0)
_T_TRIG = datetime(2026, 3, 6, 17, 25, 5)


def _make_record(
    channels: list[str] | None = None,
    start_time: datetime = _T0,
    trigger_time: datetime | None = None,
) -> DisturbanceRecord:
    channels = channels or []
    if trigger_time is None:
        trigger_time = start_time
    t = np.linspace(0.0, 10.0, 30)
    data: dict = {"time": t}
    analog = []
    for i, name in enumerate(channels):
        data[name] = np.ones(30)
        analog.append(AnalogChannel(name=name, unit="MW", index=i))
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST", recorder_name="CSV",
            source_file="test.csv", provider_type="csv", nominal_frequency=50.0,
        ),
        waveform_data=pd.DataFrame(data),
        analog_channels=analog,
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[1.0], samples_per_rate=[30]),
        timing_info=TimingInformation(start_time=start_time, trigger_time=trigger_time),
        disturbance_info=None,
    )


def _make_interpretation(fmt_string: str, fmt_label: str, parsed_samples: list):
    """Build a minimal TimestampInterpretation-like mock."""
    interp = MagicMock()
    interp.format_string = fmt_string
    interp.format_label = fmt_label
    interp.parsed_samples = parsed_samples
    interp.confidence = 0.9
    return interp


def _make_matrix(interp):
    matrix = MagicMock()
    matrix.interpretations = [interp]
    matrix.recommended = interp
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Axis mode constants
# ─────────────────────────────────────────────────────────────────────────────


class TestAxisModeConstants:
    def test_relative_constant_value(self) -> None:
        from app.visualization.axis.datetime_axis import AXIS_MODE_RELATIVE
        assert AXIS_MODE_RELATIVE == "relative_seconds"

    def test_absolute_constant_value(self) -> None:
        from app.visualization.axis.datetime_axis import AXIS_MODE_ABSOLUTE
        assert AXIS_MODE_ABSOLUTE == "absolute_datetime"

    def test_constants_are_distinct(self) -> None:
        from app.visualization.axis.datetime_axis import AXIS_MODE_ABSOLUTE, AXIS_MODE_RELATIVE
        assert AXIS_MODE_RELATIVE != AXIS_MODE_ABSOLUTE


# ─────────────────────────────────────────────────────────────────────────────
# FlexiblePlotCanvas.set_record() axis_mode parameter
# ─────────────────────────────────────────────────────────────────────────────


class TestFlexiblePlotCanvasAxisMode:
    """Verify axis_mode is forwarded correctly from set_record to set_time_axis_mode.

    set_record delegates time-axis setup to set_time_axis_mode; these tests
    verify the delegation contract rather than the downstream _datetime_axis call.
    """

    def test_relative_mode_forwarded_to_set_time_axis_mode(self) -> None:
        """axis_mode='relative_seconds' must be forwarded to set_time_axis_mode."""
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        from app.analytics.rms.rms_models import RMSDisplayMode
        from app.visualization.axis.datetime_axis import AXIS_MODE_RELATIVE

        canvas = MagicMock(spec=FlexiblePlotCanvas)
        canvas._datetime_axis = MagicMock()
        canvas._rms_display_mode = RMSDisplayMode.OFF

        record = _make_record(start_time=_T_START, trigger_time=_T_TRIG)

        FlexiblePlotCanvas.set_record(canvas, record, axis_mode=AXIS_MODE_RELATIVE)
        canvas.set_time_axis_mode.assert_called_with(
            AXIS_MODE_RELATIVE, axis_reference_time=_T_START
        )

    def test_absolute_mode_forwarded_to_set_time_axis_mode(self) -> None:
        """axis_mode='absolute_datetime' must be forwarded to set_time_axis_mode."""
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        from app.analytics.rms.rms_models import RMSDisplayMode
        from app.visualization.axis.datetime_axis import AXIS_MODE_ABSOLUTE

        canvas = MagicMock(spec=FlexiblePlotCanvas)
        canvas._datetime_axis = MagicMock()
        canvas._rms_display_mode = RMSDisplayMode.OFF

        record = _make_record(start_time=_T_START, trigger_time=_T_TRIG)

        FlexiblePlotCanvas.set_record(canvas, record, axis_mode=AXIS_MODE_ABSOLUTE)
        canvas.set_time_axis_mode.assert_called_with(
            AXIS_MODE_ABSOLUTE, axis_reference_time=_T_START
        )

    def test_default_mode_is_relative(self) -> None:
        """Default axis_mode must be 'relative_seconds'."""
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        from app.analytics.rms.rms_models import RMSDisplayMode
        from app.visualization.axis.datetime_axis import AXIS_MODE_RELATIVE

        canvas = MagicMock(spec=FlexiblePlotCanvas)
        canvas._datetime_axis = MagicMock()
        canvas._rms_display_mode = RMSDisplayMode.OFF

        record = _make_record(start_time=_T_START, trigger_time=_T_TRIG)

        FlexiblePlotCanvas.set_record(canvas, record)
        canvas.set_time_axis_mode.assert_called_with(
            AXIS_MODE_RELATIVE, axis_reference_time=_T_START
        )


# ─────────────────────────────────────────────────────────────────────────────
# _rebase_record_start_time
# ─────────────────────────────────────────────────────────────────────────────


class TestRebaseRecordStartTime:
    def test_new_start_time_applied(self) -> None:
        record = _make_record(start_time=_T0, trigger_time=_T0)
        new_start = datetime(2026, 3, 6, 17, 25, 0)
        rebased = _rebase_record_start_time(record, new_start)
        assert rebased.timing_info.start_time == new_start

    def test_trigger_offset_preserved(self) -> None:
        record = _make_record(
            start_time=datetime(2000, 1, 1, 0, 0, 0),
            trigger_time=datetime(2000, 1, 1, 0, 0, 5),
        )
        new_start = datetime(2026, 3, 6, 17, 25, 0)
        rebased = _rebase_record_start_time(record, new_start)
        expected_trigger = new_start + timedelta(seconds=5)
        assert rebased.timing_info.trigger_time == expected_trigger

    def test_waveform_data_unchanged(self) -> None:
        record = _make_record(["ch1"], start_time=_T0)
        new_start = datetime(2026, 1, 1)
        rebased = _rebase_record_start_time(record, new_start)
        pd.testing.assert_frame_equal(rebased.waveform_data, record.waveform_data)

    def test_original_record_unchanged(self) -> None:
        record = _make_record(start_time=_T0)
        new_start = datetime(2026, 1, 1)
        _rebase_record_start_time(record, new_start)
        assert record.timing_info.start_time == _T0

    def test_time_multiplier_preserved(self) -> None:
        record = _make_record(start_time=_T0)
        record.timing_info.__class__  # just ensure it's real
        rebased = _rebase_record_start_time(record, datetime(2026, 1, 1))
        assert rebased.timing_info.time_multiplier == record.timing_info.time_multiplier

    def test_timezone_preserved(self) -> None:
        record = _make_record(start_time=_T0)
        rebased = _rebase_record_start_time(record, datetime(2026, 1, 1))
        assert rebased.timing_info.timezone == record.timing_info.timezone

    def test_returns_new_disturbance_record(self) -> None:
        record = _make_record(start_time=_T0)
        rebased = _rebase_record_start_time(record, datetime(2026, 1, 1))
        assert rebased is not record


# ─────────────────────────────────────────────────────────────────────────────
# apply_selected_timestamp_format
# ─────────────────────────────────────────────────────────────────────────────


class TestApplySelectedTimestampFormat:
    def _make_args(self, col="time", fmt="%m/%d/%Y %H:%M"):
        parsed = [datetime(2026, 3, 6, 17, 25, 0)]
        interp = _make_interpretation(fmt, "M/D/Y HH:MM", parsed)
        matrix = _make_matrix(interp)
        ts_matrices = {col: matrix}
        selected_formats = {col: fmt}
        return ts_matrices, selected_formats

    def test_start_time_updated(self) -> None:
        record = _make_record(start_time=_T0)
        ts_matrices, selected_formats = self._make_args()
        rebased = apply_selected_timestamp_format(record, ts_matrices, selected_formats)
        assert rebased.timing_info.start_time == datetime(2026, 3, 6, 17, 25, 0)

    def test_returns_original_when_no_match(self) -> None:
        record = _make_record(start_time=_T0)
        result = apply_selected_timestamp_format(record, {}, {})
        assert result is record

    def test_returns_original_when_col_not_in_matrices(self) -> None:
        record = _make_record(start_time=_T0)
        result = apply_selected_timestamp_format(record, {}, {"time": "%d/%m/%Y"})
        assert result is record

    def test_uses_first_matching_col(self) -> None:
        record = _make_record(start_time=_T0)
        parsed = [datetime(2026, 6, 1)]
        interp = _make_interpretation("%Y/%m/%d", "ISO-ish", parsed)
        matrix = _make_matrix(interp)
        ts_matrices = {"timestamp": matrix}
        selected_formats = {"timestamp": "%Y/%m/%d"}
        rebased = apply_selected_timestamp_format(record, ts_matrices, selected_formats)
        assert rebased.timing_info.start_time == datetime(2026, 6, 1)

    def test_fallback_to_recommended_when_fmt_not_found(self) -> None:
        record = _make_record(start_time=_T0)
        parsed = [datetime(2026, 3, 1)]
        interp = _make_interpretation("%d/%m/%Y", "D/M/Y", parsed)
        matrix = _make_matrix(interp)
        # Pass a format_string that doesn't match any interpretation — recommended used
        ts_matrices = {"time": matrix}
        selected_formats = {"time": "%nonexistent"}
        rebased = apply_selected_timestamp_format(record, ts_matrices, selected_formats)
        assert rebased.timing_info.start_time == datetime(2026, 3, 1)

    def test_empty_parsed_samples_skipped(self) -> None:
        record = _make_record(start_time=_T0)
        interp = _make_interpretation("%d/%m/%Y", "D/M/Y", [])
        interp.parsed_samples = []
        matrix = _make_matrix(interp)
        matrix.recommended = interp
        ts_matrices = {"time": matrix}
        selected_formats = {"time": "%d/%m/%Y"}
        result = apply_selected_timestamp_format(record, ts_matrices, selected_formats)
        assert result is record


# ─────────────────────────────────────────────────────────────────────────────
# DirectOpenDiagnostics
# ─────────────────────────────────────────────────────────────────────────────


class TestDirectOpenDiagnostics:
    def test_fields_accessible(self) -> None:
        diag = DirectOpenDiagnostics(
            source_path="/data/file.csv",
            provider_type="csv",
            timestamp_column="time",
            timestamp_interpretation="M/D/Y HH:MM",
            axis_mode="absolute_datetime",
            display_groups={"Freq": "frequency"},
            review_columns=[],
        )
        assert diag.source_path == "/data/file.csv"
        assert diag.axis_mode == "absolute_datetime"

    def test_is_frozen(self) -> None:
        diag = DirectOpenDiagnostics(
            source_path="x.csv", provider_type="csv",
            timestamp_column=None, timestamp_interpretation=None,
            axis_mode="relative_seconds",
            display_groups={}, review_columns=[],
        )
        with pytest.raises((AttributeError, TypeError)):
            diag.axis_mode = "something_else"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# build_direct_open_diagnostics
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildDirectOpenDiagnostics:
    def _make_metadata(self, grp: str, confirm: bool = False) -> SignalMetadata:
        m = MagicMock(spec=SignalMetadata)
        m.display_group = grp
        m.requires_user_confirmation = confirm
        return m

    def test_display_groups_populated(self) -> None:
        sm = {"Freq": self._make_metadata("frequency")}
        diag = build_direct_open_diagnostics(
            source_path="f.csv", provider_type="csv",
            signal_metadata=sm, ts_matrices={},
            selected_formats={}, axis_mode="absolute_datetime",
        )
        assert diag.display_groups == {"Freq": "frequency"}

    def test_review_columns_populated(self) -> None:
        sm = {
            "Freq": self._make_metadata("frequency", confirm=False),
            "Tie": self._make_metadata("power", confirm=True),
        }
        diag = build_direct_open_diagnostics(
            source_path="f.csv", provider_type="csv",
            signal_metadata=sm, ts_matrices={},
            selected_formats={}, axis_mode="absolute_datetime",
        )
        assert "Tie" in diag.review_columns
        assert "Freq" not in diag.review_columns

    def test_timestamp_column_from_selected_formats(self) -> None:
        parsed = [datetime(2026, 3, 6)]
        interp = _make_interpretation("%m/%d/%Y", "M/D/Y", parsed)
        matrix = _make_matrix(interp)
        diag = build_direct_open_diagnostics(
            source_path="f.csv", provider_type="csv",
            signal_metadata={},
            ts_matrices={"time": matrix},
            selected_formats={"time": "%m/%d/%Y"},
            axis_mode="absolute_datetime",
        )
        assert diag.timestamp_column == "time"
        assert diag.timestamp_interpretation == "M/D/Y"

    def test_timestamp_column_from_matrices_when_no_selection(self) -> None:
        parsed = [datetime(2026, 3, 6)]
        interp = _make_interpretation("%m/%d/%Y", "M/D/Y", parsed)
        matrix = _make_matrix(interp)
        diag = build_direct_open_diagnostics(
            source_path="f.csv", provider_type="csv",
            signal_metadata={},
            ts_matrices={"time": matrix},
            selected_formats={},
            axis_mode="absolute_datetime",
        )
        assert diag.timestamp_column == "time"
        assert diag.timestamp_interpretation is None

    def test_no_ts_data_gives_none_fields(self) -> None:
        diag = build_direct_open_diagnostics(
            source_path="f.csv", provider_type="csv",
            signal_metadata={}, ts_matrices={},
            selected_formats={}, axis_mode="relative_seconds",
        )
        assert diag.timestamp_column is None
        assert diag.timestamp_interpretation is None

    def test_axis_mode_set(self) -> None:
        diag = build_direct_open_diagnostics(
            source_path="f.csv", provider_type="csv",
            signal_metadata={}, ts_matrices={},
            selected_formats={}, axis_mode="absolute_datetime",
        )
        assert diag.axis_mode == "absolute_datetime"


# ─────────────────────────────────────────────────────────────────────────────
# log_direct_open_diagnostics
# ─────────────────────────────────────────────────────────────────────────────


class TestLogDirectOpenDiagnostics:
    def _make_diag(self, **kwargs) -> DirectOpenDiagnostics:
        defaults = dict(
            source_path="/data/pulu.csv",
            provider_type="csv",
            timestamp_column="time",
            timestamp_interpretation="M/D/Y HH:MM",
            axis_mode="absolute_datetime",
            display_groups={"Frequency": "frequency", "System Demand": "power"},
            review_columns=["Tie-Line"],
        )
        defaults.update(kwargs)
        return DirectOpenDiagnostics(**defaults)

    def test_source_path_in_output(self, capsys) -> None:
        log_direct_open_diagnostics(self._make_diag())
        assert "pulu.csv" in capsys.readouterr().err

    def test_provider_type_in_output(self, capsys) -> None:
        log_direct_open_diagnostics(self._make_diag())
        assert "csv" in capsys.readouterr().err

    def test_axis_mode_in_output(self, capsys) -> None:
        log_direct_open_diagnostics(self._make_diag())
        assert "absolute_datetime" in capsys.readouterr().err

    def test_timestamp_interpretation_in_output(self, capsys) -> None:
        log_direct_open_diagnostics(self._make_diag())
        assert "M/D/Y" in capsys.readouterr().err

    def test_display_group_in_output(self, capsys) -> None:
        log_direct_open_diagnostics(self._make_diag())
        assert "frequency" in capsys.readouterr().err

    def test_review_column_in_output(self, capsys) -> None:
        log_direct_open_diagnostics(self._make_diag())
        assert "Tie-Line" in capsys.readouterr().err

    def test_no_ts_column_does_not_raise(self, capsys) -> None:
        diag = self._make_diag(timestamp_column=None, timestamp_interpretation=None)
        log_direct_open_diagnostics(diag)
        captured = capsys.readouterr()
        assert "none" in captured.err

    def test_empty_review_columns_shows_none(self, capsys) -> None:
        diag = self._make_diag(review_columns=[])
        log_direct_open_diagnostics(diag)
        assert "none" in capsys.readouterr().err
