"""Runtime smoke tests for Qt/pyqtgraph widgets.

These tests intentionally instantiate real widgets. Import-only tests cannot
catch pyqtgraph constructor/API mismatches or scene lifecycle issues.
"""
from __future__ import annotations

import os
import sys
import dataclasses
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pyqtgraph as pg
import pytest
from PyQt6.QtWidgets import QApplication, QDialog, QSplitter


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_flexible_plot_canvas_instantiates(qapp: QApplication) -> None:
    pg.setConfigOptions(
        useOpenGL=True,
        antialias=False,
        foreground="w",
        background="#1E1E1E",
    )
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    canvas = FlexiblePlotCanvas()
    try:
        assert canvas._primary_plot is not None
    finally:
        canvas.close()
        qapp.processEvents()


def test_flexible_plot_canvas_set_record_before_show_supports_multi_axis(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_mixed_disturbance_record
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    result = make_mixed_disturbance_record()
    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(result.record)
        assert len(canvas._axis_manager.parameter_names()) > 1
    finally:
        canvas.close()
        qapp.processEvents()


def test_sparse_low_rate_record_uses_markers_and_neighbor_context(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.models import SamplingInformation
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    base = make_high_rate_record().record
    channel = dataclasses.replace(base.analog_channels[0], name="MW", unit="MW")
    record = dataclasses.replace(
        base,
        waveform_data=pd.DataFrame({
            "time": [2340.0, 2400.0],
            "MW": [100.0, 101.0],
        }),
        analog_channels=[channel],
        sampling_info=SamplingInformation(
            sampling_rates=[1.0 / 60.0],
            samples_per_rate=[2],
        ),
    )

    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(record)
        canvas._primary_plot.setXRange(2348.0, 2355.0, padding=0)
        qapp.processEvents()

        curve = canvas._axis_manager.get_curves()["MW"]
        x, y = curve.getData()
        y_range = curve.getViewBox().viewRange()[1]
        assert curve.opts["symbol"] == "o"
        assert list(x) == [2340.0, 2400.0]
        assert list(y) == [100.0, 101.0]
        assert y_range[0] < 100.0
        assert y_range[1] > 101.0
    finally:
        canvas.close()
        qapp.processEvents()


def test_sparse_low_rate_record_initial_view_shows_full_time_extent(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.models import SamplingInformation
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    base = make_high_rate_record().record
    channel = dataclasses.replace(base.analog_channels[0], name="MW", unit="MW")
    record = dataclasses.replace(
        base,
        waveform_data=pd.DataFrame({
            "time": [0.0, 60.0, 120.0],
            "MW": [100.0, 101.0, 102.0],
        }),
        analog_channels=[channel],
        sampling_info=SamplingInformation(
            sampling_rates=[1.0 / 60.0],
            samples_per_rate=[3],
        ),
    )

    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(record)
        qapp.processEvents()

        x_range = canvas._primary_plot.getViewBox().viewRange()[0]
        curve = canvas._axis_manager.get_curves()["MW"]
        x, y = curve.getData()

        assert x_range[0] == pytest.approx(0.0)
        assert x_range[1] == pytest.approx(120.0)
        assert list(x) == [0.0, 60.0, 120.0]
        assert list(y) == [100.0, 101.0, 102.0]
    finally:
        canvas.close()
        qapp.processEvents()


def test_high_rate_record_initial_view_remains_trigger_zoomed(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    record = make_high_rate_record().record
    trigger_s = (
        record.timing_info.trigger_time - record.timing_info.start_time
    ).total_seconds()

    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(record)
        qapp.processEvents()

        x_start, x_end = canvas._primary_plot.getViewBox().viewRange()[0]
        assert x_start == pytest.approx(max(0.0, trigger_s - 0.2))
        assert x_end == pytest.approx(min(float(canvas._time_cache[-1]), trigger_s + 0.2))
    finally:
        canvas.close()
        qapp.processEvents()


def _load_direct_result(path: Path):
    from app.data.direct_load_intelligence import (
        build_signal_metadata,
        detect_timestamp_ambiguity,
    )
    from app.data.intelligence import IntelligenceManager
    from app.providers import ProviderManager, ComtradeProvider, CsvProvider, ExcelProvider
    from app.ui.main_window.main_window import _DirectOpenResult

    manager = ProviderManager()
    manager.register_provider(ComtradeProvider())
    manager.register_provider(CsvProvider())
    manager.register_provider(ExcelProvider())
    record = manager.load(path)
    suffix = path.suffix.lower()
    provider_type = (
        "comtrade" if suffix in (".cfg", ".comtrade")
        else "excel" if suffix in (".xlsx", ".xls")
        else "csv"
    )
    if provider_type in ("csv", "excel"):
        signal_metadata = build_signal_metadata(
            record, IntelligenceManager(), path.stem, provider_type
        )
        ts_ambiguous, ts_matrices = detect_timestamp_ambiguity(path, record)
    else:
        signal_metadata = {}
        ts_ambiguous = False
        ts_matrices = {}
    return _DirectOpenResult(
        record=record,
        path=path,
        provider_type=provider_type,
        signal_metadata=signal_metadata,
        ts_ambiguous=ts_ambiguous,
        ts_matrices=ts_matrices,
    )


def _accept_review_dialog():
    patcher = patch("app.ui.dialogs.data_review_dialog.DataReviewDialog")
    mock_cls = patcher.start()
    instance = mock_cls.return_value
    instance.exec.return_value = QDialog.DialogCode.Accepted
    instance.selected_timestamp_formats = {}
    return patcher


def test_direct_csv_open_routes_to_grouped_visible_panels(qapp: QApplication) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = _load_direct_result(Path("samples/csv/pulu_20260306.csv"))
    win = PowerwaveMainWindow()
    dialog_patch = _accept_review_dialog()
    try:
        win.show()
        qapp.processEvents()
        with patch.object(win._vis_manager, "set_record", wraps=win._vis_manager.set_record) as set_record, \
             patch.object(
                 win._vis_manager,
                 "display_grouped_record",
                 wraps=win._vis_manager.display_grouped_record,
             ) as display_grouped:
            win._on_record_loaded(result)
            for _ in range(5):
                qapp.processEvents()

            display_grouped.assert_called_once()
            set_record.assert_not_called()

        assert list(win._panel_canvases) == ["power", "frequency"]
        splitter = win.centralWidget()
        assert isinstance(splitter, QSplitter)
        assert splitter.count() == 2
        assert win._grouped_timeline is None
        assert not win._timeline.isVisible()

        power = win._panel_canvases["power"]
        frequency = win._panel_canvases["frequency"]
        assert set(power._axis_manager.get_curves()) == {"System Demand", "Tie-Line"}
        assert set(frequency._axis_manager.get_curves()) == {"Frequency"}
        assert power._primary_plot.getViewBox().viewRange()[0] == pytest.approx([0.0, 3840.0])
        assert frequency._primary_plot.getViewBox().viewRange()[0] == pytest.approx([0.0, 3840.0])
        assert power._datetime_axis._start_time == result.record.timing_info.start_time
        assert frequency._datetime_axis._start_time == result.record.timing_info.start_time
    finally:
        dialog_patch.stop()
        win.close()
        qapp.processEvents()


def test_legacy_plain_csv_record_still_routes_to_grouped_display(qapp: QApplication) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = _load_direct_result(Path("samples/csv/pulu_20260306.csv"))
    win = PowerwaveMainWindow()
    dialog_patch = _accept_review_dialog()
    try:
        with patch.object(win._vis_manager, "set_record", wraps=win._vis_manager.set_record) as set_record, \
             patch.object(
                 win._vis_manager,
                 "display_grouped_record",
                 wraps=win._vis_manager.display_grouped_record,
             ) as display_grouped:
            win._on_record_loaded(result.record)
            qapp.processEvents()

            display_grouped.assert_called_once()
            set_record.assert_not_called()
        assert list(win._panel_canvases) == ["power", "frequency"]
        assert isinstance(win.centralWidget(), QSplitter)
        assert win.centralWidget().count() == 2
    finally:
        dialog_patch.stop()
        win.close()
        qapp.processEvents()


def test_comtrade_direct_open_keeps_standard_analog_digital_layout(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = _load_direct_result(Path("samples/comtrade/pulu_20260306.cfg"))
    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()
        with patch.object(win._vis_manager, "set_record", wraps=win._vis_manager.set_record) as set_record, \
             patch.object(
                 win._vis_manager,
                 "display_grouped_record",
                 wraps=win._vis_manager.display_grouped_record,
             ) as display_grouped:
            win._on_record_loaded(result)
            qapp.processEvents()

            set_record.assert_called_once()
            display_grouped.assert_not_called()

        splitter = win.centralWidget()
        assert isinstance(splitter, QSplitter)
        assert splitter.count() == 2
        assert win._timeline.isVisible()
        assert len(win._canvas._axis_manager.get_curves()) == 42
        assert len(result.record.digital_channels) == 88
        assert win._canvas._datetime_axis._start_time is None
    finally:
        win.close()
        qapp.processEvents()
