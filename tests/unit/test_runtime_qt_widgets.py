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

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PyQt6 import sip
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


def test_synchronization_manager_runtime_x_range_and_cursor(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.visualization.managers.synchronization_manager import SynchronizationManager
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    record = make_high_rate_record(duration_s=1.0, sampling_rate_hz=1000.0).record
    master = FlexiblePlotCanvas()
    follower = FlexiblePlotCanvas()
    manager = SynchronizationManager()
    try:
        master.set_record(record)
        follower.set_record(record)
        manager.register_many([master, follower], master_canvas=master)

        manager.synchronize_x_range(master, (0.2, 0.4))
        manager.synchronize_cursor(master, 0.3)
        qapp.processEvents()

        assert follower._primary_plot.getViewBox().viewRange()[0] == pytest.approx([0.2, 0.4])
        assert follower._cursor.value() == pytest.approx(0.3)
    finally:
        manager.clear()
        master.close()
        follower.close()
        qapp.processEvents()


def test_synchronization_manager_runtime_digital_timeline_follows_canvas(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.visualization.managers.synchronization_manager import SynchronizationManager
    from app.visualization.widgets.digital_event_timeline import DigitalEventTimeline
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    record = make_high_rate_record(duration_s=1.0, sampling_rate_hz=1000.0).record
    canvas = FlexiblePlotCanvas()
    timeline = DigitalEventTimeline()
    manager = SynchronizationManager()
    try:
        canvas.set_record(record)
        timeline.set_record(record)
        manager.register_canvas(canvas, set_as_master=True)
        manager.register_canvas(timeline)

        manager.synchronize_x_range(canvas, (0.1, 0.35))
        manager.synchronize_cursor(canvas, 0.22)
        qapp.processEvents()

        assert timeline.getPlotItem().getViewBox().viewRange()[0] == pytest.approx([0.1, 0.35])
        assert timeline._cursor.value() == pytest.approx(0.22)
    finally:
        manager.clear()
        canvas.close()
        timeline.close()
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
    instance.confirmed_column_rows = {}
    return patcher


def _process_events(qapp: QApplication, count: int = 5) -> None:
    for _ in range(count):
        qapp.processEvents()


def _mapped_view_pixel_x(canvas, x_value: float) -> float:
    viewbox = canvas._primary_plot.getViewBox()
    scene_point = viewbox.mapViewToScene(pg.Point(float(x_value), 0.0))
    return float(canvas.viewportTransform().map(scene_point).x())


def _mapped_timeline_pixel_x(timeline, x_value: float) -> float:
    viewbox = timeline.getPlotItem().getViewBox()
    scene_point = viewbox.mapViewToScene(pg.Point(float(x_value), 0.0))
    return float(timeline.viewportTransform().map(scene_point).x())


def _assert_same_x_pixel_mapping(canvases, x_values, tolerance: float = 1.0) -> None:
    deltas: list[float] = []
    for x_value in x_values:
        pixels = [_mapped_view_pixel_x(canvas, x_value) for canvas in canvases]
        assert max(pixels) - min(pixels) <= tolerance
        if len(pixels) >= 2:
            deltas.append(pixels[1] - pixels[0])
    if len(deltas) >= 2:
        assert max(deltas) - min(deltas) <= tolerance


def _assert_standard_widgets_alive(win) -> None:
    assert not sip.isdeleted(win._canvas)
    assert not sip.isdeleted(win._timeline)
    assert win._vis_manager.canvas is win._canvas
    assert win._vis_manager.timeline is win._timeline


def _assert_no_deleted_sync_registrations(win) -> None:
    sync = win._vis_manager.synchronization_manager
    for entry in sync._registered.values():
        assert not sip.isdeleted(entry.canvas)
        assert not sip.isdeleted(entry.plot_item)
        assert not sip.isdeleted(entry.viewbox)
        if entry.cursor is not None:
            assert not sip.isdeleted(entry.cursor)


def _synthetic_single_source_session():
    from app.data.multi_source_session import MultiSourceSession, SourceRecord
    from app.data.synthetic import make_mixed_disturbance_record

    result = make_mixed_disturbance_record()
    return MultiSourceSession([
        SourceRecord(
            source_id="synthetic",
            provider_type="synthetic",
            record=result.record,
            signal_metadata=result.signal_metadata,
            original_start_time=result.record.timing_info.start_time,
            sampling_rates=result.record.sampling_info.sampling_rates,
        )
    ])


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
        assert power._panel_title == "Power"
        assert frequency._panel_title == "Frequency"
        assert power._primary_plot.getAxis("left").labelUnits == "MW"
        assert not power._primary_plot.getAxis("left").autoSIPrefix
        assert frequency._primary_plot.getAxis("left").labelUnits == "Hz"
        assert not frequency._primary_plot.getAxis("left").autoSIPrefix
        assert power._primary_plot.getViewBox().viewRange()[0] == pytest.approx([0.0, 3840.0])
        assert frequency._primary_plot.getViewBox().viewRange()[0] == pytest.approx([0.0, 3840.0])
        assert np.array_equal(power._time_cache, frequency._time_cache)
        assert power._datetime_axis._start_time == result.record.timing_info.start_time
        assert frequency._datetime_axis._start_time == result.record.timing_info.start_time
        _assert_same_x_pixel_mapping([power, frequency], [0.0, 1920.0, 3840.0])

        power._primary_plot.setXRange(600.0, 1200.0, padding=0)
        power._cursor.setValue(900.0)
        _process_events(qapp)
        assert frequency._primary_plot.getViewBox().viewRange()[0] == pytest.approx([600.0, 1200.0])
        assert power._primary_plot.getViewBox().viewRange()[0] == pytest.approx([600.0, 1200.0])
        assert frequency._cursor.value() == pytest.approx(900.0)
        _assert_same_x_pixel_mapping([power, frequency], [600.0, 900.0, 1200.0])

        win.resize(1200, 900)
        _process_events(qapp)
        assert frequency._primary_plot.getViewBox().viewRange()[0] == pytest.approx([600.0, 1200.0])
        _assert_same_x_pixel_mapping([power, frequency], [600.0, 900.0, 1200.0])
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


def test_synthetic_grouped_panels_keep_x_pixel_alignment(qapp: QApplication) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()
        win._on_load_synthetic_mixed()
        _process_events(qapp, count=8)

        # Only check visible canvases — hidden panels (e.g. harmonic THD/spectrum)
        # have no valid pixel geometry and are excluded from alignment assertions.
        canvases = [c for c in win._panel_canvases.values() if c.isVisible()]
        assert len(canvases) >= 3
        first_range = canvases[0]._primary_plot.getViewBox().viewRange()[0]
        for canvas in canvases[1:]:
            assert canvas._primary_plot.getViewBox().viewRange()[0] == pytest.approx(first_range)

        x_start, x_end = first_range
        x_mid = (x_start + x_end) / 2.0
        _assert_same_x_pixel_mapping(canvases, [x_start, x_mid, x_end])

        canvases[0]._primary_plot.setXRange(x_mid - 0.05, x_mid + 0.05, padding=0)
        canvases[0]._cursor.setValue(x_mid)
        _process_events(qapp)
        for canvas in canvases:
            assert canvas._primary_plot.getViewBox().viewRange()[0] == pytest.approx(
                [x_mid - 0.05, x_mid + 0.05]
            )
            assert canvas._cursor.value() == pytest.approx(x_mid)
        _assert_same_x_pixel_mapping(canvases, [x_mid - 0.05, x_mid, x_mid + 0.05])
    finally:
        win.close()
        qapp.processEvents()


def test_synthetic_multi_source_panels_keep_x_pixel_alignment(qapp: QApplication) -> None:
    from app.data.multi_source_session import MultiSourceSession, SourceRecord
    from app.data.synthetic import make_mixed_disturbance_record
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = make_mixed_disturbance_record()
    session = MultiSourceSession([
        SourceRecord(
            source_id="synthetic",
            provider_type="synthetic",
            record=result.record,
            signal_metadata=result.signal_metadata,
            original_start_time=result.record.timing_info.start_time,
            sampling_rates=result.record.sampling_info.sampling_rates,
        )
    ])
    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()
        win._on_multi_source_loaded(session)
        _process_events(qapp, count=8)

        canvases = list(win._panel_canvases.values())
        assert len(canvases) >= 3
        x_range = canvases[0]._primary_plot.getViewBox().viewRange()[0]
        for canvas in canvases[1:]:
            assert canvas._primary_plot.getViewBox().viewRange()[0] == pytest.approx(x_range)

        x_start, x_end = x_range
        _assert_same_x_pixel_mapping(canvases, [x_start, (x_start + x_end) / 2.0, x_end])
    finally:
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
        assert len(win._canvas._axis_manager.get_curves()) == 8
        assert win._signal_browser.entry_count() == 42 + 88
        assert len(result.record.digital_channels) == 88
        assert win._canvas._datetime_axis._start_time is None
    finally:
        win.close()
        qapp.processEvents()


def test_comtrade_direct_open_can_switch_to_absolute_timestamp_mode(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow
    from app.visualization.axis.datetime_axis import TimeDisplayMode

    result = _load_direct_result(Path("samples/comtrade/pulu_20260306.cfg"))
    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()
        win._on_record_loaded(result)
        _process_events(qapp, count=4)

        initial_range = win._canvas._primary_plot.getViewBox().viewRange()[0]
        win._canvas._cursor.setValue(0.5)
        _process_events(qapp)
        assert win._canvas._datetime_axis._start_time is None
        assert win._timeline._datetime_axis._start_time is None

        win._on_time_axis_mode_changed(TimeDisplayMode.ABSOLUTE)
        _process_events(qapp)

        assert win._time_display_mode is TimeDisplayMode.ABSOLUTE
        assert win._canvas._datetime_axis._start_time == result.record.timing_info.start_time
        assert win._timeline._datetime_axis._start_time == result.record.timing_info.start_time
        assert win._canvas._primary_plot.getViewBox().viewRange()[0] == pytest.approx(initial_range)
        assert win._canvas._cursor.value() == pytest.approx(0.5)
        labels = win._canvas._datetime_axis.tickStrings([0.5], 1.0, 0.01)
        assert result.record.timing_info.trigger_time.strftime("%H:%M:%S") in labels[0]

        win._on_time_axis_mode_changed(TimeDisplayMode.RELATIVE)
        _process_events(qapp)
        assert win._canvas._datetime_axis._start_time is None
        assert win._timeline._datetime_axis._start_time is None
    finally:
        win.close()
        qapp.processEvents()


def test_comtrade_rms_overlay_remains_aligned_after_time_axis_switch(
    qapp: QApplication,
) -> None:
    from app.analytics.rms.rms_models import RMSDisplayMode
    from app.ui.main_window.main_window import PowerwaveMainWindow
    from app.visualization.axis.datetime_axis import TimeDisplayMode

    result = _load_direct_result(Path("samples/comtrade/pulu_20260306.cfg"))
    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()
        win._on_record_loaded(result)
        _process_events(qapp, count=4)

        win._on_rms_mode_changed(RMSDisplayMode.OVERLAY)
        _process_events(qapp, count=4)
        rms_names = set(win._canvas._rms_curves)
        assert rms_names
        rms_time_before = {
            name: values.copy()
            for name, values in win._canvas._rms_time_cache.items()
        }

        win._on_time_axis_mode_changed(TimeDisplayMode.ABSOLUTE)
        _process_events(qapp, count=4)

        assert set(win._canvas._rms_curves) == rms_names
        for name in rms_names:
            np.testing.assert_array_equal(
                win._canvas._rms_time_cache[name],
                rms_time_before[name],
            )
    finally:
        win.close()
        qapp.processEvents()


def test_comtrade_rms_window_switch_recomputes_cached_envelope(
    qapp: QApplication,
) -> None:
    from app.analytics.rms.rms_models import RMSDisplayMode, RMSWindowMode
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = _load_direct_result(Path("samples/comtrade/pulu_20260306.cfg"))
    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()
        win._on_record_loaded(result)
        _process_events(qapp, count=4)

        win._on_rms_mode_changed(RMSDisplayMode.OVERLAY)
        _process_events(qapp, count=4)
        name = next(iter(win._canvas._rms_data_cache))
        one_cycle_len = len(win._canvas._rms_data_cache[name])

        win._on_rms_window_mode_changed(RMSWindowMode.TWO_CYCLE)
        _process_events(qapp, count=4)

        assert win._rms_config.window_mode == RMSWindowMode.TWO_CYCLE
        assert name in win._canvas._rms_data_cache
        assert len(win._canvas._rms_data_cache[name]) < one_cycle_len
        assert win._canvas._primary_plot.getViewBox().viewRange()[0] == pytest.approx(
            win._timeline.getPlotItem().getViewBox().viewRange()[0]
        )
        _assert_no_deleted_sync_registrations(win)
    finally:
        win.close()
        qapp.processEvents()


def test_direct_csv_can_open_twice_without_deleted_timeline(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = _load_direct_result(Path("samples/csv/pulu_20260306.csv"))
    win = PowerwaveMainWindow()
    dialog_patch = _accept_review_dialog()
    try:
        win.show()
        qapp.processEvents()

        win._on_record_loaded(result)
        _process_events(qapp, count=8)
        win._on_record_loaded(result)
        _process_events(qapp, count=8)

        _assert_standard_widgets_alive(win)
        assert list(win._panel_canvases) == ["power", "frequency"]
        assert win._grouped_timeline is None
        assert not win._timeline.isVisible()
        _assert_no_deleted_sync_registrations(win)
    finally:
        dialog_patch.stop()
        win.close()
        qapp.processEvents()


def test_direct_comtrade_can_open_twice_without_stale_widgets(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = _load_direct_result(Path("samples/comtrade/pulu_20260306.cfg"))
    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()

        win._on_record_loaded(result)
        _process_events(qapp, count=4)
        win._on_record_loaded(result)
        _process_events(qapp, count=4)

        _assert_standard_widgets_alive(win)
        splitter = win.centralWidget()
        assert isinstance(splitter, QSplitter)
        assert splitter.count() == 2
        assert win._timeline.isVisible()
        assert len(win._canvas._axis_manager.get_curves()) == 8
        assert win._signal_browser.entry_count() == 42 + 88
        assert win._vis_manager.synchronization_manager.registered_count == 2
        _assert_no_deleted_sync_registrations(win)
    finally:
        win.close()
        qapp.processEvents()


def test_direct_csv_to_comtrade_restores_standard_timeline(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    csv_result = _load_direct_result(Path("samples/csv/pulu_20260306.csv"))
    comtrade_result = _load_direct_result(Path("samples/comtrade/pulu_20260306.cfg"))
    win = PowerwaveMainWindow()
    dialog_patch = _accept_review_dialog()
    try:
        win.show()
        qapp.processEvents()

        win._on_record_loaded(csv_result)
        _process_events(qapp, count=8)
        assert win._panel_canvases
        assert not win._timeline.isVisible()

        win._on_record_loaded(comtrade_result)
        _process_events(qapp, count=8)

        _assert_standard_widgets_alive(win)
        assert win._panel_canvases == {}
        assert win._grouped_timeline is None
        assert win._timeline.isVisible()
        assert len(win._canvas._axis_manager.get_curves()) == 8
        assert win._signal_browser.entry_count() == 42 + 88
        assert win._vis_manager.synchronization_manager.registered_count == 2
        _assert_no_deleted_sync_registrations(win)
    finally:
        dialog_patch.stop()
        win.close()
        qapp.processEvents()


def test_direct_comtrade_to_csv_keeps_standard_widgets_detached_not_deleted(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    comtrade_result = _load_direct_result(Path("samples/comtrade/pulu_20260306.cfg"))
    csv_result = _load_direct_result(Path("samples/csv/pulu_20260306.csv"))
    win = PowerwaveMainWindow()
    dialog_patch = _accept_review_dialog()
    try:
        win.show()
        qapp.processEvents()

        win._on_record_loaded(comtrade_result)
        _process_events(qapp, count=4)
        win._on_record_loaded(csv_result)
        _process_events(qapp, count=8)

        _assert_standard_widgets_alive(win)
        assert list(win._panel_canvases) == ["power", "frequency"]
        assert win._grouped_timeline is None
        assert not win._timeline.isVisible()
        _assert_no_deleted_sync_registrations(win)
    finally:
        dialog_patch.stop()
        win.close()
        qapp.processEvents()


def test_multi_source_to_direct_csv_does_not_reuse_deleted_timeline(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    csv_result = _load_direct_result(Path("samples/csv/pulu_20260306.csv"))
    session = _synthetic_single_source_session()
    win = PowerwaveMainWindow()
    dialog_patch = _accept_review_dialog()
    try:
        win.show()
        qapp.processEvents()

        win._on_multi_source_loaded(session)
        _process_events(qapp, count=8)
        assert win._panel_canvases

        win._on_record_loaded(csv_result)
        _process_events(qapp, count=8)

        _assert_standard_widgets_alive(win)
        assert list(win._panel_canvases) == ["power", "frequency"]
        _assert_no_deleted_sync_registrations(win)
    finally:
        dialog_patch.stop()
        win.close()
        qapp.processEvents()


def test_direct_csv_to_multi_source_keeps_sync_registry_clean(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    csv_result = _load_direct_result(Path("samples/csv/pulu_20260306.csv"))
    session = _synthetic_single_source_session()
    win = PowerwaveMainWindow()
    dialog_patch = _accept_review_dialog()
    try:
        win.show()
        qapp.processEvents()

        win._on_record_loaded(csv_result)
        _process_events(qapp, count=8)
        win._on_multi_source_loaded(session)
        _process_events(qapp, count=8)

        _assert_standard_widgets_alive(win)
        assert win._panel_canvases
        assert win._vis_manager.synchronization_manager.registered_count == len(
            win._panel_canvases
        )
        _assert_no_deleted_sync_registrations(win)
    finally:
        dialog_patch.stop()
        win.close()
        qapp.processEvents()


def test_grouped_layout_restore_recreates_deleted_standard_widgets(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = _load_direct_result(Path("samples/comtrade/pulu_20260306.cfg"))
    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()
        old_timeline = win._timeline
        sip.delete(old_timeline)
        _process_events(qapp, count=8)
        assert sip.isdeleted(old_timeline)

        win._restore_standard_layout()
        _process_events(qapp, count=4)
        win._on_record_loaded(result)
        _process_events(qapp, count=4)

        _assert_standard_widgets_alive(win)
        assert win._timeline is not old_timeline
        assert win._timeline.isVisible()
        assert win._vis_manager.synchronization_manager.registered_count == 2
        _assert_no_deleted_sync_registrations(win)
    finally:
        win.close()
        qapp.processEvents()


def test_signal_browser_can_reveal_hidden_comtrade_channel_without_reload(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = _load_direct_result(Path("samples/comtrade/pulu_20260306.cfg"))
    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()
        win._on_record_loaded(result)
        _process_events(qapp, count=4)

        hidden_name = result.record.analog_channels[10].name
        assert hidden_name not in win._canvas._axis_manager.get_curves()
        entry_key = next(
            key
            for key, target in win._signal_entry_targets.items()
            if target == ("analog", "standard", hidden_name)
        )

        before_range = win._canvas._primary_plot.getViewBox().viewRange()[0]
        win._canvas._cursor.setValue(0.25)
        _process_events(qapp)
        win._on_signal_visibility_changed(entry_key, True)
        _process_events(qapp, count=4)

        assert hidden_name in win._canvas._axis_manager.get_curves()
        assert len(win._canvas._axis_manager.get_curves()) == 9
        assert win._canvas._primary_plot.getViewBox().viewRange()[0] == pytest.approx(before_range)
        assert win._canvas._cursor.value() == pytest.approx(0.25)
        assert win._vis_manager.synchronization_manager.registered_count == 2
        _assert_no_deleted_sync_registrations(win)
    finally:
        win.close()
        qapp.processEvents()


def test_signal_browser_hides_grouped_csv_axis_and_preserves_sync(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = _load_direct_result(Path("samples/csv/pulu_20260306.csv"))
    win = PowerwaveMainWindow()
    dialog_patch = _accept_review_dialog()
    try:
        win.show()
        qapp.processEvents()
        win._on_record_loaded(result)
        _process_events(qapp, count=8)

        power = win._panel_canvases["power"]
        frequency = win._panel_canvases["frequency"]
        power._primary_plot.setXRange(600.0, 1200.0, padding=0)
        power._cursor.setValue(900.0)
        _process_events(qapp)
        assert power.right_axis_count() == 0
        assert power._axis_manager.axis_count() == 1

        entry_key = next(
            key
            for key, target in win._signal_entry_targets.items()
            if target == ("analog", "power", "Tie-Line")
        )
        win._on_signal_visibility_changed(entry_key, False)
        _process_events(qapp, count=8)

        assert set(power._axis_manager.get_curves()) == {"System Demand"}
        assert power.right_axis_count() == 0
        assert power._axis_manager.axis_count() == 1
        assert frequency._primary_plot.getViewBox().viewRange()[0] == pytest.approx([600.0, 1200.0])
        assert frequency._cursor.value() == pytest.approx(900.0)
        _assert_same_x_pixel_mapping([power, frequency], [600.0, 900.0, 1200.0])
        _assert_no_deleted_sync_registrations(win)
    finally:
        dialog_patch.stop()
        win.close()
        qapp.processEvents()


def test_axis_mode_switches_between_shared_and_dedicated_csv_axes(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow
    from app.visualization.axis_management import AxisDisplayMode

    result = _load_direct_result(Path("samples/csv/pulu_20260306.csv"))
    win = PowerwaveMainWindow()
    dialog_patch = _accept_review_dialog()
    try:
        win.show()
        qapp.processEvents()
        win._on_record_loaded(result)
        _process_events(qapp, count=8)

        power = win._panel_canvases["power"]
        frequency = win._panel_canvases["frequency"]
        assert power.axis_display_mode() == AxisDisplayMode.SHARED
        assert power._axis_manager.axis_count() == 1
        assert power.right_axis_count() == 0

        win._on_axis_display_mode_changed(AxisDisplayMode.DEDICATED)
        _process_events(qapp, count=8)
        assert power.axis_display_mode() == AxisDisplayMode.DEDICATED
        assert power._axis_manager.axis_count() == 2
        assert power.right_axis_count() == 1

        win._on_axis_display_mode_changed(AxisDisplayMode.SHARED)
        _process_events(qapp, count=8)
        assert power.axis_display_mode() == AxisDisplayMode.SHARED
        assert power._axis_manager.axis_count() == 1
        _assert_same_x_pixel_mapping([power, frequency], [0.0, 1800.0, 3840.0])
    finally:
        dialog_patch.stop()
        win.close()
        qapp.processEvents()


def test_comtrade_standard_analog_and_digital_timeline_are_pixel_aligned(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = _load_direct_result(Path("samples/comtrade/pulu_20260306.cfg"))
    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()
        win._on_record_loaded(result)
        _process_events(qapp, count=8)

        canvas = win._canvas
        timeline = win._timeline
        canvas._primary_plot.setXRange(0.0, 0.2, padding=0)
        _process_events(qapp, count=8)

        for x_value in [0.0, 0.1, 0.2]:
            analog_px = _mapped_view_pixel_x(canvas, x_value)
            digital_px = _mapped_timeline_pixel_x(timeline, x_value)
            assert abs(analog_px - digital_px) <= 1.0
    finally:
        win.close()
        qapp.processEvents()


def test_signal_visibility_removes_rms_overlay_with_hidden_channel(
    qapp: QApplication,
) -> None:
    from app.analytics.rms.rms_models import RMSDisplayMode
    from app.data.synthetic import make_high_rate_record
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    record = make_high_rate_record(duration_s=1.0, sampling_rate_hz=1000.0).record
    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(record)
        canvas.set_rms_display_mode(RMSDisplayMode.OVERLAY)
        _process_events(qapp)

        visible = canvas.visible_channel_names()
        hidden = visible[0]
        assert hidden in canvas._rms_curves

        canvas.set_visible_channels(visible[1:])
        _process_events(qapp)

        assert hidden not in canvas._axis_manager.get_curves()
        assert hidden not in canvas._rms_curves
        assert set(canvas._rms_curves).issubset(set(canvas.visible_channel_names()))
    finally:
        canvas.close()
        qapp.processEvents()


def test_signal_browser_can_reveal_hidden_digital_track(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    result = _load_direct_result(Path("samples/comtrade/pulu_20260306.cfg"))
    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()
        win._on_record_loaded(result)
        _process_events(qapp, count=4)

        hidden_name = result.record.digital_channels[20].name
        assert len(win._timeline._tracks) == 16
        assert hidden_name not in win._timeline._tracks
        entry_key = next(
            key
            for key, target in win._signal_entry_targets.items()
            if target == ("digital", "digital", hidden_name)
        )

        win._on_signal_visibility_changed(entry_key, True)
        _process_events(qapp, count=4)

        assert len(win._timeline._tracks) == 17
        assert hidden_name in win._timeline._tracks
        assert win._vis_manager.synchronization_manager.registered_count == 2
        _assert_no_deleted_sync_registrations(win)
    finally:
        win.close()
        qapp.processEvents()


def test_signal_browser_supports_multi_source_panel_visibility(
    qapp: QApplication,
) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow

    session = _synthetic_single_source_session()
    win = PowerwaveMainWindow()
    try:
        win.show()
        qapp.processEvents()
        win._on_multi_source_loaded(session)
        _process_events(qapp, count=8)

        panel_key = next(key for key in win._panel_canvases if key.startswith("synthetic/"))
        canvas = win._panel_canvases[panel_key]
        signal_name = canvas.visible_channel_names()[0]
        entry_key = next(
            key
            for key, target in win._signal_entry_targets.items()
            if target == ("analog", panel_key, signal_name)
        )

        before_range = canvas._primary_plot.getViewBox().viewRange()[0]
        win._on_signal_visibility_changed(entry_key, False)
        _process_events(qapp, count=8)

        assert signal_name not in canvas._axis_manager.get_curves()
        assert canvas._primary_plot.getViewBox().viewRange()[0] == pytest.approx(before_range)
        assert win._vis_manager.synchronization_manager.registered_count == len(
            win._panel_canvases
        )
        _assert_no_deleted_sync_registrations(win)
    finally:
        win.close()
        qapp.processEvents()
