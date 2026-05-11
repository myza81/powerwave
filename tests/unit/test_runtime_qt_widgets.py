"""Runtime smoke tests for Qt/pyqtgraph widgets.

These tests intentionally instantiate real widgets. Import-only tests cannot
catch pyqtgraph constructor/API mismatches or scene lifecycle issues.
"""
from __future__ import annotations

import os
import sys
import dataclasses

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pyqtgraph as pg
import pytest
from PyQt6.QtWidgets import QApplication


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
