"""Diagnostic tests for D4.4.2 — confirm Y-range and layout issues with PULU-like data.

These tests reproduce the exact rendering conditions of the PULU CSV power panel
(System Demand ~18 700 MW + Tie-Line ~100 MW) and the frequency panel (~50 Hz)
to measure actual ViewBox Y-ranges and identify why System Demand and Frequency
appear invisible.

Run with:
    python -m pytest tests/unit/test_d442_diagnostic.py -v -s
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PyQt6.QtWidgets import QApplication

from app.models import DisturbanceRecord
from app.models.channels import AnalogChannel
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    pg.setConfigOptions(useOpenGL=False, antialias=False, foreground="w", background="#1E1E1E")
    return app


def _make_sparse_record(channels: dict[str, list[float]]) -> DisturbanceRecord:
    """Build a sparse (1-sample/minute) DisturbanceRecord like PULU CSV."""
    n = max(len(v) for v in channels.values())
    t = [float(i * 60) for i in range(n)]
    start = datetime(2026, 3, 6, 17, 25, 0)
    col_data: dict = {"time": np.array(t)}
    analog = []
    for i, (name, vals) in enumerate(channels.items()):
        col_data[name] = np.array(vals[:n])
        analog.append(AnalogChannel(name=name, unit="MW", index=i))
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST", recorder_name="CSV",
            source_file="pulu.csv", provider_type="csv", nominal_frequency=50.0,
        ),
        waveform_data=pd.DataFrame(col_data),
        analog_channels=analog,
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[1.0 / 60.0], samples_per_rate=[n]
        ),
        timing_info=TimingInformation(start_time=start, trigger_time=start),
        disturbance_info=None,
    )


_SYSTEM_DEMAND = [18738.85, 18751.21, 18739.43, 18771.59, 18698.91,
                  18678.16, 18683.19, 18659.56, 18690.96, 18712.34]
_TIE_LINE      = [108.16, 80.64, 80.32, 57.28, 79.36,
                  112.0, 80.0, 95.04, 68.16, 75.20]
_FREQUENCY     = [50.02, 50.02, 50.01, 49.98, 49.99,
                  50.01, 50.01, 50.02, 50.01, 50.00]


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: inspect actual ViewBox ranges
# ─────────────────────────────────────────────────────────────────────────────


class TestPowerPanelYRangeDiagnosis:
    """Confirm that System Demand and Tie-Line each have independent, correct Y ranges."""

    def test_system_demand_and_tie_line_use_independent_viewboxes(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"System Demand": _SYSTEM_DEMAND, "Tie-Line": _TIE_LINE})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode="absolute_datetime")
            qapp.processEvents()

            curves = canvas._axis_manager.get_curves()
            sd_vb = curves["System Demand"].getViewBox()
            tl_vb = curves["Tie-Line"].getViewBox()

            # They must use DIFFERENT ViewBoxes (independent Y axes)
            assert sd_vb is not tl_vb, (
                "System Demand and Tie-Line share the same ViewBox — "
                "they cannot have independent Y ranges"
            )
        finally:
            canvas.close()
            qapp.processEvents()

    def test_system_demand_viewbox_y_range_covers_18000_mw(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"System Demand": _SYSTEM_DEMAND, "Tie-Line": _TIE_LINE})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode="absolute_datetime")
            qapp.processEvents()

            sd_vb = canvas._axis_manager.get_curves()["System Demand"].getViewBox()
            y_lo, y_hi = sd_vb.viewRange()[1]
            print(f"\n  System Demand ViewBox Y range: [{y_lo:.1f}, {y_hi:.1f}]")
            print(f"  Expected to cover ~18659 – 18772")

            assert y_lo < 18659.56, f"Y lower bound {y_lo} too high — System Demand not visible"
            assert y_hi > 18771.59, f"Y upper bound {y_hi} too low — System Demand not visible"
        finally:
            canvas.close()
            qapp.processEvents()

    def test_tie_line_viewbox_y_range_covers_100_mw(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"System Demand": _SYSTEM_DEMAND, "Tie-Line": _TIE_LINE})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode="absolute_datetime")
            qapp.processEvents()

            tl_vb = canvas._axis_manager.get_curves()["Tie-Line"].getViewBox()
            y_lo, y_hi = tl_vb.viewRange()[1]
            print(f"\n  Tie-Line ViewBox Y range: [{y_lo:.2f}, {y_hi:.2f}]")
            print(f"  Expected to cover ~57 – 112")

            assert y_lo < 57.28, f"Y lower bound {y_lo} too high — Tie-Line not visible"
            assert y_hi > 112.0, f"Y upper bound {y_hi} too low — Tie-Line not visible"
        finally:
            canvas.close()
            qapp.processEvents()

    def test_system_demand_curve_has_finite_data(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"System Demand": _SYSTEM_DEMAND, "Tie-Line": _TIE_LINE})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode="absolute_datetime")
            qapp.processEvents()

            curve = canvas._axis_manager.get_curves()["System Demand"]
            x, y = curve.getData()
            print(f"\n  System Demand curve data points: {len(x) if x is not None else 0}")
            assert x is not None and len(x) > 0, "System Demand curve has no x data"
            assert y is not None and len(y) > 0, "System Demand curve has no y data"
            assert np.all(np.isfinite(y)), "System Demand curve has non-finite y values"
        finally:
            canvas.close()
            qapp.processEvents()


class TestFrequencyPanelYRangeDiagnosis:
    """Confirm that Frequency panel has correct Y range and visible curve."""

    def test_frequency_curve_has_finite_data(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"Frequency": _FREQUENCY})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode="absolute_datetime")
            qapp.processEvents()

            curve = canvas._axis_manager.get_curves()["Frequency"]
            x, y = curve.getData()
            print(f"\n  Frequency curve data points: {len(x) if x is not None else 0}")
            assert x is not None and len(x) > 0
            assert y is not None and len(y) > 0
        finally:
            canvas.close()
            qapp.processEvents()

    def test_frequency_viewbox_y_range_covers_50_hz(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"Frequency": _FREQUENCY})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode="absolute_datetime")
            qapp.processEvents()

            vb = canvas._axis_manager.get_curves()["Frequency"].getViewBox()
            y_lo, y_hi = vb.viewRange()[1]
            print(f"\n  Frequency ViewBox Y range: [{y_lo:.4f}, {y_hi:.4f}]")
            print(f"  Expected to cover ~49.98 – 50.02")

            assert y_lo < 49.98, f"Y lower {y_lo:.4f} > 49.98 — 50 Hz not visible"
            assert y_hi > 50.02, f"Y upper {y_hi:.4f} < 50.02 — 50 Hz not visible"
        finally:
            canvas.close()
            qapp.processEvents()

    def test_frequency_viewbox_y_range_is_non_degenerate(self, qapp) -> None:
        """Y range must be wide enough that a curve is visible (not zero height)."""
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"Frequency": _FREQUENCY})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode="absolute_datetime")
            qapp.processEvents()

            vb = canvas._axis_manager.get_curves()["Frequency"].getViewBox()
            y_lo, y_hi = vb.viewRange()[1]
            y_span = y_hi - y_lo
            print(f"\n  Frequency Y span: {y_span:.6f} Hz")

            assert y_span > 0.001, f"Y span {y_span:.6f} is too small — curve not distinguishable"
        finally:
            canvas.close()
            qapp.processEvents()
