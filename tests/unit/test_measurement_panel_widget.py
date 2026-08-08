"""Unit tests for MeasurementPanel's rendering of per-curve measurement
results (Sprint 1A), especially the explicit "Unavailable" row treatment.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication

from app.ui.widgets.measurement_panel import MeasurementPanel
from app.visualization.interaction.measurement_engine import (
    CurveSample,
    compute_measurements,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _curve(name: str, unit: str, time, values) -> CurveSample:
    return CurveSample(name=name, unit=unit, time=np.asarray(time, dtype=np.float64), values=np.asarray(values, dtype=np.float64))


class TestAvailableRowRendering:
    def test_available_row_shows_numeric_values(self, qapp) -> None:
        t = np.linspace(0, 1, 11)
        curve = _curve("A", "kV", t, np.arange(11, dtype=float))
        result = compute_measurements(0.2, 0.5, [curve])

        panel = MeasurementPanel()
        panel.update_measurements(result)
        assert panel._table.rowCount() == 1
        assert panel._table.item(0, panel._COL_CHANNEL).text() == "A"
        assert panel._table.item(0, panel._COL_DY).text() != "Unavailable"


class TestUnavailableRowRendering:
    def test_unavailable_row_shows_explicit_marker_in_every_stat_column(self, qapp) -> None:
        empty_curve = _curve("Empty", "V", [], [])
        result = compute_measurements(0.0, 1.0, [empty_curve])

        panel = MeasurementPanel()
        panel.update_measurements(result)
        assert panel._table.rowCount() == 1
        for col in (panel._COL_DY, panel._COL_RMS, panel._COL_MEAN, panel._COL_PEAK, panel._COL_PP):
            assert panel._table.item(0, col).text() == "Unavailable"

    def test_unavailable_row_tooltip_carries_the_reason(self, qapp) -> None:
        empty_curve = _curve("Empty", "V", [], [])
        result = compute_measurements(0.0, 1.0, [empty_curve])

        panel = MeasurementPanel()
        panel.update_measurements(result)
        item = panel._table.item(0, panel._COL_DY)
        assert item.toolTip()  # non-empty
        assert "No data" in item.toolTip()

    def test_mixed_available_and_unavailable_curves_both_get_rows(self, qapp) -> None:
        good = _curve("Good", "V", np.linspace(0, 1, 50), np.ones(50))
        empty = _curve("Empty", "V", [], [])
        result = compute_measurements(0.0, 1.0, [good, empty])

        panel = MeasurementPanel()
        panel.update_measurements(result)
        assert panel._table.rowCount() == 2
        labels = [panel._table.item(r, panel._COL_CHANNEL).text() for r in range(2)]
        assert labels == ["Good", "Empty"]
        assert panel._table.item(1, panel._COL_DY).text() == "Unavailable"
        assert panel._table.item(0, panel._COL_DY).text() != "Unavailable"

    def test_clear_on_invalid_result(self, qapp) -> None:
        panel = MeasurementPanel()
        good = _curve("Good", "V", np.linspace(0, 1, 50), np.ones(50))
        panel.update_measurements(compute_measurements(0.0, 1.0, [good]))
        assert panel._table.rowCount() == 1

        panel.update_measurements(None)
        assert panel._table.rowCount() == 0
