"""Visibility guarantee tests for D4.4.2 — Panel Scaling & Stabilization.

Covers:
  - Sparse-mode detection for PULU-like data (1 sample/min)
  - _finite_y_range padding logic
  - _force_y_ranges pins correct per-channel Y extents after set_record()
  - axis_mode=AXIS_MODE_ABSOLUTE sets datetime axis start_time
  - axis_mode=AXIS_MODE_RELATIVE passes None to datetime axis (elapsed-seconds)
  - Multi-magnitude channels (System Demand ~18700 MW, Tie-Line ~100 MW) have
    independent ViewBox Y ranges that do not bleed into each other
  - VisualizationManager.set_record() / display_grouped_record() pass axis_mode
    through to the underlying canvas

Run with:
    python -m pytest tests/unit/test_d442_panel_visibility.py -v -s
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

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
from app.visualization.axis.datetime_axis import AXIS_MODE_ABSOLUTE, AXIS_MODE_RELATIVE
from app.visualization.widgets.flexible_plot_canvas import (
    _finite_y_range,
    _is_sparse_timeseries,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    pg.setConfigOptions(useOpenGL=False, antialias=False, foreground="w", background="#1E1E1E")
    return app


def _make_sparse_record(channels: dict[str, list[float]]) -> DisturbanceRecord:
    """Build a 1-sample/minute DisturbanceRecord (like PULU CSV data)."""
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


def _make_high_rate_record(channel_name: str = "VA") -> DisturbanceRecord:
    """Build a 6400 Hz DisturbanceRecord (COMTRADE-like)."""
    n = 6400
    t = np.arange(n, dtype=np.float64) / 6400.0
    start = datetime(2024, 1, 1, 0, 0, 0)
    data = np.sin(2 * np.pi * 50.0 * t)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="GRID", recorder_name="COMTRADE",
            source_file="test.cfg", provider_type="comtrade", nominal_frequency=50.0,
        ),
        waveform_data=pd.DataFrame({"time": t, channel_name: data}),
        analog_channels=[AnalogChannel(name=channel_name, unit="kV", index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[6400.0], samples_per_rate=[n]),
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
# TestSparseDetection
# ─────────────────────────────────────────────────────────────────────────────


class TestSparseDetection:
    """_is_sparse_timeseries() identifies PULU-like low-rate data."""

    def test_one_sample_per_minute_is_sparse(self) -> None:
        record = _make_sparse_record({"SD": _SYSTEM_DEMAND})
        t = record.waveform_data["time"].to_numpy(dtype=np.float64)
        assert _is_sparse_timeseries(record, t) is True

    def test_rate_below_threshold_triggers_sparse(self) -> None:
        record = _make_sparse_record({"SD": _SYSTEM_DEMAND})
        t = record.waveform_data["time"].to_numpy(dtype=np.float64)
        # sampling_rates contains 1/60 ≈ 0.0167 Hz, well below 2 Hz threshold
        assert max(record.sampling_info.sampling_rates) < 2.0
        assert _is_sparse_timeseries(record, t) is True

    def test_high_rate_is_not_sparse(self) -> None:
        record = _make_high_rate_record()
        t = record.waveform_data["time"].to_numpy(dtype=np.float64)
        assert _is_sparse_timeseries(record, t) is False

    def test_two_samples_or_fewer_is_sparse(self) -> None:
        """Edge: ≤ 2 samples cannot determine interval — treated as sparse."""
        # Create a record with a fast sample rate but only 2 data points
        start = datetime(2024, 1, 1)
        record = DisturbanceRecord(
            metadata=RecordingMetadata(
                station_name="T", recorder_name="R",
                source_file="t.csv", provider_type="csv", nominal_frequency=50.0,
            ),
            waveform_data=pd.DataFrame({"time": [0.0, 0.001], "X": [1.0, 2.0]}),
            analog_channels=[AnalogChannel(name="X", unit="V", index=0)],
            digital_channels=[],
            sampling_info=SamplingInformation(sampling_rates=[1000.0], samples_per_rate=[2]),
            timing_info=TimingInformation(start_time=start, trigger_time=start),
            disturbance_info=None,
        )
        t = record.waveform_data["time"].to_numpy(dtype=np.float64)
        assert _is_sparse_timeseries(record, t) is True


# ─────────────────────────────────────────────────────────────────────────────
# TestFiniteYRange
# ─────────────────────────────────────────────────────────────────────────────


class TestFiniteYRange:
    """_finite_y_range() correctly computes padded Y bounds."""

    def test_returns_none_for_empty_array(self) -> None:
        assert _finite_y_range(np.array([])) is None

    def test_returns_none_for_all_nan(self) -> None:
        assert _finite_y_range(np.array([np.nan, np.nan])) is None

    def test_returns_none_for_all_inf(self) -> None:
        assert _finite_y_range(np.array([np.inf, -np.inf])) is None

    def test_range_contains_data_values(self) -> None:
        data = np.array(_SYSTEM_DEMAND)
        lo, hi = _finite_y_range(data)  # type: ignore[misc]
        assert lo < min(_SYSTEM_DEMAND)
        assert hi > max(_SYSTEM_DEMAND)

    def test_constant_array_gets_absolute_padding(self) -> None:
        data = np.full(10, 50.0)
        lo, hi = _finite_y_range(data)  # type: ignore[misc]
        assert hi > lo, "Constant-value array must get nonzero padding"
        assert (hi - lo) >= 1.0, "Minimum padding for constant is 1.0"

    def test_nan_values_ignored(self) -> None:
        data = np.array([1.0, np.nan, 3.0])
        lo, hi = _finite_y_range(data)  # type: ignore[misc]
        assert lo < 1.0
        assert hi > 3.0

    def test_high_magnitude_data_has_proportional_padding(self) -> None:
        data = np.array(_SYSTEM_DEMAND)
        lo, hi = _finite_y_range(data)  # type: ignore[misc]
        span = hi - lo
        data_span = max(_SYSTEM_DEMAND) - min(_SYSTEM_DEMAND)
        assert span > data_span, "Padded range must exceed raw data span"

    def test_frequency_data_has_nonzero_span(self) -> None:
        data = np.array(_FREQUENCY)
        lo, hi = _finite_y_range(data)  # type: ignore[misc]
        assert (hi - lo) > 0.001, "Frequency Y range must be non-degenerate"


# ─────────────────────────────────────────────────────────────────────────────
# TestAxisModePolicy (mock-based, no Qt window)
# ─────────────────────────────────────────────────────────────────────────────


class TestAxisModePolicy:
    """axis_mode threading: FlexiblePlotCanvas passes the correct value to DatetimeAxisItem."""

    def test_axis_mode_constants_exist(self) -> None:
        assert AXIS_MODE_RELATIVE == "relative_seconds"
        assert AXIS_MODE_ABSOLUTE == "absolute_datetime"

    def test_absolute_mode_sets_start_time(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"SD": _SYSTEM_DEMAND})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
            qapp.processEvents()
            assert canvas._datetime_axis._start_time == record.timing_info.start_time
        finally:
            canvas.close()
            qapp.processEvents()

    def test_relative_mode_clears_start_time(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_high_rate_record()
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_RELATIVE)
            qapp.processEvents()
            assert canvas._datetime_axis._start_time is None
        finally:
            canvas.close()
            qapp.processEvents()

    def test_default_mode_is_relative(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_high_rate_record()
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record)  # no axis_mode kwarg
            qapp.processEvents()
            assert canvas._datetime_axis._start_time is None
        finally:
            canvas.close()
            qapp.processEvents()

    def test_second_set_record_absolute_then_relative_clears_start_time(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        sparse = _make_sparse_record({"SD": _SYSTEM_DEMAND})
        fast = _make_high_rate_record()
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(sparse, axis_mode=AXIS_MODE_ABSOLUTE)
            assert canvas._datetime_axis._start_time is not None
            canvas.set_record(fast, axis_mode=AXIS_MODE_RELATIVE)
            qapp.processEvents()
            assert canvas._datetime_axis._start_time is None
        finally:
            canvas.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# TestSparseViewBoxRanges
# ─────────────────────────────────────────────────────────────────────────────


class TestSparseViewBoxRanges:
    """After set_record() with sparse data, ViewBox Y ranges must cover the data."""

    def test_system_demand_y_range_covers_full_dataset(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"System Demand": _SYSTEM_DEMAND})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
            qapp.processEvents()

            vb = canvas._axis_manager.get_curves()["System Demand"].getViewBox()
            y_lo, y_hi = vb.viewRange()[1]
            assert y_lo < min(_SYSTEM_DEMAND), (
                f"Lower Y bound {y_lo} must be below data minimum {min(_SYSTEM_DEMAND)}"
            )
            assert y_hi > max(_SYSTEM_DEMAND), (
                f"Upper Y bound {y_hi} must be above data maximum {max(_SYSTEM_DEMAND)}"
            )
        finally:
            canvas.close()
            qapp.processEvents()

    def test_frequency_y_range_is_non_degenerate(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"Frequency": _FREQUENCY})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
            qapp.processEvents()

            vb = canvas._axis_manager.get_curves()["Frequency"].getViewBox()
            y_lo, y_hi = vb.viewRange()[1]
            assert (y_hi - y_lo) > 0.001, (
                f"Frequency Y span {y_hi - y_lo:.6f} is too small — curve invisible"
            )
        finally:
            canvas.close()
            qapp.processEvents()

    def test_frequency_y_range_covers_50hz(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"Frequency": _FREQUENCY})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
            qapp.processEvents()

            vb = canvas._axis_manager.get_curves()["Frequency"].getViewBox()
            y_lo, y_hi = vb.viewRange()[1]
            assert y_lo < min(_FREQUENCY), f"Lower bound {y_lo:.4f} must be below {min(_FREQUENCY)}"
            assert y_hi > max(_FREQUENCY), f"Upper bound {y_hi:.4f} must be above {max(_FREQUENCY)}"
        finally:
            canvas.close()
            qapp.processEvents()

    def test_sparse_mode_flag_set_for_pulu_data(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"System Demand": _SYSTEM_DEMAND})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
            assert canvas._sparse_mode is True
        finally:
            canvas.close()
            qapp.processEvents()

    def test_sparse_mode_flag_not_set_for_high_rate(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_high_rate_record()
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record)
            assert canvas._sparse_mode is False
        finally:
            canvas.close()
            qapp.processEvents()

    def test_data_cache_populated_after_set_record(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"System Demand": _SYSTEM_DEMAND, "Tie-Line": _TIE_LINE})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
            assert "System Demand" in canvas._data_cache
            assert "Tie-Line" in canvas._data_cache
            assert len(canvas._data_cache["System Demand"]) == len(_SYSTEM_DEMAND)
            assert len(canvas._data_cache["Tie-Line"]) == len(_TIE_LINE)
        finally:
            canvas.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# TestMultiMagnitudeIndependence
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiMagnitudeIndependence:
    """System Demand (~18700 MW) and Tie-Line (~100 MW) must have non-overlapping Y ranges."""

    def test_viewboxes_are_different_objects(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"System Demand": _SYSTEM_DEMAND, "Tie-Line": _TIE_LINE})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
            qapp.processEvents()

            curves = canvas._axis_manager.get_curves()
            sd_vb = curves["System Demand"].getViewBox()
            tl_vb = curves["Tie-Line"].getViewBox()
            assert sd_vb is not tl_vb
        finally:
            canvas.close()
            qapp.processEvents()

    def test_system_demand_y_range_does_not_overlap_tie_line_range(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"System Demand": _SYSTEM_DEMAND, "Tie-Line": _TIE_LINE})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
            qapp.processEvents()

            curves = canvas._axis_manager.get_curves()
            sd_lo, sd_hi = curves["System Demand"].getViewBox().viewRange()[1]
            tl_lo, tl_hi = curves["Tie-Line"].getViewBox().viewRange()[1]

            # These ranges must NOT overlap: System Demand is ~18000–18800, Tie-Line is ~50–120
            assert sd_lo > tl_hi or tl_lo > sd_hi, (
                f"System Demand [{sd_lo:.0f}, {sd_hi:.0f}] and "
                f"Tie-Line [{tl_lo:.0f}, {tl_hi:.0f}] unexpectedly overlap — "
                "independent Y axes are not working"
            )
        finally:
            canvas.close()
            qapp.processEvents()

    def test_system_demand_range_in_18000_mw_band(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"System Demand": _SYSTEM_DEMAND, "Tie-Line": _TIE_LINE})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
            qapp.processEvents()

            sd_vb = canvas._axis_manager.get_curves()["System Demand"].getViewBox()
            y_lo, y_hi = sd_vb.viewRange()[1]
            # System Demand must be near 18000 MW, not near 0 or 100 MW
            assert y_lo > 1000.0, f"System Demand lower bound {y_lo:.0f} unexpectedly low"
            assert y_hi < 25000.0, f"System Demand upper bound {y_hi:.0f} unexpectedly high"
        finally:
            canvas.close()
            qapp.processEvents()

    def test_tie_line_range_in_100_mw_band(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({"System Demand": _SYSTEM_DEMAND, "Tie-Line": _TIE_LINE})
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
            qapp.processEvents()

            tl_vb = canvas._axis_manager.get_curves()["Tie-Line"].getViewBox()
            y_lo, y_hi = tl_vb.viewRange()[1]
            # Tie-Line must be near 50–120 MW
            assert y_lo < 200.0, f"Tie-Line lower bound {y_lo:.0f} unexpectedly high"
            assert y_hi < 500.0, f"Tie-Line upper bound {y_hi:.0f} unexpectedly high"
            assert y_lo > -500.0, f"Tie-Line lower bound {y_lo:.0f} unexpectedly negative"
        finally:
            canvas.close()
            qapp.processEvents()

    def test_three_channels_all_visible(self, qapp) -> None:
        """Adding a third channel (Frequency ~50 Hz) alongside power channels."""
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        record = _make_sparse_record({
            "System Demand": _SYSTEM_DEMAND,
            "Tie-Line": _TIE_LINE,
            "Frequency": _FREQUENCY,
        })
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
            qapp.processEvents()

            curves = canvas._axis_manager.get_curves()
            assert len(curves) == 3

            for name, expected_data in [
                ("System Demand", _SYSTEM_DEMAND),
                ("Tie-Line", _TIE_LINE),
                ("Frequency", _FREQUENCY),
            ]:
                vb = curves[name].getViewBox()
                y_lo, y_hi = vb.viewRange()[1]
                assert y_lo < min(expected_data), (
                    f"{name}: lower bound {y_lo} not below data minimum {min(expected_data)}"
                )
                assert y_hi > max(expected_data), (
                    f"{name}: upper bound {y_hi} not above data maximum {max(expected_data)}"
                )
        finally:
            canvas.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# TestVisualizationManagerAxisMode (mock-based, no Qt window)
# ─────────────────────────────────────────────────────────────────────────────


class TestVisualizationManagerAxisMode:
    """VisualizationManager threads axis_mode through to canvas.set_record()."""

    def _make_manager(self):
        from app.visualization.managers.visualization_manager import VisualizationManager
        canvas = MagicMock()
        timeline = MagicMock()
        return VisualizationManager(canvas, timeline), canvas, timeline

    def test_set_record_default_mode_is_relative(self) -> None:
        mgr, canvas, _ = self._make_manager()
        record = MagicMock()
        mgr.set_record(record)
        canvas.set_record.assert_called_once_with(record, axis_mode=AXIS_MODE_RELATIVE)

    def test_set_record_passes_absolute_mode(self) -> None:
        mgr, canvas, _ = self._make_manager()
        record = MagicMock()
        mgr.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
        canvas.set_record.assert_called_once_with(record, axis_mode=AXIS_MODE_ABSOLUTE)

    def test_display_grouped_record_default_mode_is_relative(self) -> None:
        mgr, _, _ = self._make_manager()
        from app.data.synthetic import make_mixed_disturbance_record
        result = make_mixed_disturbance_record()
        received_modes: list[str] = []

        def factory() -> MagicMock:
            m = MagicMock()
            m.set_record.side_effect = lambda r, **kw: received_modes.append(kw.get("axis_mode", ""))
            return m

        mgr.display_grouped_record(result.record, result.signal_metadata, canvas_factory=factory)
        assert all(mode == AXIS_MODE_RELATIVE for mode in received_modes), (
            f"Expected all panels to use {AXIS_MODE_RELATIVE!r}, got {received_modes}"
        )

    def test_display_grouped_record_passes_absolute_mode(self) -> None:
        mgr, _, _ = self._make_manager()
        from app.data.synthetic import make_mixed_disturbance_record
        result = make_mixed_disturbance_record()
        received_modes: list[str] = []

        def factory() -> MagicMock:
            m = MagicMock()
            m.set_record.side_effect = lambda r, **kw: received_modes.append(kw.get("axis_mode", ""))
            return m

        mgr.display_grouped_record(
            result.record, result.signal_metadata,
            canvas_factory=factory,
            axis_mode=AXIS_MODE_ABSOLUTE,
        )
        assert all(mode == AXIS_MODE_ABSOLUTE for mode in received_modes), (
            f"Expected all panels to use {AXIS_MODE_ABSOLUTE!r}, got {received_modes}"
        )

    def test_csv_path_uses_absolute_mode(self) -> None:
        """Simulate what main_window._handle_direct_csv_excel() does."""
        mgr, canvas, _ = self._make_manager()
        record = MagicMock()
        mgr.set_record(record, axis_mode=AXIS_MODE_ABSOLUTE)
        _, call_kwargs = canvas.set_record.call_args
        assert call_kwargs["axis_mode"] == AXIS_MODE_ABSOLUTE

    def test_comtrade_path_uses_relative_mode(self) -> None:
        """Simulate what the COMTRADE open path does (default axis_mode)."""
        mgr, canvas, _ = self._make_manager()
        record = MagicMock()
        mgr.set_record(record)  # default
        _, call_kwargs = canvas.set_record.call_args
        assert call_kwargs["axis_mode"] == AXIS_MODE_RELATIVE
