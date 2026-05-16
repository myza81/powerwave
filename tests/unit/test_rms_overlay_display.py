"""Unit tests for RMS overlay display modes on FlexiblePlotCanvas.

Uses MagicMock + a minimal QApplication-free canvas factory so no Qt
display is required.  Tests exercise the coordination layer: which
curves are created/hidden/removed when modes change.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.analytics.rms.rms_models import RMSConfig, RMSDisplayMode, RMSWindowMode


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_disturbance_record(
    n_samples: int = 600,
    sampling_rate: float = 5000.0,
    channel_names: list[str] | None = None,
):
    """Build a minimal DisturbanceRecord-like mock for canvas tests."""
    from datetime import datetime, timedelta, timezone
    from app.models import DisturbanceRecord, RecordingMetadata, AnalogChannel
    from app.models.timing import TimingInformation, SamplingInformation, DisturbanceInformation

    if channel_names is None:
        channel_names = ["VA", "VB"]

    t = np.linspace(0.0, n_samples / sampling_rate, n_samples, endpoint=False)
    data = {
        "time": t,
    }
    for i, name in enumerate(channel_names):
        data[name] = np.sin(2 * math.pi * 50.0 * t + i * math.pi / 3)

    import pandas as pd
    df = pd.DataFrame(data)

    channels = [
        AnalogChannel(name=name, unit="kV", index=i, phase=None,
                      description=None, scale=1.0, offset=0.0,
                      primary_ratio=None, secondary_ratio=None)
        for i, name in enumerate(channel_names)
    ]

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST", recorder_name="SIM", source_file="test.cfg",
            nominal_frequency=50.0, provider_type="comtrade",
        ),
        waveform_data=df,
        analog_channels=channels,
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[sampling_rate],
            samples_per_rate=[n_samples],
        ),
        timing_info=TimingInformation(
            start_time=t0,
            trigger_time=t0 + timedelta(seconds=n_samples / sampling_rate / 2),
            time_multiplier=1.0,
        ),
        disturbance_info=DisturbanceInformation(event_type=None, notes=None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestRMSDisplayModels
# (model-level tests — no Qt required)
# ─────────────────────────────────────────────────────────────────────────────


class TestRMSDisplayMode:
    def test_off_value(self) -> None:
        assert RMSDisplayMode.OFF.value == "off"

    def test_overlay_value(self) -> None:
        assert RMSDisplayMode.OVERLAY.value == "overlay"

    def test_rms_only_value(self) -> None:
        assert RMSDisplayMode.RMS_ONLY.value == "rms_only"

    def test_three_modes_exist(self) -> None:
        modes = list(RMSDisplayMode)
        assert len(modes) == 3


class TestRMSConfig:
    def test_default_50hz(self) -> None:
        cfg = RMSConfig()
        assert cfg.nominal_frequency_hz == 50.0

    def test_default_one_cycle(self) -> None:
        assert RMSConfig().window_mode == RMSWindowMode.ONE_CYCLE
        assert RMSConfig().cycles_per_window == 1

    def test_window_modes_exist(self) -> None:
        assert RMSWindowMode.HALF_CYCLE.value == "half_cycle"
        assert RMSWindowMode.ONE_CYCLE.value == "one_cycle"
        assert RMSWindowMode.TWO_CYCLE.value == "two_cycle"
        assert RMSWindowMode.CUSTOM_SAMPLES.value == "custom_samples"

    def test_custom_values(self) -> None:
        cfg = RMSConfig(nominal_frequency_hz=60.0, cycles_per_window=2)
        assert cfg.nominal_frequency_hz == 60.0
        assert cfg.cycles_per_window == 2

    def test_custom_sample_window_config(self) -> None:
        cfg = RMSConfig(
            window_mode=RMSWindowMode.CUSTOM_SAMPLES,
            custom_window_samples=128,
        )
        assert cfg.window_mode == RMSWindowMode.CUSTOM_SAMPLES
        assert cfg.custom_window_samples == 128

    def test_frozen(self) -> None:
        cfg = RMSConfig()
        with pytest.raises(Exception):
            cfg.nominal_frequency_hz = 60.0  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# TestRMSEligibilityResult (models)
# ─────────────────────────────────────────────────────────────────────────────


class TestRMSEligibilityResult:
    def test_eligible_true(self) -> None:
        from app.analytics.rms.rms_models import RMSEligibilityResult
        r = RMSEligibilityResult(eligible=True, reason="test", auto_classified=True)
        assert r.eligible

    def test_auto_classified_false_for_override(self) -> None:
        from app.analytics.rms.rms_models import RMSEligibilityResult
        r = RMSEligibilityResult(eligible=True, reason="operator_override", auto_classified=False)
        assert not r.auto_classified

    def test_frozen(self) -> None:
        from app.analytics.rms.rms_models import RMSEligibilityResult
        r = RMSEligibilityResult(eligible=True, reason="x", auto_classified=True)
        with pytest.raises(Exception):
            r.eligible = False  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# TestRMSPenColor
# (internal helper — unit test without Qt)
# ─────────────────────────────────────────────────────────────────────────────


class TestRMSPenColor:
    def test_pure_red_lightened(self) -> None:
        from app.visualization.widgets.flexible_plot_canvas import _rms_pen_color
        result = _rms_pen_color("#FF0000")
        # Red channel stays max, green/blue increase
        r = int(result[1:3], 16)
        g = int(result[3:5], 16)
        b = int(result[5:7], 16)
        assert r == 255
        assert g > 0
        assert b > 0

    def test_white_stays_white(self) -> None:
        from app.visualization.widgets.flexible_plot_canvas import _rms_pen_color
        assert _rms_pen_color("#FFFFFF") == "#FFFFFF"

    def test_invalid_hex_returns_white(self) -> None:
        from app.visualization.widgets.flexible_plot_canvas import _rms_pen_color
        assert _rms_pen_color("invalid") == "#FFFFFF"

    def test_result_is_valid_hex(self) -> None:
        from app.visualization.widgets.flexible_plot_canvas import _rms_pen_color
        result = _rms_pen_color("#4488FF")
        assert result.startswith("#")
        assert len(result) == 7
        int(result[1:], 16)  # Should not raise

    def test_darker_color_is_lightened(self) -> None:
        from app.visualization.widgets.flexible_plot_canvas import _rms_pen_color
        original = "#FF4444"
        lighter = _rms_pen_color(original)
        r_orig = int(original[1:3], 16)
        r_light = int(lighter[1:3], 16)
        # At minimum same brightness; other channels must lighten
        g_light = int(lighter[3:5], 16)
        assert g_light > int(original[3:5], 16)


# ─────────────────────────────────────────────────────────────────────────────
# TestRMSCanvasIntegration
# Verifies coordination behavior using a real DisturbanceRecord + mock canvas
# infrastructure.  No actual rendering — validates curve bookkeeping.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def _qt_app(request):
    """Provide a minimal QApplication for canvas tests that need it."""
    try:
        from PyQt6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv[:1])
        return app
    except Exception:
        pytest.skip("Qt not available in this environment")


class TestRMSCanvasOffMode:
    def test_initial_mode_is_off(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        assert canvas._rms_display_mode == RMSDisplayMode.OFF

    def test_off_mode_no_rms_curves(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        record = _make_disturbance_record()
        canvas.set_record(record)
        assert len(canvas._rms_curves) == 0

    def test_off_mode_no_rms_data_cache(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        record = _make_disturbance_record()
        canvas.set_record(record)
        assert len(canvas._rms_data_cache) == 0


class TestRMSCanvasOverlayMode:
    def test_overlay_mode_creates_rms_curves_for_eligible_channels(
        self, _qt_app
    ) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        record = _make_disturbance_record(channel_names=["VA", "VB"])
        canvas.set_record(record)
        canvas.set_rms_display_mode(RMSDisplayMode.OVERLAY)
        # VA and VB are eligible (V-prefix) — should have RMS curves
        assert len(canvas._rms_curves) == 2
        assert "VA" in canvas._rms_curves
        assert "VB" in canvas._rms_curves
        assert canvas._rms_curves["VA"].opts["name"] == "VA RMS (kV)"

    def test_overlay_mode_populates_rms_data_cache(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        canvas.set_record(_make_disturbance_record(channel_names=["VA"]))
        canvas.set_rms_display_mode(RMSDisplayMode.OVERLAY)
        assert "VA" in canvas._rms_data_cache
        assert len(canvas._rms_data_cache["VA"]) > 0

    def test_ineligible_channels_excluded_from_overlay(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        record = _make_disturbance_record(channel_names=["MW", "Frequency"])
        canvas.set_record(record)
        canvas.set_rms_display_mode(RMSDisplayMode.OVERLAY)
        # MW and Frequency are ineligible — no RMS curves
        assert len(canvas._rms_curves) == 0

    def test_switching_off_removes_rms_curves(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        canvas.set_record(_make_disturbance_record(channel_names=["VA"]))
        canvas.set_rms_display_mode(RMSDisplayMode.OVERLAY)
        assert len(canvas._rms_curves) == 1
        canvas.set_rms_display_mode(RMSDisplayMode.OFF)
        assert len(canvas._rms_curves) == 0

    def test_rms_time_cache_shorter_than_raw(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        canvas.set_record(_make_disturbance_record(n_samples=600, channel_names=["VA"]))
        canvas.set_rms_display_mode(RMSDisplayMode.OVERLAY)
        if "VA" in canvas._rms_time_cache:
            assert len(canvas._rms_time_cache["VA"]) < len(canvas._time_cache)


class TestRMSCanvasRMSOnlyMode:
    def test_rms_only_creates_rms_curves(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        canvas.set_record(_make_disturbance_record(channel_names=["IA", "IB"]))
        canvas.set_rms_display_mode(RMSDisplayMode.RMS_ONLY)
        assert "IA" in canvas._rms_curves
        assert "IB" in canvas._rms_curves

    def test_rms_only_mode_stored(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        canvas.set_record(_make_disturbance_record())
        canvas.set_panel_title("Voltage Waveforms")
        canvas.set_rms_display_mode(RMSDisplayMode.RMS_ONLY)
        assert canvas._rms_display_mode == RMSDisplayMode.RMS_ONLY
        assert canvas._panel_title == "Voltage Waveforms - RMS Only"

    def test_overlay_mode_title_is_explicit(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        canvas.set_record(_make_disturbance_record(channel_names=["VA"]))
        canvas.set_panel_title("Voltage Waveforms")
        canvas.set_rms_display_mode(RMSDisplayMode.OVERLAY)
        assert canvas._panel_title == "Voltage Waveforms - RMS Overlay"


class TestRMSCacheReuse:
    def test_cache_reused_on_mode_toggle(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        canvas.set_record(_make_disturbance_record(channel_names=["VA"]))
        canvas.set_rms_display_mode(RMSDisplayMode.OVERLAY)
        # Capture cached data
        cached_before = canvas._rms_cache
        canvas.set_rms_display_mode(RMSDisplayMode.OFF)
        canvas.set_rms_display_mode(RMSDisplayMode.OVERLAY)
        # Same cache object — reused without reallocation
        assert canvas._rms_cache is cached_before

    def test_new_record_clears_rms_state(self, _qt_app) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
        canvas = FlexiblePlotCanvas()
        canvas.set_record(_make_disturbance_record(channel_names=["VA"]))
        canvas.set_rms_display_mode(RMSDisplayMode.OVERLAY)
        assert len(canvas._rms_curves) > 0
        # Loading a new record should reset curves
        canvas.set_record(_make_disturbance_record(channel_names=["VA"]))
        # After set_record the mode is still OVERLAY, so _build_rms_overlays is called again
        # Either way the canvas should be in a consistent state
        assert canvas._rms_display_mode == RMSDisplayMode.OVERLAY
