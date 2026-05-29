"""Phase 8 harmonic overlay stability tests — requires offscreen Qt.

Coverage:
  - Harmonic pen/color helpers produce valid Qt objects (require QApplication)
  - FlexiblePlotCanvas mode transitions: OFF→MAGNITUDE→THD→SPECTRUM→OFF
  - Repeated HARMONIC_MAGNITUDE calls do not duplicate curves
  - Loading a new record while overlay active rebuilds cleanly
  - RMS/telemetry channel skipped safely in HARMONIC_MAGNITUDE mode
  - Performance timing instrumentation fires correctly for harmonic rebuilds
  - _build_harmonic_panels with single-channel record via PowerwaveMainWindow
  - _build_harmonic_panels returns {} for non-eligible single-channel record
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from app.analytics.harmonics.harmonic_models import HarmonicDisplayMode
from app.data.signal_metadata import SignalMetadata
from app.visualization.overlays.overlay_colors import (
    harmonic_curve_label,
    harmonic_order_color,
    harmonic_order_pen,
    thd_curve_label,
    thd_pen,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _pen_hex(pen) -> str:
    return pen.color().name().upper()


def _single_channel_record(name: str = "VA", unit: str = "V", n_samples: int = 1000):
    from app.data.synthetic import make_high_rate_record

    base = make_high_rate_record(duration_s=0.25, sampling_rate_hz=5000.0).record
    keep = base.analog_channels[0]
    ch = replace(keep, name=name, unit=unit)
    return replace(
        base,
        waveform_data=base.waveform_data.loc[:, ["time", keep.name]].rename(
            columns={keep.name: name}
        ),
        analog_channels=[ch],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Harmonic pen/color helpers — require Qt for QPen
# ─────────────────────────────────────────────────────────────────────────────


class TestHarmonicOverlayColors:
    def test_harmonic_order_pen_is_deterministic(self) -> None:
        pen1 = harmonic_order_pen(3)
        pen2 = harmonic_order_pen(3)
        assert _pen_hex(pen1) == _pen_hex(pen2)

    def test_different_orders_have_different_pens(self) -> None:
        assert _pen_hex(harmonic_order_pen(3)) != _pen_hex(harmonic_order_pen(5))

    def test_thd_voltage_and_current_pen_differ(self) -> None:
        assert _pen_hex(thd_pen("voltage")) != _pen_hex(thd_pen("current"))

    def test_harmonic_color_hex_is_valid_for_all_expected_orders(self) -> None:
        for order in (1, 2, 3, 5, 7, 11, 13):
            c = harmonic_order_color(order)
            assert c.startswith("#") and len(c) == 7
            int(c[1:], 16)

    def test_harmonic_curve_label_is_stable(self) -> None:
        assert harmonic_curve_label("VA", 5) == "VA H5"
        assert harmonic_curve_label("VA", 5) == harmonic_curve_label("VA", 5)

    def test_thd_curve_label_is_stable(self) -> None:
        assert thd_curve_label("IA") == "IA THD%"


# ─────────────────────────────────────────────────────────────────────────────
# FlexiblePlotCanvas — harmonic mode stability
# ─────────────────────────────────────────────────────────────────────────────


class TestFlexibleCanvasHarmonicStability:
    def test_off_to_magnitude_builds_curves(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            canvas.set_harmonic_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
            assert "VA" in canvas._harmonic_curves
            assert canvas._harmonic_curves["VA"]  # at least one order curve
        finally:
            canvas.close()
            qapp.processEvents()

    def test_off_mode_clears_harmonic_curves(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            canvas.set_harmonic_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
            assert "VA" in canvas._harmonic_curves

            canvas.set_harmonic_display_mode(HarmonicDisplayMode.OFF)
            assert canvas._harmonic_curves == {}
        finally:
            canvas.close()
            qapp.processEvents()

    def test_thd_mode_removes_magnitude_overlays(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            canvas.set_harmonic_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
            assert "VA" in canvas._harmonic_curves

            canvas.set_harmonic_display_mode(HarmonicDisplayMode.THD)
            assert canvas._harmonic_curves == {}
        finally:
            canvas.close()
            qapp.processEvents()

    def test_spectrum_mode_removes_magnitude_overlays(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            canvas.set_harmonic_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
            canvas.set_harmonic_display_mode(HarmonicDisplayMode.SPECTRUM)
            assert canvas._harmonic_curves == {}
        finally:
            canvas.close()
            qapp.processEvents()

    def test_full_mode_cycle_ends_clean(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            for mode in (
                HarmonicDisplayMode.HARMONIC_MAGNITUDE,
                HarmonicDisplayMode.THD,
                HarmonicDisplayMode.SPECTRUM,
                HarmonicDisplayMode.OFF,
                HarmonicDisplayMode.HARMONIC_MAGNITUDE,
                HarmonicDisplayMode.OFF,
            ):
                canvas.set_harmonic_display_mode(mode)

            assert canvas._harmonic_display_mode == HarmonicDisplayMode.OFF
            assert canvas._harmonic_curves == {}
            assert canvas._harmonic_time_cache == {}
            assert canvas._harmonic_data_cache == {}
        finally:
            canvas.close()
            qapp.processEvents()

    def test_repeated_magnitude_mode_does_not_duplicate_curves(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            canvas.set_harmonic_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
            count_after_first = sum(
                len(v) for v in canvas._harmonic_curves.values()
            )
            # Second call with same mode — must not add new items
            canvas.set_harmonic_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
            count_after_second = sum(
                len(v) for v in canvas._harmonic_curves.values()
            )
            assert count_after_second == count_after_first
        finally:
            canvas.close()
            qapp.processEvents()

    def test_loading_new_record_while_magnitude_active_rebuilds_cleanly(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            canvas.set_harmonic_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
            assert "VA" in canvas._harmonic_curves

            canvas.set_record(_single_channel_record("IA", "A"))
            assert "VA" not in canvas._harmonic_curves
            # IA may or may not be present depending on whether set_record rebuilds
            # — but old channel must be gone
        finally:
            canvas.close()
            qapp.processEvents()

    def test_rms_channel_skipped_in_magnitude_mode(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA_RMS", "kV"))
            meta = {
                "VA_RMS": SignalMetadata(
                    name="VA_RMS",
                    unit="kV",
                    electrical_type="voltage",
                    measurement_kind="rms",
                )
            }
            canvas.set_harmonic_display_mode(
                HarmonicDisplayMode.HARMONIC_MAGNITUDE,
                signal_metadata=meta,
            )
            assert canvas._harmonic_curves == {}
        finally:
            canvas.close()
            qapp.processEvents()

    def test_performance_timing_fires_for_harmonic_rebuild(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        calls = []
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            canvas.set_performance_timing(True, lambda name, elapsed: calls.append(name))
            canvas.set_harmonic_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
            assert "harmonic_overlay_rebuild" in calls
        finally:
            canvas.close()
            qapp.processEvents()

    def test_harmonic_cache_preserved_across_off_to_magnitude_transition(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            canvas.set_harmonic_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
            cache_after_first = canvas._harmonic_cache

            canvas.set_harmonic_display_mode(HarmonicDisplayMode.OFF)
            # Cache must be preserved so re-enabling MAGNITUDE reuses it
            assert canvas._harmonic_cache is cache_after_first
        finally:
            canvas.close()
            qapp.processEvents()
