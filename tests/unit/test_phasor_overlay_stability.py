from __future__ import annotations

import os
import sys
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from app.analytics.phasors.phasor_models import PhaseLabel, PhasorDisplayMode
from app.data.signal_metadata import SignalMetadata
from app.visualization.overlays.overlay_colors import (
    OverlayTheme,
    is_waveform_phasor_candidate,
    phasor_curve_label,
    phasor_pen,
    sequence_curve_label,
    sequence_pen,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _pen_hex(pen) -> str:
    return pen.color().name().upper()


def _single_channel_record(name: str = "VA", unit: str = "V"):
    from app.data.synthetic import make_high_rate_record

    base = make_high_rate_record(duration_s=0.25, sampling_rate_hz=1000.0).record
    keep = base.analog_channels[0]
    ch = replace(keep, name=name, unit=unit)
    return replace(
        base,
        waveform_data=base.waveform_data.loc[:, ["time", keep.name]].rename(columns={keep.name: name}),
        analog_channels=[ch],
    )


class TestOverlayThemePolicy:
    def test_phasor_pen_is_deterministic_for_same_channel(self) -> None:
        first = phasor_pen("VA", PhasorDisplayMode.MAGNITUDE, signal_role="voltage")
        second = phasor_pen("VA", PhasorDisplayMode.MAGNITUDE, signal_role="voltage")
        assert _pen_hex(first) == _pen_hex(second)

    def test_angle_pen_is_related_but_distinguishable(self) -> None:
        mag = phasor_pen("VA", PhasorDisplayMode.MAGNITUDE, signal_role="voltage")
        angle = phasor_pen("VA", PhasorDisplayMode.ANGLE, signal_role="voltage")
        assert _pen_hex(mag) != _pen_hex(angle)
        assert angle.style() != mag.style()

    def test_theme_override_changes_phase_color(self) -> None:
        theme = OverlayTheme(
            voltage_phase_colors={PhaseLabel.A: "#123456", PhaseLabel.UNKNOWN: "#CCCCCC"},
        )
        pen = phasor_pen(
            "VA",
            PhasorDisplayMode.MAGNITUDE,
            theme=theme,
            signal_role="voltage",
            phase=PhaseLabel.A,
        )
        assert _pen_hex(pen) == "#123456"

    def test_sequence_pen_and_label_are_stable(self) -> None:
        assert _pen_hex(sequence_pen("V1")) == _pen_hex(sequence_pen("positive"))
        assert sequence_curve_label("V", "negative") == "V2 Magnitude"
        assert phasor_curve_label("IA", PhasorDisplayMode.ANGLE) == "IA Angle"

    def test_rms_metadata_is_not_waveform_candidate(self) -> None:
        meta = SignalMetadata(
            name="VA_RMS",
            unit="kV",
            electrical_type="voltage",
            measurement_kind="rms",
        )
        assert is_waveform_phasor_candidate(meta) is False


class TestFlexibleCanvasPhasorStability:
    def test_repeated_mode_switching_does_not_duplicate_curves(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            canvas.set_phasor_display_mode(PhasorDisplayMode.MAGNITUDE)
            assert set(canvas._phasor_curves) == {"VA"}
            first_curve = canvas._phasor_curves["VA"]

            canvas.set_phasor_display_mode(PhasorDisplayMode.MAGNITUDE)
            assert canvas._phasor_curves["VA"] is first_curve

            for mode in (
                PhasorDisplayMode.OFF,
                PhasorDisplayMode.ANGLE,
                PhasorDisplayMode.OFF,
                PhasorDisplayMode.MAGNITUDE,
                PhasorDisplayMode.OFF,
            ):
                canvas.set_phasor_display_mode(mode)

            assert canvas._phasor_curves == {}
            assert canvas._phasor_time_cache == {}
            assert canvas._phasor_data_cache == {}
        finally:
            canvas.close()
            qapp.processEvents()

    def test_loading_new_record_while_overlay_active_rebuilds_cleanly(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            canvas.set_phasor_display_mode(PhasorDisplayMode.MAGNITUDE)
            assert set(canvas._phasor_curves) == {"VA"}

            canvas.set_record(_single_channel_record("IA", "A"))
            assert set(canvas._phasor_curves) == {"IA"}
            assert "VA" not in canvas._phasor_data_cache
        finally:
            canvas.close()
            qapp.processEvents()

    def test_rms_like_voltage_channel_is_skipped_safely(self, qapp) -> None:
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
            canvas.set_phasor_display_mode(
                PhasorDisplayMode.MAGNITUDE,
                signal_metadata=meta,
            )
            assert canvas._phasor_curves == {}
        finally:
            canvas.close()
            qapp.processEvents()

    def test_performance_timing_is_optional(self, qapp) -> None:
        from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

        calls = []
        canvas = FlexiblePlotCanvas()
        try:
            canvas.set_record(_single_channel_record("VA", "V"))
            canvas.set_performance_timing(False, lambda *args: calls.append(args))
            canvas.set_phasor_display_mode(PhasorDisplayMode.MAGNITUDE)
            assert calls == []

            canvas.set_phasor_display_mode(PhasorDisplayMode.OFF)
            canvas.set_performance_timing(True, lambda *args: calls.append(args))
            canvas.set_phasor_display_mode(PhasorDisplayMode.MAGNITUDE)
            assert any(name == "phasor_overlay_rebuild" for name, _elapsed in calls)
        finally:
            canvas.close()
            qapp.processEvents()


class TestSequencePanelStability:
    def test_missing_three_phase_groups_returns_no_panels(self, qapp) -> None:
        from app.ui.main_window.main_window import PowerwaveMainWindow

        win = PowerwaveMainWindow()
        try:
            panels = win._build_sequence_panels(
                _single_channel_record("VA", "V"),
                signal_metadata=None,
                canvas_factory=lambda: None,
            )
            assert panels == {}
        finally:
            win.close()
            qapp.processEvents()
