"""Unit tests for Phase 8 harmonic rendering integration.

Coverage:
  - HarmonicCurveOverlay lifecycle (attach/detach/dispose, no duplicate curves)
  - Harmonic color/pen helpers (deterministic output, valid hex)
  - _make_harmonic_record builds a valid synthetic DisturbanceRecord
  - _apply_harmonic_display_mode routing to waveform and panel canvases
  - _build_harmonic_panels with no eligible channels returns {}
  - _build_harmonic_panels with voltage-only channels builds correct panels
  - Panel keys filtered by _HARMONIC_PANEL_KEYS in routing

No QApplication required; PyQtGraph plot items are mocked or avoided.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.analytics.harmonics.harmonic_models import HarmonicDisplayMode
from app.analytics.harmonics.harmonic_registry import HarmonicRegistry


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _utc() -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_disturbance_record(channel_names: list[str], n_samples: int = 1000):
    from app.models import AnalogChannel, DisturbanceRecord
    from app.models.metadata import RecordingMetadata
    from app.models.timing import SamplingInformation, TimingInformation

    fs = 5000.0
    t = np.arange(n_samples) / fs
    df = pd.DataFrame({"time": t})
    channels = []
    for i, name in enumerate(channel_names):
        amp = 100.0 * (i + 1)
        phase_deg = -120.0 * i
        df[name] = amp * np.sin(2 * np.pi * 50.0 * t + np.radians(phase_deg))
        channels.append(AnalogChannel(name=name, unit="V", index=i))

    start = _utc()
    trigger = start + timedelta(seconds=t[n_samples // 2])
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="Test",
            recorder_name="TestRecorder",
            source_file="test.cfg",
            provider_type="comtrade",
            nominal_frequency=50.0,
        ),
        waveform_data=df,
        analog_channels=channels,
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[fs],
            samples_per_rate=[n_samples],
        ),
        timing_info=TimingInformation(
            start_time=start,
            trigger_time=trigger,
        ),
        disturbance_info=None,
    )


class _FakeCanvas:
    """Lightweight fake for FlexiblePlotCanvas — no Qt required."""

    def __init__(self) -> None:
        self._harmonic_display_mode = HarmonicDisplayMode.OFF
        self._visible = True
        self.harmonic_mode_calls: list[HarmonicDisplayMode] = []
        self.record_set: Any = None
        self._title: str = ""

    def set_harmonic_display_mode(self, mode: HarmonicDisplayMode, **kwargs: Any) -> None:
        self._harmonic_display_mode = mode
        self.harmonic_mode_calls.append(mode)

    def set_record(self, record: Any) -> None:
        self.record_set = record

    def set_panel_title(self, title: str) -> None:
        self._title = title

    def setVisible(self, v: bool) -> None:
        self._visible = v

    def isVisible(self) -> bool:
        return self._visible


# ─────────────────────────────────────────────────────────────────────────────
# HarmonicCurveOverlay — lifecycle (no Qt)
# ─────────────────────────────────────────────────────────────────────────────


class TestHarmonicCurveOverlayInit:
    def test_starts_not_disposed(self) -> None:
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay
        o = HarmonicCurveOverlay()
        assert not o.disposed

    def test_starts_not_attached(self) -> None:
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay
        o = HarmonicCurveOverlay()
        assert o.canvas is None

    def test_channel_order_pairs_empty_before_attach(self) -> None:
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay
        o = HarmonicCurveOverlay()
        assert o.channel_order_pairs() == []


class TestHarmonicCurveOverlayAttach:
    def test_attach_extracts_primary_plot(self) -> None:
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay
        fake_canvas = MagicMock()
        fake_canvas._primary_plot = MagicMock()
        o = HarmonicCurveOverlay()
        o.attach(fake_canvas)
        assert o._plot_item is fake_canvas._primary_plot
        assert o.canvas is fake_canvas

    def test_attach_fallback_to_addItem(self) -> None:
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay
        fake_plot = MagicMock(spec=["addItem"])
        o = HarmonicCurveOverlay()
        o.attach(fake_plot)
        assert o._plot_item is fake_plot

    def test_attach_no_plot_interface_sets_none(self) -> None:
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay
        o = HarmonicCurveOverlay()
        o.attach(object())
        assert o._plot_item is None

    def test_detach_clears_plot_item(self) -> None:
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay
        fake_canvas = MagicMock()
        fake_canvas._primary_plot = MagicMock()
        o = HarmonicCurveOverlay()
        o.attach(fake_canvas)
        o.detach()
        assert o._plot_item is None
        assert o.canvas is None

    def test_dispose_marks_disposed(self) -> None:
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay
        o = HarmonicCurveOverlay()
        o.dispose()
        assert o.disposed

    def test_attach_after_dispose_raises(self) -> None:
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay
        o = HarmonicCurveOverlay()
        o.dispose()
        with pytest.raises(RuntimeError, match="disposed"):
            o.attach(MagicMock())


class TestHarmonicCurveOverlayUpdateChannel:
    def test_update_channel_before_attach_is_noop(self) -> None:
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay
        o = HarmonicCurveOverlay()
        t = np.linspace(0, 0.1, 50)
        mag = np.ones(50) * 5.0
        o.update_channel("VA", t, mag, order=3)
        assert o.channel_order_pairs() == []

    def test_update_channel_after_attach_creates_pair(self) -> None:
        import pyqtgraph as pg
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay

        fake_canvas = MagicMock()
        fake_plot = MagicMock()
        fake_canvas._primary_plot = fake_plot
        fake_curve = MagicMock(spec=pg.PlotDataItem)

        with patch("app.visualization.overlays.curve_store.pg.PlotDataItem", return_value=fake_curve):
            o = HarmonicCurveOverlay()
            o.attach(fake_canvas)
            t = np.linspace(0, 0.1, 50)
            mag = np.ones(50) * 5.0
            o.update_channel("VA", t, mag, order=3)
            assert ("VA", 3) in o.channel_order_pairs()
            fake_curve.setData.assert_called_once()

    def test_update_channel_same_pair_no_duplicate(self) -> None:
        import pyqtgraph as pg
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay

        fake_canvas = MagicMock()
        fake_plot = MagicMock()
        fake_canvas._primary_plot = fake_plot
        fake_curve = MagicMock(spec=pg.PlotDataItem)

        with patch("app.visualization.overlays.curve_store.pg.PlotDataItem", return_value=fake_curve):
            o = HarmonicCurveOverlay()
            o.attach(fake_canvas)
            t = np.linspace(0, 0.1, 50)
            mag = np.ones(50) * 5.0
            o.update_channel("VA", t, mag, order=5)
            o.update_channel("VA", t, mag, order=5)
            assert o.channel_order_pairs().count(("VA", 5)) == 1

    def test_remove_channel_clears_all_orders(self) -> None:
        import pyqtgraph as pg
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay

        fake_canvas = MagicMock()
        fake_plot = MagicMock()
        fake_canvas._primary_plot = fake_plot
        fake_curve = MagicMock(spec=pg.PlotDataItem)
        fake_curve.getViewBox = MagicMock(return_value=None)

        with patch("app.visualization.overlays.curve_store.pg.PlotDataItem", return_value=fake_curve):
            o = HarmonicCurveOverlay()
            o.attach(fake_canvas)
            t = np.linspace(0, 0.1, 50)
            mag = np.ones(50) * 5.0
            for order in (3, 5, 7):
                o.update_channel("VA", t, mag, order=order)
            assert len(o.channel_order_pairs()) == 3
            o.remove_channel("VA")
            assert o.channel_order_pairs() == []

    def test_remove_order_removes_only_one(self) -> None:
        import pyqtgraph as pg
        from app.visualization.overlays.harmonic_overlay import HarmonicCurveOverlay

        fake_canvas = MagicMock()
        fake_plot = MagicMock()
        fake_canvas._primary_plot = fake_plot
        fake_curve = MagicMock(spec=pg.PlotDataItem)
        fake_curve.getViewBox = MagicMock(return_value=None)

        with patch("app.visualization.overlays.curve_store.pg.PlotDataItem", return_value=fake_curve):
            o = HarmonicCurveOverlay()
            o.attach(fake_canvas)
            t = np.linspace(0, 0.1, 50)
            mag = np.ones(50) * 5.0
            o.update_channel("VA", t, mag, order=3)
            o.update_channel("VA", t, mag, order=5)
            o.remove_order("VA", 3)
            pairs = o.channel_order_pairs()
            assert ("VA", 3) not in pairs
            assert ("VA", 5) in pairs


# ─────────────────────────────────────────────────────────────────────────────
# Harmonic color/pen helpers — pure logic
# ─────────────────────────────────────────────────────────────────────────────


class TestHarmonicColorHelpers:
    def test_harmonic_order_color_returns_valid_hex(self) -> None:
        from app.visualization.overlays.overlay_colors import harmonic_order_color
        for order in (1, 3, 5, 7, 11, 13):
            c = harmonic_order_color(order)
            assert c.startswith("#"), f"order {order}: not a hex string"
            assert len(c) == 7, f"order {order}: wrong length"
            int(c[1:], 16)  # must be valid hex

    def test_harmonic_order_color_is_deterministic(self) -> None:
        from app.visualization.overlays.overlay_colors import harmonic_order_color
        assert harmonic_order_color(3) == harmonic_order_color(3)
        assert harmonic_order_color(5) != harmonic_order_color(7)

    def test_harmonic_order_color_unknown_order_returns_fallback(self) -> None:
        from app.visualization.overlays.overlay_colors import harmonic_order_color
        c = harmonic_order_color(99)
        assert c.startswith("#")
        assert len(c) == 7

    def test_harmonic_order_color_known_values(self) -> None:
        from app.visualization.overlays.overlay_colors import harmonic_order_color
        assert harmonic_order_color(3) == "#FF6600"
        assert harmonic_order_color(5) == "#FF00CC"
        assert harmonic_order_color(7) == "#00CCFF"

    def test_harmonic_curve_label_format(self) -> None:
        from app.visualization.overlays.overlay_colors import harmonic_curve_label
        assert harmonic_curve_label("VA", 3) == "VA H3"
        assert harmonic_curve_label("IA", 13) == "IA H13"

    def test_thd_curve_label_format(self) -> None:
        from app.visualization.overlays.overlay_colors import thd_curve_label
        assert thd_curve_label("VA") == "VA THD%"

    def test_thd_pen_voltage_and_current_differ(self) -> None:
        from app.visualization.overlays.overlay_colors import thd_pen
        pv = thd_pen("voltage")
        pi = thd_pen("current")
        assert pv.color().name() != pi.color().name()

    def test_thd_pen_default_is_voltage(self) -> None:
        from app.visualization.overlays.overlay_colors import thd_pen
        pd_default = thd_pen()
        pv = thd_pen("voltage")
        assert pd_default.color().name() == pv.color().name()

    def test_harmonic_order_pen_returns_pen_object(self) -> None:
        import pyqtgraph as pg
        from app.visualization.overlays.overlay_colors import harmonic_order_pen
        for order in (3, 5, 7):
            pen = harmonic_order_pen(order)
            assert isinstance(pen, pg.Qt.QtGui.QPen)


# ─────────────────────────────────────────────────────────────────────────────
# _make_harmonic_record — pure logic
# ─────────────────────────────────────────────────────────────────────────────


class TestMakeHarmonicRecord:
    def test_builds_record_with_correct_channels(self) -> None:
        from app.ui.main_window.main_window import _make_harmonic_record

        src = _make_disturbance_record(["VA", "VB", "VC"])
        t = np.linspace(0, 0.1, 100)
        data = {"VA": np.ones(100) * 3.5, "VB": np.ones(100) * 2.1}
        result = _make_harmonic_record(src, t, data, "%")

        names = [ch.name for ch in result.analog_channels]
        assert "VA" in names
        assert "VB" in names
        assert len(result.analog_channels) == 2

    def test_unit_is_set_on_all_channels(self) -> None:
        from app.ui.main_window.main_window import _make_harmonic_record

        src = _make_disturbance_record(["VA"])
        t = np.linspace(0, 0.1, 100)
        data = {"H3": np.ones(100), "H5": np.ones(100)}
        result = _make_harmonic_record(src, t, data, "V RMS")

        for ch in result.analog_channels:
            assert ch.unit == "V RMS"

    def test_time_column_present(self) -> None:
        from app.ui.main_window.main_window import _make_harmonic_record

        src = _make_disturbance_record(["VA"])
        t = np.linspace(0, 0.1, 100)
        data = {"VA": np.ones(100)}
        result = _make_harmonic_record(src, t, data, "%")

        assert "time" in result.waveform_data.columns

    def test_shares_timing_and_metadata_with_source(self) -> None:
        from app.ui.main_window.main_window import _make_harmonic_record

        src = _make_disturbance_record(["VA"])
        t = np.linspace(0, 0.1, 100)
        data = {"VA": np.ones(100)}
        result = _make_harmonic_record(src, t, data, "%")

        assert result.timing_info is src.timing_info
        assert result.metadata is src.metadata

    def test_no_digital_channels(self) -> None:
        from app.ui.main_window.main_window import _make_harmonic_record

        src = _make_disturbance_record(["VA"])
        t = np.linspace(0, 0.1, 100)
        data = {"VA": np.ones(100)}
        result = _make_harmonic_record(src, t, data, "%")

        assert result.digital_channels == []

    def test_data_values_match_input(self) -> None:
        from app.ui.main_window.main_window import _make_harmonic_record

        src = _make_disturbance_record(["VA"])
        t = np.linspace(0, 0.1, 50)
        expected = np.arange(50, dtype=float) * 0.1
        data = {"VA": expected}
        result = _make_harmonic_record(src, t, data, "%")

        np.testing.assert_array_almost_equal(
            result.waveform_data["VA"].to_numpy(), expected
        )


# ─────────────────────────────────────────────────────────────────────────────
# _apply_harmonic_display_mode routing — fake canvases, no Qt
# ─────────────────────────────────────────────────────────────────────────────


_HARMONIC_PANEL_KEYS = frozenset({
    "thd_voltage",
    "thd_current",
    "harmonic_spectrum_voltage",
    "harmonic_spectrum_current",
})


class _FakeMainWindow:
    """Minimal stub of PowerwaveMainWindow for harmonic routing tests."""

    def __init__(self, panel_canvases: dict, main_canvas=None) -> None:
        self._panel_canvases = panel_canvases
        self._canvas = main_canvas or _FakeCanvas()
        self._harmonic_registry = HarmonicRegistry()

    def _qt_widget_alive(self, w: Any) -> bool:
        return w is not None

    def _apply_harmonic_display_mode(self) -> None:
        """Inline copy of the real implementation for testing without Qt."""
        mode = self._harmonic_registry.display_mode

        if self._panel_canvases:
            waveform_canvases = [
                c for k, c in self._panel_canvases.items()
                if k not in _HARMONIC_PANEL_KEYS and self._qt_widget_alive(c)
            ]
        elif self._qt_widget_alive(self._canvas):
            waveform_canvases = [self._canvas]
        else:
            waveform_canvases = []

        for canvas in waveform_canvases:
            canvas.set_harmonic_display_mode(mode)

        thd_visible = (mode == HarmonicDisplayMode.THD)
        for key in ("thd_voltage", "thd_current"):
            canvas = (self._panel_canvases or {}).get(key)
            if canvas is not None and self._qt_widget_alive(canvas):
                canvas.setVisible(thd_visible)

        spectrum_visible = (mode == HarmonicDisplayMode.SPECTRUM)
        for key in ("harmonic_spectrum_voltage", "harmonic_spectrum_current"):
            canvas = (self._panel_canvases or {}).get(key)
            if canvas is not None and self._qt_widget_alive(canvas):
                canvas.setVisible(spectrum_visible)


class TestApplyHarmonicDisplayModeRouting:
    def test_off_mode_calls_off_on_waveform_canvases(self) -> None:
        volt = _FakeCanvas()
        curr = _FakeCanvas()
        win = _FakeMainWindow({"voltage_raw": volt, "current_raw": curr})
        win._harmonic_registry.set_display_mode(HarmonicDisplayMode.OFF)
        win._apply_harmonic_display_mode()

        assert volt.harmonic_mode_calls[-1] == HarmonicDisplayMode.OFF
        assert curr.harmonic_mode_calls[-1] == HarmonicDisplayMode.OFF

    def test_magnitude_mode_calls_magnitude_on_waveform_canvases(self) -> None:
        volt = _FakeCanvas()
        win = _FakeMainWindow({"voltage_raw": volt})
        win._harmonic_registry.set_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
        win._apply_harmonic_display_mode()

        assert volt.harmonic_mode_calls[-1] == HarmonicDisplayMode.HARMONIC_MAGNITUDE

    def test_thd_mode_shows_thd_panels(self) -> None:
        volt = _FakeCanvas()
        thd_v = _FakeCanvas()
        thd_i = _FakeCanvas()
        thd_v.setVisible(False)
        thd_i.setVisible(False)
        win = _FakeMainWindow({
            "voltage_raw": volt,
            "thd_voltage": thd_v,
            "thd_current": thd_i,
        })
        win._harmonic_registry.set_display_mode(HarmonicDisplayMode.THD)
        win._apply_harmonic_display_mode()

        assert thd_v.isVisible()
        assert thd_i.isVisible()

    def test_thd_mode_hides_spectrum_panels(self) -> None:
        spec_v = _FakeCanvas()
        spec_v.setVisible(True)
        win = _FakeMainWindow({
            "voltage_raw": _FakeCanvas(),
            "harmonic_spectrum_voltage": spec_v,
        })
        win._harmonic_registry.set_display_mode(HarmonicDisplayMode.THD)
        win._apply_harmonic_display_mode()

        assert not spec_v.isVisible()

    def test_spectrum_mode_shows_spectrum_panels(self) -> None:
        spec_v = _FakeCanvas()
        spec_i = _FakeCanvas()
        spec_v.setVisible(False)
        spec_i.setVisible(False)
        win = _FakeMainWindow({
            "voltage_raw": _FakeCanvas(),
            "harmonic_spectrum_voltage": spec_v,
            "harmonic_spectrum_current": spec_i,
        })
        win._harmonic_registry.set_display_mode(HarmonicDisplayMode.SPECTRUM)
        win._apply_harmonic_display_mode()

        assert spec_v.isVisible()
        assert spec_i.isVisible()

    def test_spectrum_mode_hides_thd_panels(self) -> None:
        thd_v = _FakeCanvas()
        thd_v.setVisible(True)
        win = _FakeMainWindow({
            "voltage_raw": _FakeCanvas(),
            "thd_voltage": thd_v,
        })
        win._harmonic_registry.set_display_mode(HarmonicDisplayMode.SPECTRUM)
        win._apply_harmonic_display_mode()

        assert not thd_v.isVisible()

    def test_off_mode_hides_all_harmonic_panels(self) -> None:
        thd_v = _FakeCanvas()
        thd_i = _FakeCanvas()
        spec_v = _FakeCanvas()
        for c in (thd_v, thd_i, spec_v):
            c.setVisible(True)
        win = _FakeMainWindow({
            "voltage_raw": _FakeCanvas(),
            "thd_voltage": thd_v,
            "thd_current": thd_i,
            "harmonic_spectrum_voltage": spec_v,
        })
        win._harmonic_registry.set_display_mode(HarmonicDisplayMode.OFF)
        win._apply_harmonic_display_mode()

        assert not thd_v.isVisible()
        assert not thd_i.isVisible()
        assert not spec_v.isVisible()

    def test_harmonic_panels_do_not_receive_set_harmonic_mode(self) -> None:
        thd_v = _FakeCanvas()
        volt = _FakeCanvas()
        win = _FakeMainWindow({
            "voltage_raw": volt,
            "thd_voltage": thd_v,
        })
        win._harmonic_registry.set_display_mode(HarmonicDisplayMode.THD)
        win._apply_harmonic_display_mode()

        assert thd_v.harmonic_mode_calls == []
        assert volt.harmonic_mode_calls[-1] == HarmonicDisplayMode.THD

    def test_no_panel_canvases_uses_main_canvas(self) -> None:
        main = _FakeCanvas()
        win = _FakeMainWindow({}, main_canvas=main)
        win._harmonic_registry.set_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
        win._apply_harmonic_display_mode()

        assert main.harmonic_mode_calls[-1] == HarmonicDisplayMode.HARMONIC_MAGNITUDE

    def test_multi_source_harmonic_panel_keys_excluded_from_waveform_list(self) -> None:
        thd_v = _FakeCanvas()
        volt = _FakeCanvas()
        win = _FakeMainWindow({
            "SRC/voltage_raw": volt,
            "thd_voltage": thd_v,
        })
        win._harmonic_registry.set_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
        win._apply_harmonic_display_mode()

        assert thd_v.harmonic_mode_calls == []
        assert volt.harmonic_mode_calls[-1] == HarmonicDisplayMode.HARMONIC_MAGNITUDE


# ─────────────────────────────────────────────────────────────────────────────
# _build_harmonic_panels — standalone logic tests (no Qt)
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildHarmonicPanels:
    def _run_build(self, record, signal_metadata=None):
        from app.ui.main_window.main_window import _make_harmonic_record
        from app.analytics.harmonics import (
            HarmonicCache,
            HarmonicConfig,
            HarmonicRegistry,
            classify_harmonic_role,
            compute_harmonic_window_samples,
            extract_harmonics,
        )
        from app.analytics.harmonics.harmonic_metrics import compute_thd_array
        from app.analytics.harmonics.harmonic_models import HarmonicChannelRole

        if record is None:
            return {}

        registry = HarmonicRegistry()
        config = registry.config

        try:
            time_col = record.waveform_data["time"].to_numpy(dtype=np.float64)
        except Exception:
            return {}

        rates = [r for r in record.sampling_info.sampling_rates if r > 0]
        sample_rate_hz = float(rates[0]) if rates else 0.0
        if sample_rate_hz <= 0:
            return {}

        window = compute_harmonic_window_samples(sample_rate_hz, config)
        overlap_clamped = max(0.0, min(config.overlap, 0.999))
        hop = max(1, round(window * (1.0 - overlap_clamped)))

        cache = HarmonicCache()
        thd_voltage: dict = {}
        thd_current: dict = {}
        spec_voltage: dict = {}
        spec_current: dict = {}
        harmonic_time = None
        _SPECTRUM_ORDERS = [3, 5, 7, 11, 13]

        for ch in record.analog_channels:
            name = ch.name
            meta = (signal_metadata or {}).get(name)
            role = classify_harmonic_role(name, ch.unit, meta).role
            if role == HarmonicChannelRole.UNKNOWN:
                continue

            try:
                raw_data = record.waveform_data[name].to_numpy(dtype=np.float64)
            except Exception:
                continue

            cached = cache.get(name, window, hop, config.nominal_hz, config.max_order)
            if cached is not None:
                h_result = cached
            else:
                try:
                    h_result = extract_harmonics(raw_data, sample_rate_hz, config, time=time_col)
                except Exception:
                    continue
                cache.put(name, window, hop, config.nominal_hz, config.max_order, h_result)

            if h_result.n_windows == 0:
                continue

            harmonic_time = h_result.harmonic_time
            thd_arr = compute_thd_array(h_result.magnitudes) * 100.0

            if role == HarmonicChannelRole.VOLTAGE_HARMONIC:
                thd_voltage[name] = thd_arr
                if not spec_voltage:
                    for order in _SPECTRUM_ORDERS:
                        mag = h_result.get_magnitude(order)
                        if mag is not None:
                            spec_voltage[f"H{order}"] = mag
            else:
                thd_current[name] = thd_arr
                if not spec_current:
                    for order in _SPECTRUM_ORDERS:
                        mag = h_result.get_magnitude(order)
                        if mag is not None:
                            spec_current[f"H{order}"] = mag

        if harmonic_time is None:
            return {}

        result = {}
        for panel_key, data_dict, unit, title in [
            ("thd_voltage",               thd_voltage,   "%",     "THD — Voltage (%)"),
            ("thd_current",               thd_current,   "%",     "THD — Current (%)"),
            ("harmonic_spectrum_voltage",  spec_voltage, "V RMS", "Harmonic Spectrum — Voltage"),
            ("harmonic_spectrum_current",  spec_current, "A RMS", "Harmonic Spectrum — Current"),
        ]:
            if not data_dict:
                continue
            syn_record = _make_harmonic_record(record, harmonic_time, data_dict, unit)
            canvas = _FakeCanvas()
            canvas.set_record(syn_record)
            canvas.set_panel_title(title)
            canvas.setVisible(False)
            result[panel_key] = canvas

        return result

    def test_returns_empty_for_unknown_channels(self) -> None:
        from app.models import AnalogChannel, DisturbanceRecord
        from app.models.metadata import RecordingMetadata
        from app.models.timing import SamplingInformation, TimingInformation

        fs = 5000.0
        n = 1000
        t = np.arange(n) / fs
        df = pd.DataFrame({"time": t, "Freq": np.ones(n) * 50.0, "MW": np.ones(n)})
        src = DisturbanceRecord(
            metadata=RecordingMetadata(
                station_name="T", recorder_name="R", source_file="x.cfg",
                provider_type="comtrade", nominal_frequency=50.0
            ),
            waveform_data=df,
            analog_channels=[
                AnalogChannel(name="Freq", unit="Hz", index=0),
                AnalogChannel(name="MW", unit="MW", index=1),
            ],
            digital_channels=[],
            sampling_info=SamplingInformation(sampling_rates=[fs], samples_per_rate=[n]),
            timing_info=TimingInformation(
                start_time=_utc(), trigger_time=_utc() + timedelta(seconds=0.1)
            ),
            disturbance_info=None,
        )
        result = self._run_build(src)
        assert result == {}

    def test_voltage_only_record_builds_thd_and_spectrum_voltage(self) -> None:
        record = _make_disturbance_record(["VA"])
        result = self._run_build(record)
        assert "thd_voltage" in result
        assert "harmonic_spectrum_voltage" in result
        assert "thd_current" not in result
        assert "harmonic_spectrum_current" not in result

    def test_thd_voltage_panel_starts_hidden(self) -> None:
        record = _make_disturbance_record(["VA"])
        result = self._run_build(record)
        assert not result["thd_voltage"].isVisible()

    def test_thd_voltage_panel_has_record_set(self) -> None:
        record = _make_disturbance_record(["VA"])
        result = self._run_build(record)
        canvas = result["thd_voltage"]
        assert canvas.record_set is not None

    def test_thd_voltage_channel_names_match_source(self) -> None:
        record = _make_disturbance_record(["VA", "VB"])
        result = self._run_build(record)
        canvas = result["thd_voltage"]
        names = [ch.name for ch in canvas.record_set.analog_channels]
        assert "VA" in names
        assert "VB" in names

    def test_spectrum_voltage_panel_has_harmonic_order_channels(self) -> None:
        record = _make_disturbance_record(["VA"])
        result = self._run_build(record)
        canvas = result["harmonic_spectrum_voltage"]
        names = [ch.name for ch in canvas.record_set.analog_channels]
        assert any(n.startswith("H") for n in names)

    def test_current_channels_build_thd_current(self) -> None:
        from app.models import AnalogChannel, DisturbanceRecord
        from app.models.metadata import RecordingMetadata
        from app.models.timing import SamplingInformation, TimingInformation

        fs = 5000.0
        n = 1000
        t = np.arange(n) / fs
        df = pd.DataFrame({"time": t, "IA": 10.0 * np.sin(2 * np.pi * 50.0 * t)})
        src = DisturbanceRecord(
            metadata=RecordingMetadata(
                station_name="T", recorder_name="R", source_file="x.cfg",
                provider_type="comtrade", nominal_frequency=50.0
            ),
            waveform_data=df,
            analog_channels=[AnalogChannel(name="IA", unit="A", index=0)],
            digital_channels=[],
            sampling_info=SamplingInformation(sampling_rates=[fs], samples_per_rate=[n]),
            timing_info=TimingInformation(
                start_time=_utc(), trigger_time=_utc() + timedelta(seconds=0.1)
            ),
            disturbance_info=None,
        )
        result = self._run_build(src)
        assert "thd_current" in result
        assert "harmonic_spectrum_current" in result

    def test_mixed_channels_build_both_sets(self) -> None:
        from app.models import AnalogChannel, DisturbanceRecord
        from app.models.metadata import RecordingMetadata
        from app.models.timing import SamplingInformation, TimingInformation

        fs = 5000.0
        n = 1000
        t = np.arange(n) / fs
        df = pd.DataFrame({
            "time": t,
            "VA": 100.0 * np.sin(2 * np.pi * 50.0 * t),
            "IA": 10.0 * np.sin(2 * np.pi * 50.0 * t),
        })
        src = DisturbanceRecord(
            metadata=RecordingMetadata(
                station_name="T", recorder_name="R", source_file="x.cfg",
                provider_type="comtrade", nominal_frequency=50.0
            ),
            waveform_data=df,
            analog_channels=[
                AnalogChannel(name="VA", unit="V", index=0),
                AnalogChannel(name="IA", unit="A", index=1),
            ],
            digital_channels=[],
            sampling_info=SamplingInformation(sampling_rates=[fs], samples_per_rate=[n]),
            timing_info=TimingInformation(
                start_time=_utc(), trigger_time=_utc() + timedelta(seconds=0.1)
            ),
            disturbance_info=None,
        )
        result = self._run_build(src)
        assert "thd_voltage" in result
        assert "thd_current" in result
        assert "harmonic_spectrum_voltage" in result
        assert "harmonic_spectrum_current" in result
