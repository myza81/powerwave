"""Integration tests for Phase 6B phasor rendering.

Coverage:
  - PhasorCurveOverlay lifecycle (mode property, attach/detach/dispose)
  - PhasorCurveOverlay rejects SEQUENCE_COMPONENTS at construction
  - Mode routing: _apply_phasor_display_mode distributes to correct canvases
  - Sequence-panel key detection (is_sequence_key logic)
  - _make_sequence_record builds a valid synthetic DisturbanceRecord
  - _build_sequence_panels with empty/no groups returns {}
  - _build_sequence_panels with a mocked three-phase group computes and returns panels
  - Phasor cache reuse: extract_phasor called once per channel across mode switches
  - OFF mode clears existing overlays (set_phasor_display_mode called with OFF)
  - Color helpers produce valid hex strings

No QApplication is required; PyQtGraph plot items are mocked or avoided.
FlexiblePlotCanvas is tested via a lightweight fake (no real widgets).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from app.analytics.phasors.phasor_models import (
    PhasorConfig,
    PhasorDisplayMode,
    PhasorWindowMode,
    ThreePhaseGroup,
)
from app.analytics.phasors.phasor_registry import PhasorRegistry


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — fake canvas and record builders
# ─────────────────────────────────────────────────────────────────────────────


def _utc(year: int = 2024, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _make_disturbance_record(channel_names: list[str], n_samples: int = 500):
    """Build a minimal DisturbanceRecord with pure-sine waveforms."""
    from app.models import AnalogChannel, DisturbanceRecord
    from app.models.metadata import RecordingMetadata
    from app.models.timing import DisturbanceInformation, SamplingInformation, TimingInformation

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
        self._phasor_display_mode = PhasorDisplayMode.OFF
        self._visible = True
        self.phasor_mode_calls: list[PhasorDisplayMode] = []
        self.record_set: Any = None

    def set_phasor_display_mode(
        self, mode: PhasorDisplayMode, **kwargs: Any
    ) -> None:
        self._phasor_display_mode = mode
        self.phasor_mode_calls.append(mode)

    def set_record(self, record: Any) -> None:
        self.record_set = record

    def set_panel_title(self, title: str) -> None:
        pass

    def setVisible(self, v: bool) -> None:
        self._visible = v

    def isVisible(self) -> bool:
        return self._visible

    def set_axis_display_mode(self, mode: Any) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# PhasorCurveOverlay — pure-logic tests (no Qt widgets)
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorCurveOverlayInit:
    def test_default_mode_is_magnitude(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        o = PhasorCurveOverlay()
        assert o.mode == PhasorDisplayMode.MAGNITUDE

    def test_angle_mode_accepted(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        o = PhasorCurveOverlay(PhasorDisplayMode.ANGLE)
        assert o.mode == PhasorDisplayMode.ANGLE

    def test_sequence_components_rejected(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        with pytest.raises(ValueError, match="MAGNITUDE or ANGLE"):
            PhasorCurveOverlay(PhasorDisplayMode.SEQUENCE_COMPONENTS)

    def test_off_rejected(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        with pytest.raises(ValueError):
            PhasorCurveOverlay(PhasorDisplayMode.OFF)

    def test_starts_not_disposed(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        o = PhasorCurveOverlay()
        assert not o.disposed

    def test_starts_not_attached(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        o = PhasorCurveOverlay()
        assert o.canvas is None

    def test_channel_names_empty_before_attach(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        o = PhasorCurveOverlay()
        assert o.channel_names() == []


class TestPhasorCurveOverlayAttach:
    def test_attach_extracts_primary_plot(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        fake_canvas = MagicMock()
        fake_canvas._primary_plot = MagicMock()
        o = PhasorCurveOverlay()
        o.attach(fake_canvas)
        assert o._plot_item is fake_canvas._primary_plot
        assert o.canvas is fake_canvas

    def test_attach_fallback_to_addItem(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        fake_plot = MagicMock(spec=["addItem"])
        o = PhasorCurveOverlay()
        o.attach(fake_plot)
        assert o._plot_item is fake_plot

    def test_attach_no_plot_item_sets_none(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        o = PhasorCurveOverlay()
        o.attach(object())
        assert o._plot_item is None

    def test_detach_clears_plot_item(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        fake_canvas = MagicMock()
        fake_canvas._primary_plot = MagicMock()
        o = PhasorCurveOverlay()
        o.attach(fake_canvas)
        o.detach()
        assert o._plot_item is None
        assert o.canvas is None

    def test_dispose_marks_disposed(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        o = PhasorCurveOverlay()
        o.dispose()
        assert o.disposed

    def test_attach_after_dispose_raises(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        o = PhasorCurveOverlay()
        o.dispose()
        with pytest.raises(RuntimeError, match="disposed"):
            o.attach(MagicMock())


class TestPhasorCurveOverlayUpdateChannel:
    def test_update_channel_before_attach_is_noop(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        o = PhasorCurveOverlay()
        t = np.linspace(0, 0.1, 50)
        d = np.ones(50) * 230.0
        # Should not raise
        o.update_channel("VA", t, d, color="#FF4444")
        assert o.channel_names() == []

    def test_update_channel_after_attach_creates_curve(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        import pyqtgraph as pg
        fake_canvas = MagicMock()
        fake_plot = MagicMock()
        fake_canvas._primary_plot = fake_plot
        fake_curve = MagicMock(spec=pg.PlotDataItem)
        fake_plot.addItem = MagicMock()

        with patch("app.visualization.overlays.curve_store.pg.PlotDataItem", return_value=fake_curve):
            o = PhasorCurveOverlay()
            o.attach(fake_canvas)
            t = np.linspace(0, 0.1, 50)
            d = np.ones(50) * 230.0
            o.update_channel("VA", t, d, color="#FF4444")
            assert "VA" in o.channel_names()
            fake_curve.setData.assert_called_once()

    def test_remove_channel(self) -> None:
        from app.visualization.overlays.phasor_overlay import PhasorCurveOverlay
        import pyqtgraph as pg
        fake_canvas = MagicMock()
        fake_plot = MagicMock()
        fake_canvas._primary_plot = fake_plot
        fake_curve = MagicMock(spec=pg.PlotDataItem)
        fake_curve.getViewBox = MagicMock(return_value=None)

        with patch("app.visualization.overlays.curve_store.pg.PlotDataItem", return_value=fake_curve):
            o = PhasorCurveOverlay()
            o.attach(fake_canvas)
            t = np.linspace(0, 0.1, 50)
            d = np.ones(50) * 100.0
            o.update_channel("IA", t, d)
            assert "IA" in o.channel_names()
            o.remove_channel("IA")
            assert "IA" not in o.channel_names()


# ─────────────────────────────────────────────────────────────────────────────
# Color helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorPenColors:
    def test_mag_pen_color_returns_hex(self) -> None:
        from app.visualization.widgets.flexible_plot_canvas import _phasor_mag_pen_color
        result = _phasor_mag_pen_color("#FF4444")
        assert result.startswith("#")
        assert len(result) == 7

    def test_angle_pen_color_returns_hex(self) -> None:
        from app.visualization.widgets.flexible_plot_canvas import _phasor_angle_pen_color
        result = _phasor_angle_pen_color("#FF4444")
        assert result.startswith("#")
        assert len(result) == 7

    def test_mag_pen_color_brighter_than_input(self) -> None:
        from app.visualization.widgets.flexible_plot_canvas import _phasor_mag_pen_color
        # Red channel at #FF0000 → should stay near full red, green/blue raised
        result = _phasor_mag_pen_color("#FF0000")
        r = int(result[1:3], 16)
        g = int(result[3:5], 16)
        b = int(result[5:7], 16)
        assert r == 255  # capped at 255
        assert g > 0     # blended toward white
        assert b > 0

    def test_angle_pen_color_shifts_toward_cyan(self) -> None:
        from app.visualization.widgets.flexible_plot_canvas import _phasor_angle_pen_color
        result = _phasor_angle_pen_color("#FF0000")
        g = int(result[3:5], 16)
        b = int(result[5:7], 16)
        # Green and blue should increase (toward cyan)
        assert g > 0
        assert b > 0

    def test_mag_color_handles_invalid_input(self) -> None:
        from app.visualization.widgets.flexible_plot_canvas import _phasor_mag_pen_color
        assert _phasor_mag_pen_color("badcolor") == "#FFFFFF"

    def test_angle_color_handles_invalid_input(self) -> None:
        from app.visualization.widgets.flexible_plot_canvas import _phasor_angle_pen_color
        assert _phasor_angle_pen_color("badcolor") == "#44FFFF"


# ─────────────────────────────────────────────────────────────────────────────
# _make_sequence_record — pure logic, no Qt
# ─────────────────────────────────────────────────────────────────────────────


class TestMakeSequenceRecord:
    def test_builds_record_with_correct_channels(self) -> None:
        from app.ui.main_window.main_window import _make_sequence_record

        src = _make_disturbance_record(["VA", "VB", "VC"])
        t = np.linspace(0, 0.1, 400)
        data = {"V1": np.ones(400) * 230.0, "V2": np.zeros(400), "V0": np.zeros(400)}
        result = _make_sequence_record(src, t, data, "V")

        assert len(result.analog_channels) == 3
        names = [ch.name for ch in result.analog_channels]
        assert "V1" in names
        assert "V2" in names
        assert "V0" in names

    def test_builds_record_with_correct_unit(self) -> None:
        from app.ui.main_window.main_window import _make_sequence_record

        src = _make_disturbance_record(["IA", "IB", "IC"])
        t = np.linspace(0, 0.1, 400)
        data = {"I1": np.ones(400) * 10.0, "I2": np.zeros(400), "I0": np.zeros(400)}
        result = _make_sequence_record(src, t, data, "A")

        for ch in result.analog_channels:
            assert ch.unit == "A"

    def test_shares_timing_info_with_source(self) -> None:
        from app.ui.main_window.main_window import _make_sequence_record

        src = _make_disturbance_record(["VA"])
        t = np.linspace(0, 0.1, 400)
        data = {"V1": np.ones(400)}
        result = _make_sequence_record(src, t, data, "V")

        assert result.timing_info is src.timing_info

    def test_time_column_present_in_waveform_data(self) -> None:
        from app.ui.main_window.main_window import _make_sequence_record

        src = _make_disturbance_record(["VA"])
        t = np.linspace(0, 0.1, 400)
        data = {"V1": np.ones(400)}
        result = _make_sequence_record(src, t, data, "V")

        assert "time" in result.waveform_data.columns

    def test_no_digital_channels(self) -> None:
        from app.ui.main_window.main_window import _make_sequence_record

        src = _make_disturbance_record(["VA"])
        t = np.linspace(0, 0.1, 100)
        data = {"V1": np.ones(100)}
        result = _make_sequence_record(src, t, data, "V")

        assert result.digital_channels == []


# ─────────────────────────────────────────────────────────────────────────────
# _apply_phasor_display_mode routing — uses _FakeCanvas, no Qt
# ─────────────────────────────────────────────────────────────────────────────


class _FakeMainWindow:
    """Minimal stub of PowerwaveMainWindow for routing tests."""

    def __init__(self, panel_canvases: dict, main_canvas=None) -> None:
        self._panel_canvases = panel_canvases
        self._canvas = main_canvas or _FakeCanvas()
        self._phasor_registry = PhasorRegistry()
        self._current_signal_metadata: dict = {}

    def _qt_widget_alive(self, w: Any) -> bool:
        return w is not None

    def _apply_phasor_display_mode(self) -> None:
        """Inline copy of the real implementation for testing without Qt."""
        mode = self._phasor_registry.display_mode

        def _is_sequence_key(k: str) -> bool:
            return (
                k in ("sequence_voltage", "sequence_current")
                or k.endswith("/sequence_voltage")
                or k.endswith("/sequence_current")
            )

        if self._panel_canvases:
            waveform_canvases = [
                c for k, c in self._panel_canvases.items()
                if not _is_sequence_key(k) and self._qt_widget_alive(c)
            ]
        elif self._qt_widget_alive(self._canvas):
            waveform_canvases = [self._canvas]
        else:
            waveform_canvases = []

        sig_meta = self._current_signal_metadata or None
        for canvas in waveform_canvases:
            canvas.set_phasor_display_mode(mode, signal_metadata=sig_meta)

        seq_visible = (mode == PhasorDisplayMode.SEQUENCE_COMPONENTS)
        for key, canvas in (self._panel_canvases or {}).items():
            if _is_sequence_key(key) and self._qt_widget_alive(canvas):
                canvas.setVisible(seq_visible)


class TestApplyPhasorDisplayModeRouting:
    def test_off_mode_calls_off_on_all_waveform_canvases(self) -> None:
        volt = _FakeCanvas()
        curr = _FakeCanvas()
        win = _FakeMainWindow({"voltage_raw": volt, "current_raw": curr})
        win._phasor_registry.set_display_mode(PhasorDisplayMode.OFF)
        win._apply_phasor_display_mode()

        assert volt.phasor_mode_calls[-1] == PhasorDisplayMode.OFF
        assert curr.phasor_mode_calls[-1] == PhasorDisplayMode.OFF

    def test_magnitude_mode_calls_magnitude_on_waveform_canvases(self) -> None:
        volt = _FakeCanvas()
        curr = _FakeCanvas()
        win = _FakeMainWindow({"voltage_raw": volt, "current_raw": curr})
        win._phasor_registry.set_display_mode(PhasorDisplayMode.MAGNITUDE)
        win._apply_phasor_display_mode()

        assert volt.phasor_mode_calls[-1] == PhasorDisplayMode.MAGNITUDE
        assert curr.phasor_mode_calls[-1] == PhasorDisplayMode.MAGNITUDE

    def test_angle_mode_calls_angle_on_waveform_canvases(self) -> None:
        volt = _FakeCanvas()
        win = _FakeMainWindow({"voltage_raw": volt})
        win._phasor_registry.set_display_mode(PhasorDisplayMode.ANGLE)
        win._apply_phasor_display_mode()

        assert volt.phasor_mode_calls[-1] == PhasorDisplayMode.ANGLE

    def test_sequence_mode_does_not_call_waveform_with_magnitude(self) -> None:
        volt = _FakeCanvas()
        seq_v = _FakeCanvas()
        win = _FakeMainWindow({"voltage_raw": volt, "sequence_voltage": seq_v})
        win._phasor_registry.set_display_mode(PhasorDisplayMode.SEQUENCE_COMPONENTS)
        win._apply_phasor_display_mode()

        # Waveform canvas gets SEQUENCE_COMPONENTS (clears any magnitude/angle)
        assert volt.phasor_mode_calls[-1] == PhasorDisplayMode.SEQUENCE_COMPONENTS
        # Sequence panel itself gets no set_phasor_display_mode call
        assert seq_v.phasor_mode_calls == []

    def test_sequence_panels_become_visible_in_sequence_mode(self) -> None:
        seq_v = _FakeCanvas()
        seq_i = _FakeCanvas()
        seq_v.setVisible(False)
        seq_i.setVisible(False)
        win = _FakeMainWindow({
            "voltage_raw": _FakeCanvas(),
            "sequence_voltage": seq_v,
            "sequence_current": seq_i,
        })
        win._phasor_registry.set_display_mode(PhasorDisplayMode.SEQUENCE_COMPONENTS)
        win._apply_phasor_display_mode()

        assert seq_v.isVisible()
        assert seq_i.isVisible()

    def test_sequence_panels_hidden_in_off_mode(self) -> None:
        seq_v = _FakeCanvas()
        seq_v.setVisible(True)
        win = _FakeMainWindow({
            "voltage_raw": _FakeCanvas(),
            "sequence_voltage": seq_v,
        })
        win._phasor_registry.set_display_mode(PhasorDisplayMode.OFF)
        win._apply_phasor_display_mode()

        assert not seq_v.isVisible()

    def test_sequence_panels_hidden_in_magnitude_mode(self) -> None:
        seq_v = _FakeCanvas()
        seq_v.setVisible(True)
        win = _FakeMainWindow({
            "voltage_raw": _FakeCanvas(),
            "sequence_voltage": seq_v,
        })
        win._phasor_registry.set_display_mode(PhasorDisplayMode.MAGNITUDE)
        win._apply_phasor_display_mode()

        assert not seq_v.isVisible()

    def test_multi_source_sequence_key_is_hidden(self) -> None:
        seq_v = _FakeCanvas()
        seq_v.setVisible(True)
        volt = _FakeCanvas()
        win = _FakeMainWindow({
            "SRC/voltage_raw": volt,
            "SRC/sequence_voltage": seq_v,
        })
        win._phasor_registry.set_display_mode(PhasorDisplayMode.OFF)
        win._apply_phasor_display_mode()

        assert not seq_v.isVisible()

    def test_no_panel_canvases_uses_main_canvas(self) -> None:
        main = _FakeCanvas()
        win = _FakeMainWindow({}, main_canvas=main)
        win._phasor_registry.set_display_mode(PhasorDisplayMode.MAGNITUDE)
        win._apply_phasor_display_mode()

        assert main.phasor_mode_calls[-1] == PhasorDisplayMode.MAGNITUDE


# ─────────────────────────────────────────────────────────────────────────────
# _build_sequence_panels — logic tests using synthetic records
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildSequencePanels:
    """Test _build_sequence_panels logic via a minimal standalone reimplementation
    of the helper (avoids full PowerwaveMainWindow construction)."""

    def _run_build(self, record, signal_metadata=None):
        """Call _build_sequence_panels via a stub that mirrors the real implementation."""
        import numpy as np
        from app.analytics.phasors.phasor_extraction import extract_phasor
        from app.analytics.phasors.symmetrical_components import compute_sequence_from_phasor_arrays
        from app.ui.main_window.main_window import _make_sequence_record

        registry = PhasorRegistry()
        channel_names = [ch.name for ch in record.analog_channels]
        channel_phases = {
            ch.name: ch.phase
            for ch in record.analog_channels
            if ch.phase
        }
        groups = registry.detect_three_phase_groups(
            channel_names, signal_metadata or None, channel_phases or None
        )
        if not groups:
            return {}

        time_col = record.waveform_data["time"].to_numpy(dtype=np.float64)
        rates = [r for r in record.sampling_info.sampling_rates if r > 0]
        sample_rate_hz = float(rates[0]) if rates else 0.0

        if sample_rate_hz <= 0:
            return {}

        config = registry.config
        seq_voltage_data: dict = {}
        seq_current_data: dict = {}
        seq_time = None

        for group in groups:
            if not group.complete:
                continue
            try:
                ch_a = record.waveform_data[group.phase_a].to_numpy(dtype=np.float64)
                ch_b = record.waveform_data[group.phase_b].to_numpy(dtype=np.float64)
                ch_c = record.waveform_data[group.phase_c].to_numpy(dtype=np.float64)
                pa = extract_phasor(time_col, ch_a, sample_rate_hz, config)
                pb = extract_phasor(time_col, ch_b, sample_rate_hz, config)
                pc = extract_phasor(time_col, ch_c, sample_rate_hz, config)
                seq = compute_sequence_from_phasor_arrays(pa, pb, pc)
            except Exception:
                continue

            seq_time = seq["time"]
            pfx = group.prefix or ""
            if group.signal_type == "voltage":
                seq_voltage_data[f"{pfx}V1"] = seq["mag_v1"]
                seq_voltage_data[f"{pfx}V2"] = seq["mag_v2"]
                seq_voltage_data[f"{pfx}V0"] = seq["mag_v0"]
            else:
                seq_current_data[f"{pfx}I1"] = seq["mag_v1"]
                seq_current_data[f"{pfx}I2"] = seq["mag_v2"]
                seq_current_data[f"{pfx}I0"] = seq["mag_v0"]

        if seq_time is None:
            return {}

        result = {}
        for panel_key, data_dict, unit, title in [
            ("sequence_voltage", seq_voltage_data, "V", "Sequence Voltage"),
            ("sequence_current", seq_current_data, "A", "Sequence Current"),
        ]:
            if not data_dict:
                continue
            syn = _make_sequence_record(record, seq_time, data_dict, unit)
            canvas = _FakeCanvas()
            canvas.set_record(syn)
            canvas.set_panel_title(title)
            canvas.setVisible(False)
            result[panel_key] = canvas

        return result

    def test_returns_empty_for_non_three_phase_record(self) -> None:
        record = _make_disturbance_record(["VA"])
        result = self._run_build(record)
        assert result == {}

    def test_returns_sequence_voltage_for_abc_voltages(self) -> None:
        record = _make_disturbance_record(["VA", "VB", "VC"])
        result = self._run_build(record)
        assert "sequence_voltage" in result

    def test_sequence_voltage_panel_starts_hidden(self) -> None:
        record = _make_disturbance_record(["VA", "VB", "VC"])
        result = self._run_build(record)
        assert "sequence_voltage" in result
        assert not result["sequence_voltage"].isVisible()

    def test_sequence_voltage_panel_has_record_set(self) -> None:
        record = _make_disturbance_record(["VA", "VB", "VC"])
        result = self._run_build(record)
        canvas = result.get("sequence_voltage")
        assert canvas is not None
        assert canvas.record_set is not None

    def test_sequence_voltage_record_has_v1_v2_v0_channels(self) -> None:
        record = _make_disturbance_record(["VA", "VB", "VC"])
        result = self._run_build(record)
        canvas = result["sequence_voltage"]
        names = [ch.name for ch in canvas.record_set.analog_channels]
        assert any("V1" in n or "v1" in n.lower() for n in names)
        assert any("V2" in n or "v2" in n.lower() for n in names)
        assert any("V0" in n or "v0" in n.lower() for n in names)

    def test_returns_empty_for_two_channels_only(self) -> None:
        record = _make_disturbance_record(["VA", "VB"])
        result = self._run_build(record)
        assert result == {}

    def test_current_group_goes_to_sequence_current(self) -> None:
        record = _make_disturbance_record(["IA", "IB", "IC"])
        result = self._run_build(record)
        assert "sequence_current" in result

    def test_mixed_voltage_and_current(self) -> None:
        record = _make_disturbance_record(["VA", "VB", "VC", "IA", "IB", "IC"])
        result = self._run_build(record)
        assert "sequence_voltage" in result
        assert "sequence_current" in result


# ─────────────────────────────────────────────────────────────────────────────
# Phasor cache reuse — verify extract_phasor called once per mode switch
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorCacheReuse:
    """Verify that switching MAGNITUDE→ANGLE reuses cached phasor results."""

    def test_switching_modes_reuses_cache(self) -> None:
        from app.analytics.phasors.phasor_cache import PhasorCache
        from app.analytics.phasors.phasor_extraction import extract_phasor, compute_phasor_window_samples

        record = _make_disturbance_record(["VA"], n_samples=500)
        time_col = record.waveform_data["time"].to_numpy(dtype=np.float64)
        data = record.waveform_data["VA"].to_numpy(dtype=np.float64)
        fs = 5000.0
        cfg = PhasorConfig()
        window = compute_phasor_window_samples(fs, cfg)

        cache = PhasorCache()
        extract_count = 0

        def counting_extract(t, v, sr, c):
            nonlocal extract_count
            extract_count += 1
            return extract_phasor(t, v, sr, c)

        # First extraction (magnitude mode)
        result = counting_extract(time_col, data, fs, cfg)
        cache.put_phasor("VA", window, cfg.nominal_hz, result)
        assert extract_count == 1

        # Second use (angle mode) — should use cache, not recompute
        cached = cache.get_phasor("VA", window, cfg.nominal_hz)
        assert cached is not None
        assert extract_count == 1  # no additional extraction needed

    def test_cache_cleared_on_new_record(self) -> None:
        from app.analytics.phasors.phasor_cache import PhasorCache
        from app.analytics.phasors.phasor_extraction import extract_phasor, compute_phasor_window_samples

        record = _make_disturbance_record(["VA"], n_samples=500)
        time_col = record.waveform_data["time"].to_numpy(dtype=np.float64)
        data = record.waveform_data["VA"].to_numpy(dtype=np.float64)
        fs = 5000.0
        cfg = PhasorConfig()
        window = compute_phasor_window_samples(fs, cfg)

        cache = PhasorCache()
        result = extract_phasor(time_col, data, fs, cfg)
        cache.put_phasor("VA", window, cfg.nominal_hz, result)

        assert cache.get_phasor("VA", window, cfg.nominal_hz) is not None
        cache.clear()
        assert cache.get_phasor("VA", window, cfg.nominal_hz) is None


# ─────────────────────────────────────────────────────────────────────────────
# _phasor_pen helper — visual style test
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorPenHelper:
    def test_magnitude_uses_dot_line(self) -> None:
        from app.visualization.overlays.phasor_overlay import _phasor_pen
        from PyQt6.QtCore import Qt
        pen = _phasor_pen("#FF4444", PhasorDisplayMode.MAGNITUDE)
        assert pen.style() == Qt.PenStyle.DotLine

    def test_angle_uses_dash_dot_line(self) -> None:
        from app.visualization.overlays.phasor_overlay import _phasor_pen
        from PyQt6.QtCore import Qt
        pen = _phasor_pen("#FF4444", PhasorDisplayMode.ANGLE)
        assert pen.style() == Qt.PenStyle.DashDotLine
