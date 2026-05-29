"""Phase 8.5 harmonic visualization stabilization tests."""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from app.analytics.harmonics.harmonic_models import HarmonicDisplayMode
from app.data.signal_metadata import SignalMetadata


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _single_channel_record(name: str = "VA", unit: str = "V"):
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


def _mixed_channel_record():
    from app.data.synthetic import make_high_rate_record

    base = make_high_rate_record(duration_s=0.25, sampling_rate_hz=5000.0).record
    time = base.waveform_data["time"].to_numpy(dtype=np.float64)
    df = base.waveform_data.loc[:, ["time", base.analog_channels[0].name]].rename(
        columns={base.analog_channels[0].name: "VA"}
    )
    df["Freq"] = np.full_like(time, 50.0)
    return replace(
        base,
        waveform_data=df,
        analog_channels=[
            replace(base.analog_channels[0], name="VA", unit="V", index=0),
            replace(base.analog_channels[0], name="Freq", unit="Hz", index=1),
        ],
    )


def test_identical_harmonic_viewport_update_skips_redundant_setdata(qapp) -> None:
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(_single_channel_record("VA", "V"))
        canvas.set_harmonic_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
        harmonic_curve = next(iter(canvas._harmonic_curves["VA"].values()))
        original = harmonic_curve.setData
        spy = MagicMock(wraps=original)
        harmonic_curve.setData = spy
        canvas._curve_data_signatures.clear()

        canvas._update_viewport(0.02, 0.12)
        first_count = spy.call_count
        canvas._update_viewport(0.02, 0.12)

        assert first_count == 1
        assert spy.call_count == first_count
    finally:
        canvas.close()
        qapp.processEvents()


def test_repeated_off_switching_is_idempotent(qapp) -> None:
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(_single_channel_record("VA", "V"))
        for _ in range(5):
            canvas.set_harmonic_display_mode(HarmonicDisplayMode.OFF)
        assert canvas._harmonic_curves == {}
        assert canvas._harmonic_time_cache == {}
        assert canvas._harmonic_data_cache == {}
    finally:
        canvas.close()
        qapp.processEvents()


def test_unsupported_channels_do_not_populate_harmonic_cache(qapp) -> None:
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(_single_channel_record("Freq", "Hz"))
        canvas.set_harmonic_display_mode(HarmonicDisplayMode.HARMONIC_MAGNITUDE)
        assert canvas._harmonic_curves == {}
        assert canvas._harmonic_cache is None or len(canvas._harmonic_cache) == 0
    finally:
        canvas.close()
        qapp.processEvents()


def test_partial_harmonic_support_skips_telemetry_but_keeps_waveform(qapp) -> None:
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    metadata = {
        "VA": SignalMetadata(
            name="VA",
            unit="V",
            electrical_type="voltage",
            measurement_kind="waveform",
        ),
        "Freq": SignalMetadata(
            name="Freq",
            unit="Hz",
            electrical_type="frequency",
            measurement_kind="telemetry",
        ),
    }
    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(_mixed_channel_record())
        canvas.set_harmonic_display_mode(
            HarmonicDisplayMode.HARMONIC_MAGNITUDE,
            signal_metadata=metadata,
        )
        assert "VA" in canvas._harmonic_curves
        assert "Freq" not in canvas._harmonic_curves
    finally:
        canvas.close()
        qapp.processEvents()


def test_harmonic_panels_remain_cursor_synchronized(qapp) -> None:
    from app.visualization.managers.synchronization_manager import SynchronizationManager
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    master = FlexiblePlotCanvas()
    spectrum = FlexiblePlotCanvas()
    manager = SynchronizationManager(rate_limit_hz=240)
    try:
        master.set_record(_single_channel_record("VA", "V"))
        spectrum.set_record(_single_channel_record("H3", "V RMS"))
        manager.register_many([master, spectrum], master_canvas=master)

        manager.synchronize_cursor(master, 0.075)

        assert spectrum._cursor.value() == pytest.approx(0.075)
    finally:
        manager.clear()
        master.close()
        spectrum.close()
        qapp.processEvents()

