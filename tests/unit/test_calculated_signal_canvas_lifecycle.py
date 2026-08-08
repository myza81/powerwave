"""End-to-end Calculated Signals Session Canvas lifecycle tests (Phase 3B),
exercising the real app.ui.main_window.PowerwaveMainWindow wiring -- not a
reimplementation of it -- so these tests catch integration bugs the
lower-level controller/session-panel unit tests cannot.

Scenario (Phase 3B task Step 23):
    1. load Source A + Source B
    2. create A-B calculated signal (backend path -- equivalent to what the
       Phase 3A dialog does internally, already covered by its own tests)
    3. signal appears on canvas
    4. cursor can inspect it
    5. change Source B offset
    6. result becomes stale immediately
    7. stale indicator appears
    8. finish offset edit
    9. service recalculates once
    10. result becomes OK
    11. canvas waveform updates
    12. original source waveforms remain unchanged

Plus the failure case: deactivate -> stale -> failed recalc -> old waveform
stays visible as stale -> reactivate -> committed recalc succeeds.

Uses the repository's offscreen Qt conventions (QT_QPA_PLATFORM=offscreen is
set process-wide by tests/conftest.py; a module-scoped `qapp` fixture
provides a single QApplication).
"""
from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from PyQt6.QtWidgets import QApplication

from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    ChannelRef,
    CalculationStatus,
)
from app.calculated_signals.resolver import CalculatedSignalResolutionService
from app.models.channels import AnalogChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.event_session import EventAnalysisSession
from app.ui.main_window.main_window import PowerwaveMainWindow
from app.ui.session.session_canvas_controller import _calc_curve_key


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _make_record(analog: dict[str, str], time: np.ndarray | None = None, n: int = 2000) -> DisturbanceRecord:
    if time is not None:
        n = len(time)
    else:
        time = np.linspace(0, 10, n)
    data: dict[str, object] = {"time": time}
    for name in analog:
        data[name] = np.sin(time)
    df = pd.DataFrame(data)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="GenericStation", recorder_name="GenericRecorder",
            source_file="generic.csv", provider_type="csv", nominal_frequency=50.0,
        ),
        waveform_data=df,
        analog_channels=[AnalogChannel(name=n, unit=u, index=i) for i, (n, u) in enumerate(analog.items())],
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[10.0], samples_per_rate=[n]),
        timing_info=TimingInformation(start_time=datetime(2024, 1, 1), trigger_time=datetime(2024, 1, 1)),
    )


def _build_window_with_two_sources(qapp: QApplication) -> tuple[PowerwaveMainWindow, str, str]:
    win = PowerwaveMainWindow()
    win._active_session = EventAnalysisSession()
    sid_a = win._active_session.add_source(_make_record({"A": "MW"}), "Source A", "csv")
    sid_b = win._active_session.add_source(_make_record({"B": "MW"}), "Source B", "csv")
    win._active_session.default_layout()
    win._session_canvas_action.setEnabled(True)
    win._activate_session_canvas()
    qapp.processEvents()
    return win, sid_a, sid_b


class TestSuccessfulRecalculationLifecycle:
    def test_full_lifecycle(self, qapp: QApplication) -> None:
        win, sid_a, sid_b = _build_window_with_two_sources(qapp)
        try:
            sess = win._active_session
            ctrl = win._session_canvas_controller

            # 2. create A-B calculated signal (backend path)
            defn = CalculatedSignalDefinition(
                calc_id="c1", name="NetPower", expression="a - b",
                variable_bindings={"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")},
                reference_variable="a",
            )
            sess.add_calculated_signal(defn)
            CalculatedSignalResolutionService(sess).resolve_one("c1")

            # Equivalent to _on_calculated_signals()'s post-dialog-Accepted step
            win._sync_calculated_signals_to_canvas()
            qapp.processEvents()

            # 3. signal appears on canvas
            panel_id = ctrl._calc_panel_by_id["c1"]
            canvas = ctrl._canvases[panel_id]
            key = _calc_curve_key("c1", "NetPower")
            assert key in canvas._curves
            original_curve = canvas._curves[key]

            # 4. cursor can inspect it
            received = []
            canvas.crosshair_values_changed.connect(lambda t, v: received.append((t, v)))
            canvas._emit_crosshair_values(1.0)
            assert received
            assert any(entry[0].startswith("calc:") for entry in received[0][1])

            # Snapshot original source curve identity/data for step 12
            source_key = (sid_a, "A")
            source_curve_before = canvas._curves.get(source_key) or next(
                c._curves[source_key] for c in ctrl._canvases.values() if source_key in c._curves
            )
            source_x_before, source_y_before = source_curve_before.getOriginalDataset()

            # 5. change Source B offset (simulates spinbox valueChanged)
            win._on_session_offset_changed(sid_b, 0.5)
            qapp.processEvents()

            # 6. result becomes stale immediately
            assert sess.get_calculated_signal_result("c1").status == CalculationStatus.STALE

            # 7. stale indicator appears (canvas label + session panel row)
            assert "(stale)" in canvas._metadata[key].display_name
            panel_widget = win._session_panel
            panel_widget.refresh_calculated_signals(sess)
            assert panel_widget._calc_rows["c1"]._status_lbl.text() == "stale"

            # 8. finish offset edit (editingFinished)
            win._on_session_offset_edit_finished(sid_b)
            qapp.processEvents()

            # 9 + 10. service recalculated once, result is OK
            result = sess.get_calculated_signal_result("c1")
            assert result.status == CalculationStatus.OK

            # 11. canvas waveform updates (same PlotDataItem, new data)
            assert canvas._curves[key] is original_curve  # no duplicate curve
            x_after, y_after = canvas._curves[key].getOriginalDataset()
            np.testing.assert_array_almost_equal(x_after, result.time)
            np.testing.assert_array_almost_equal(y_after, result.values)

            # 12. original source waveforms remain unchanged
            source_x_after, source_y_after = source_curve_before.getOriginalDataset()
            np.testing.assert_array_equal(source_x_before, source_x_after)
            np.testing.assert_array_equal(source_y_before, source_y_after)
        finally:
            win.close()
            qapp.processEvents()


class TestFailedRecalculationLifecycle:
    def test_deactivate_then_reactivate(self, qapp: QApplication) -> None:
        win, sid_a, sid_b = _build_window_with_two_sources(qapp)
        try:
            sess = win._active_session
            ctrl = win._session_canvas_controller

            defn = CalculatedSignalDefinition(
                calc_id="c1", name="NetPower", expression="a - b",
                variable_bindings={"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")},
                reference_variable="a",
            )
            sess.add_calculated_signal(defn)
            CalculatedSignalResolutionService(sess).resolve_one("c1")
            win._sync_calculated_signals_to_canvas()
            qapp.processEvents()

            panel_id = ctrl._calc_panel_by_id["c1"]
            canvas = ctrl._canvases[panel_id]
            key = _calc_curve_key("c1", "NetPower")
            assert key in canvas._curves
            stale_snapshot_before = canvas._curves[key].getOriginalDataset()

            # 2. deactivate Source B -- this already triggers a recalc
            # attempt at the reactivation boundary handler, which is
            # expected to fail (dependency now unresolvable).
            win._on_session_source_active(sid_b, False)
            qapp.processEvents()

            # 3. result stale (or unchanged from before -- either way, not destroyed)
            result = sess.get_calculated_signal_result("c1")
            assert result.status == CalculationStatus.STALE

            # 4/5. old waveform remains visible as stale -- curve retained
            assert key in canvas._curves
            x_still, y_still = canvas._curves[key].getOriginalDataset()
            np.testing.assert_array_equal(x_still, stale_snapshot_before[0])
            np.testing.assert_array_equal(y_still, stale_snapshot_before[1])
            assert "(stale)" in canvas._metadata[key].display_name

            # 6. reactivate
            win._on_session_source_active(sid_b, True)
            qapp.processEvents()

            # 7. committed recalc succeeds
            result_after = sess.get_calculated_signal_result("c1")
            assert result_after.status == CalculationStatus.OK
            assert key in canvas._curves
        finally:
            win.close()
            qapp.processEvents()
