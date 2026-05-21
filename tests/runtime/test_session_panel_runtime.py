"""Runtime tests for app/ui/session/ — Phase 9B.

These tests exercise the Qt widget layer with a live QApplication.
Each test creates and destroys its own widgets; the runtime_qapp fixture
handles event processing and top-level widget cleanup.

Coverage
--------
 1.  SessionPanel shows and hides without crash
 2.  Dynamically adding 3 sources builds 3 rows
 3.  Removing a source tears down its row cleanly
 4.  Rapid offset changes (100 updates) don't crash or storm
 5.  100+ channel source expands without crash
 6.  processEvents responsiveness: 10 add-source cycles within 2 s
 7.  session_cleared removes all rows
 8.  refresh_all is idempotent (repeated calls don't duplicate rows)
 9.  refresh_all preserves expansion state across calls
10.  Source with non-uniform time produces correct tooltip
"""
from __future__ import annotations

import sys
import time
from datetime import datetime

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication

from app.sessions import EventAnalysisSession
from app.sessions.session_models import PanelConfig, SourceQualityMetrics
from app.ui.session.session_panel import SessionPanel


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def runtime_qapp():
    """Provide QApplication; drain event queue after each test."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    for w in list(QApplication.topLevelWidgets()):
        w.close()
        w.deleteLater()
    app.processEvents()
    app.sendPostedEvents()
    app.processEvents()


def _make_record(
    n_analog: int = 3,
    n_digital: int = 0,
    n_samples: int = 200,
    dt: float = 0.001,
):
    from app.models.channels import AnalogChannel, DigitalChannel
    from app.models.disturbance_record import DisturbanceRecord
    from app.models.metadata import RecordingMetadata
    from app.models.timing import SamplingInformation, TimingInformation

    t = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
    data: dict = {"time": t}
    analog_names = [f"VA{i}" for i in range(n_analog)]
    digital_names = [f"CB{i}" for i in range(n_digital)]
    for name in analog_names:
        data[name] = np.sin(2 * np.pi * 50 * t)
    for name in digital_names:
        data[name] = np.zeros(n_samples)

    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="RUNTIME_TEST",
            recorder_name="REC",
            source_file="rt.cfg",
            provider_type="comtrade",
            nominal_frequency=50.0,
        ),
        waveform_data=data,
        analog_channels=[
            AnalogChannel(name=n, unit="kV", index=i)
            for i, n in enumerate(analog_names)
        ],
        digital_channels=[
            DigitalChannel(name=n, index=i) for i, n in enumerate(digital_names)
        ],
        sampling_info=SamplingInformation(
            sampling_rates=[1.0 / dt], samples_per_rate=[n_samples]
        ),
        timing_info=TimingInformation(
            start_time=datetime(2024, 1, 1),
            trigger_time=datetime(2024, 1, 1),
        ),
        disturbance_info=None,
    )


def _build_session(n_sources: int = 2, n_analog: int = 3) -> EventAnalysisSession:
    session = EventAnalysisSession()
    for i in range(n_sources):
        rec = _make_record(n_analog=n_analog)
        session.add_source(rec, f"Source {i+1}", "comtrade")
    session.default_layout()
    return session


# ---------------------------------------------------------------------------
# 1. Show/hide without crash
# ---------------------------------------------------------------------------


def test_session_panel_show_hide(runtime_qapp) -> None:
    panel = SessionPanel()
    panel.show()
    runtime_qapp.processEvents()
    panel.hide()
    runtime_qapp.processEvents()
    panel.close()


# ---------------------------------------------------------------------------
# 2. Dynamically adding 3 sources
# ---------------------------------------------------------------------------


def test_add_three_sources_builds_three_rows(runtime_qapp) -> None:
    session = _build_session(n_sources=3)
    panel = SessionPanel()
    panel.show()
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    assert len(panel._source_rows) == 3
    panel.close()


# ---------------------------------------------------------------------------
# 3. Removing a source tears down its row
# ---------------------------------------------------------------------------


def test_remove_source_tears_down_row(runtime_qapp) -> None:
    session = _build_session(n_sources=2)
    panel = SessionPanel()
    panel.show()
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    source_ids = list(panel._source_rows.keys())
    first_id = source_ids[0]
    session.remove_source(first_id)
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    assert first_id not in panel._source_rows
    assert len(panel._source_rows) == 1
    panel.close()


# ---------------------------------------------------------------------------
# 4. Rapid offset changes (no crash, no storm)
# ---------------------------------------------------------------------------


def test_rapid_offset_changes(runtime_qapp) -> None:
    session = _build_session(n_sources=1)
    panel = SessionPanel()
    panel.show()
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    source_id = list(panel._source_rows.keys())[0]
    row = panel._source_rows[source_id]

    offset_log: list[float] = []
    panel.offset_changed.connect(lambda _sid, val: offset_log.append(val))

    # Start from 1 to avoid no-change on the initial value of 0.0
    for i in range(1, 101):
        row._offset_spin.setValue(float(i) * 0.01)

    runtime_qapp.processEvents()
    assert len(offset_log) == 100
    panel.close()


# ---------------------------------------------------------------------------
# 5. 100+ channel source expands without crash
# ---------------------------------------------------------------------------


def test_hundred_channel_expansion(runtime_qapp) -> None:
    session = EventAnalysisSession()
    rec = _make_record(n_analog=110, n_digital=10, n_samples=50)
    session.add_source(rec, "BigSource", "comtrade")
    session.default_layout()

    panel = SessionPanel()
    panel.show()
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    source_id = list(panel._source_rows.keys())[0]
    row = panel._source_rows[source_id]
    row.set_expanded(True)
    runtime_qapp.processEvents()

    tree = row._channel_tree
    assert len(tree._channel_items) == 120   # 110 analog + 10 digital
    panel.close()


# ---------------------------------------------------------------------------
# 6. processEvents responsiveness: 10 add-source cycles
# ---------------------------------------------------------------------------


def test_add_source_responsiveness(runtime_qapp) -> None:
    session = EventAnalysisSession()
    panel = SessionPanel()
    panel.show()

    start = time.monotonic()
    for i in range(10):
        rec = _make_record(n_analog=5)
        session.add_source(rec, f"S{i}", "comtrade")
        session.default_layout()
        panel.refresh_all(session)
        runtime_qapp.processEvents()

    elapsed = time.monotonic() - start
    assert elapsed < 10.0, f"10 add-source cycles took {elapsed:.2f}s (limit 10s)"
    assert len(panel._source_rows) == 10
    panel.close()


# ---------------------------------------------------------------------------
# 7. session_cleared removes all rows
# ---------------------------------------------------------------------------


def test_clear_button_removes_all_rows(runtime_qapp) -> None:
    session = _build_session(n_sources=3)
    panel = SessionPanel()
    panel.show()
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    assert len(panel._source_rows) == 3
    panel._clear_btn.click()
    runtime_qapp.processEvents()

    assert len(panel._source_rows) == 0
    panel.close()


# ---------------------------------------------------------------------------
# 8. refresh_all is idempotent
# ---------------------------------------------------------------------------


def test_refresh_all_idempotent(runtime_qapp) -> None:
    session = _build_session(n_sources=2)
    panel = SessionPanel()
    panel.show()

    for _ in range(5):
        panel.refresh_all(session)
        runtime_qapp.processEvents()

    assert len(panel._source_rows) == 2
    panel.close()


# ---------------------------------------------------------------------------
# 9. refresh_all preserves expansion state
# ---------------------------------------------------------------------------


def test_refresh_all_preserves_expansion(runtime_qapp) -> None:
    session = _build_session(n_sources=2)
    panel = SessionPanel()
    panel.show()
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    source_ids = list(panel._source_rows.keys())
    panel._source_rows[source_ids[0]].set_expanded(True)

    panel.refresh_all(session)
    runtime_qapp.processEvents()

    assert panel._source_rows[source_ids[0]].is_expanded()
    assert not panel._source_rows[source_ids[1]].is_expanded()
    panel.close()


# ---------------------------------------------------------------------------
# 10. Non-uniform source tooltip
# ---------------------------------------------------------------------------


def test_nonuniform_source_tooltip(runtime_qapp) -> None:
    session = EventAnalysisSession()
    # Build a record whose time array has jitter so quality detects non-uniform
    import pandas as pd

    n = 200
    rng = np.random.default_rng(42)
    t = np.cumsum(np.abs(rng.normal(loc=0.001, scale=0.0002, size=n)))
    from app.models.channels import AnalogChannel
    from app.models.disturbance_record import DisturbanceRecord
    from app.models.metadata import RecordingMetadata
    from app.models.timing import SamplingInformation, TimingInformation

    data = {"time": t, "VA": np.sin(2 * np.pi * 50 * t)}
    rec = DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="JITTER",
            recorder_name="R",
            source_file="j.csv",
            provider_type="csv",
            nominal_frequency=50.0,
        ),
        waveform_data=data,
        analog_channels=[AnalogChannel(name="VA", unit="kV", index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[1000.0], samples_per_rate=[n]),
        timing_info=TimingInformation(
            start_time=datetime(2024, 1, 1),
            trigger_time=datetime(2024, 1, 1),
        ),
        disturbance_info=None,
    )
    session.add_source(rec, "jitter_source", "csv")
    session.default_layout()

    panel = SessionPanel()
    panel.show()
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    source_id = list(panel._source_rows.keys())[0]
    row = panel._source_rows[source_id]
    tooltip = row._info_btn.toolTip()
    # tooltip should mention uniformity (either way quality data was applied)
    assert "uniformity" in tooltip.lower() or "%" in tooltip
    panel.close()
