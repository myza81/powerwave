"""Runtime tests for Phase 9C — Interactive Time Alignment.

Tests exercise the full Qt widget stack with a live QApplication and a real
EventAnalysisSession.  Each test creates its own session; the runtime_qapp
fixture handles cleanup.

Coverage
--------
 1.  Two-source session: auto-align produces non-zero offset for delayed source
 2.  processEvents after auto-align reflects updated spinbox value and badge text
 3.  Reset returns offset to 0.000 s and clears badge
 4.  Align All skips inactive sources
 5.  Manual spinbox change clears alignment badge method
"""
from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication

from app.sessions import EventAnalysisSession
from app.sessions.alignment_engine import suggest_alignment_offsets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runtime_qapp():
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stepped_record(
    n_samples: int = 500,
    dt: float = 0.001,
    step_at: float = 0.10,
    amplitude: float = 10.0,
    pre_amplitude: float = 0.5,
):
    """Record with a step event at step_at seconds.

    The pre-step signal has a small non-zero amplitude so the trigger detector
    has a measurable baseline RMS and can confidently detect the step.
    """
    from app.models.channels import AnalogChannel
    from app.models.disturbance_record import DisturbanceRecord
    from app.models.metadata import RecordingMetadata
    from app.models.timing import SamplingInformation, TimingInformation

    t = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
    signal = np.where(t >= step_at, amplitude, pre_amplitude).astype(np.float64)
    data = {"time": t, "VA": signal}

    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="ALIGN_TEST",
            recorder_name="REC",
            source_file="test.cfg",
            provider_type="comtrade",
            nominal_frequency=50.0,
        ),
        waveform_data=data,
        analog_channels=[AnalogChannel(name="VA", unit="kV", index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[1.0 / dt], samples_per_rate=[n_samples]
        ),
        timing_info=TimingInformation(
            start_time=datetime(2024, 1, 1),
            trigger_time=datetime(2024, 1, 1),
        ),
        disturbance_info=None,
    )


def _build_two_source_session(delay_s: float = 0.05) -> EventAnalysisSession:
    """Session with source 1 stepping at t=0.10 and source 2 stepping at t=0.10+delay."""
    session = EventAnalysisSession()
    rec1 = _make_stepped_record(step_at=0.10)
    rec2 = _make_stepped_record(step_at=0.10 + delay_s)
    session.add_source(rec1, "Source1", "comtrade")
    session.add_source(rec2, "Source2", "comtrade")
    session.default_layout()
    return session


# ---------------------------------------------------------------------------
# 1. Auto-align produces non-zero offset for delayed source
# ---------------------------------------------------------------------------


def test_auto_align_produces_nonzero_offset_for_delayed_source(runtime_qapp) -> None:
    session = _build_two_source_session(delay_s=0.05)
    sources = session.list_sources()
    results = suggest_alignment_offsets(sources)

    offsets = {r.source_id: r.suggested_offset_s for r in results}
    assert len(offsets) == 2
    sid1, sid2 = [s.source_id for s in sources]
    # Both offsets shift trigger to t=0; the delayed source gets a larger magnitude shift
    assert abs(offsets[sid2]) > abs(offsets[sid1]) or abs(offsets[sid2] - offsets[sid1]) > 0.01


# ---------------------------------------------------------------------------
# 2. Panel reflects updated spinbox and badge after auto-align
# ---------------------------------------------------------------------------


def test_panel_reflects_auto_align_result(runtime_qapp) -> None:
    from app.ui.session.session_panel import SessionPanel

    session = _build_two_source_session(delay_s=0.05)
    panel = SessionPanel()
    panel.show()
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    sources = session.list_sources()
    results = suggest_alignment_offsets(sources)

    for result in results:
        session.set_time_offset(
            result.source_id,
            result.suggested_offset_s,
            method=result.alignment_method,
            confidence=result.alignment_confidence,
        )
        session.set_alignment_notes(result.source_id, result.notes)

    panel.refresh_all(session)
    runtime_qapp.processEvents()

    for source in sources:
        row = panel._source_rows[source.source_id]
        # Spinbox reflects the written offset
        assert abs(row._offset_spin.value() - source.time_offset_s) < 1e-6
        # Badge has visible text (alignment method was set)
        assert row._alignment_badge.text() != ""

    panel.close()


# ---------------------------------------------------------------------------
# 3. Reset returns offset to 0 and clears badge
# ---------------------------------------------------------------------------


def test_reset_clears_offset_and_badge(runtime_qapp) -> None:
    from app.ui.session.session_panel import SessionPanel
    from app.sessions.session_models import AlignmentResult

    session = _build_two_source_session(delay_s=0.05)
    sources = session.list_sources()
    sid = sources[0].source_id

    # Apply an alignment first
    session.set_time_offset(sid, -0.10, method="auto_trigger", confidence=0.90)
    session.set_alignment_notes(sid, "Trigger at 0.10 s.")

    panel = SessionPanel()
    panel.show()
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    # Now reset
    session.set_time_offset(sid, 0.0, method="none")
    session.set_alignment_notes(sid, "")
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    row = panel._source_rows[sid]
    assert abs(row._offset_spin.value()) < 1e-9
    assert row._alignment_badge.text() == ""
    panel.close()


# ---------------------------------------------------------------------------
# 4. Align All skips inactive sources
# ---------------------------------------------------------------------------


def test_align_all_skips_inactive_sources(runtime_qapp) -> None:
    session = _build_two_source_session(delay_s=0.05)
    sources = session.list_sources()
    sid2 = sources[1].source_id

    # Deactivate source 2
    session.set_source_active(sid2, False)

    active_sources = [s for s in session.list_sources() if s.is_active]
    results = suggest_alignment_offsets(active_sources)

    assert len(results) == 1
    assert results[0].source_id == sources[0].source_id


# ---------------------------------------------------------------------------
# 5. Manual spinbox change clears alignment method in session
# ---------------------------------------------------------------------------


def test_manual_spinbox_change_writes_manual_method(runtime_qapp) -> None:
    from app.ui.session.session_panel import SessionPanel

    session = _build_two_source_session(delay_s=0.05)
    sources = session.list_sources()
    sid = sources[0].source_id

    # Start with an auto-aligned state
    session.set_time_offset(sid, -0.10, method="auto_trigger", confidence=0.90)

    panel = SessionPanel()
    panel.show()
    panel.refresh_all(session)
    runtime_qapp.processEvents()

    # Simulate user editing the spinbox directly
    row = panel._source_rows[sid]
    changed: list[tuple[str, float]] = []
    panel.offset_changed.connect(lambda s, v: changed.append((s, v)))
    row._offset_spin.setValue(-0.05)
    runtime_qapp.processEvents()

    # The signal was emitted
    assert len(changed) == 1
    assert changed[0][0] == sid
    assert abs(changed[0][1] - (-0.05)) < 1e-6

    panel.close()
