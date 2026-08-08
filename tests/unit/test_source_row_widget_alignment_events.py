"""Sprint 1C — widget-level event-routing tests for SourceRowWidget's
committed offset contract (offset_edit_finished).

Pure widget tests: no EventAnalysisSession, no main window. These verify
the SIGNAL CONTRACT itself -- exactly what fires, how many times, and
under what conditions -- independent of what main_window.py does with it.
Uses the repository's offscreen Qt conventions (QT_QPA_PLATFORM=offscreen
is set process-wide by tests/conftest.py; a module-scoped `qapp` fixture
provides a single QApplication).
"""
from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication

from app.models.channels import AnalogChannel, DigitalChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.session_models import PanelConfig, SessionSource
from app.ui.session.source_row_widget import SourceRowWidget


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _make_record() -> DisturbanceRecord:
    n = 100
    t = np.linspace(0, 0.099, n)
    data = {"time": t, "VA0": np.sin(2 * np.pi * 50 * t)}
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="GenericStation", recorder_name="GenericRecorder",
            source_file="generic.csv", provider_type="csv", nominal_frequency=50.0,
        ),
        waveform_data=data,
        analog_channels=[AnalogChannel(name="VA0", unit="kV", index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[1000.0], samples_per_rate=[n]),
        timing_info=TimingInformation(start_time=datetime(2024, 1, 1), trigger_time=datetime(2024, 1, 1)),
    )


def _make_source(time_offset_s: float = 0.0) -> SessionSource:
    return SessionSource(
        source_id="src-1", display_name="Source A", record=_make_record(),
        provider_type="csv", origin_path=None, time_offset_s=time_offset_s,
        is_active=True, alignment_method="none", alignment_confidence=None,
    )


def _make_row(time_offset_s: float = 0.0) -> SourceRowWidget:
    row = SourceRowWidget(_make_source(time_offset_s), [], [], [])
    row._sample_interval_s = 0.001
    return row


class _Recorder:
    """Collects (event_name, args) tuples from connected signals."""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple]] = []

    def wire(self, row: SourceRowWidget) -> None:
        row.offset_changed.connect(lambda sid, v: self.events.append(("changed", (sid, v))))
        row.offset_edit_finished.connect(lambda sid: self.events.append(("committed", (sid,))))

    def count(self, name: str) -> int:
        return sum(1 for e, _ in self.events if e == name)


# ─────────────────────────────────────────────────────────────────────────────
# Fine nudge
# ─────────────────────────────────────────────────────────────────────────────


class TestFineNudgeEventContract:
    def test_fine_right_changes_spinbox_value(self, qapp) -> None:
        row = _make_row(0.0)
        row._on_fine_right()
        assert row._offset_spin.value() == pytest.approx(0.001)
        row.close()

    def test_fine_left_changes_spinbox_value(self, qapp) -> None:
        row = _make_row(0.0)
        row._on_fine_left()
        assert row._offset_spin.value() == pytest.approx(-0.001)
        row.close()

    def test_fine_right_emits_exactly_one_changed_and_one_committed(self, qapp) -> None:
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)
        row._on_fine_right()
        assert rec.count("changed") == 1
        assert rec.count("committed") == 1
        row.close()

    def test_fine_left_emits_exactly_one_changed_and_one_committed(self, qapp) -> None:
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)
        row._on_fine_left()
        assert rec.count("changed") == 1
        assert rec.count("committed") == 1
        row.close()

    def test_committed_event_carries_the_source_id(self, qapp) -> None:
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)
        row._on_fine_right()
        assert rec.events[-1] == ("committed", ("src-1",))
        row.close()

    def test_committed_value_matches_new_spinbox_value(self, qapp) -> None:
        """The committed event fires AFTER the value has already changed --
        by the time it's observed, the spinbox reflects the new offset."""
        row = _make_row(0.0)
        observed = []
        row.offset_edit_finished.connect(lambda sid: observed.append(row._offset_spin.value()))
        row._on_fine_right()
        assert observed == [pytest.approx(0.001)]
        row.close()

    def test_round_trip_right_then_left_returns_to_original_with_two_commits(self, qapp) -> None:
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)
        row._on_fine_right()
        row._on_fine_left()
        assert row._offset_spin.value() == pytest.approx(0.0)
        assert rec.count("committed") == 2

    def test_three_separate_clicks_produce_three_committed_events(self, qapp) -> None:
        """Each click is a distinct deliberate adjustment -- not debounced."""
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)
        row._on_fine_right()
        row._on_fine_right()
        row._on_fine_right()
        assert rec.count("committed") == 3
        assert rec.count("changed") == 3
        assert row._offset_spin.value() == pytest.approx(0.003)
        row.close()


# ─────────────────────────────────────────────────────────────────────────────
# Typed edit / editingFinished
# ─────────────────────────────────────────────────────────────────────────────


class TestTypedEditEventContract:
    def test_change_then_editing_finished_emits_one_committed_event(self, qapp) -> None:
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)
        row._offset_spin.setValue(0.5)  # simulates KeyboardTracking ticks while typing
        row._on_offset_spin_editing_finished()
        assert rec.count("committed") == 1
        row.close()

    def test_unchanged_editing_finished_emits_zero_committed_events(self, qapp) -> None:
        """The EAV-identified no-change guard: editingFinished with no
        actual value change must not fire a committed event."""
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)
        row._on_offset_spin_editing_finished()
        assert rec.count("committed") == 0
        row.close()

    def test_multiple_ticks_then_one_editing_finished_still_one_commit(self, qapp) -> None:
        """Simulates typing several digits (each keystroke fires
        valueChanged via KeyboardTracking) followed by a single Enter --
        exactly one committed event, not one per keystroke."""
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)
        row._offset_spin.setValue(0.1)
        row._offset_spin.setValue(0.12)
        row._offset_spin.setValue(0.123)
        row._on_offset_spin_editing_finished()
        assert rec.count("committed") == 1
        assert rec.count("changed") == 3
        row.close()

    def test_editing_finished_twice_in_a_row_only_first_commits(self, qapp) -> None:
        """Enter pressed twice without further edits -- e.g. focus loss
        after Enter still firing editingFinished again -- must not
        double-commit."""
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)
        row._offset_spin.setValue(0.2)
        row._on_offset_spin_editing_finished()
        row._on_offset_spin_editing_finished()
        assert rec.count("committed") == 1
        row.close()

    def test_change_back_to_original_value_still_commits(self, qapp) -> None:
        """Typing away and then back to the exact original value across two
        separate edit sessions is still two real, distinct commits."""
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)
        row._offset_spin.setValue(0.3)
        row._on_offset_spin_editing_finished()
        row._offset_spin.setValue(0.0)
        row._on_offset_spin_editing_finished()
        assert rec.count("committed") == 2
        row.close()


# ─────────────────────────────────────────────────────────────────────────────
# refresh() / external programmatic push keeps tracking in sync
# ─────────────────────────────────────────────────────────────────────────────


class TestExternalRefreshTracking:
    def test_refresh_updates_committed_baseline(self, qapp) -> None:
        """After an external push (Reset/Auto-align/Set-as-reference all
        call refresh() with a new offset), editingFinished with no further
        change must not re-fire a committed event."""
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)

        pushed_source = _make_source(time_offset_s=1.234)
        row.refresh(pushed_source, None, [])
        assert row._offset_spin.value() == pytest.approx(1.234)

        row._on_offset_spin_editing_finished()
        assert rec.count("committed") == 0
        row.close()

    def test_refresh_baseline_then_real_edit_commits_once(self, qapp) -> None:
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)

        pushed_source = _make_source(time_offset_s=1.234)
        row.refresh(pushed_source, None, [])
        row._offset_spin.setValue(2.0)
        row._on_offset_spin_editing_finished()
        assert rec.count("committed") == 1
        row.close()

    def test_refresh_does_not_itself_emit_any_event(self, qapp) -> None:
        """refresh() is a pure display sync -- it must never itself fire
        offset_changed or offset_edit_finished (guarded by self._updating)."""
        row = _make_row(0.0)
        rec = _Recorder()
        rec.wire(row)
        row.refresh(_make_source(time_offset_s=5.0), None, [])
        assert rec.events == []
        row.close()

    def test_nudge_after_external_refresh_uses_refreshed_baseline(self, qapp) -> None:
        """A fine nudge immediately after Reset/Auto-align must nudge from
        the freshly pushed value, and still commit exactly once."""
        row = _make_row(0.0)
        row.refresh(_make_source(time_offset_s=1.0), None, [])
        rec = _Recorder()
        rec.wire(row)
        row._on_fine_right()
        assert row._offset_spin.value() == pytest.approx(1.001)
        assert rec.count("committed") == 1
        row.close()
