"""Unit tests for app/ui/session/ — Phase 9B Session Workspace Panel.

Coverage
--------
 1.  SessionPanel constructs without error
 2.  add_source_row adds a row that is visible in the panel
 3.  add_source_row is idempotent (duplicate source_id ignored)
 4.  remove_source_row removes the row
 5.  remove_source_row on unknown id is a no-op
 6.  source_remove_requested signal emitted when Remove button clicked
 7.  offset_changed signal emitted when spinbox value changes
 8.  source_active_changed signal emitted when active checkbox toggled
 9.  auto_align_requested signal emitted when Auto button clicked
10.  channel_visibility_changed signal re-emitted from channel tree
11.  channel_panel_changed signal re-emitted from panel combo
12.  session_cleared signal emitted on Clear Session click
13.  expansion state preserved across remove + re-add
14.  refresh_source_row updates offset display without rebuilding row
15.  quality tooltip set on ⓘ button
16.  refresh_all adds rows for new sources and removes stale ones
17.  alignment badge text for auto_trigger high confidence
18.  alignment badge text for none method is empty
19.  fine-left nudge decrements offset by sample interval
20.  fine-right nudge increments offset by sample interval
21.  ChannelTreeWidget.populate builds correct item counts
22.  ChannelTreeWidget.update_channel_visibility sets checkbox without signal
23.  ChannelTreeWidget.update_panel_choices swaps combo options without signal
24.  SourceRowWidget.set_expanded shows/hides channel tree
25.  SourceRowWidget.refresh does not fire offset_changed
26.  SourceRowWidget.refresh does not fire active_changed
27.  offset spinbox range is ±9999.999 with 3 decimal places
28.  offset spinbox step is 0.001
"""
from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from app.sessions import EventAnalysisSession
from app.sessions.session_models import (
    PanelConfig,
    SessionChannel,
    SessionSource,
    SourceQualityMetrics,
)
from app.ui.session.channel_tree_widget import ChannelTreeWidget
from app.ui.session.session_panel import SessionPanel
from app.ui.session.source_row_widget import SourceRowWidget

# ---------------------------------------------------------------------------
# QApplication fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(n_analog: int = 3, n_digital: int = 2):
    """Return a minimal DisturbanceRecord mock."""
    import pandas as pd
    from app.models.channels import AnalogChannel, DigitalChannel
    from app.models.disturbance_record import DisturbanceRecord
    from app.models.metadata import RecordingMetadata
    from app.models.timing import SamplingInformation, TimingInformation

    n = 100
    t = np.linspace(0, 0.099, n)
    data = {"time": t}
    analog_names = [f"VA{i}" for i in range(n_analog)]
    digital_names = [f"CB{i}" for i in range(n_digital)]
    for name in analog_names:
        data[name] = np.sin(2 * np.pi * 50 * t)
    for name in digital_names:
        data[name] = np.zeros(n)

    analog_channels = [
        AnalogChannel(name=n, unit="kV", index=i) for i, n in enumerate(analog_names)
    ]
    digital_channels = [
        DigitalChannel(name=n, index=i) for i, n in enumerate(digital_names)
    ]
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST",
            recorder_name="REC",
            source_file="test.cfg",
            provider_type="comtrade",
            nominal_frequency=50.0,
        ),
        waveform_data=data,
        analog_channels=analog_channels,
        digital_channels=digital_channels,
        sampling_info=SamplingInformation(
            sampling_rates=[1000.0], samples_per_rate=[n]
        ),
        timing_info=TimingInformation(
            start_time=datetime(2024, 1, 1),
            trigger_time=datetime(2024, 1, 1),
        ),
        disturbance_info=None,
    )


def _make_session_source(
    source_id: str = "src-1",
    display_name: str = "Test Source",
    provider_type: str = "comtrade",
    time_offset_s: float = 0.0,
    alignment_method: str = "none",
    alignment_confidence: float | None = None,
    is_active: bool = True,
) -> SessionSource:
    record = _make_record()
    return SessionSource(
        source_id=source_id,
        display_name=display_name,
        record=record,
        provider_type=provider_type,
        origin_path=None,
        time_offset_s=time_offset_s,
        is_active=is_active,
        alignment_method=alignment_method,
        alignment_confidence=alignment_confidence,
    )


def _make_metrics(
    source_id: str = "src-1",
    rate_hz: float = 1000.0,
    stability: float = 0.98,
    uniform: bool = True,
) -> SourceQualityMetrics:
    return SourceQualityMetrics(
        source_id=source_id,
        sample_count=100,
        inferred_sample_rate_hz=rate_hz,
        sample_rate_stability=stability,
        missing_data_pct=0.0,
        duplicate_timestamp_pct=0.0,
        interpolated_pct=0.0,
        resampling_ratio=1.0,
        time_is_uniform=uniform,
    )


def _make_panels() -> list[PanelConfig]:
    return [
        PanelConfig(panel_id="voltage", title="Voltage"),
        PanelConfig(panel_id="current", title="Current"),
        PanelConfig(panel_id="other", title="Other Analog"),
    ]


def _make_channels(source_id: str = "src-1") -> tuple[list[SessionChannel], list[SessionChannel]]:
    analog = [
        SessionChannel(
            source_id=source_id,
            channel_name=f"VA{i}",
            channel_type="analog",
            display_name=f"VA{i}",
            color_hex=None,
            line_style="solid",
            is_visible=True,
            panel_id="voltage",
        )
        for i in range(3)
    ]
    digital = [
        SessionChannel(
            source_id=source_id,
            channel_name=f"CB{i}",
            channel_type="digital",
            display_name=f"CB{i}",
            color_hex=None,
            line_style="solid",
            is_visible=True,
            panel_id="other",
        )
        for i in range(2)
    ]
    return analog, digital


# ---------------------------------------------------------------------------
# 1. SessionPanel construction
# ---------------------------------------------------------------------------


def test_session_panel_constructs(qapp) -> None:
    panel = SessionPanel()
    assert panel is not None
    panel.close()


# ---------------------------------------------------------------------------
# 2–4. Source row add / remove
# ---------------------------------------------------------------------------


def test_add_source_row_appears(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source()
    analog, digital = _make_channels()
    panels = _make_panels()
    panel.add_source_row(source, analog, digital, panels)
    assert source.source_id in panel._source_rows
    panel.close()


def test_add_source_row_idempotent(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source()
    analog, digital = _make_channels()
    panels = _make_panels()
    panel.add_source_row(source, analog, digital, panels)
    panel.add_source_row(source, analog, digital, panels)  # duplicate ignored
    assert len(panel._source_rows) == 1
    panel.close()


def test_remove_source_row_removes(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source()
    analog, digital = _make_channels()
    panels = _make_panels()
    panel.add_source_row(source, analog, digital, panels)
    panel.remove_source_row(source.source_id)
    assert source.source_id not in panel._source_rows
    panel.close()


def test_remove_source_row_unknown_is_noop(qapp) -> None:
    panel = SessionPanel()
    panel.remove_source_row("nonexistent-id")  # must not raise
    panel.close()


# ---------------------------------------------------------------------------
# 5–9. Signal emissions
# ---------------------------------------------------------------------------


def test_source_remove_requested_signal(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source()
    analog, digital = _make_channels()
    panels = _make_panels()
    panel.add_source_row(source, analog, digital, panels)

    emitted: list[str] = []
    panel.source_remove_requested.connect(emitted.append)

    row = panel._source_rows[source.source_id]
    row._remove_btn.click()

    assert emitted == [source.source_id]
    panel.close()


def test_offset_changed_signal_from_spinbox(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source()
    analog, digital = _make_channels()
    panels = _make_panels()
    panel.add_source_row(source, analog, digital, panels)

    emitted: list[tuple] = []
    panel.offset_changed.connect(lambda sid, val: emitted.append((sid, val)))

    row = panel._source_rows[source.source_id]
    row._offset_spin.setValue(1.234)

    assert len(emitted) == 1
    assert emitted[0][0] == source.source_id
    assert abs(emitted[0][1] - 1.234) < 1e-6
    panel.close()


def test_source_active_changed_signal(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source(is_active=True)
    analog, digital = _make_channels()
    panels = _make_panels()
    panel.add_source_row(source, analog, digital, panels)

    emitted: list[tuple] = []
    panel.source_active_changed.connect(lambda sid, active: emitted.append((sid, active)))

    row = panel._source_rows[source.source_id]
    row._active_cb.setChecked(False)

    assert len(emitted) == 1
    assert emitted[0] == (source.source_id, False)
    panel.close()


def test_auto_align_requested_signal(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source()
    analog, digital = _make_channels()
    panels = _make_panels()
    panel.add_source_row(source, analog, digital, panels)

    emitted: list[str] = []
    panel.auto_align_requested.connect(emitted.append)

    row = panel._source_rows[source.source_id]
    row._auto_align_btn.click()

    assert emitted == [source.source_id]
    panel.close()


# ---------------------------------------------------------------------------
# 10–11. Channel signal propagation
# ---------------------------------------------------------------------------


def test_channel_visibility_changed_propagated(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source()
    analog, digital = _make_channels()
    panels = _make_panels()
    panel.add_source_row(source, analog, digital, panels)

    emitted: list[tuple] = []
    panel.channel_visibility_changed.connect(
        lambda sid, ch, vis: emitted.append((sid, ch, vis))
    )

    row = panel._source_rows[source.source_id]
    row.set_expanded(True)
    tree = row._channel_tree
    item = tree._channel_items.get("VA0")
    assert item is not None
    item.setCheckState(0, Qt.CheckState.Unchecked)

    assert any(e[1] == "VA0" for e in emitted)
    panel.close()


def test_channel_panel_changed_propagated(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source()
    analog, digital = _make_channels()
    panels = _make_panels()
    panel.add_source_row(source, analog, digital, panels)

    emitted: list[tuple] = []
    panel.channel_panel_changed.connect(
        lambda sid, ch, pid: emitted.append((sid, ch, pid))
    )

    row = panel._source_rows[source.source_id]
    row.set_expanded(True)
    tree = row._channel_tree
    combo = tree._panel_combos.get("VA0")
    assert combo is not None
    combo.setCurrentIndex(1)  # switch to index 1 (Current panel)

    assert any(e[1] == "VA0" for e in emitted)
    panel.close()


# ---------------------------------------------------------------------------
# 12. session_cleared signal
# ---------------------------------------------------------------------------


def test_session_cleared_signal(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source()
    analog, digital = _make_channels()
    panels = _make_panels()
    panel.add_source_row(source, analog, digital, panels)

    emitted: list[int] = []
    panel.session_cleared.connect(lambda: emitted.append(1))

    # Sprint 1D: Clear Session now confirms first when the panel has
    # meaningful work (a populated source row here) -- simulate the user
    # confirming so this test keeps exercising the clear itself.
    with patch(
        "app.ui.session.session_panel.confirm_destructive_action",
        return_value=True,
    ):
        panel._clear_btn.click()

    assert len(emitted) == 1
    assert len(panel._source_rows) == 0
    panel.close()


# ---------------------------------------------------------------------------
# 13. Expansion state preserved across remove + re-add
# ---------------------------------------------------------------------------


def test_expansion_state_preserved_across_readd(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source()
    analog, digital = _make_channels()
    panels = _make_panels()

    panel.add_source_row(source, analog, digital, panels)
    row = panel._source_rows[source.source_id]
    row.set_expanded(True)
    assert row.is_expanded()

    panel.remove_source_row(source.source_id)
    assert panel._expansion_state.get(source.source_id) is True

    panel.add_source_row(source, analog, digital, panels)
    new_row = panel._source_rows[source.source_id]
    assert new_row.is_expanded()
    panel.close()


# ---------------------------------------------------------------------------
# 14. refresh_source_row updates offset without rebuild
# ---------------------------------------------------------------------------


def test_refresh_source_row_updates_offset(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source(time_offset_s=0.0)
    analog, digital = _make_channels()
    panels = _make_panels()
    panel.add_source_row(source, analog, digital, panels)
    row_before = panel._source_rows[source.source_id]

    updated_source = _make_session_source(
        source_id=source.source_id, time_offset_s=3.141
    )
    panel.refresh_source_row(source.source_id, updated_source, None, panels)

    row_after = panel._source_rows[source.source_id]
    assert row_before is row_after   # same object, not rebuilt
    assert abs(row_after._offset_spin.value() - 3.141) < 1e-6
    panel.close()


# ---------------------------------------------------------------------------
# 15. Quality tooltip
# ---------------------------------------------------------------------------


def test_quality_tooltip_set_from_metrics(qapp) -> None:
    panel = SessionPanel()
    source = _make_session_source()
    analog, digital = _make_channels()
    panels = _make_panels()
    metrics = _make_metrics(stability=0.61, uniform=False)
    panel.add_source_row(source, analog, digital, panels, metrics=metrics)

    row = panel._source_rows[source.source_id]
    tooltip = row._info_btn.toolTip()
    assert "61%" in tooltip
    assert "Non-uniform" in tooltip
    panel.close()


# ---------------------------------------------------------------------------
# 16. refresh_all syncs session state
# ---------------------------------------------------------------------------


def test_refresh_all_adds_and_removes(qapp) -> None:
    session = EventAnalysisSession()
    rec1 = _make_record()
    rec2 = _make_record()
    sid1 = session.add_source(rec1, "Source A", "comtrade")
    session.default_layout()

    panel = SessionPanel()
    panel.refresh_all(session)
    assert sid1 in panel._source_rows
    assert len(panel._source_rows) == 1

    sid2 = session.add_source(rec2, "Source B", "csv")
    session.default_layout()
    panel.refresh_all(session)
    assert sid2 in panel._source_rows
    assert len(panel._source_rows) == 2

    session.remove_source(sid1)
    panel.refresh_all(session)
    assert sid1 not in panel._source_rows
    assert len(panel._source_rows) == 1

    panel.close()


# ---------------------------------------------------------------------------
# 17–18. Alignment badge text
# ---------------------------------------------------------------------------


def test_alignment_badge_high_confidence(qapp) -> None:
    source = _make_session_source(
        alignment_method="auto_trigger", alignment_confidence=0.91
    )
    panels = _make_panels()
    analog, digital = _make_channels()
    row = SourceRowWidget(source, analog, digital, panels)
    text = row._alignment_badge.text()
    assert "High" in text
    assert "0.91" in text
    row.close()


def test_alignment_badge_none_method_is_empty(qapp) -> None:
    source = _make_session_source(alignment_method="none")
    panels = _make_panels()
    analog, digital = _make_channels()
    row = SourceRowWidget(source, analog, digital, panels)
    assert row._alignment_badge.text() == ""
    row.close()


# ---------------------------------------------------------------------------
# 19–20. Fine nudge buttons
# ---------------------------------------------------------------------------


def test_fine_left_nudge_decrements_by_sample_interval(qapp) -> None:
    source = _make_session_source(time_offset_s=1.0)
    panels = _make_panels()
    analog, digital = _make_channels()
    metrics = _make_metrics(rate_hz=1000.0)
    row = SourceRowWidget(source, analog, digital, panels, metrics=metrics)

    emitted: list[float] = []
    row.offset_changed.connect(lambda _sid, val: emitted.append(val))

    row._fine_left_btn.click()
    assert len(emitted) == 1
    assert abs(emitted[0] - (1.0 - 0.001)) < 1e-9
    row.close()


def test_fine_right_nudge_increments_by_sample_interval(qapp) -> None:
    source = _make_session_source(time_offset_s=0.0)
    panels = _make_panels()
    analog, digital = _make_channels()
    metrics = _make_metrics(rate_hz=500.0)
    row = SourceRowWidget(source, analog, digital, panels, metrics=metrics)

    emitted: list[float] = []
    row.offset_changed.connect(lambda _sid, val: emitted.append(val))

    row._fine_right_btn.click()
    assert len(emitted) == 1
    expected_step = 1.0 / 500.0
    assert abs(emitted[0] - expected_step) < 1e-9
    row.close()


# ---------------------------------------------------------------------------
# 21–23. ChannelTreeWidget
# ---------------------------------------------------------------------------


def test_channel_tree_populate_item_counts(qapp) -> None:
    analog, digital = _make_channels()
    tree = ChannelTreeWidget("src-1")
    panels = _make_panels()
    tree.populate(analog, digital, panels)
    assert len(tree._channel_items) == 5  # 3 analog + 2 digital


def test_channel_tree_update_visibility_no_signal(qapp) -> None:
    analog, digital = _make_channels()
    tree = ChannelTreeWidget("src-1")
    panels = _make_panels()
    tree.populate(analog, digital, panels)

    emitted: list = []
    tree.channel_visibility_changed.connect(lambda *args: emitted.append(args))

    tree.update_channel_visibility("VA0", False)
    item = tree._channel_items["VA0"]
    assert item.checkState(0) == Qt.CheckState.Unchecked
    assert len(emitted) == 0   # no signal emitted during programmatic update


def test_channel_tree_update_panel_choices_no_signal(qapp) -> None:
    analog, digital = _make_channels()
    tree = ChannelTreeWidget("src-1")
    panels = _make_panels()
    tree.populate(analog, digital, panels)

    emitted: list = []
    tree.channel_panel_changed.connect(lambda *args: emitted.append(args))

    new_panels = [
        PanelConfig(panel_id="voltage", title="Voltage"),
        PanelConfig(panel_id="power", title="Power"),
    ]
    tree.update_panel_choices(new_panels)
    combo = tree._panel_combos["VA0"]
    assert combo.count() == 3  # 2 panels + "＋ New panel…" sentinel
    assert len(emitted) == 0   # no signal storm


# ---------------------------------------------------------------------------
# 24. SourceRowWidget expand/collapse
# ---------------------------------------------------------------------------


def test_source_row_expand_shows_channel_tree(qapp) -> None:
    source = _make_session_source()
    panels = _make_panels()
    analog, digital = _make_channels()
    row = SourceRowWidget(source, analog, digital, panels)

    assert row._channel_tree.isHidden()   # hidden before expand
    row.set_expanded(True)
    assert not row._channel_tree.isHidden()  # no longer explicitly hidden
    assert row.is_expanded()
    row.set_expanded(False)
    assert row._channel_tree.isHidden()   # hidden again after collapse
    row.close()


# ---------------------------------------------------------------------------
# 25–26. refresh() does not fire spurious signals
# ---------------------------------------------------------------------------


def test_refresh_does_not_fire_offset_changed(qapp) -> None:
    source = _make_session_source(time_offset_s=0.0)
    panels = _make_panels()
    analog, digital = _make_channels()
    row = SourceRowWidget(source, analog, digital, panels)

    emitted: list = []
    row.offset_changed.connect(lambda *a: emitted.append(a))

    updated = _make_session_source(source_id=source.source_id, time_offset_s=5.0)
    row.refresh(updated, None, panels)

    assert len(emitted) == 0
    assert abs(row._offset_spin.value() - 5.0) < 1e-6
    row.close()


def test_refresh_does_not_fire_active_changed(qapp) -> None:
    source = _make_session_source(is_active=True)
    panels = _make_panels()
    analog, digital = _make_channels()
    row = SourceRowWidget(source, analog, digital, panels)

    emitted: list = []
    row.active_changed.connect(lambda *a: emitted.append(a))

    updated = _make_session_source(source_id=source.source_id, is_active=False)
    row.refresh(updated, None, panels)

    assert len(emitted) == 0
    assert not row._active_cb.isChecked()
    row.close()


# ---------------------------------------------------------------------------
# 27–28. Offset spinbox configuration
# ---------------------------------------------------------------------------


def test_offset_spinbox_range_and_precision(qapp) -> None:
    """Millisecond precision, and a range wide enough for absolute alignment.

    Stage 1 raised the bound from ±9999.999 s (2.78 h) to ±366 days: an
    absolute-timestamp offset can legitimately exceed the old limit (the GPTH
    event's COMTRADE record is +4183.806 s from the SCADA trend, and records
    hours apart go further), and a too-narrow spinbox silently clamped the
    displayed value away from the true SessionSource.time_offset_s.
    """
    source = _make_session_source()
    panels = _make_panels()
    analog, digital = _make_channels()
    row = SourceRowWidget(source, analog, digital, panels)
    spin = row._offset_spin
    assert spin.minimum() == pytest.approx(-31_622_400.0, abs=1e-6)
    assert spin.maximum() == pytest.approx(31_622_400.0, abs=1e-6)
    assert spin.decimals() == 3
    # A real absolute-alignment offset must survive a round trip unclamped.
    spin.setValue(4183.805733)
    assert spin.value() == pytest.approx(4183.806, abs=1e-6)
    row.close()


def test_offset_spinbox_step_is_one_millisecond(qapp) -> None:
    source = _make_session_source()
    panels = _make_panels()
    analog, digital = _make_channels()
    row = SourceRowWidget(source, analog, digital, panels)
    assert row._offset_spin.singleStep() == pytest.approx(0.001, abs=1e-9)
    row.close()


# ---------------------------------------------------------------------------
# Phase 9C tests (29–36)
# ---------------------------------------------------------------------------


def test_reset_button_emits_offset_reset_requested(qapp) -> None:
    """29. Reset button emits offset_reset_requested(source_id)."""
    source = _make_session_source(source_id="src-reset")
    panels = _make_panels()
    analog, digital = _make_channels("src-reset")
    row = SourceRowWidget(source, analog, digital, panels)

    received: list[str] = []
    row.offset_reset_requested.connect(received.append)
    row._reset_btn.click()

    assert received == ["src-reset"]
    row.close()


def test_align_all_button_emits_auto_align_all(qapp) -> None:
    """30. Align All toolbar button emits auto_align_requested('all')."""
    panel = SessionPanel()
    received: list[str] = []
    panel.auto_align_requested.connect(received.append)
    panel._align_all_btn.click()
    assert received == ["all"]
    panel.close()


def test_offset_reset_requested_re_emitted_by_session_panel(qapp) -> None:
    """31. SessionPanel re-emits offset_reset_requested from source rows."""
    source = _make_session_source(source_id="src-re")
    panels = _make_panels()
    analog, digital = _make_channels("src-re")

    panel = SessionPanel()
    panel.add_source_row(source, analog, digital, panels)

    received: list[str] = []
    panel.offset_reset_requested.connect(received.append)
    panel._source_rows["src-re"]._reset_btn.click()
    assert received == ["src-re"]
    panel.close()


def test_alignment_badge_notes_tooltip(qapp) -> None:
    """32. Badge tooltip shows alignment notes when provided."""
    source = _make_session_source(
        source_id="src-notes",
        alignment_method="auto_trigger",
        alignment_confidence=0.85,
    )
    panels = _make_panels()
    analog, digital = _make_channels("src-notes")
    row = SourceRowWidget(source, analog, digital, panels)
    notes = "Trigger at t=0.0123 s (confidence 0.85)."
    row._apply_alignment(source, notes=notes)
    assert row._alignment_badge.toolTip() == notes
    row.close()


def test_alignment_badge_default_tooltip_when_no_notes(qapp) -> None:
    """33. Badge tooltip falls back to generic text when notes is empty."""
    source = _make_session_source(
        source_id="src-notip",
        alignment_method="auto_trigger",
        alignment_confidence=0.9,
    )
    panels = _make_panels()
    analog, digital = _make_channels("src-notip")
    row = SourceRowWidget(source, analog, digital, panels)
    row._apply_alignment(source, notes="")
    assert row._alignment_badge.toolTip() == "Alignment method and confidence"
    row.close()


def test_alignment_badge_manual_method(qapp) -> None:
    """34. Manual method with no confidence shows the method label in grey.

    Stage 3 renders the human-readable label ("Manual") rather than the
    internal identifier ("manual"), so an absolute-timestamp source reads
    "Absolute timestamp" instead of "absolute_timestamp".
    """
    source = _make_session_source(
        source_id="src-manual",
        alignment_method="manual",
        alignment_confidence=None,
    )
    panels = _make_panels()
    analog, digital = _make_channels("src-manual")
    row = SourceRowWidget(source, analog, digital, panels)
    assert "Manual" in row._alignment_badge.text()
    assert "#888888" in row._alignment_badge.styleSheet()
    row.close()


def test_alignment_badge_absolute_timestamp_method(qapp) -> None:
    """Stage 3: the internal identifier never reaches the engineer."""
    source = _make_session_source(
        source_id="src-abs",
        alignment_method="absolute_timestamp",
        alignment_confidence=None,
    )
    panels = _make_panels()
    analog, digital = _make_channels("src-abs")
    row = SourceRowWidget(source, analog, digital, panels)
    text = row._alignment_badge.text()
    assert "Absolute timestamp" in text
    assert "absolute_timestamp" not in text
    row.close()


def test_alignment_badge_low_confidence(qapp) -> None:
    """35. Confidence < 0.40 shows 'Low' in red."""
    source = _make_session_source(
        source_id="src-low",
        alignment_method="auto_trigger",
        alignment_confidence=0.20,
    )
    panels = _make_panels()
    analog, digital = _make_channels("src-low")
    row = SourceRowWidget(source, analog, digital, panels)
    assert "Low" in row._alignment_badge.text()
    assert "#cc3333" in row._alignment_badge.styleSheet()
    row.close()


def test_add_source_row_passes_alignment_notes_to_badge(qapp) -> None:
    """36. add_source_row with alignment_notes sets badge tooltip."""
    source = _make_session_source(
        source_id="src-note-add",
        alignment_method="auto_trigger",
        alignment_confidence=0.80,
    )
    panels = _make_panels()
    analog, digital = _make_channels("src-note-add")

    panel = SessionPanel()
    notes = "Trigger at 0.005 s. Suggested offset: -0.0050 s."
    panel.add_source_row(source, analog, digital, panels, alignment_notes=notes)

    row = panel._source_rows["src-note-add"]
    assert row._alignment_badge.toolTip() == notes
    panel.close()
