"""Unit tests for _apply_time_offset and display_multi_source_session (Phase D3).

Uses MagicMock canvas/timeline factories — no QApplication required.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from unittest.mock import MagicMock

import pandas as pd
import numpy as np
import pytest

from app.data.multi_source_session import MultiSourceSession, SourceRecord
from app.data.synthetic import make_high_rate_record, make_low_rate_record
from app.visualization.managers.visualization_manager import (
    _DisplayRecord,
    VisualizationManager,
    _apply_time_offset,
    _select_initial_multi_source_viewport,
)


def _make_manager() -> tuple[VisualizationManager, MagicMock, MagicMock]:
    canvas = MagicMock()
    timeline = MagicMock()
    return VisualizationManager(canvas, timeline), canvas, timeline


def _make_source(source_id: str = "src", start_offset_s: float = 0.0) -> SourceRecord:
    result = make_high_rate_record()
    base = result.record.timing_info.start_time
    import datetime as dt_mod
    start = base + dt_mod.timedelta(seconds=start_offset_s)
    return SourceRecord(
        source_id=source_id,
        provider_type="comtrade",
        record=result.record,
        signal_metadata=result.signal_metadata,
        original_start_time=start,
        sampling_rates=list(result.record.sampling_info.sampling_rates),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestApplyTimeOffset
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyTimeOffset:
    def test_zero_offset_returns_same_object(self) -> None:
        result = make_high_rate_record()
        out = _apply_time_offset(result.record, 0.0)
        assert out is result.record

    def test_nonzero_offset_shifts_time_column(self) -> None:
        result = make_high_rate_record()
        original_t0 = float(result.record.waveform_data["time"].iloc[0])
        shifted = _apply_time_offset(result.record, 5.0)
        shifted_t0 = float(shifted.waveform_data["time"].iloc[0])
        assert shifted_t0 == pytest.approx(original_t0 + 5.0)

    def test_original_record_not_mutated(self) -> None:
        result = make_high_rate_record()
        original_t0 = float(result.record.waveform_data["time"].iloc[0])
        _apply_time_offset(result.record, 3.0)
        assert float(result.record.waveform_data["time"].iloc[0]) == pytest.approx(original_t0)

    def test_metadata_shared_by_reference(self) -> None:
        result = make_high_rate_record()
        shifted = _apply_time_offset(result.record, 1.0)
        assert shifted.metadata is result.record.metadata

    def test_analog_channels_shared_by_reference(self) -> None:
        result = make_high_rate_record()
        shifted = _apply_time_offset(result.record, 1.0)
        assert shifted.analog_channels is result.record.analog_channels

    def test_all_rows_shifted_uniformly(self) -> None:
        result = make_high_rate_record()
        original = result.record.waveform_data["time"].to_numpy(dtype=np.float64)
        shifted = _apply_time_offset(result.record, 2.5)
        new_t = shifted.waveform_data["time"].to_numpy(dtype=np.float64)
        np.testing.assert_array_almost_equal(new_t, original + 2.5)


# ─────────────────────────────────────────────────────────────────────────────
# TestDisplayMultiSourceSession
# ─────────────────────────────────────────────────────────────────────────────


class TestDisplayMultiSourceSession:
    def test_empty_session_returns_empty_dict(self) -> None:
        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        panels = mgr.display_multi_source_session(session, canvas_factory=MagicMock)
        assert panels == {}

    def test_returns_dict(self) -> None:
        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        session.add_source(_make_source("a"))
        panels = mgr.display_multi_source_session(session, canvas_factory=MagicMock)
        assert isinstance(panels, dict)

    def test_panel_keys_include_source_id(self) -> None:
        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        session.add_source(_make_source("src_a"))
        panels = mgr.display_multi_source_session(session, canvas_factory=MagicMock)
        assert all(k.startswith("src_a/") for k in panels)

    def test_two_sources_produce_distinct_panel_keys(self) -> None:
        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        session.add_source(_make_source("src_a"))
        session.add_source(_make_source("src_b"))
        panels = mgr.display_multi_source_session(session, canvas_factory=MagicMock)
        keys = list(panels.keys())
        assert len(keys) == len(set(keys))

    def test_set_record_called_on_each_canvas(self) -> None:
        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        session.add_source(_make_source("a"))
        created: list[MagicMock] = []

        def factory() -> MagicMock:
            m = MagicMock()
            created.append(m)
            return m

        mgr.display_multi_source_session(session, canvas_factory=factory)
        assert len(created) > 0
        for canvas in created:
            canvas.set_record.assert_called_once()

    def test_default_multi_source_axis_mode_is_absolute(self) -> None:
        from app.visualization.axis.datetime_axis import AXIS_MODE_ABSOLUTE

        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        session.add_source(_make_source("a"))
        received_modes: list[str] = []

        def factory() -> MagicMock:
            m = MagicMock()
            m.set_record.side_effect = lambda _record, **kw: received_modes.append(
                kw.get("axis_mode", "")
            )
            return m

        mgr.display_multi_source_session(session, canvas_factory=factory)

        assert received_modes
        assert all(mode == AXIS_MODE_ABSOLUTE for mode in received_modes)

    def test_multi_source_absolute_mode_uses_common_reference_start(self) -> None:
        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        early = _make_source("early", start_offset_s=0.0)
        late = _make_source("late", start_offset_s=2.0)
        session.add_source(early)
        session.add_source(late)
        received_refs: list[dt.datetime] = []

        def factory() -> MagicMock:
            m = MagicMock()
            m.set_record.side_effect = lambda _record, **kw: received_refs.append(
                kw.get("axis_reference_time")
            )
            return m

        mgr.display_multi_source_session(session, canvas_factory=factory)

        assert received_refs
        assert all(ref == early.original_start_time for ref in received_refs)

    def test_panel_canvases_property_updated(self) -> None:
        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        session.add_source(_make_source("a"))
        panels = mgr.display_multi_source_session(session, canvas_factory=MagicMock)
        assert mgr.panel_canvases == panels

    def test_record_property_set_to_first_source(self) -> None:
        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        src1 = _make_source("first")
        src2 = _make_source("second")
        session.add_source(src1)
        session.add_source(src2)
        mgr.display_multi_source_session(session, canvas_factory=MagicMock)
        assert mgr.record is src1.record

    def test_time_offset_applied_for_later_source(self) -> None:
        """Second source starting 2 s later should have its time column shifted."""
        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        session.add_source(_make_source("early", start_offset_s=0.0))
        session.add_source(_make_source("late", start_offset_s=2.0))

        panels = mgr.display_multi_source_session(session, canvas_factory=MagicMock)

        early_keys = [k for k in panels if k.startswith("early/")]
        late_keys = [k for k in panels if k.startswith("late/")]
        assert len(early_keys) > 0
        assert len(late_keys) > 0

        # Inspect the first panel of each source for its shifted t=0
        early_record = panels[early_keys[0]].set_record.call_args[0][0]
        late_record = panels[late_keys[0]].set_record.call_args[0][0]
        t0_early = float(early_record.waveform_data["time"].iloc[0])
        t0_late = float(late_record.waveform_data["time"].iloc[0])

        # "late" source is 2 s after "early", so t0_late ≈ t0_early + 2.0
        assert t0_late == pytest.approx(t0_early + 2.0, abs=0.01)

    def test_mixed_source_session_with_high_and_low_rate(self) -> None:
        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        hi = make_high_rate_record()
        lo = make_low_rate_record()
        session.add_source(SourceRecord(
            source_id="hi",
            provider_type="comtrade",
            record=hi.record,
            signal_metadata=hi.signal_metadata,
            original_start_time=hi.record.timing_info.start_time,
            sampling_rates=[6400.0],
        ))
        session.add_source(SourceRecord(
            source_id="lo",
            provider_type="csv",
            record=lo.record,
            signal_metadata=lo.signal_metadata,
            original_start_time=lo.record.timing_info.start_time,
            sampling_rates=[100.0],
        ))
        panels = mgr.display_multi_source_session(session, canvas_factory=MagicMock)
        assert any("hi/" in k for k in panels)
        assert any("lo/" in k for k in panels)

    def test_initial_viewport_uses_aligned_high_rate_event(self) -> None:
        mgr, _, _ = _make_manager()
        session = MultiSourceSession()
        hi = make_high_rate_record()
        lo = make_low_rate_record()
        base = hi.record.timing_info.start_time

        session.add_source(SourceRecord(
            source_id="lo",
            provider_type="csv",
            record=lo.record,
            signal_metadata=lo.signal_metadata,
            original_start_time=base,
            sampling_rates=[100.0],
        ))
        session.add_source(SourceRecord(
            source_id="hi",
            provider_type="comtrade",
            record=hi.record,
            signal_metadata=hi.signal_metadata,
            original_start_time=base + dt.timedelta(seconds=100.0),
            sampling_rates=[6400.0],
        ))

        panels = mgr.display_multi_source_session(session, canvas_factory=MagicMock)
        hi_panel = next(canvas for key, canvas in panels.items() if key.startswith("hi/"))
        t_start, t_end = hi_panel._primary_plot.setXRange.call_args[0]

        assert t_start > 90.0
        assert t_start < 101.0 < t_end


class TestInitialMultiSourceViewport:
    def test_event_view_expands_to_neighboring_low_rate_samples(self) -> None:
        hi = make_high_rate_record()
        high_display = _apply_time_offset(hi.record, 100.0)

        low_df = pd.DataFrame({
            "time": [0.0, 60.0, 120.0],
            "MW": [1.0, 2.0, 3.0],
        })
        low_record = dataclasses.replace(
            hi.record,
            waveform_data=low_df,
            analog_channels=[
                dataclasses.replace(hi.record.analog_channels[0], name="MW", unit="MW")
            ],
        )

        viewport = _select_initial_multi_source_viewport([
            _DisplayRecord(high_display, "hi", 100.0, 6400.0),
            _DisplayRecord(low_record, "lo", 0.0, 1.0 / 60.0),
        ])

        assert viewport is not None
        t_start, t_end = viewport
        assert t_start < 60.0
        assert t_start < 101.0 < t_end
        assert t_end > 120.0

    def test_without_usable_trigger_falls_back_to_full_extent(self) -> None:
        hi = make_high_rate_record()
        display = _apply_time_offset(hi.record, 100.0)

        viewport = _select_initial_multi_source_viewport([
            _DisplayRecord(display, "hi", 0.0, 6400.0),
        ])

        assert viewport is not None
        t_start, t_end = viewport
        assert t_start < 100.0
        assert t_end > float(display.waveform_data["time"].iloc[-1])
