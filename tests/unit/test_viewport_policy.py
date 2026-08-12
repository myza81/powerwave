"""Stage 2 — event-focused initial viewport for mixed-rate multi-source sessions.

Coverage map (numbers match the Stage 2 test requirements):

Core viewport policy
  1  short 5000 Hz record inside a 60 s sparse record
  2  real-event timing values produce approximately 4137 -> 4203
  3  both bracketing Excel samples are included
  4  viewport contains the entire COMTRADE event
  5  full _session_window() remains unchanged
  6  waveform_data["time"] remains unchanged
  7  time_offset_s remains unchanged

Load order
  8  Excel -> COMTRADE
  9  COMTRADE -> Excel                     (equivalent initial viewport)

Single source
 10  COMTRADE only -> no event-focus policy
 11  Excel only    -> no event-focus policy

Similar records
 12  COMTRADE + COMTRADE similar duration -> benefit guard returns None

No trigger
 13  Excel + Excel -> None
 14  trigger == start is never treated as an event

Event-domain mismatch
 15  event outside another source -> that source is skipped
 16  viewport remains finite and valid

Multiple sources
 17  COMTRADE + PMU + sparse trend
 18  highest qualifying sample-rate source anchors
 19  each in-range source can enlarge bounds through real neighbouring samples

Interaction
 20  initial viewport applied across all synchronized canvases
 21  Fit All restores _session_window()
 22  Fit All does not alter offsets
 23  zoom/pan after initialization is not overwritten by an automatic re-snap
 24  adding a source after analyst zoom does not discard that zoom

Regression
 25  _session_window() golden values unchanged
 26  RMS/phasor/harmonic/navigator still receive the full session domain
 27  Stage 1 absolute origin unchanged
 28  Stage 1 alignment methods unchanged
 29  Set-as-Reference behaviour unchanged
 30  calculated-signal alignment behaviour unchanged

Fixtures carry the real timing values of the validated GPTH event with
synthetic waveform arrays. No confidential event file is committed.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.models.channels import AnalogChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.event_session import EventAnalysisSession
from app.visualization import viewport_policy
from app.visualization.viewport_policy import (
    MAX_BENEFIT_FRACTION,
    nearest_sample_bounds,
    select_initial_viewport,
    time_extent,
    trigger_display_time,
)

EXCEL_START = datetime(2026, 7, 25, 12, 0, 0)
COMTRADE_START = datetime(2026, 7, 25, 13, 9, 43, 805733)
COMTRADE_TRIGGER = datetime(2026, 7, 25, 13, 9, 44, 305733)
COMTRADE_DURATION_S = 7.0198
COMTRADE_OFFSET = 4183.805733

EXCEL_1309_X = 4140.0
EXCEL_1310_X = 4200.0


def _record(
    *,
    start: datetime,
    trigger: datetime | None = None,
    n: int,
    dt: float,
    rate: float | None = None,
    name: str = "Va",
    unit: str = "kV",
) -> DisturbanceRecord:
    t = np.arange(n, dtype=np.float64) * dt
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST",
            recorder_name="TEST",
            source_file="test",
            provider_type="test",
            nominal_frequency=50.0,
        ),
        waveform_data=pd.DataFrame({"time": t, name: np.sin(t)}),
        analog_channels=[AnalogChannel(name=name, unit=unit, index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[rate if rate is not None else (1.0 / dt if dt else 0.0)],
            samples_per_rate=[n],
        ),
        timing_info=TimingInformation(
            start_time=start,
            trigger_time=trigger if trigger is not None else start,
            timing_reference="absolute",
        ),
    )


def _excel_record(start: datetime = EXCEL_START, n: int = 121) -> DisturbanceRecord:
    """121 samples at 60 s — the validated SCADA export geometry, no real trigger."""
    return _record(start=start, n=n, dt=60.0, name="mw_total", unit="MW")


def _comtrade_record(
    start: datetime = COMTRADE_START,
    trigger: datetime | None = COMTRADE_TRIGGER,
) -> DisturbanceRecord:
    """7.0198 s at 5000 Hz declared, trigger +0.5 s from start."""
    n = 3510
    dt = COMTRADE_DURATION_S / (n - 1)
    return _record(start=start, trigger=trigger, n=n, dt=dt, rate=5000.0)


def _pmu_record(start: datetime, seconds: float = 120.0) -> DisturbanceRecord:
    """50 fps PMU stream, no independent trigger."""
    n = int(seconds * 50)
    return _record(start=start, n=n, dt=0.02, rate=50.0, name="freq", unit="Hz")


def _event_session() -> tuple[EventAnalysisSession, str, str]:
    session = EventAnalysisSession()
    excel_id = session.add_source(_excel_record(), "Excel", "normalized_excel")
    comtrade_id = session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.apply_absolute_alignment()
    return session, excel_id, comtrade_id


# ---------------------------------------------------------------------------
# 1-7 — core policy on the validated geometry
# ---------------------------------------------------------------------------


def test_1_short_high_rate_record_inside_sparse_record() -> None:
    session, _, _ = _event_session()
    window = select_initial_viewport(session)

    assert window is not None
    lo, hi = window
    assert hi - lo == pytest.approx(66.0, abs=1e-6)   # 60 s logical + 5% each side


def test_2_real_event_timing_produces_expected_window() -> None:
    session, _, _ = _event_session()
    lo, hi = select_initial_viewport(session)

    assert lo == pytest.approx(4137.0, abs=1e-6)
    assert hi == pytest.approx(4203.0, abs=1e-6)

    # Same window expressed as wall clock, via the Stage 1 origin.
    origin = session.absolute_time_origin
    assert origin + timedelta(seconds=lo) == datetime(2026, 7, 25, 13, 8, 57)
    assert origin + timedelta(seconds=hi) == datetime(2026, 7, 25, 13, 10, 3)


def test_3_both_bracketing_excel_samples_are_included() -> None:
    session, _, _ = _event_session()
    lo, hi = select_initial_viewport(session)

    assert lo < EXCEL_1309_X < hi
    assert lo < EXCEL_1310_X < hi


def test_4_viewport_contains_the_entire_comtrade_event() -> None:
    session, _, comtrade_id = _event_session()
    lo, hi = select_initial_viewport(session)

    aligned = session.build_aligned_data(comtrade_id, "Va", -1e9, 1e9)
    assert lo < aligned.time[0]
    assert aligned.time[-1] < hi
    trigger_x = COMTRADE_OFFSET + 0.5
    assert lo < trigger_x < hi


def test_5_and_25_session_window_is_unchanged_by_viewport_selection() -> None:
    """The DATA DOMAIN must not follow the display choice. Golden values."""
    from app.ui.session.session_canvas_controller import _session_window

    session, _, _ = _event_session()
    before = _session_window(session)
    select_initial_viewport(session)
    after = _session_window(session)

    assert before == after
    # Excel 0..7200 union COMTRADE 4183.8..4190.8, plus the 2% display margin.
    assert after[0] == pytest.approx(-144.0, abs=1e-6)
    assert after[1] == pytest.approx(7344.0, abs=1e-6)


def test_6_waveform_time_arrays_unchanged() -> None:
    excel_rec, comtrade_rec = _excel_record(), _comtrade_record()
    excel_before = excel_rec.waveform_data["time"].to_numpy(copy=True)
    comtrade_before = comtrade_rec.waveform_data["time"].to_numpy(copy=True)

    session = EventAnalysisSession()
    session.add_source(excel_rec, "Excel", "normalized_excel")
    session.add_source(comtrade_rec, "COMTRADE", "comtrade")
    session.apply_absolute_alignment()
    select_initial_viewport(session)

    np.testing.assert_array_equal(excel_rec.waveform_data["time"].to_numpy(), excel_before)
    np.testing.assert_array_equal(
        comtrade_rec.waveform_data["time"].to_numpy(), comtrade_before
    )


def test_7_time_offsets_unchanged_by_viewport_selection() -> None:
    session, excel_id, comtrade_id = _event_session()
    before = {s.source_id: (s.time_offset_s, s.alignment_method) for s in session.list_sources()}

    select_initial_viewport(session)

    after = {s.source_id: (s.time_offset_s, s.alignment_method) for s in session.list_sources()}
    assert after == before
    assert session.get_time_offset(comtrade_id) == pytest.approx(COMTRADE_OFFSET, abs=1e-9)
    assert session.get_time_offset(excel_id) == 0.0


# ---------------------------------------------------------------------------
# 8-9 — load order
# ---------------------------------------------------------------------------


def test_8_9_load_order_produces_equivalent_viewport() -> None:
    forward, _, _ = _event_session()

    reverse = EventAnalysisSession()
    reverse.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    reverse.apply_absolute_alignment()
    reverse.add_source(_excel_record(), "Excel", "normalized_excel")
    reverse.apply_absolute_alignment()

    assert select_initial_viewport(forward) == select_initial_viewport(reverse)


# ---------------------------------------------------------------------------
# 10-11 — single source
# ---------------------------------------------------------------------------


def test_10_comtrade_only_gets_no_event_focus() -> None:
    session = EventAnalysisSession()
    session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.apply_absolute_alignment()

    assert select_initial_viewport(session) is None


def test_11_excel_only_gets_no_event_focus() -> None:
    session = EventAnalysisSession()
    session.add_source(_excel_record(), "Excel", "normalized_excel")
    session.apply_absolute_alignment()

    assert select_initial_viewport(session) is None


def test_10b_inactive_source_makes_it_single_source_again() -> None:
    session, excel_id, _ = _event_session()
    assert select_initial_viewport(session) is not None

    session.set_source_active(excel_id, False)
    assert select_initial_viewport(session) is None


# ---------------------------------------------------------------------------
# 12 — benefit guard
# ---------------------------------------------------------------------------


def test_12_two_similar_comtrade_records_return_none() -> None:
    """Two 7 s records of the same event: focusing buys nothing."""
    session = EventAnalysisSession()
    session.add_source(_comtrade_record(), "A", "comtrade")
    session.add_source(
        _comtrade_record(
            start=COMTRADE_START + timedelta(seconds=3),
            trigger=COMTRADE_TRIGGER + timedelta(seconds=3),
        ),
        "B",
        "comtrade",
    )
    session.apply_absolute_alignment()

    assert select_initial_viewport(session) is None


def test_12b_benefit_guard_threshold_is_the_documented_rule() -> None:
    session, _, _ = _event_session()
    # Domain is ~7200 s and the logical window 60 s, so a threshold small
    # enough to make 60/7200 "not beneficial" must suppress focusing.
    assert select_initial_viewport(session, max_benefit_fraction=1e-4) is None
    assert select_initial_viewport(session, max_benefit_fraction=MAX_BENEFIT_FRACTION) is not None


# ---------------------------------------------------------------------------
# 13-14 — no trigger
# ---------------------------------------------------------------------------


def test_13_two_long_trends_return_none() -> None:
    session = EventAnalysisSession()
    session.add_source(_excel_record(), "TrendA", "normalized_excel")
    session.add_source(
        _excel_record(start=EXCEL_START + timedelta(minutes=30)), "TrendB", "normalized_excel"
    )
    session.apply_absolute_alignment()

    assert select_initial_viewport(session) is None


def test_14_trigger_equal_to_start_is_not_an_event_marker() -> None:
    """The Import Wizard forces trigger_time == start_time for CSV/Excel."""
    excel = _excel_record()
    assert excel.timing_info.trigger_time == excel.timing_info.start_time
    assert trigger_display_time(excel, 0.0) is None

    comtrade = _comtrade_record()
    assert trigger_display_time(comtrade, COMTRADE_OFFSET) == pytest.approx(
        COMTRADE_OFFSET + 0.5, abs=1e-9
    )


def test_14b_sparse_trend_never_anchors_the_viewport_at_its_first_sample() -> None:
    """Regression for the failure mode the trigger==start rule prevents."""
    session = EventAnalysisSession()
    session.add_source(_excel_record(), "Excel", "normalized_excel")
    session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.apply_absolute_alignment()

    lo, hi = select_initial_viewport(session)
    assert lo > 4000.0, "viewport anchored near the trend's first sample, not the event"


# ---------------------------------------------------------------------------
# 15-16 — event outside another source
# ---------------------------------------------------------------------------


def test_15_16_source_not_spanning_the_event_is_skipped() -> None:
    """A trend recorded 14:00-16:00 must not stretch a 13:09 event viewport."""
    session = EventAnalysisSession()
    session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.add_source(
        _excel_record(start=datetime(2026, 7, 25, 14, 0, 0)), "LaterTrend", "normalized_excel"
    )
    session.apply_absolute_alignment()

    window = select_initial_viewport(session)
    assert window is not None
    lo, hi = window
    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi > lo
    # Anchored on the COMTRADE extent only; nowhere near the 14:00 trend.
    assert hi - lo == pytest.approx(COMTRADE_DURATION_S * 1.1, abs=1e-3)
    assert hi < 3000.0


# ---------------------------------------------------------------------------
# 17-19 — three or more sources
# ---------------------------------------------------------------------------


def test_17_18_19_three_source_mixed_rate_session() -> None:
    """COMTRADE 5000 Hz + PMU 50 fps + 60 s trend."""
    session = EventAnalysisSession()
    session.add_source(_excel_record(), "Trend", "normalized_excel")
    # PMU covering 13:09:00 -> 13:11:00, so the event falls inside it.
    session.add_source(
        _pmu_record(datetime(2026, 7, 25, 13, 9, 0), seconds=120.0), "PMU", "normalized_csv"
    )
    session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.apply_absolute_alignment()

    window = select_initial_viewport(session)
    assert window is not None
    lo, hi = window

    # 18: the 5000 Hz source anchors (both PMU and trend lack a real trigger,
    #     and even with one the COMTRADE rate is highest).
    trigger_x = COMTRADE_OFFSET + 0.5
    assert lo < trigger_x < hi

    # 19: both in-range sources enlarged the bounds through real samples.
    #     The trend's bracketing samples are the widest contributor.
    assert lo < EXCEL_1309_X and EXCEL_1310_X < hi
    assert hi - lo == pytest.approx(66.0, abs=1e-6)


def test_19b_bracketing_uses_real_samples_only() -> None:
    excel = _excel_record()
    bounds = nearest_sample_bounds(excel, COMTRADE_OFFSET + 0.5, 0.0)

    assert bounds == (EXCEL_1309_X, EXCEL_1310_X)
    # Both values are genuine recorded sample positions, not interpolated.
    t = excel.waveform_data["time"].to_numpy()
    assert bounds[0] in t and bounds[1] in t


def test_time_extent_applies_the_scalar_offset() -> None:
    comtrade = _comtrade_record()
    assert time_extent(comtrade, 0.0) == pytest.approx((0.0, COMTRADE_DURATION_S), abs=1e-9)
    assert time_extent(comtrade, COMTRADE_OFFSET) == pytest.approx(
        (COMTRADE_OFFSET, COMTRADE_OFFSET + COMTRADE_DURATION_S), abs=1e-6
    )


def test_effective_sample_rate_prefers_declared_rate() -> None:
    assert viewport_policy.effective_sample_rate(_comtrade_record()) == 5000.0
    assert viewport_policy.effective_sample_rate(_pmu_record(EXCEL_START, 10.0)) == 50.0


# ---------------------------------------------------------------------------
# 20-24 — controller interaction (Qt)
# ---------------------------------------------------------------------------


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _built_controller(session):
    from app.ui.session.session_canvas_controller import SessionCanvasController

    ctrl = SessionCanvasController()
    session.default_layout()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)
    return ctrl


def test_20_initial_viewport_applied_to_all_synchronized_canvases(qapp) -> None:
    session, _, _ = _event_session()
    ctrl = _built_controller(session)

    window = ctrl.apply_initial_viewport(session)

    assert window is not None
    assert len(ctrl._canvases) >= 2, "expected separate panels for MW and voltage"
    for pid, canvas in ctrl._canvases.items():
        x_range = canvas.getViewBox().viewRange()[0]
        assert x_range[0] == pytest.approx(window[0], abs=1e-3), pid
        assert x_range[1] == pytest.approx(window[1], abs=1e-3), pid


def test_21_22_fit_all_restores_full_domain_without_touching_offsets(qapp) -> None:
    from app.ui.session.session_canvas_controller import _session_window

    session, excel_id, comtrade_id = _event_session()
    ctrl = _built_controller(session)
    ctrl.apply_initial_viewport(session)
    offsets_before = {s.source_id: s.time_offset_s for s in session.list_sources()}

    ctrl.normalize_all_to_session_window(session)

    domain = _session_window(session)
    for canvas in ctrl._canvases.values():
        x_range = canvas.getViewBox().viewRange()[0]
        assert x_range[0] == pytest.approx(domain[0], abs=1e-3)
        assert x_range[1] == pytest.approx(domain[1], abs=1e-3)
    assert {s.source_id: s.time_offset_s for s in session.list_sources()} == offsets_before


def test_23_analyst_zoom_is_not_re_snapped(qapp) -> None:
    session, _, _ = _event_session()
    ctrl = _built_controller(session)
    ctrl.apply_initial_viewport(session)

    # Analyst zooms somewhere of their own choosing.
    for canvas in ctrl._canvases.values():
        canvas.normalize_viewport(4184.0, 4186.0)

    # A second activation must not pull the view back.
    assert ctrl.apply_initial_viewport(session) is None
    for canvas in ctrl._canvases.values():
        x_range = canvas.getViewBox().viewRange()[0]
        assert x_range[0] == pytest.approx(4184.0, abs=1e-3)
        assert x_range[1] == pytest.approx(4186.0, abs=1e-3)


def test_24_adding_a_source_after_zoom_preserves_the_zoom(qapp) -> None:
    session, _, _ = _event_session()
    ctrl = _built_controller(session)
    ctrl.apply_initial_viewport(session)
    for canvas in ctrl._canvases.values():
        canvas.normalize_viewport(4184.0, 4186.0)

    session.add_source(
        _record(start=datetime(2026, 7, 25, 13, 9, 50), n=100, dt=0.001, rate=1000.0,
                name="Ia", unit="kA"),
        "Third",
        "comtrade",
    )
    session.apply_absolute_alignment()
    session.default_layout()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)
    ctrl.apply_initial_viewport(session)

    for pid, canvas in ctrl._canvases.items():
        x_range = canvas.getViewBox().viewRange()[0]
        assert x_range[0] == pytest.approx(4184.0, abs=1e-3), pid
        assert x_range[1] == pytest.approx(4186.0, abs=1e-3), pid


def test_23b_single_source_session_still_fits_full_domain(qapp) -> None:
    from app.ui.session.session_canvas_controller import _session_window

    session = EventAnalysisSession()
    session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.apply_absolute_alignment()
    ctrl = _built_controller(session)

    assert ctrl.apply_initial_viewport(session) is None
    domain = _session_window(session)
    for canvas in ctrl._canvases.values():
        x_range = canvas.getViewBox().viewRange()[0]
        assert x_range[0] == pytest.approx(domain[0], abs=1e-3)
        assert x_range[1] == pytest.approx(domain[1], abs=1e-3)
    # Never marked initialised, so a later second source still gets a decision.
    assert ctrl._viewport_initialized_session is None


def test_23c_second_source_after_single_source_activation_gets_event_focus(qapp) -> None:
    session = EventAnalysisSession()
    session.add_source(_excel_record(), "Excel", "normalized_excel")
    session.apply_absolute_alignment()
    ctrl = _built_controller(session)
    assert ctrl.apply_initial_viewport(session) is None

    session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.apply_absolute_alignment()
    session.default_layout()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    window = ctrl.apply_initial_viewport(session)
    assert window is not None
    assert window[0] == pytest.approx(4137.0, abs=1e-6)
    assert window[1] == pytest.approx(4203.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 26-30 — regression against Stage 1 and analytics consumers
# ---------------------------------------------------------------------------


def test_26_overlays_and_navigator_receive_the_full_domain(qapp) -> None:
    """RMS/phasor/harmonic/navigator bounds come from _session_window(), not
    from the initial viewport."""
    from app.ui.session import session_canvas_controller as scc

    session, _, _ = _event_session()
    ctrl = _built_controller(session)
    ctrl.apply_initial_viewport(session)

    seen: list[tuple[float, float]] = []
    real_window = scc._session_window

    def _spy(sess):
        result = real_window(sess)
        seen.append(result)
        return result

    scc._session_window = _spy
    try:
        ctrl._refresh_rms_overlays(session)
        ctrl._refresh_phasor_overlays(session)
        ctrl._refresh_harmonic_overlays(session)
        ctrl._populate_navigator(session)
        ctrl.refresh_all(session)
    finally:
        scc._session_window = real_window

    domain = real_window(session)
    assert seen, "no consumer queried the session window"
    assert all(w == domain for w in seen)
    assert domain[0] == pytest.approx(-144.0, abs=1e-6)
    assert domain[1] == pytest.approx(7344.0, abs=1e-6)


def test_27_28_stage_1_state_unchanged_by_viewport_work() -> None:
    session, excel_id, comtrade_id = _event_session()
    origin_before = session.absolute_time_origin
    methods_before = {s.source_id: s.alignment_method for s in session.list_sources()}

    select_initial_viewport(session)

    assert session.absolute_time_origin == origin_before == EXCEL_START
    assert {s.source_id: s.alignment_method for s in session.list_sources()} == methods_before
    assert methods_before[comtrade_id] == "absolute_timestamp"


def test_29_set_as_reference_unchanged_and_viewport_follows() -> None:
    session, excel_id, comtrade_id = _event_session()

    ref_offset = session.get_time_offset(comtrade_id)
    session.rebase_absolute_time_origin(ref_offset)
    for source in session.list_sources():
        new_offset = 0.0 if source.source_id == comtrade_id else source.time_offset_s - ref_offset
        session.set_time_offset(
            source.source_id, new_offset,
            method=source.alignment_method, confidence=source.alignment_confidence,
        )

    assert session.absolute_time_origin == COMTRADE_START
    assert session.get_source(comtrade_id).alignment_method == "absolute_timestamp"

    lo, hi = select_initial_viewport(session)
    # Same physical window, expressed in the rebased coordinate system.
    assert lo == pytest.approx(4137.0 - COMTRADE_OFFSET, abs=1e-6)
    assert hi == pytest.approx(4203.0 - COMTRADE_OFFSET, abs=1e-6)
    origin = session.absolute_time_origin
    assert origin + timedelta(seconds=lo) == datetime(2026, 7, 25, 13, 8, 57)
    assert origin + timedelta(seconds=hi) == datetime(2026, 7, 25, 13, 10, 3)


def test_30_calculated_signal_alignment_behaviour_unchanged() -> None:
    from app.calculated_signals.models import (
        CalculatedSignalDefinition,
        CalculatedSignalResult,
        CalculationStatus,
        ChannelRef,
    )
    from datetime import timezone

    session, _, comtrade_id = _event_session()
    definition = CalculatedSignalDefinition(
        calc_id="calc-stage2",
        name="calc",
        expression="a * 2",
        variable_bindings={"a": ChannelRef(source_id=comtrade_id, channel_name="Va")},
        reference_variable="a",
    )
    calc_id = session.add_calculated_signal(
        definition,
        CalculatedSignalResult(
            calc_id=definition.calc_id,
            time=np.array([0.0, 1.0]),
            values=np.array([1.0, 2.0]),
            validity_mask=np.array([True, True]),
            unit="kV",
            status=CalculationStatus.OK,
            error_message=None,
            computed_at=datetime(2026, 7, 25, 14, 0, tzinfo=timezone.utc),
        ),
    )

    select_initial_viewport(session)

    assert session.get_calculated_signal_result(calc_id).status == CalculationStatus.OK


def test_policy_never_mutates_the_session() -> None:
    session, _, _ = _event_session()
    snapshot = [
        (s.source_id, s.time_offset_s, s.alignment_method, s.is_active)
        for s in session.list_sources()
    ]
    origin = session.absolute_time_origin
    panels = [p.panel_id for p in session.list_panels()]

    select_initial_viewport(session)

    assert [
        (s.source_id, s.time_offset_s, s.alignment_method, s.is_active)
        for s in session.list_sources()
    ] == snapshot
    assert session.absolute_time_origin == origin
    assert [p.panel_id for p in session.list_panels()] == panels
