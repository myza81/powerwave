"""Stage 1 — automatic absolute-time alignment with a stable session origin.

Coverage map (numbers match the Stage 1 test requirements):

Absolute alignment
  1  two absolute sources with different starts
  2  real Excel/COMTRADE timing values -> 0.0 and +4183.805733
  3  reversed load order produces physically identical placement
  4  source earlier than the current origin gets a negative offset
  5  the origin does NOT move when that earlier source is added
  6  absolute + relative_elapsed
  7  absolute + synthetic_elapsed
  8  epoch sentinel rejected
  9  missing start_time rejected
 10  single-source behaviour unchanged

Analyst interaction
 11  manual source not overwritten by later source additions
 12  auto_trigger source not overwritten
 13  correlation source not overwritten
 14  Set as Reference preserves physical separation
 15  Set as Reference preserves absolute axis labels
 16  Set as Reference preserves alignment methods
 17  source added AFTER Set as Reference uses the rebased origin
 18  manual offset + later earlier source does not move existing geometry

Data integrity
 19  original waveform_data["time"] arrays unchanged
 20  trigger marker correct after automatic alignment
 21  build_aligned_data() returns shifted X coordinates
 22  axis labels correct at first sample / trigger / middle / last
 23  applying alignment twice is idempotent

Regression
 24  session-window behaviour unchanged
 25  trigger/correlation entry points remain callable and unchanged
 26  calculated signals invalidated only when an offset actually changes

Fixtures carry the REAL timing values of the GPTH 275kV - BEN5K COMTRADE
record and the GPTH-PLNG YODC SNDC L2 Tripping SCADA export, with synthetic
waveform arrays. No confidential event file is committed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.models.channels import AnalogChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions import absolute_alignment
from app.sessions.absolute_alignment import (
    ABSOLUTE_TIMESTAMP,
    plan_absolute_alignment,
)
from app.sessions.event_session import EventAnalysisSession

# ---------------------------------------------------------------------------
# Real timing values from the validated event
# ---------------------------------------------------------------------------

EXCEL_START = datetime(2026, 7, 25, 12, 0, 0)
COMTRADE_START = datetime(2026, 7, 25, 13, 9, 43, 805733)
COMTRADE_TRIGGER = datetime(2026, 7, 25, 13, 9, 44, 305733)
COMTRADE_DURATION_S = 7.0198
EXPECTED_COMTRADE_OFFSET = 4183.805733          # (COMTRADE_START - EXCEL_START)
EPOCH_SENTINEL = datetime(2000, 1, 1, 0, 0, 0)  # provider "no real timestamp" marker


def _record(
    *,
    start: datetime | None,
    trigger: datetime | None = None,
    n: int = 20,
    dt: float = 1.0,
    timing_reference: str = "absolute",
    name: str = "Va",
    unit: str = "kV",
) -> DisturbanceRecord:
    """Minimal valid DisturbanceRecord with an elapsed-seconds time axis."""
    t = np.arange(n, dtype=np.float64) * dt
    values = np.sin(t)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST",
            recorder_name="TEST",
            source_file="test",
            provider_type="test",
            nominal_frequency=50.0,
        ),
        waveform_data=pd.DataFrame({"time": t, name: values}),
        analog_channels=[AnalogChannel(name=name, unit=unit, index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[1.0 / dt if dt else 0.0], samples_per_rate=[n]
        ),
        timing_info=TimingInformation(
            start_time=start,
            trigger_time=trigger if trigger is not None else start,
            timing_reference=timing_reference,
        ),
    )


def _excel_record() -> DisturbanceRecord:
    """121 samples at 60 s from 12:00:00 — the real SCADA export's geometry."""
    return _record(start=EXCEL_START, n=121, dt=60.0, name="mw_total", unit="MW")


def _comtrade_record() -> DisturbanceRecord:
    """5000 Hz, 7.0198 s from 13:09:43.805733, trigger +0.5 s.

    Uses 3510 samples at 2 ms rather than 35 100 at 0.2 ms: identical start,
    trigger and duration, which is all this suite measures, without a 35 k-row
    frame per fixture.
    """
    n = 3510
    dt = COMTRADE_DURATION_S / (n - 1)
    return _record(start=COMTRADE_START, trigger=COMTRADE_TRIGGER, n=n, dt=dt)


def _two_source_session() -> tuple[EventAnalysisSession, str, str]:
    """Excel then COMTRADE, aligned. Returns (session, excel_id, comtrade_id)."""
    session = EventAnalysisSession()
    excel_id = session.add_source(_excel_record(), "Excel", "normalized_excel")
    comtrade_id = session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.apply_absolute_alignment()
    return session, excel_id, comtrade_id


def _set_as_reference(session: EventAnalysisSession, source_id: str) -> None:
    """The coordinate-rebase half of PowerwaveMainWindow._on_session_set_as_reference.

    Mirrors the handler's session-level effect exactly (rebase the origin, then
    translate every offset uniformly while preserving each source's method) so
    the behaviour can be asserted without constructing a QMainWindow.
    """
    ref_offset = session.get_time_offset(source_id)
    session.rebase_absolute_time_origin(ref_offset)
    for source in session.list_sources():
        new_offset = 0.0 if source.source_id == source_id else source.time_offset_s - ref_offset
        session.set_time_offset(
            source.source_id,
            new_offset,
            method=source.alignment_method,
            confidence=source.alignment_confidence,
        )


def _label_at(session: EventAnalysisSession, x: float) -> datetime:
    """Absolute timestamp the X axis renders at session coordinate *x*.

    Reproduces SessionCanvasController._compute_session_reference_time() +
    DatetimeAxisItem.tickStrings() arithmetic without importing Qt: the
    session's explicit absolute_time_origin is authoritative (Stage 3.1), and
    the legacy min(start - offset) derivation is only the fallback, and only
    decides whether an absolute axis applies at all.
    """
    derived = [
        s.record.timing_info.start_time - timedelta(seconds=s.time_offset_s)
        for s in session.list_sources()
        if s.is_active
        and s.record.timing_info.start_time is not None
        and s.record.timing_info.timing_reference == "absolute"
    ]
    if not derived:
        raise AssertionError("no wall-clock-anchored active source; axis is relative")
    origin = session.absolute_time_origin
    return (origin if origin is not None else min(derived)) + timedelta(seconds=x)


# ---------------------------------------------------------------------------
# 1-5, 10 — origin establishment and stability
# ---------------------------------------------------------------------------


def test_1_two_absolute_sources_with_different_starts() -> None:
    session = EventAnalysisSession()
    a = session.add_source(_record(start=datetime(2026, 1, 1, 10, 0, 0)), "A", "comtrade")
    b = session.add_source(_record(start=datetime(2026, 1, 1, 10, 0, 30)), "B", "comtrade")

    plan = session.apply_absolute_alignment()

    assert plan.origin == datetime(2026, 1, 1, 10, 0, 0)
    assert plan.origin_established is True
    assert session.get_time_offset(a) == 0.0
    assert session.get_time_offset(b) == 30.0
    assert session.get_source(a).alignment_method == ABSOLUTE_TIMESTAMP
    assert session.get_source(b).alignment_method == ABSOLUTE_TIMESTAMP


def test_2_real_event_timing_values() -> None:
    session, excel_id, comtrade_id = _two_source_session()

    assert session.absolute_time_origin == EXCEL_START
    assert session.get_time_offset(excel_id) == 0.0
    assert session.get_time_offset(comtrade_id) == pytest.approx(
        EXPECTED_COMTRADE_OFFSET, abs=1e-9
    )


def test_3_reversed_load_order_is_physically_identical() -> None:
    forward, f_excel, f_comtrade = _two_source_session()

    reverse = EventAnalysisSession()
    r_comtrade = reverse.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    reverse.apply_absolute_alignment()          # single eligible source -> no-op
    r_excel = reverse.add_source(_excel_record(), "Excel", "normalized_excel")
    reverse.apply_absolute_alignment()

    assert reverse.absolute_time_origin == forward.absolute_time_origin
    assert reverse.get_time_offset(r_excel) == forward.get_time_offset(f_excel)
    assert reverse.get_time_offset(r_comtrade) == forward.get_time_offset(f_comtrade)
    # The physical separation is what actually matters, and it is identical.
    assert (
        reverse.get_time_offset(r_comtrade) - reverse.get_time_offset(r_excel)
    ) == pytest.approx(
        forward.get_time_offset(f_comtrade) - forward.get_time_offset(f_excel), abs=1e-9
    )


def test_4_source_earlier_than_origin_gets_negative_offset() -> None:
    session, _, _ = _two_source_session()
    early = session.add_source(
        _record(start=datetime(2026, 7, 25, 11, 55, 0)), "Early", "comtrade"
    )
    session.apply_absolute_alignment()

    assert session.get_time_offset(early) == pytest.approx(-300.0, abs=1e-9)


def test_5_origin_does_not_move_when_earlier_source_added() -> None:
    session, excel_id, comtrade_id = _two_source_session()
    origin_before = session.absolute_time_origin
    excel_before = session.get_time_offset(excel_id)
    comtrade_before = session.get_time_offset(comtrade_id)

    session.add_source(_record(start=datetime(2026, 7, 25, 11, 55, 0)), "Early", "comtrade")
    session.apply_absolute_alignment()

    assert session.absolute_time_origin == origin_before
    assert session.get_time_offset(excel_id) == excel_before
    assert session.get_time_offset(comtrade_id) == comtrade_before


def test_10_single_source_behaviour_unchanged() -> None:
    session = EventAnalysisSession()
    sid = session.add_source(_comtrade_record(), "COMTRADE", "comtrade")

    plan = session.apply_absolute_alignment()

    assert session.absolute_time_origin is None
    assert plan.origin is None
    assert plan.origin_established is False
    assert plan.derived_any is False
    assert plan.skipped_analyst_owned == ()   # nothing analyst-owned; just too few sources
    assert session.get_time_offset(sid) == 0.0
    assert session.get_source(sid).alignment_method == "none"


# ---------------------------------------------------------------------------
# 6-9 — eligibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["relative_elapsed", "synthetic_elapsed", "sample_index"])
def test_6_7_non_absolute_modes_are_never_auto_aligned(mode: str) -> None:
    session = EventAnalysisSession()
    abs_a = session.add_source(_record(start=datetime(2026, 1, 1, 10, 0, 0)), "A", "comtrade")
    abs_b = session.add_source(_record(start=datetime(2026, 1, 1, 10, 0, 30)), "B", "comtrade")
    rel = session.add_source(
        _record(start=datetime(2026, 1, 1, 9, 0, 0), timing_reference=mode),
        "Relative",
        "normalized_csv",
    )

    plan = session.apply_absolute_alignment()

    assert rel in plan.skipped_ineligible
    assert session.get_time_offset(rel) == 0.0
    assert session.get_source(rel).alignment_method == "none"
    # The ineligible source's earlier start must not have become the origin.
    assert session.absolute_time_origin == datetime(2026, 1, 1, 10, 0, 0)
    assert session.get_time_offset(abs_a) == 0.0
    assert session.get_time_offset(abs_b) == 30.0


def test_8_epoch_sentinel_is_rejected() -> None:
    session = EventAnalysisSession()
    real = session.add_source(_record(start=datetime(2026, 1, 1, 10, 0, 0)), "Real", "comtrade")
    fake = session.add_source(_record(start=EPOCH_SENTINEL), "Fake", "normalized_csv")

    plan = session.apply_absolute_alignment()

    # Only one genuinely-anchored source -> no origin at all.
    assert fake in plan.skipped_ineligible
    assert session.absolute_time_origin is None
    assert session.get_time_offset(fake) == 0.0
    assert session.get_time_offset(real) == 0.0


def test_9_missing_start_time_is_rejected() -> None:
    session = EventAnalysisSession()
    a = session.add_source(_record(start=datetime(2026, 1, 1, 10, 0, 0)), "A", "comtrade")
    b = session.add_source(_record(start=datetime(2026, 1, 1, 10, 0, 30)), "B", "comtrade")
    none_start = session.add_source(_record(start=None), "NoStart", "normalized_csv")

    plan = session.apply_absolute_alignment()

    assert none_start in plan.skipped_ineligible
    assert session.get_time_offset(none_start) == 0.0
    assert session.get_time_offset(a) == 0.0
    assert session.get_time_offset(b) == 30.0


# ---------------------------------------------------------------------------
# 11-13, 18 — analyst-owned alignment is never overwritten
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,confidence", [("manual", None), ("auto_trigger", 0.82), ("correlation", 0.5)]
)
def test_11_12_13_analyst_owned_methods_are_not_overwritten(
    method: str, confidence: float | None
) -> None:
    session, excel_id, comtrade_id = _two_source_session()
    session.set_time_offset(comtrade_id, 1234.5, method=method, confidence=confidence)

    session.add_source(_record(start=datetime(2026, 7, 25, 13, 30, 0)), "Later", "comtrade")
    plan = session.apply_absolute_alignment()

    assert session.get_time_offset(comtrade_id) == 1234.5
    assert session.get_source(comtrade_id).alignment_method == method
    assert session.get_source(comtrade_id).alignment_confidence == confidence
    assert comtrade_id in plan.skipped_analyst_owned
    assert comtrade_id not in plan.offsets


def test_18_manual_offset_survives_a_later_earlier_source() -> None:
    session, excel_id, comtrade_id = _two_source_session()
    session.set_time_offset(comtrade_id, EXPECTED_COMTRADE_OFFSET + 0.25, method="manual")
    origin_before = session.absolute_time_origin
    excel_before = session.get_time_offset(excel_id)
    comtrade_before = session.get_time_offset(comtrade_id)

    early = session.add_source(
        _record(start=datetime(2026, 7, 25, 11, 55, 0)), "Early", "comtrade"
    )
    session.apply_absolute_alignment()

    assert session.absolute_time_origin == origin_before          # origin stable
    assert session.get_time_offset(comtrade_id) == comtrade_before  # manual untouched
    assert session.get_source(comtrade_id).alignment_method == "manual"
    assert session.get_time_offset(excel_id) == excel_before       # geometry unchanged
    assert session.get_time_offset(early) == pytest.approx(-300.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 14-17 — Set as Reference
# ---------------------------------------------------------------------------


def test_14_set_as_reference_preserves_physical_separation() -> None:
    session, excel_id, comtrade_id = _two_source_session()
    separation_before = (
        session.get_time_offset(comtrade_id) - session.get_time_offset(excel_id)
    )

    _set_as_reference(session, comtrade_id)

    assert session.get_time_offset(comtrade_id) == 0.0
    assert session.get_time_offset(excel_id) == pytest.approx(
        -EXPECTED_COMTRADE_OFFSET, abs=1e-9
    )
    assert (
        session.get_time_offset(comtrade_id) - session.get_time_offset(excel_id)
    ) == pytest.approx(separation_before, abs=1e-9)


def test_15_set_as_reference_preserves_absolute_axis_labels() -> None:
    session, excel_id, comtrade_id = _two_source_session()
    _set_as_reference(session, comtrade_id)

    assert session.absolute_time_origin == COMTRADE_START

    trigger_x = session.get_time_offset(comtrade_id) + 0.5
    assert trigger_x == pytest.approx(0.5, abs=1e-9)
    assert _label_at(session, trigger_x) == COMTRADE_TRIGGER

    excel_1309_x = session.get_time_offset(excel_id) + 4140.0
    assert excel_1309_x == pytest.approx(-43.805733, abs=1e-9)
    assert _label_at(session, excel_1309_x) == datetime(2026, 7, 25, 13, 9, 0)


def test_16_set_as_reference_preserves_alignment_methods() -> None:
    session, excel_id, comtrade_id = _two_source_session()
    session.set_time_offset(
        comtrade_id, EXPECTED_COMTRADE_OFFSET, method="auto_trigger", confidence=0.9
    )

    _set_as_reference(session, comtrade_id)

    assert session.get_source(excel_id).alignment_method == ABSOLUTE_TIMESTAMP
    assert session.get_source(comtrade_id).alignment_method == "auto_trigger"
    assert session.get_source(comtrade_id).alignment_confidence == 0.9


def test_17_source_added_after_set_as_reference_uses_rebased_origin() -> None:
    session, _, comtrade_id = _two_source_session()
    _set_as_reference(session, comtrade_id)
    assert session.absolute_time_origin == COMTRADE_START

    later = session.add_source(
        _record(start=datetime(2026, 7, 25, 13, 10, 10)), "Later", "comtrade"
    )
    session.apply_absolute_alignment()

    expected = (datetime(2026, 7, 25, 13, 10, 10) - COMTRADE_START).total_seconds()
    assert session.get_time_offset(later) == pytest.approx(expected, abs=1e-9)
    assert expected == pytest.approx(26.194267, abs=1e-9)
    # NOT derived against the original 12:00:00 origin.
    assert session.get_time_offset(later) != pytest.approx(4210.0, abs=1e-3)
    assert session.absolute_time_origin == COMTRADE_START


# ---------------------------------------------------------------------------
# 19-23 — data integrity
# ---------------------------------------------------------------------------


def test_19_waveform_time_arrays_are_unchanged() -> None:
    excel_rec, comtrade_rec = _excel_record(), _comtrade_record()
    excel_before = excel_rec.waveform_data["time"].to_numpy(copy=True)
    comtrade_before = comtrade_rec.waveform_data["time"].to_numpy(copy=True)

    session = EventAnalysisSession()
    session.add_source(excel_rec, "Excel", "normalized_excel")
    cid = session.add_source(comtrade_rec, "COMTRADE", "comtrade")
    session.apply_absolute_alignment()
    session.build_aligned_data(cid, "Va", -1e9, 1e9)

    np.testing.assert_array_equal(excel_rec.waveform_data["time"].to_numpy(), excel_before)
    np.testing.assert_array_equal(
        comtrade_rec.waveform_data["time"].to_numpy(), comtrade_before
    )
    assert comtrade_rec.waveform_data["time"].iloc[0] == 0.0


def test_20_trigger_marker_position_after_alignment() -> None:
    session, _, comtrade_id = _two_source_session()
    source = session.get_source(comtrade_id)

    # SessionCanvasController._source_trigger_t arithmetic, Qt-free.
    timing = source.record.timing_info
    trigger_x = (
        timing.trigger_time - timing.start_time
    ).total_seconds() + source.time_offset_s

    assert trigger_x == pytest.approx(4184.305733, abs=1e-9)
    assert _label_at(session, trigger_x) == COMTRADE_TRIGGER


def test_21_build_aligned_data_returns_shifted_coordinates() -> None:
    session, excel_id, comtrade_id = _two_source_session()

    aligned = session.build_aligned_data(comtrade_id, "Va", -1e9, 1e9)
    assert aligned.time_offset_s == pytest.approx(EXPECTED_COMTRADE_OFFSET, abs=1e-9)
    assert aligned.time[0] == pytest.approx(EXPECTED_COMTRADE_OFFSET, abs=1e-9)
    assert aligned.time[-1] == pytest.approx(
        EXPECTED_COMTRADE_OFFSET + COMTRADE_DURATION_S, abs=1e-6
    )

    excel_aligned = session.build_aligned_data(excel_id, "mw_total", -1e9, 1e9)
    assert excel_aligned.time[0] == 0.0
    # The COMTRADE record lies strictly between the Excel 13:09 and 13:10 samples.
    assert 4140.0 < aligned.time[0] < aligned.time[-1] < 4200.0


def test_22_axis_labels_correct_at_four_probe_points() -> None:
    session, _, comtrade_id = _two_source_session()
    offset = session.get_time_offset(comtrade_id)

    for elapsed in (0.0, 0.5, COMTRADE_DURATION_S / 2.0, COMTRADE_DURATION_S):
        actual = COMTRADE_START + timedelta(seconds=elapsed)
        shown = _label_at(session, offset + elapsed)
        assert shown == actual, f"label wrong at elapsed={elapsed}"


def test_23_applying_alignment_twice_is_idempotent() -> None:
    session, excel_id, comtrade_id = _two_source_session()
    origin = session.absolute_time_origin
    offsets = {s.source_id: s.time_offset_s for s in session.list_sources()}

    second = session.apply_absolute_alignment()

    assert session.absolute_time_origin == origin
    assert second.origin_established is False
    assert {s.source_id: s.time_offset_s for s in session.list_sources()} == offsets


# ---------------------------------------------------------------------------
# 24-26 — regression
# ---------------------------------------------------------------------------


def test_24_session_window_semantics_unchanged() -> None:
    """_session_window() must remain the UNION of offset-shifted source ranges.

    Stage 1 deliberately does not touch the viewport; this pins that promise.
    """
    from app.ui.session.session_canvas_controller import _session_window

    session, _, _ = _two_source_session()
    t0, t1 = _session_window(session)

    # Excel 0..7200 unioned with COMTRADE 4183.806..4190.826, plus a 2% margin.
    span = 7200.0
    margin = span * 0.02
    assert t0 == pytest.approx(-margin, abs=1e-6)
    assert t1 == pytest.approx(7200.0 + margin, abs=1e-6)


def test_25_trigger_and_correlation_entry_points_unchanged() -> None:
    from app.sessions.alignment_engine import suggest_alignment_offsets
    from app.sessions.session_models import ALIGNMENT_METHODS

    session, _, _ = _two_source_session()
    results = suggest_alignment_offsets(session.list_sources())

    assert len(results) == 2
    assert all(r.alignment_method == "auto_trigger" for r in results)
    assert {"none", "manual", "auto_trigger", "correlation", "imported"} <= ALIGNMENT_METHODS
    assert ABSOLUTE_TIMESTAMP in ALIGNMENT_METHODS


def test_26_calculated_signals_invalidated_only_on_real_change() -> None:
    from app.calculated_signals.models import (
        CalculatedSignalDefinition,
        CalculatedSignalResult,
        CalculationStatus,
        ChannelRef,
    )

    session, excel_id, comtrade_id = _two_source_session()
    ref = ChannelRef(source_id=comtrade_id, channel_name="Va")
    definition = CalculatedSignalDefinition(
        calc_id="calc-stage1-test",
        name="test_calc",
        expression="a * 2",
        variable_bindings={"a": ref},
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
            computed_at=datetime(2026, 7, 25, 14, 0, 0, tzinfo=timezone.utc),
        ),
    )

    # Idempotent re-alignment changes no offset -> result stays OK.
    session.apply_absolute_alignment()
    assert session.get_calculated_signal_result(calc_id).status == CalculationStatus.OK

    # Adding an earlier source does not move the COMTRADE offset -> still OK.
    session.add_source(_record(start=datetime(2026, 7, 25, 11, 55, 0)), "Early", "comtrade")
    session.apply_absolute_alignment()
    assert session.get_calculated_signal_result(calc_id).status == CalculationStatus.OK

    # A real offset change does invalidate.
    session.set_time_offset(comtrade_id, 99.0, method="manual")
    assert session.get_calculated_signal_result(calc_id).status == CalculationStatus.STALE


# ---------------------------------------------------------------------------
# Policy-module unit checks (no session)
# ---------------------------------------------------------------------------


def test_plan_is_pure_and_does_not_mutate_sources() -> None:
    session = EventAnalysisSession()
    session.add_source(_excel_record(), "Excel", "normalized_excel")
    session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    sources = session.list_sources()
    before = [(s.time_offset_s, s.alignment_method) for s in sources]

    plan = plan_absolute_alignment(sources, None)

    assert [(s.time_offset_s, s.alignment_method) for s in sources] == before
    assert plan.origin == EXCEL_START
    assert plan.clock_verified is False


def test_inactive_sources_remain_eligible() -> None:
    """Toggling a source off must not change what its timestamps mean."""
    session = EventAnalysisSession()
    a = session.add_source(_excel_record(), "Excel", "normalized_excel")
    b = session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.set_source_active(b, False)

    session.apply_absolute_alignment()

    assert session.absolute_time_origin == EXCEL_START
    assert session.get_time_offset(b) == pytest.approx(EXPECTED_COMTRADE_OFFSET, abs=1e-9)
    assert session.get_time_offset(a) == 0.0


def test_derive_offset_matches_the_documented_rule() -> None:
    assert absolute_alignment.derive_offset(COMTRADE_START, EXCEL_START) == pytest.approx(
        EXPECTED_COMTRADE_OFFSET, abs=1e-9
    )
    assert absolute_alignment.derive_offset(EXCEL_START, COMTRADE_START) == pytest.approx(
        -EXPECTED_COMTRADE_OFFSET, abs=1e-9
    )
