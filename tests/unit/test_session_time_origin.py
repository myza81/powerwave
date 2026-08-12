"""Stage 3.1 — absolute_time_origin is the authoritative X-axis origin.

Regression suite for a confirmed defect: the X-axis origin was re-derived on
every repaint as ``min(start_time - time_offset_s)``, which equals the session
origin only while every offset is exactly timestamp-derived. After any analyst
adjustment the two diverged, and because the derivation took the MINIMUM,
moving one source 250 ms later dragged the whole axis 250 ms earlier — which
both relabelled every unrelated source and cancelled out the adjustment so the
analyst could not see it.

Coverage map (numbers match the Stage 3.1 test requirements):

  1  pure absolute alignment still labels both sources exactly
  2  manual +250 ms leaves absolute_time_origin unchanged
  3  Excel labels unchanged after a COMTRADE manual adjustment
  4  COMTRADE labels shift +250 ms
  5  Set-as-Reference labels remain exact
  6  manual adjustment after Set-as-Reference moves only the adjusted source
  7  auto_trigger offset does not redefine the origin
  8  correlation offset does not redefine the origin
  9  save/reload preserves the corrected display labels
 10  single-source behaviour unchanged (legacy fallback)
 11  relative-time session behaviour unchanged
 12  legacy persisted offsets without an origin remain loadable
 13  Stage 2 viewport coordinates still work with adjusted offsets
 14  _session_window() semantics unchanged
 15  source waveform arrays untouched
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
from app.ui.session.session_canvas_controller import (
    SessionCanvasController,
    _session_window,
)

EXCEL_START = datetime(2026, 7, 25, 12, 0, 0)
COMTRADE_START = datetime(2026, 7, 25, 13, 9, 43, 805733)
COMTRADE_TRIGGER = datetime(2026, 7, 25, 13, 9, 44, 305733)
COMTRADE_DURATION_S = 7.0198
COMTRADE_OFFSET = 4183.805733
EXCEL_1309_X = 4140.0


def _record(
    *,
    start: datetime,
    trigger: datetime | None = None,
    n: int,
    dt: float,
    name: str = "Va",
    timing_reference: str = "absolute",
) -> DisturbanceRecord:
    t = np.arange(n, dtype=np.float64) * dt
    return DisturbanceRecord(
        metadata=RecordingMetadata("TEST", "TEST", "test", "test", 50.0),
        waveform_data=pd.DataFrame({"time": t, name: np.sin(t)}),
        analog_channels=[AnalogChannel(name=name, unit="kV", index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation([1.0 / dt if dt else 0.0], [n]),
        timing_info=TimingInformation(
            start_time=start,
            trigger_time=trigger if trigger is not None else start,
            timing_reference=timing_reference,
        ),
    )


def _excel_record() -> DisturbanceRecord:
    return _record(start=EXCEL_START, n=121, dt=60.0, name="mw_total")


def _comtrade_record() -> DisturbanceRecord:
    n = 3510
    return _record(
        start=COMTRADE_START,
        trigger=COMTRADE_TRIGGER,
        n=n,
        dt=COMTRADE_DURATION_S / (n - 1),
    )


def _session() -> tuple[EventAnalysisSession, str, str]:
    session = EventAnalysisSession()
    excel_id = session.add_source(_excel_record(), "Excel", "normalized_excel")
    comtrade_id = session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.apply_absolute_alignment()
    return session, excel_id, comtrade_id


def _ctrl() -> SessionCanvasController:
    return SessionCanvasController()


def _axis_ref(session) -> datetime | None:
    """The real production code path, not a re-implementation."""
    return _ctrl()._compute_session_reference_time(session)


def _label_at(session, x: float) -> datetime:
    ref = _axis_ref(session)
    assert ref is not None, "expected an absolute axis"
    return ref + timedelta(seconds=x)


def _trigger_label(session, comtrade_id: str) -> datetime:
    """Label rendered where the COMTRADE trigger marker is drawn."""
    source = session.get_source(comtrade_id)
    timing = source.record.timing_info
    x = (timing.trigger_time - timing.start_time).total_seconds() + source.time_offset_s
    return _label_at(session, x)


def _set_as_reference(session, source_id: str) -> None:
    ref_offset = session.get_time_offset(source_id)
    session.rebase_absolute_time_origin(ref_offset)
    for source in session.list_sources():
        session.set_time_offset(
            source.source_id,
            0.0 if source.source_id == source_id else source.time_offset_s - ref_offset,
            method=source.alignment_method,
            confidence=source.alignment_confidence,
        )


# ---------------------------------------------------------------------------
# 1 — baseline
# ---------------------------------------------------------------------------


def test_1_pure_absolute_alignment_labels_are_exact() -> None:
    session, _, comtrade_id = _session()

    assert _axis_ref(session) == EXCEL_START == session.absolute_time_origin
    assert _label_at(session, EXCEL_1309_X) == datetime(2026, 7, 25, 13, 9, 0)
    assert _label_at(session, 4200.0) == datetime(2026, 7, 25, 13, 10, 0)
    assert _trigger_label(session, comtrade_id) == COMTRADE_TRIGGER


# ---------------------------------------------------------------------------
# 2-4 — the confirmed defect
# ---------------------------------------------------------------------------


def test_2_manual_correction_leaves_origin_unchanged() -> None:
    session, _, comtrade_id = _session()
    session.set_time_offset(comtrade_id, COMTRADE_OFFSET + 0.250, method="manual")

    assert session.absolute_time_origin == EXCEL_START
    assert _axis_ref(session) == EXCEL_START


def test_3_unrelated_source_labels_unchanged_after_manual_correction() -> None:
    session, _, comtrade_id = _session()
    before = _label_at(session, EXCEL_1309_X)

    session.set_time_offset(comtrade_id, COMTRADE_OFFSET + 0.250, method="manual")

    assert _label_at(session, EXCEL_1309_X) == before == datetime(2026, 7, 25, 13, 9, 0)


def test_4_adjusted_source_labels_shift_by_the_correction() -> None:
    session, _, comtrade_id = _session()
    session.set_time_offset(comtrade_id, COMTRADE_OFFSET + 0.250, method="manual")

    assert session.get_time_offset(comtrade_id) == pytest.approx(4184.055733, abs=1e-9)
    assert _trigger_label(session, comtrade_id) == datetime(
        2026, 7, 25, 13, 9, 44, 555733
    )
    # The adjustment is visible, not cancelled out by a moving axis.
    assert _trigger_label(session, comtrade_id) != COMTRADE_TRIGGER


def test_4b_negative_correction_is_symmetric() -> None:
    session, _, comtrade_id = _session()
    session.set_time_offset(comtrade_id, COMTRADE_OFFSET - 0.250, method="manual")

    assert _axis_ref(session) == EXCEL_START
    assert _label_at(session, EXCEL_1309_X) == datetime(2026, 7, 25, 13, 9, 0)
    assert _trigger_label(session, comtrade_id) == datetime(
        2026, 7, 25, 13, 9, 44, 55733
    )


# ---------------------------------------------------------------------------
# 5-6 — Set as Reference
# ---------------------------------------------------------------------------


def test_5_set_as_reference_labels_remain_exact() -> None:
    session, excel_id, comtrade_id = _session()
    _set_as_reference(session, comtrade_id)

    assert session.absolute_time_origin == COMTRADE_START
    assert _axis_ref(session) == COMTRADE_START
    assert _label_at(session, 0.0) == COMTRADE_START
    assert _trigger_label(session, comtrade_id) == COMTRADE_TRIGGER
    excel_1309_x = session.get_time_offset(excel_id) + EXCEL_1309_X
    assert excel_1309_x == pytest.approx(-43.805733, abs=1e-9)
    assert _label_at(session, excel_1309_x) == datetime(2026, 7, 25, 13, 9, 0)


def test_6_manual_adjustment_after_rebase_moves_only_that_source() -> None:
    session, excel_id, comtrade_id = _session()
    _set_as_reference(session, comtrade_id)
    comtrade_label_before = _trigger_label(session, comtrade_id)

    session.set_time_offset(
        excel_id, session.get_time_offset(excel_id) + 0.100, method="manual"
    )

    assert session.absolute_time_origin == COMTRADE_START
    assert _axis_ref(session) == COMTRADE_START
    assert _trigger_label(session, comtrade_id) == comtrade_label_before == COMTRADE_TRIGGER
    excel_1309_x = session.get_time_offset(excel_id) + EXCEL_1309_X
    assert _label_at(session, excel_1309_x) == datetime(
        2026, 7, 25, 13, 9, 0, 100000
    )


# ---------------------------------------------------------------------------
# 7-8 — trigger and correlation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,confidence,offset",
    [("auto_trigger", 0.82, 4180.0), ("correlation", 0.91, 4181.0)],
)
def test_7_8_analytical_alignment_does_not_redefine_the_origin(
    method: str, confidence: float, offset: float
) -> None:
    session, _, comtrade_id = _session()
    session.set_time_offset(comtrade_id, offset, method=method, confidence=confidence)

    assert session.absolute_time_origin == EXCEL_START
    assert _axis_ref(session) == EXCEL_START
    # Unrelated source untouched.
    assert _label_at(session, EXCEL_1309_X) == datetime(2026, 7, 25, 13, 9, 0)
    # Adjusted source sits where the analytical method put it.
    assert _trigger_label(session, comtrade_id) == EXCEL_START + timedelta(
        seconds=offset + 0.5
    )


# ---------------------------------------------------------------------------
# 9 — persistence
# ---------------------------------------------------------------------------


def test_9_save_reload_preserves_corrected_labels(tmp_path) -> None:
    import yaml

    from app.data.manifest_generator import generate_manifest
    from app.data.manifest_loader import parse_alignment_state
    from app.ui.main_window.main_window import PowerwaveMainWindow

    session, excel_id, comtrade_id = _session()
    session.set_time_offset(comtrade_id, COMTRADE_OFFSET + 0.250, method="manual")

    path = tmp_path / "event.yaml"
    generate_manifest(session, "event", path)
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    alignment = parse_alignment_state(manifest)

    records = {s.source_id: s.record for s in session.list_sources()}
    restored = EventAnalysisSession()
    id_map = {
        manifest_id: restored.add_source(record, manifest_id, "restored")
        for manifest_id, record in records.items()
    }
    PowerwaveMainWindow._restore_manifest_alignment(None, restored, alignment, id_map)
    restored.apply_absolute_alignment()

    assert restored.absolute_time_origin == EXCEL_START
    assert restored.get_time_offset(id_map[comtrade_id]) == pytest.approx(
        4184.055733, abs=1e-9
    )
    assert restored.get_source(id_map[comtrade_id]).alignment_method == "manual"
    assert _label_at(restored, EXCEL_1309_X) == datetime(2026, 7, 25, 13, 9, 0)
    assert _trigger_label(restored, id_map[comtrade_id]) == datetime(
        2026, 7, 25, 13, 9, 44, 555733
    )


# ---------------------------------------------------------------------------
# 10-12 — legacy fallback paths
# ---------------------------------------------------------------------------


def test_10_single_source_uses_the_legacy_derivation() -> None:
    session = EventAnalysisSession()
    session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.apply_absolute_alignment()

    assert session.absolute_time_origin is None
    assert _axis_ref(session) == COMTRADE_START      # unchanged behaviour


def test_10b_single_source_with_manual_offset_uses_legacy_derivation() -> None:
    session = EventAnalysisSession()
    sid = session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.set_time_offset(sid, 5.0, method="manual")

    assert session.absolute_time_origin is None
    assert _axis_ref(session) == COMTRADE_START - timedelta(seconds=5.0)


def test_11_relative_only_session_has_no_absolute_axis() -> None:
    session = EventAnalysisSession()
    for name in ("A", "B"):
        session.add_source(
            _record(start=EXCEL_START, n=20, dt=1.0, timing_reference="relative_elapsed"),
            name,
            "normalized_csv",
        )
    session.apply_absolute_alignment()

    assert session.absolute_time_origin is None
    assert _axis_ref(session) is None


def test_11b_stale_origin_does_not_force_an_absolute_axis() -> None:
    """All wall-clock-anchored sources deactivated: the axis stays relative
    even though an origin lingers from before."""
    session, excel_id, comtrade_id = _session()
    assert session.absolute_time_origin == EXCEL_START

    session.set_source_active(excel_id, False)
    session.set_source_active(comtrade_id, False)
    session.add_source(
        _record(start=EXCEL_START, n=20, dt=1.0, timing_reference="relative_elapsed"),
        "Relative",
        "normalized_csv",
    )

    assert session.absolute_time_origin == EXCEL_START   # not cleared
    assert _axis_ref(session) is None                     # but not used


def test_12_legacy_offsets_without_origin_use_the_legacy_derivation() -> None:
    """Stage 3 keeps such geometry opaque; the axis falls back as before."""
    session, excel_id, comtrade_id = _session()
    session.restore_absolute_time_origin(None)
    session.set_time_offset(comtrade_id, 1234.5, method="imported")

    assert session.absolute_time_origin is None
    assert _axis_ref(session) == min(
        EXCEL_START, COMTRADE_START - timedelta(seconds=1234.5)
    )


# ---------------------------------------------------------------------------
# 13-15 — Stage 2 and data-integrity regression
# ---------------------------------------------------------------------------


def test_13_viewport_policy_still_works_with_an_adjusted_offset() -> None:
    from app.visualization.viewport_policy import select_initial_viewport

    session, _, comtrade_id = _session()
    session.set_time_offset(comtrade_id, COMTRADE_OFFSET + 0.250, method="manual")

    window = select_initial_viewport(session)

    assert window is not None
    lo, hi = window
    # Still brackets the two real Excel samples and the whole event.
    assert lo < EXCEL_1309_X < hi
    assert lo < 4200.0 < hi
    aligned = session.build_aligned_data(comtrade_id, "Va", -1e9, 1e9)
    assert lo < aligned.time[0] and aligned.time[-1] < hi


def test_14_session_window_semantics_unchanged() -> None:
    session, _, comtrade_id = _session()
    before = _session_window(session)
    assert before[0] == pytest.approx(-144.0, abs=1e-6)
    assert before[1] == pytest.approx(7344.0, abs=1e-6)

    session.set_time_offset(comtrade_id, COMTRADE_OFFSET + 0.250, method="manual")
    after = _session_window(session)

    # Still the union of offset-shifted extents plus a 2% margin: the COMTRADE
    # shift is far inside the Excel span, so the domain is unmoved.
    assert after == before


def test_15_waveform_arrays_untouched() -> None:
    session, excel_id, comtrade_id = _session()
    comtrade_record = session.get_source(comtrade_id).record
    excel_record = session.get_source(excel_id).record
    ct_before = comtrade_record.waveform_data["time"].to_numpy(copy=True)
    ex_before = excel_record.waveform_data["time"].to_numpy(copy=True)

    session.set_time_offset(comtrade_id, COMTRADE_OFFSET + 0.250, method="manual")
    _axis_ref(session)
    session.build_aligned_data(comtrade_id, "Va", -1e9, 1e9)

    np.testing.assert_array_equal(
        comtrade_record.waveform_data["time"].to_numpy(), ct_before
    )
    np.testing.assert_array_equal(
        excel_record.waveform_data["time"].to_numpy(), ex_before
    )
    assert comtrade_record.waveform_data["time"].iloc[0] == 0.0


def test_derivation_helper_is_still_available_and_pure() -> None:
    """The legacy calculation is retained, not deleted."""
    session, _, comtrade_id = _session()
    ctrl = _ctrl()
    session.set_time_offset(comtrade_id, COMTRADE_OFFSET + 0.250, method="manual")

    derived = ctrl._derive_reference_time_from_sources(session)
    authoritative = ctrl._compute_session_reference_time(session)

    assert derived == EXCEL_START - timedelta(seconds=0.250)   # the old answer
    assert authoritative == EXCEL_START                         # the correct one
    assert derived != authoritative
