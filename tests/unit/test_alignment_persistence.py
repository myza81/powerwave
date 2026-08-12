"""Stage 3 — alignment persistence, source identity mapping, and minimal UI.

Coverage map (numbers match the Stage 3 test requirements):

Manifest round trip
  1  automatic absolute alignment save/reload
  2  exact absolute_time_origin restoration
  3  exact offsets restoration
  4  alignment methods restoration
  5  microseconds preserved
  6  load order does not affect restored geometry
  7  Set-as-Reference save/reload
  8  manual offset save/reload
  9  auto_trigger state save/reload
 10  correlation state save/reload

Legacy compatibility
 11  old manifest with offsets but no methods
 12  old manifest with no alignment block
 13  single-source legacy manifest
 14  missing optional new fields
 15  unknown future alignment method handled safely

Source mapping
 16  stable mapping from manifest source identity to live SessionSource
 17  source missing on reload
 18  source order changed during reload
 19  duplicate display name case

Axis correctness
 20  automatic state labels correct after reload
 21  rebased state labels correct after reload
 22  manual-corrected state labels preserve corrected geometry

UI
 23  absolute_timestamp renders as human-readable text
 24  all-timestamp-aligned session summary
 25  mixed alignment-method summary
 26  relative-only session
 27  timezone-naive absolute pair shows clock verification as not verified
 28  UI never claims GPS/PTP/synchronized clocks

Regression
 31  _session_window() unchanged
 32  source data arrays untouched
 33  manifest without new fields remains loadable
 34  calculated signals unaffected by save/reload geometry restoration

Fixtures carry the real GPTH event timing values with synthetic waveform
arrays. No confidential event file is committed.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from app.data.manifest_generator import generate_manifest
from app.data.manifest_loader import parse_alignment_state
from app.data.multi_source_session import SessionAlignmentState
from app.models.channels import AnalogChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.alignment_summary import (
    ClockVerification,
    method_label,
    summarize_alignment,
)
from app.sessions.event_session import EventAnalysisSession
from app.sessions.session_models import ALIGNMENT_METHODS

EXCEL_START = datetime(2026, 7, 25, 12, 0, 0)
COMTRADE_START = datetime(2026, 7, 25, 13, 9, 43, 805733)
COMTRADE_TRIGGER = datetime(2026, 7, 25, 13, 9, 44, 305733)
COMTRADE_DURATION_S = 7.0198
COMTRADE_OFFSET = 4183.805733
EPOCH_SENTINEL = datetime(2000, 1, 1, 0, 0, 0)


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
    t = np.arange(n, dtype=np.float64) * dt
    return DisturbanceRecord(
        metadata=RecordingMetadata("TEST", "TEST", "test", "test", 50.0),
        waveform_data=pd.DataFrame({"time": t, name: np.sin(t)}),
        analog_channels=[AnalogChannel(name=name, unit=unit, index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation([1.0 / dt if dt else 0.0], [n]),
        timing_info=TimingInformation(
            start_time=start,
            trigger_time=trigger if trigger is not None else start,
            timing_reference=timing_reference,
        ),
    )


def _excel_record() -> DisturbanceRecord:
    return _record(start=EXCEL_START, n=121, dt=60.0, name="mw_total", unit="MW")


def _comtrade_record() -> DisturbanceRecord:
    n = 3510
    return _record(
        start=COMTRADE_START,
        trigger=COMTRADE_TRIGGER,
        n=n,
        dt=COMTRADE_DURATION_S / (n - 1),
    )


def _aligned_session() -> tuple[EventAnalysisSession, str, str]:
    session = EventAnalysisSession()
    excel_id = session.add_source(_excel_record(), "Excel", "normalized_excel")
    comtrade_id = session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    session.apply_absolute_alignment()
    return session, excel_id, comtrade_id


# ---------------------------------------------------------------------------
# Save/reload harness
#
# Exercises the real generator, the real alignment parser, and the real
# restore logic from PowerwaveMainWindow._restore_manifest_alignment — the
# latter as an unbound call, so the persistence contract is tested without
# constructing a QMainWindow or resolving source files from disk.
# ---------------------------------------------------------------------------


def _save(session: EventAnalysisSession, tmp_path: Path) -> dict:
    path = tmp_path / "event.yaml"
    generate_manifest(session, "event_test", path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _reload(
    manifest: dict,
    records_by_manifest_id: dict[str, DisturbanceRecord],
    *,
    order: list[str] | None = None,
) -> tuple[EventAnalysisSession, dict[str, str], list[str]]:
    """Rebuild a session the way _on_multi_source_loaded does."""
    from app.ui.main_window.main_window import PowerwaveMainWindow

    alignment = parse_alignment_state(manifest)
    manifest_ids = order if order is not None else [
        str(s["source_id"]) for s in manifest.get("sources", [])
    ]

    session = EventAnalysisSession()
    id_map: dict[str, str] = {}
    for manifest_id in manifest_ids:
        record = records_by_manifest_id.get(manifest_id)
        if record is None:
            continue        # source unavailable on reload
        id_map[manifest_id] = session.add_source(record, manifest_id, "restored")

    missing = PowerwaveMainWindow._restore_manifest_alignment(
        None, session, alignment, id_map
    )
    # Mirrors _on_multi_source_loaded: offsets restored without a recorded
    # origin are opaque, so no origin is derived for them.
    if not (alignment.has_offsets and not alignment.has_trustworthy_origin):
        session.apply_absolute_alignment()
    return session, id_map, missing


def _round_trip(session, tmp_path, records, **kw):
    return _reload(_save(session, tmp_path), records, **kw)


def _records_for(session: EventAnalysisSession) -> dict[str, DisturbanceRecord]:
    return {s.source_id: s.record for s in session.list_sources()}


def _label_at(session: EventAnalysisSession, x: float) -> datetime:
    """Absolute timestamp the X axis renders at session coordinate x.

    Mirrors SessionCanvasController._compute_session_reference_time(): the
    explicit absolute_time_origin wins (Stage 3.1), with the legacy
    min(start - offset) derivation as the fallback for sessions that never
    established one.
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
# 1-6 — automatic absolute alignment round trip
# ---------------------------------------------------------------------------


def test_1_2_3_4_automatic_alignment_round_trip(tmp_path) -> None:
    session, excel_id, comtrade_id = _aligned_session()
    records = _records_for(session)

    restored, id_map, missing = _round_trip(session, tmp_path, records)

    assert missing == []
    assert restored.absolute_time_origin == EXCEL_START            # 2
    assert restored.get_time_offset(id_map[excel_id]) == 0.0        # 3
    assert restored.get_time_offset(id_map[comtrade_id]) == pytest.approx(
        COMTRADE_OFFSET, abs=1e-9
    )
    assert restored.get_source(id_map[comtrade_id]).alignment_method == "absolute_timestamp"  # 4
    assert restored.get_source(id_map[excel_id]).alignment_method == "absolute_timestamp"


def test_5_microseconds_preserved_in_origin_and_offsets(tmp_path) -> None:
    session, _, comtrade_id = _aligned_session()
    # Rebase so the origin itself carries microseconds.
    ref_offset = session.get_time_offset(comtrade_id)
    session.rebase_absolute_time_origin(ref_offset)
    for s in session.list_sources():
        session.set_time_offset(
            s.source_id,
            0.0 if s.source_id == comtrade_id else s.time_offset_s - ref_offset,
            method=s.alignment_method,
        )
    manifest = _save(session, tmp_path)

    assert manifest["alignment"]["absolute_time_origin"] == "2026-07-25T13:09:43.805733"

    restored, id_map, _ = _reload(manifest, _records_for(session))
    assert restored.absolute_time_origin == COMTRADE_START
    assert restored.absolute_time_origin.microsecond == 805733


def test_6_load_order_does_not_affect_restored_geometry(tmp_path) -> None:
    session, excel_id, comtrade_id = _aligned_session()
    records = _records_for(session)
    manifest = _save(session, tmp_path)

    forward, fmap, _ = _reload(manifest, records, order=[excel_id, comtrade_id])
    reverse, rmap, _ = _reload(manifest, records, order=[comtrade_id, excel_id])

    assert forward.absolute_time_origin == reverse.absolute_time_origin
    for manifest_id in (excel_id, comtrade_id):
        assert forward.get_time_offset(fmap[manifest_id]) == reverse.get_time_offset(
            rmap[manifest_id]
        )


# ---------------------------------------------------------------------------
# 7-10 — analyst-owned geometry round trips
# ---------------------------------------------------------------------------


def test_7_set_as_reference_round_trip(tmp_path) -> None:
    session, excel_id, comtrade_id = _aligned_session()
    ref_offset = session.get_time_offset(comtrade_id)
    session.rebase_absolute_time_origin(ref_offset)
    for s in session.list_sources():
        session.set_time_offset(
            s.source_id,
            0.0 if s.source_id == comtrade_id else s.time_offset_s - ref_offset,
            method=s.alignment_method,
            confidence=s.alignment_confidence,
        )
    methods_before = {s.source_id: s.alignment_method for s in session.list_sources()}

    restored, id_map, _ = _round_trip(session, tmp_path, _records_for(session))

    assert restored.absolute_time_origin == COMTRADE_START
    assert restored.get_time_offset(id_map[comtrade_id]) == 0.0
    assert restored.get_time_offset(id_map[excel_id]) == pytest.approx(
        -COMTRADE_OFFSET, abs=1e-9
    )
    for manifest_id, method in methods_before.items():
        assert restored.get_source(id_map[manifest_id]).alignment_method == method


def test_8_manual_offset_round_trip(tmp_path) -> None:
    session, _, comtrade_id = _aligned_session()
    corrected = COMTRADE_OFFSET + 0.250
    session.set_time_offset(comtrade_id, corrected, method="manual")

    restored, id_map, _ = _round_trip(session, tmp_path, _records_for(session))

    live = restored.get_source(id_map[comtrade_id])
    assert live.time_offset_s == pytest.approx(corrected, abs=1e-9)
    assert live.alignment_method == "manual"
    # Crucially NOT re-derived back to the timestamp-implied value.
    assert live.time_offset_s != pytest.approx(COMTRADE_OFFSET, abs=1e-6)


@pytest.mark.parametrize(
    "method,confidence,note",
    [("auto_trigger", 0.82, "Trigger detected at t=0.5000 s"), ("correlation", 0.91, "")],
)
def test_9_10_trigger_and_correlation_round_trip(
    tmp_path, method: str, confidence: float, note: str
) -> None:
    session, _, comtrade_id = _aligned_session()
    session.set_time_offset(comtrade_id, 4180.0, method=method, confidence=confidence)
    if note:
        session.set_alignment_notes(comtrade_id, note)

    restored, id_map, _ = _round_trip(session, tmp_path, _records_for(session))

    live = restored.get_source(id_map[comtrade_id])
    assert live.time_offset_s == pytest.approx(4180.0, abs=1e-9)
    assert live.alignment_method == method
    assert live.alignment_confidence == pytest.approx(confidence, abs=1e-9)
    if note:
        assert restored.get_alignment_notes(id_map[comtrade_id]) == note


# ---------------------------------------------------------------------------
# 11-15 — legacy manifests
# ---------------------------------------------------------------------------


def test_11_legacy_offsets_without_methods_or_origin() -> None:
    """Legacy A: offsets preserved, method conservative, origin left unset."""
    session = EventAnalysisSession()
    excel_id = session.add_source(_excel_record(), "Excel", "normalized_excel")
    comtrade_id = session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    legacy = {
        "event_id": "legacy",
        "sources": [{"source_id": excel_id}, {"source_id": comtrade_id}],
        "alignment": {
            "reference_source": excel_id,
            "offsets_seconds": {excel_id: 0.0, comtrade_id: 1234.5},
        },
    }

    restored, id_map, _ = _reload(legacy, _records_for(session))

    assert restored.get_time_offset(id_map[comtrade_id]) == pytest.approx(1234.5, abs=1e-9)
    assert restored.get_source(id_map[comtrade_id]).alignment_method == "imported"
    # No origin was saved, so none is invented; 'imported' is not derivable, so
    # the later apply_absolute_alignment() cannot overwrite the saved numbers.
    assert restored.absolute_time_origin is None


def test_12_legacy_manifest_with_no_alignment_block() -> None:
    """Legacy B: nothing to restore; Stage 1 derives normally."""
    session = EventAnalysisSession()
    excel_id = session.add_source(_excel_record(), "Excel", "normalized_excel")
    comtrade_id = session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    legacy = {
        "event_id": "legacy",
        "sources": [{"source_id": excel_id}, {"source_id": comtrade_id}],
    }

    restored, id_map, missing = _reload(legacy, _records_for(session))

    assert missing == []
    assert restored.absolute_time_origin == EXCEL_START
    assert restored.get_time_offset(id_map[comtrade_id]) == pytest.approx(
        COMTRADE_OFFSET, abs=1e-9
    )
    assert restored.get_source(id_map[comtrade_id]).alignment_method == "absolute_timestamp"


def test_13_single_source_legacy_manifest_unchanged() -> None:
    session = EventAnalysisSession()
    comtrade_id = session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    legacy = {"event_id": "legacy", "sources": [{"source_id": comtrade_id}]}

    restored, id_map, _ = _reload(legacy, _records_for(session))

    assert restored.absolute_time_origin is None
    assert restored.get_time_offset(id_map[comtrade_id]) == 0.0
    assert restored.get_source(id_map[comtrade_id]).alignment_method == "none"


def test_14_missing_optional_fields_parse_safely() -> None:
    state = parse_alignment_state({"alignment": {"offsets_seconds": {"a": 1.0}}})
    assert state.offsets_seconds == {"a": 1.0}
    assert state.methods == {} and state.confidences == {} and state.notes == {}
    assert state.absolute_time_origin is None
    assert state.has_offsets and not state.has_trustworthy_origin

    assert parse_alignment_state({}) == SessionAlignmentState()
    assert parse_alignment_state({"alignment": None}) == SessionAlignmentState()
    assert parse_alignment_state({"alignment": "nonsense"}) == SessionAlignmentState()
    # A malformed individual entry is skipped, not fatal.
    bad = parse_alignment_state({"alignment": {"offsets_seconds": {"a": "abc", "b": 2.0}}})
    assert bad.offsets_seconds == {"b": 2.0}


def test_15_unknown_future_alignment_method_is_downgraded() -> None:
    session = EventAnalysisSession()
    excel_id = session.add_source(_excel_record(), "Excel", "normalized_excel")
    comtrade_id = session.add_source(_comtrade_record(), "COMTRADE", "comtrade")
    future = {
        "event_id": "future",
        "sources": [{"source_id": excel_id}, {"source_id": comtrade_id}],
        "alignment": {
            "absolute_time_origin": EXCEL_START.isoformat(),
            "offsets_seconds": {excel_id: 0.0, comtrade_id: 999.0},
            "methods": {excel_id: "quantum_entanglement", comtrade_id: "manual"},
        },
    }

    restored, id_map, _ = _reload(future, _records_for(session))

    assert restored.get_source(id_map[excel_id]).alignment_method == "imported"
    assert restored.get_source(id_map[comtrade_id]).alignment_method == "manual"
    assert restored.get_time_offset(id_map[comtrade_id]) == pytest.approx(999.0, abs=1e-9)
    assert "quantum_entanglement" not in ALIGNMENT_METHODS


# ---------------------------------------------------------------------------
# 16-19 — source identity mapping
# ---------------------------------------------------------------------------


def test_16_manifest_source_id_is_the_persisted_identity(tmp_path) -> None:
    session, excel_id, comtrade_id = _aligned_session()
    manifest = _save(session, tmp_path)

    manifest_ids = [str(s["source_id"]) for s in manifest["sources"]]
    assert set(manifest_ids) == {excel_id, comtrade_id}
    assert set(manifest["alignment"]["offsets_seconds"]) == {excel_id, comtrade_id}
    assert set(manifest["alignment"]["methods"]) == {excel_id, comtrade_id}

    restored, id_map, _ = _reload(manifest, _records_for(session))
    # Live ids are fresh uuid4s, never the manifest ids.
    assert set(id_map.values()).isdisjoint({excel_id, comtrade_id})
    assert restored.get_time_offset(id_map[comtrade_id]) == pytest.approx(
        COMTRADE_OFFSET, abs=1e-9
    )


def test_17_missing_source_on_reload_is_reported_not_fatal(tmp_path) -> None:
    session, excel_id, comtrade_id = _aligned_session()
    manifest = _save(session, tmp_path)
    records = _records_for(session)
    del records[comtrade_id]          # COMTRADE file unavailable

    restored, id_map, missing = _reload(manifest, records, order=[excel_id, comtrade_id])

    assert missing == [comtrade_id]
    assert comtrade_id not in id_map
    assert restored.get_time_offset(id_map[excel_id]) == 0.0
    assert restored.absolute_time_origin == EXCEL_START


def test_18_changed_source_order_still_maps_correctly(tmp_path) -> None:
    session, excel_id, comtrade_id = _aligned_session()
    manifest = _save(session, tmp_path)

    reordered, id_map, _ = _reload(
        manifest, _records_for(session), order=[comtrade_id, excel_id]
    )

    assert reordered.get_time_offset(id_map[excel_id]) == 0.0
    assert reordered.get_time_offset(id_map[comtrade_id]) == pytest.approx(
        COMTRADE_OFFSET, abs=1e-9
    )


def test_19_duplicate_display_names_do_not_collide(tmp_path) -> None:
    """Two sources may share a display name; identity is the source_id."""
    session = EventAnalysisSession()
    a = session.add_source(_comtrade_record(), "relay", "comtrade")
    b = session.add_source(
        _record(
            start=COMTRADE_START + timedelta(seconds=10),
            trigger=COMTRADE_TRIGGER + timedelta(seconds=10),
        ),
        "relay",
        "comtrade",
    )
    session.apply_absolute_alignment()
    assert a != b

    restored, id_map, _ = _round_trip(session, tmp_path, _records_for(session))

    assert restored.get_time_offset(id_map[a]) == 0.0
    assert restored.get_time_offset(id_map[b]) == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 20-22 — absolute axis correctness after reload
# ---------------------------------------------------------------------------


def test_20_automatic_state_labels_correct_after_reload(tmp_path) -> None:
    session, excel_id, comtrade_id = _aligned_session()
    restored, id_map, _ = _round_trip(session, tmp_path, _records_for(session))

    offset = restored.get_time_offset(id_map[comtrade_id])
    assert _label_at(restored, 4140.0) == datetime(2026, 7, 25, 13, 9, 0)
    assert _label_at(restored, offset) == COMTRADE_START
    assert _label_at(restored, offset + 0.5) == COMTRADE_TRIGGER
    assert _label_at(restored, offset + COMTRADE_DURATION_S) == COMTRADE_START + timedelta(
        seconds=COMTRADE_DURATION_S
    )
    assert _label_at(restored, 4200.0) == datetime(2026, 7, 25, 13, 10, 0)


def test_21_rebased_state_labels_correct_after_reload(tmp_path) -> None:
    session, excel_id, comtrade_id = _aligned_session()
    ref_offset = session.get_time_offset(comtrade_id)
    session.rebase_absolute_time_origin(ref_offset)
    for s in session.list_sources():
        session.set_time_offset(
            s.source_id,
            0.0 if s.source_id == comtrade_id else s.time_offset_s - ref_offset,
            method=s.alignment_method,
        )

    restored, id_map, _ = _round_trip(session, tmp_path, _records_for(session))

    assert _label_at(restored, 0.5) == COMTRADE_TRIGGER
    assert _label_at(restored, -43.805733) == datetime(2026, 7, 25, 13, 9, 0)
    assert _label_at(restored, 16.194267) == datetime(2026, 7, 25, 13, 10, 0)


def test_22_manual_correction_geometry_preserved_after_reload(tmp_path) -> None:
    session, _, comtrade_id = _aligned_session()
    session.set_time_offset(comtrade_id, COMTRADE_OFFSET + 0.250, method="manual")

    restored, id_map, _ = _round_trip(session, tmp_path, _records_for(session))

    offset = restored.get_time_offset(id_map[comtrade_id])
    assert offset == pytest.approx(COMTRADE_OFFSET + 0.250, abs=1e-9)
    # Stage 3.1: the origin is stable, so the analyst's +250 ms correction is
    # VISIBLE — the trigger labels 250 ms later than its recorded timestamp,
    # which is exactly what the analyst asserted. The Excel side is unmoved.
    assert restored.absolute_time_origin == EXCEL_START
    assert _label_at(restored, offset + 0.5) == COMTRADE_TRIGGER + timedelta(seconds=0.250)
    assert _label_at(restored, 4140.0) == datetime(2026, 7, 25, 13, 9, 0)


# ---------------------------------------------------------------------------
# 23-28 — UI summary
# ---------------------------------------------------------------------------


def test_23_absolute_timestamp_renders_human_readable() -> None:
    assert method_label("absolute_timestamp") == "Absolute timestamp"
    assert method_label("auto_trigger") == "Trigger detection"
    assert method_label("correlation") == "Cross-correlation"
    assert method_label("manual") == "Manual"
    assert method_label("imported") == "Imported"
    assert method_label("none") == "Not aligned"


def test_23b_source_row_badge_shows_the_label(qapp=None) -> None:
    from app.ui.session.source_row_widget import _alignment_badge_text

    session, _, comtrade_id = _aligned_session()
    badge = _alignment_badge_text(session.get_source(comtrade_id))
    assert badge == "● Absolute timestamp"
    assert "absolute_timestamp" not in badge


def test_24_all_timestamp_aligned_summary() -> None:
    session, _, _ = _aligned_session()
    summary = summarize_alignment(session)

    assert summary.headline == "Absolute timestamp"
    assert summary.is_mixed is False
    assert summary.clock_verification is ClockVerification.NOT_VERIFIED
    assert summary.not_timestamp_aligned == ()


def test_25_mixed_alignment_summary() -> None:
    session, excel_id, comtrade_id = _aligned_session()
    session.set_time_offset(comtrade_id, 1.0, method="manual")
    third = session.add_source(
        _record(start=datetime(2026, 7, 25, 13, 30, 0)), "Relay", "comtrade"
    )
    session.set_time_offset(third, 2.0, method="correlation", confidence=0.8)

    summary = summarize_alignment(session)

    assert summary.is_mixed is True
    assert summary.headline.startswith("Mixed (")
    assert "1 absolute timestamp" in summary.headline
    assert "1 manual" in summary.headline
    assert "1 correlation" in summary.headline
    names = {name for name, _ in summary.not_timestamp_aligned}
    assert names == {"COMTRADE", "Relay"}


def test_26_relative_only_session_reports_timing_reference_not_clock() -> None:
    session = EventAnalysisSession()
    session.add_source(
        _record(start=EXCEL_START, timing_reference="relative_elapsed"), "A", "normalized_csv"
    )
    session.add_source(
        _record(start=EXCEL_START, timing_reference="relative_elapsed"), "B", "normalized_csv"
    )
    session.apply_absolute_alignment()

    summary = summarize_alignment(session)

    assert summary.clock_verification is ClockVerification.NOT_APPLICABLE
    assert "Relative" in summary.clock_line
    assert "verification" not in summary.clock_line.lower()


def test_26b_epoch_sentinel_source_is_not_wall_clock_anchored() -> None:
    session = EventAnalysisSession()
    session.add_source(_record(start=EPOCH_SENTINEL), "A", "normalized_csv")
    session.add_source(_record(start=EPOCH_SENTINEL), "B", "normalized_csv")

    summary = summarize_alignment(session)
    assert summary.clock_verification is ClockVerification.NOT_APPLICABLE


def test_27_timezone_naive_absolute_pair_is_not_verified() -> None:
    session, _, _ = _aligned_session()
    summary = summarize_alignment(session)

    assert summary.clock_verification is ClockVerification.NOT_VERIFIED
    assert summary.clock_line.startswith("Not verified")
    assert "timezone" in summary.clock_line


def test_28_ui_never_claims_synchronized_clocks() -> None:
    forbidden = ("gps", "ptp", "synchronized", "synchronised", "verified clock", "accurate")
    sessions = []

    s1, _, _ = _aligned_session()
    sessions.append(s1)

    s2 = EventAnalysisSession()
    s2.add_source(_record(start=EXCEL_START, timing_reference="relative_elapsed"), "A", "csv")
    s2.add_source(_record(start=EXCEL_START, timing_reference="sample_index"), "B", "csv")
    sessions.append(s2)

    s3, _, cid = _aligned_session()
    s3.set_time_offset(cid, 1.0, method="manual")
    sessions.append(s3)

    for session in sessions:
        summary = summarize_alignment(session)
        blob = f"{summary.headline} {summary.clock_line} {summary.detail_text()}".lower()
        for word in forbidden:
            assert word not in blob, f"UI text claims {word!r}: {blob}"


def test_28b_empty_session_summary_is_safe() -> None:
    summary = summarize_alignment(EventAnalysisSession())
    assert summary.headline == "No sources"
    assert summary.clock_line == ""


# ---------------------------------------------------------------------------
# 31-34 — regression
# ---------------------------------------------------------------------------


def test_31_session_window_unchanged_after_round_trip(tmp_path) -> None:
    from app.ui.session.session_canvas_controller import _session_window

    session, _, _ = _aligned_session()
    before = _session_window(session)
    restored, _, _ = _round_trip(session, tmp_path, _records_for(session))

    assert _session_window(restored) == before
    assert before[0] == pytest.approx(-144.0, abs=1e-6)
    assert before[1] == pytest.approx(7344.0, abs=1e-6)


def test_32_source_arrays_untouched_by_persistence(tmp_path) -> None:
    session, _, comtrade_id = _aligned_session()
    record = session.get_source(comtrade_id).record
    before = record.waveform_data["time"].to_numpy(copy=True)

    restored, id_map, _ = _round_trip(session, tmp_path, _records_for(session))
    restored.build_aligned_data(id_map[comtrade_id], "Va", -1e9, 1e9)

    np.testing.assert_array_equal(record.waveform_data["time"].to_numpy(), before)
    assert record.waveform_data["time"].iloc[0] == 0.0


def test_33_manifest_without_new_fields_remains_loadable(tmp_path) -> None:
    """A manifest whose alignment block predates Stage 3 still opens."""
    session, excel_id, comtrade_id = _aligned_session()
    manifest = _save(session, tmp_path)
    for key in ("absolute_time_origin", "methods", "confidences", "notes"):
        manifest["alignment"].pop(key, None)

    restored, id_map, missing = _reload(manifest, _records_for(session))

    assert missing == []
    assert restored.get_time_offset(id_map[comtrade_id]) == pytest.approx(
        COMTRADE_OFFSET, abs=1e-9
    )
    assert restored.get_source(id_map[comtrade_id]).alignment_method == "imported"


def test_34_calculated_signals_unaffected_by_restore(tmp_path) -> None:
    from datetime import timezone

    from app.calculated_signals.models import (
        CalculatedSignalDefinition,
        CalculatedSignalResult,
        CalculationStatus,
        ChannelRef,
    )

    session, _, comtrade_id = _aligned_session()
    restored, id_map, _ = _round_trip(session, tmp_path, _records_for(session))
    live_id = id_map[comtrade_id]

    definition = CalculatedSignalDefinition(
        calc_id="calc-stage3",
        name="calc",
        expression="a * 2",
        variable_bindings={"a": ChannelRef(source_id=live_id, channel_name="Va")},
        reference_variable="a",
    )
    calc_id = restored.add_calculated_signal(
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

    # Re-running alignment on the restored session is idempotent, so a
    # calculated signal is not invalidated by the reload path.
    restored.apply_absolute_alignment()
    assert restored.get_calculated_signal_result(calc_id).status == CalculationStatus.OK


def test_generator_writes_the_documented_schema(tmp_path) -> None:
    session, excel_id, comtrade_id = _aligned_session()
    session.set_time_offset(comtrade_id, 4180.0, method="auto_trigger", confidence=0.8)
    session.set_alignment_notes(comtrade_id, "Trigger detected")

    manifest = _save(session, tmp_path)
    alignment = manifest["alignment"]

    assert alignment["absolute_time_origin"] == "2026-07-25T12:00:00"
    assert alignment["offsets_seconds"] == {excel_id: 0.0, comtrade_id: 4180.0}
    assert alignment["methods"] == {
        excel_id: "absolute_timestamp",
        comtrade_id: "auto_trigger",
    }
    assert alignment["confidences"] == {comtrade_id: 0.8}
    assert alignment["notes"] == {comtrade_id: "Trigger detected"}
    assert "reference_source" in alignment


def test_all_zero_offsets_are_still_persisted(tmp_path) -> None:
    """An all-zero geometry is a real geometry and must survive reload."""
    session = EventAnalysisSession()
    a = session.add_source(_comtrade_record(), "A", "comtrade")
    b = session.add_source(_comtrade_record(), "B", "comtrade")
    session.apply_absolute_alignment()
    assert session.get_time_offset(a) == 0.0 and session.get_time_offset(b) == 0.0

    manifest = _save(session, tmp_path)
    assert manifest["alignment"]["offsets_seconds"] == {a: 0.0, b: 0.0}

    restored, id_map, _ = _reload(manifest, _records_for(session))
    assert restored.absolute_time_origin == COMTRADE_START
