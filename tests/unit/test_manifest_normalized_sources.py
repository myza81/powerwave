"""Import-Wizard (normalized) source manifest round trip.

A session containing a Wizard-imported CSV/Excel source could be saved but not
reopened: the generator wrote ``type: normalized_excel`` (the record's own
provider_type) and the loader rejected anything outside
comtrade/csv/excel with ``ValueError: Unknown source type``.

The fix keeps the normalized types as their own RELOAD RECIPE rather than
flattening them to the physical format, because the two produce materially
different records from the same bytes -- ExcelProvider yields
('DC Total Demand', 'MW', 'active_power') where the Wizard yields
('mw_dc_total_demand', 'MW', 'mw'). Reloading a normalized source re-runs
run_import_pipeline(), which is deterministic for a given file.

Coverage map (numbers match the requirement list):

  1  Import-Wizard Excel save/reload succeeds
  2  Import-Wizard CSV save/reload succeeds
  3  timing metadata equivalent before/after
  4  time arrays equivalent before/after
  5  channel count equivalent
  6  channel names equivalent
  7  units / parameter types equivalent
  8  manifest source ids map correctly
  9  source load order does not affect alignment restoration
 10  existing `type: excel` manifest still loads
 11  existing `type: csv` manifest still loads
 12  COMTRADE manifest still loads
 13  legacy `normalized_excel` with a generic paths.path key still loads
 14  legacy `normalized_csv` handled the same way
 15  Excel + COMTRADE automatic alignment round trip
 16  Excel + COMTRADE manual alignment round trip
 17  Stage 2 event viewport still works after reload
 19  waveform_data["time"] unmutated
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from app.data.manifest_generator import generate_manifest
from app.data.manifest_loader import (
    SUPPORTED_SOURCE_TYPES,
    build_session_from_manifest,
    is_normalized_source_type,
    physical_source_type,
)
from app.import_wizard.import_pipeline import run_import_pipeline
from app.sessions.event_session import EventAnalysisSession

EXCEL_START = datetime(2026, 7, 25, 12, 0, 0)
CT_START = datetime(2026, 7, 25, 13, 9, 43, 805733)
CT_TRIGGER = datetime(2026, 7, 25, 13, 9, 44, 305733)
CT_DURATION = 7.0198
CT_OFFSET = 4183.805733


# ---------------------------------------------------------------------------
# Fixtures written to disk so the round trip is genuinely file-based
# ---------------------------------------------------------------------------


def _write_xlsx(tmp_path: Path) -> Path:
    ts = [EXCEL_START + timedelta(minutes=i) for i in range(121)]
    mw = np.full(121, 930.0)
    mw[69], mw[70], mw[71] = 939.32, 623.88, 805.36
    p = tmp_path / "scada_trend.xlsx"
    pd.DataFrame(
        {"Time": [t.strftime("%m/%d/%Y %H:%M") for t in ts], "DC Total Demand": mw}
    ).to_excel(p, index=False)
    return p


def _write_csv(tmp_path: Path) -> Path:
    ts = [EXCEL_START + timedelta(minutes=i) for i in range(121)]
    p = tmp_path / "scada_trend.csv"
    pd.DataFrame(
        {
            "Time": [t.strftime("%m/%d/%Y %H:%M") for t in ts],
            "System Demand": np.full(121, 930.0),
        }
    ).to_csv(p, index=False)
    return p


def _write_comtrade(tmp_path: Path) -> Path:
    """Minimal 1999 ASCII COMTRADE with the real GPTH start/trigger timestamps.

    1.0 s at 5000 Hz: long enough that the +0.5 s trigger falls inside the
    record's own extent, which viewport_policy requires of an event anchor.
    """
    n, fs = 5000, 5000.0
    cfg = tmp_path / "relay.cfg"
    cfg_text = (
        "GPTH 275kV - TEST,1838,1999\n"
        "1,1A,0D\n"
        "1,VR,A,,kV,1.0,0.0,0,-32768,32767,1.0,1.0,P\n"
        "50\n"
        "1\n"
        f"{fs:.3f},{n}\n"
        + CT_START.strftime("%d/%m/%Y,%H:%M:%S.%f") + "\n"
        + CT_TRIGGER.strftime("%d/%m/%Y,%H:%M:%S.%f") + "\n"
        "ASCII\n"
        "1\n"
    )
    cfg.write_text(cfg_text, encoding="latin-1")
    rows = [
        f"{i + 1},{int(round(i / fs * 1e6))},"
        f"{int(1000 * np.sin(2 * np.pi * 50 * i / fs))}"
        for i in range(n)
    ]
    (tmp_path / "relay.dat").write_text("\n".join(rows) + "\n", encoding="latin-1")
    return cfg


def _wizard_record(path: Path, provider_type: str):
    result = run_import_pipeline(str(path), provider_type=provider_type)
    assert result.success, result.validation_messages
    return result.record


def _fingerprint(record) -> dict:
    t = record.waveform_data["time"].to_numpy(dtype=float)
    return {
        "start": record.timing_info.start_time,
        "trigger": record.timing_info.trigger_time,
        "timing_reference": record.timing_info.timing_reference,
        "n": len(t),
        "t_first": float(t[0]),
        "t_last": float(t[-1]),
        "t_sum": round(float(np.sum(t)), 6),
        "analog": [(c.name, c.unit, c.parameter_type) for c in record.analog_channels],
        "digital": [c.name for c in record.digital_channels],
    }


def _session_with(records) -> EventAnalysisSession:
    session = EventAnalysisSession()
    for record, name in records:
        session.add_source(
            record, name, str(record.metadata.provider_type), record.metadata.source_file
        )
    session.apply_absolute_alignment()
    return session


def _round_trip(session: EventAnalysisSession, tmp_path: Path):
    p = tmp_path / "event.yaml"
    generate_manifest(session, "event", p)
    manifest = yaml.safe_load(p.read_text(encoding="utf-8"))
    return build_session_from_manifest(p, root=tmp_path), manifest


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_source_type_vocabulary() -> None:
    assert SUPPORTED_SOURCE_TYPES == {
        "comtrade", "csv", "excel", "normalized_csv", "normalized_excel"
    }
    assert physical_source_type("normalized_excel") == "excel"
    assert physical_source_type("normalized_csv") == "csv"
    assert physical_source_type("excel") == "excel"
    assert physical_source_type("COMTRADE") == "comtrade"
    assert is_normalized_source_type("normalized_csv") is True
    assert is_normalized_source_type("csv") is False


# ---------------------------------------------------------------------------
# 1-7 — Excel and CSV round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "writer,provider,expected_type",
    [(_write_xlsx, "excel", "normalized_excel"), (_write_csv, "csv", "normalized_csv")],
)
def test_1_2_wizard_source_round_trip(tmp_path, writer, provider, expected_type) -> None:
    src = writer(tmp_path)
    record = _wizard_record(src, provider)
    assert record.metadata.provider_type == expected_type
    before = _fingerprint(record)

    session = _session_with([(record, "Trend")])
    reloaded, manifest = _round_trip(session, tmp_path)

    assert manifest["sources"][0]["type"] == expected_type
    assert physical_source_type(provider) in manifest["sources"][0]["paths"]
    assert reloaded.source_count() == 1
    after = _fingerprint(reloaded.sources[0].record)

    assert after["start"] == before["start"]                      # 3
    assert after["trigger"] == before["trigger"]
    assert after["timing_reference"] == before["timing_reference"]
    assert (after["n"], after["t_first"], after["t_last"], after["t_sum"]) == (
        before["n"], before["t_first"], before["t_last"], before["t_sum"]
    )                                                             # 4
    assert len(after["analog"]) == len(before["analog"])           # 5
    assert [a[0] for a in after["analog"]] == [a[0] for a in before["analog"]]   # 6
    assert after["analog"] == before["analog"]                    # 7 (unit + parameter_type)
    assert after["digital"] == before["digital"]


def test_6b_reload_preserves_canonical_names_not_raw_headers(tmp_path) -> None:
    """The point of keeping the normalized type: raw-provider names differ."""
    from app.providers.excel.excel_provider import ExcelProvider

    src = _write_xlsx(tmp_path)
    wizard = _wizard_record(src, "excel")
    direct = ExcelProvider().load(src)
    assert [c.name for c in wizard.analog_channels] == ["mw_dc_total_demand"]
    assert [c.name for c in direct.analog_channels] == ["DC Total Demand"]

    reloaded, _ = _round_trip(_session_with([(wizard, "Trend")]), tmp_path)
    assert [c.name for c in reloaded.sources[0].record.analog_channels] == [
        "mw_dc_total_demand"
    ]


# ---------------------------------------------------------------------------
# 8-9 — identity mapping
# ---------------------------------------------------------------------------


def test_8_manifest_source_ids_round_trip(tmp_path) -> None:
    xlsx, cfg = _write_xlsx(tmp_path), _write_comtrade(tmp_path)
    from app.providers.comtrade.comtrade_provider import ComtradeProvider

    session = _session_with(
        [(_wizard_record(xlsx, "excel"), "Trend"), (ComtradeProvider().load(cfg), "Relay")]
    )
    saved_ids = [s.source_id for s in session.list_sources()]
    reloaded, manifest = _round_trip(session, tmp_path)

    assert [str(d["source_id"]) for d in manifest["sources"]] == saved_ids
    assert [s.source_id for s in reloaded.sources] == saved_ids
    assert set(manifest["alignment"]["offsets_seconds"]) == set(saved_ids)


def test_9_load_order_does_not_affect_restored_alignment(tmp_path) -> None:
    from app.ui.main_window.main_window import PowerwaveMainWindow
    from app.data.manifest_loader import parse_alignment_state
    from app.providers.comtrade.comtrade_provider import ComtradeProvider

    xlsx, cfg = _write_xlsx(tmp_path), _write_comtrade(tmp_path)
    session = _session_with(
        [(_wizard_record(xlsx, "excel"), "Trend"), (ComtradeProvider().load(cfg), "Relay")]
    )
    reloaded, manifest = _round_trip(session, tmp_path)
    alignment = parse_alignment_state(manifest)

    results = []
    for order in (list(reloaded.sources), list(reversed(reloaded.sources))):
        live = EventAnalysisSession()
        id_map = {
            s.source_id: live.add_source(s.record, s.source_id, s.provider_type)
            for s in order
        }
        PowerwaveMainWindow._restore_manifest_alignment(None, live, alignment, id_map)
        live.apply_absolute_alignment()
        results.append(
            (live.absolute_time_origin,
             {mid: round(live.get_time_offset(lid), 9) for mid, lid in id_map.items()})
        )
    assert results[0] == results[1]


# ---------------------------------------------------------------------------
# 10-14 — backward compatibility
# ---------------------------------------------------------------------------


def test_10_11_12_existing_physical_types_still_load(tmp_path) -> None:
    xlsx, csvp, cfg = _write_xlsx(tmp_path), _write_csv(tmp_path), _write_comtrade(tmp_path)
    manifest = {
        "event_id": "legacy",
        "sources": [
            {"source_id": "x", "type": "excel", "paths": {"excel": xlsx.name}},
            {"source_id": "c", "type": "csv", "paths": {"csv": csvp.name}},
            {"source_id": "t", "type": "comtrade", "paths": {"cfg": cfg.name}},
        ],
    }
    p = tmp_path / "legacy.yaml"
    p.write_text(yaml.dump(manifest, sort_keys=False), encoding="utf-8")

    session = build_session_from_manifest(p, root=tmp_path)

    assert session.source_count() == 3
    assert [s.provider_type for s in session.sources] == ["excel", "csv", "comtrade"]
    # Physical types keep using the raw providers -> raw header names.
    assert [c.name for c in session.get_source("x").record.analog_channels] == [
        "DC Total Demand"
    ]
    assert session.get_source("t").record.timing_info.start_time == CT_START


@pytest.mark.parametrize(
    "writer,mtype", [(_write_xlsx, "normalized_excel"), (_write_csv, "normalized_csv")]
)
def test_13_14_legacy_normalized_manifest_with_generic_path_key(
    tmp_path, writer, mtype
) -> None:
    """Manifests written before the fix stored the file under paths.path."""
    src = writer(tmp_path)
    manifest = {
        "event_id": "legacy_norm",
        "sources": [{"source_id": "n", "type": mtype, "paths": {"path": src.name}}],
    }
    p = tmp_path / "legacy_norm.yaml"
    p.write_text(yaml.dump(manifest, sort_keys=False), encoding="utf-8")

    session = build_session_from_manifest(p, root=tmp_path)

    assert session.source_count() == 1
    record = session.sources[0].record
    assert record.metadata.provider_type == mtype
    assert record.timing_info.start_time == EXCEL_START
    assert [c.name for c in record.analog_channels][0].startswith("mw_")


def test_unknown_type_still_rejected_with_a_clear_message(tmp_path) -> None:
    manifest = {
        "event_id": "bad",
        "sources": [{"source_id": "z", "type": "quantum", "paths": {"path": "x"}}],
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown source type 'quantum'"):
        build_session_from_manifest(p, root=tmp_path)


# ---------------------------------------------------------------------------
# 15-17, 19 — the real workflow
# ---------------------------------------------------------------------------


def _wizard_plus_comtrade(tmp_path):
    from app.providers.comtrade.comtrade_provider import ComtradeProvider

    xlsx, cfg = _write_xlsx(tmp_path), _write_comtrade(tmp_path)
    return _session_with(
        [(_wizard_record(xlsx, "excel"), "Trend"), (ComtradeProvider().load(cfg), "Relay")]
    )


def _restore(reloaded, manifest):
    from app.ui.main_window.main_window import PowerwaveMainWindow
    from app.data.manifest_loader import parse_alignment_state

    live = EventAnalysisSession()
    id_map = {
        s.source_id: live.add_source(s.record, s.source_id, s.provider_type)
        for s in reloaded.sources
    }
    alignment = parse_alignment_state(manifest)
    PowerwaveMainWindow._restore_manifest_alignment(None, live, alignment, id_map)
    if not (alignment.has_offsets and not alignment.has_trustworthy_origin):
        live.apply_absolute_alignment()
    return live, id_map


def test_15_excel_comtrade_automatic_alignment_round_trip(tmp_path) -> None:
    session = _wizard_plus_comtrade(tmp_path)
    ex_id, ct_id = [s.source_id for s in session.list_sources()]
    assert session.absolute_time_origin == EXCEL_START
    assert session.get_time_offset(ct_id) == pytest.approx(CT_OFFSET, abs=1e-9)

    reloaded, manifest = _round_trip(session, tmp_path)
    live, id_map = _restore(reloaded, manifest)

    assert live.absolute_time_origin == EXCEL_START
    assert live.get_time_offset(id_map[ex_id]) == 0.0
    assert live.get_time_offset(id_map[ct_id]) == pytest.approx(CT_OFFSET, abs=1e-9)
    assert live.get_source(id_map[ct_id]).alignment_method == "absolute_timestamp"

    origin = live.absolute_time_origin
    assert origin + timedelta(seconds=4140.0) == datetime(2026, 7, 25, 13, 9, 0)
    assert origin + timedelta(seconds=CT_OFFSET + 0.5) == CT_TRIGGER
    assert origin + timedelta(seconds=4200.0) == datetime(2026, 7, 25, 13, 10, 0)


def test_16_excel_comtrade_manual_alignment_round_trip(tmp_path) -> None:
    session = _wizard_plus_comtrade(tmp_path)
    ex_id, ct_id = [s.source_id for s in session.list_sources()]
    session.set_time_offset(ct_id, CT_OFFSET + 0.250, method="manual")

    reloaded, manifest = _round_trip(session, tmp_path)
    live, id_map = _restore(reloaded, manifest)

    assert live.absolute_time_origin == EXCEL_START
    assert live.get_time_offset(id_map[ex_id]) == 0.0
    assert live.get_time_offset(id_map[ct_id]) == pytest.approx(CT_OFFSET + 0.250, abs=1e-9)
    assert live.get_source(id_map[ct_id]).alignment_method == "manual"
    origin = live.absolute_time_origin
    assert origin + timedelta(seconds=4140.0) == datetime(2026, 7, 25, 13, 9, 0)
    assert origin + timedelta(seconds=CT_OFFSET + 0.750) == datetime(
        2026, 7, 25, 13, 9, 44, 555733
    )


def test_17_event_viewport_still_selected_after_reload(tmp_path) -> None:
    from app.visualization.viewport_policy import select_initial_viewport

    session = _wizard_plus_comtrade(tmp_path)
    reloaded, manifest = _round_trip(session, tmp_path)
    live, _ = _restore(reloaded, manifest)

    window = select_initial_viewport(live)

    assert window is not None
    lo, hi = window
    # Brackets both real Excel samples and the whole COMTRADE extent.
    assert lo < 4140.0 < hi and lo < 4200.0 < hi
    ct_record = next(
        s.record for s in live.list_sources()
        if s.record.metadata.provider_type.lower() == "comtrade"
    )
    ct_duration = float(ct_record.waveform_data["time"].to_numpy()[-1])
    assert lo < CT_OFFSET and CT_OFFSET + ct_duration < hi


def test_19_reloaded_time_arrays_are_unmutated(tmp_path) -> None:
    session = _wizard_plus_comtrade(tmp_path)
    reloaded, manifest = _round_trip(session, tmp_path)
    live, id_map = _restore(reloaded, manifest)

    for manifest_id, live_id in id_map.items():
        record = live.get_source(live_id).record
        t = record.waveform_data["time"].to_numpy(dtype=float)
        assert t[0] == 0.0, f"{manifest_id} time axis no longer starts at 0"
        live.build_aligned_data(
            live_id, record.analog_channels[0].name, -1e9, 1e9
        )
        np.testing.assert_array_equal(
            record.waveform_data["time"].to_numpy(dtype=float), t
        )
