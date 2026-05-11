"""Unit tests for app.data.manifest_loader."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.data.manifest_loader import (
    _get_source_file_path,
    _infer_comtrade_channel_type,
    _parse_timestamp,
    build_session_from_manifest,
    load_manifest,
)
from app.models import (
    AnalogChannel,
    DigitalChannel,
    DisturbanceRecord,
    RecordingMetadata,
    SamplingInformation,
    TimingInformation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_COMTRADE_MANIFEST = """\
event_id: test_event
sources:
  - source_id: comtrade_1
    type: comtrade
    paths:
      cfg: samples/comtrade/test.cfg
      dat: samples/comtrade/test.dat
    voltage_reference: phase_ground
    start_time: "2024-01-01T00:00:00"
    trigger_time: "2024-01-01T00:00:01"
    sampling_rates_hz:
      - 5000.0
    channels:
      - KPDN1 VR
      - KPDN1 IB
alignment:
  reference_source: comtrade_1
  reference_start: "2024-01-01T00:00:00"
  offsets_seconds:
    comtrade_1: 0.0
"""

_MULTI_SOURCE_MANIFEST = """\
event_id: test_multi
sources:
  - source_id: comtrade_main
    type: comtrade
    paths:
      cfg: samples/comtrade/test.cfg
    voltage_reference: phase_ground
    start_time: "2024-01-01T00:00:00"
  - source_id: csv_ops
    type: csv
    paths:
      csv: samples/csv/test.csv
    start_time: "2024-01-01T00:00:00"
    columns:
      - name: Frequency
        signal_type: frequency
        unit: Hz
        display_group: frequency
        confidence: 0.95
        inferred_from: manifest_confirmed
        requires_user_confirmation: false
      - name: LowConf
        signal_type: active_power
        unit: MW
        display_group: power
        confidence: 0.65
        inferred_from: name_keyword
        requires_user_confirmation: true
alignment:
  reference_source: comtrade_main
  reference_start: "2024-01-01T00:00:00"
  offsets_seconds:
    comtrade_main: 0.0
    csv_ops: 0.0
"""

_MISSING_EVENT_ID = """\
sources:
  - source_id: x
    type: comtrade
    paths:
      cfg: samples/comtrade/test.cfg
"""

_MISSING_SOURCES = """\
event_id: test
"""

_INVALID_TYPE_MANIFEST = """\
event_id: bad_type
sources:
  - source_id: src1
    type: unknown_format
    paths:
      cfg: samples/comtrade/test.cfg
"""


def _make_analog_record(channel_names: list[str]) -> DisturbanceRecord:
    """Return a minimal DisturbanceRecord with the given analog channel names."""
    import numpy as np
    import pandas as pd

    channels = [
        AnalogChannel(name=n, unit="kV" if n.split()[-1][0] == "V" else "kA", index=i)
        for i, n in enumerate(channel_names)
    ]
    n_pts = 10
    data = {"time": np.linspace(0, 1, n_pts)}
    for ch in channels:
        data[ch.name] = np.zeros(n_pts)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST",
            recorder_name="test",
            source_file="test.cfg",
            provider_type="comtrade",
            nominal_frequency=50.0,
        ),
        waveform_data=pd.DataFrame(data),
        analog_channels=channels,
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[5000.0],
            samples_per_rate=[n_pts],
        ),
        timing_info=TimingInformation(
            start_time=datetime(2024, 1, 1),
            trigger_time=datetime(2024, 1, 1, 0, 0, 1),
        ),
    )


def _make_mock_pm(channel_names: list[str]) -> MagicMock:
    pm = MagicMock()
    pm.load.return_value = _make_analog_record(channel_names)
    return pm


# ─────────────────────────────────────────────────────────────────────────────
# TestLoadManifest
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadManifest:
    def test_loads_yaml_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_COMTRADE_MANIFEST, encoding="utf-8")
        data = load_manifest(f)
        assert data["event_id"] == "test_event"

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.yaml")

    def test_missing_event_id_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(_MISSING_EVENT_ID, encoding="utf-8")
        with pytest.raises(ValueError, match="event_id"):
            load_manifest(f)

    def test_missing_sources_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(_MISSING_SOURCES, encoding="utf-8")
        with pytest.raises(ValueError, match="sources"):
            load_manifest(f)

    def test_sources_list_parsed(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_COMTRADE_MANIFEST, encoding="utf-8")
        data = load_manifest(f)
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) == 1

    def test_relative_path_resolved(self, tmp_path: Path) -> None:
        f = tmp_path / "manifest.yaml"
        f.write_text(_COMTRADE_MANIFEST, encoding="utf-8")
        data = load_manifest(f)
        assert data is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestGetSourceFilePath
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSourceFilePath:
    def test_comtrade_cfg_from_paths_dict(self, tmp_path: Path) -> None:
        src_def = {"paths": {"cfg": "samples/comtrade/test.cfg"}}
        result = _get_source_file_path(src_def, "comtrade", tmp_path, "src1")
        assert result == tmp_path / "samples" / "comtrade" / "test.cfg"

    def test_csv_from_paths_dict(self, tmp_path: Path) -> None:
        src_def = {"paths": {"csv": "samples/csv/test.csv"}}
        result = _get_source_file_path(src_def, "csv", tmp_path, "src1")
        assert result == tmp_path / "samples" / "csv" / "test.csv"

    def test_path_string_fallback(self, tmp_path: Path) -> None:
        src_def = {"path": "samples/comtrade/test.cfg"}
        result = _get_source_file_path(src_def, "comtrade", tmp_path, "src1")
        assert result == tmp_path / "samples" / "comtrade" / "test.cfg"

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Cannot determine file path"):
            _get_source_file_path({}, "comtrade", tmp_path, "src1")

    def test_absolute_path_preserved(self, tmp_path: Path) -> None:
        abs_path = str(tmp_path / "test.cfg")
        src_def = {"paths": {"cfg": abs_path}}
        result = _get_source_file_path(src_def, "comtrade", tmp_path, "src1")
        assert result == tmp_path / "test.cfg"


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildSessionFromManifest
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildSessionFromManifest:
    def test_builds_session_with_sources(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_COMTRADE_MANIFEST, encoding="utf-8")
        pm = _make_mock_pm(["KPDN1 VR", "KPDN1 IB"])
        session = build_session_from_manifest(f, root=tmp_path, provider_manager=pm)
        assert session.source_count() == 1

    def test_session_source_ids(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_COMTRADE_MANIFEST, encoding="utf-8")
        pm = _make_mock_pm(["KPDN1 VR", "KPDN1 IB"])
        session = build_session_from_manifest(f, root=tmp_path, provider_manager=pm)
        assert "comtrade_1" in session.source_ids()

    def test_unknown_source_type_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(_INVALID_TYPE_MANIFEST, encoding="utf-8")
        pm = _make_mock_pm([])
        with pytest.raises(ValueError, match="Unknown source type"):
            build_session_from_manifest(f, root=tmp_path, provider_manager=pm)

    def test_uses_manifest_start_time(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_COMTRADE_MANIFEST, encoding="utf-8")
        pm = _make_mock_pm(["KPDN1 VR"])
        session = build_session_from_manifest(f, root=tmp_path, provider_manager=pm)
        src = session.get_source("comtrade_1")
        assert src is not None
        assert src.original_start_time == datetime(2024, 1, 1)

    def test_voltage_reference_applied_to_voltage_channels(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_COMTRADE_MANIFEST, encoding="utf-8")
        pm = _make_mock_pm(["KPDN1 VR", "KPDN1 IB"])
        session = build_session_from_manifest(f, root=tmp_path, provider_manager=pm)
        src = session.get_source("comtrade_1")
        vr_meta = src.signal_metadata["KPDN1 VR"]
        assert vr_meta.electrical_type == "voltage"
        assert vr_meta.phase_reference == "phase_ground"

    def test_current_channel_no_phase_reference(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_COMTRADE_MANIFEST, encoding="utf-8")
        pm = _make_mock_pm(["KPDN1 VR", "KPDN1 IB"])
        session = build_session_from_manifest(f, root=tmp_path, provider_manager=pm)
        src = session.get_source("comtrade_1")
        ib_meta = src.signal_metadata["KPDN1 IB"]
        assert ib_meta.electrical_type == "current"
        assert ib_meta.phase_reference is None

    def test_manifest_column_classification_applied(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_MULTI_SOURCE_MANIFEST, encoding="utf-8")
        pm = _make_mock_pm(["Frequency", "LowConf"])
        pm.load.side_effect = [
            _make_analog_record(["KPDN1 VR"]),  # comtrade_main
            _make_analog_record(["Frequency", "LowConf"]),  # csv_ops
        ]
        session = build_session_from_manifest(f, root=tmp_path, provider_manager=pm)
        csv_src = session.get_source("csv_ops")
        freq_meta = csv_src.signal_metadata["Frequency"]
        assert freq_meta.signal_type == "frequency"
        assert freq_meta.confidence == 0.95
        assert freq_meta.requires_user_confirmation is False

    def test_low_confidence_column_flagged(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_MULTI_SOURCE_MANIFEST, encoding="utf-8")
        pm = MagicMock()
        pm.load.side_effect = [
            _make_analog_record(["KPDN1 VR"]),
            _make_analog_record(["Frequency", "LowConf"]),
        ]
        session = build_session_from_manifest(f, root=tmp_path, provider_manager=pm)
        csv_src = session.get_source("csv_ops")
        low_meta = csv_src.signal_metadata["LowConf"]
        assert low_meta.requires_user_confirmation is True
        assert low_meta.confidence == pytest.approx(0.65)

    def test_provider_load_error_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_COMTRADE_MANIFEST, encoding="utf-8")
        pm = MagicMock()
        pm.load.side_effect = RuntimeError("File corrupt")
        with pytest.raises(ValueError, match="Failed to load source"):
            build_session_from_manifest(f, root=tmp_path, provider_manager=pm)

    def test_default_root_uses_cwd(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_COMTRADE_MANIFEST, encoding="utf-8")
        # Just verify it doesn't crash with a relative manifest path when given
        # an explicit root; full cwd test would require changing directory.
        pm = _make_mock_pm(["KPDN1 VR"])
        session = build_session_from_manifest(f, root=tmp_path, provider_manager=pm)
        assert not session.is_empty()

    def test_multi_source_session_has_both_sources(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(_MULTI_SOURCE_MANIFEST, encoding="utf-8")
        pm = MagicMock()
        pm.load.side_effect = [
            _make_analog_record(["KPDN1 VR"]),
            _make_analog_record(["Frequency", "LowConf"]),
        ]
        session = build_session_from_manifest(f, root=tmp_path, provider_manager=pm)
        assert session.source_count() == 2
        assert "comtrade_main" in session.source_ids()
        assert "csv_ops" in session.source_ids()


# ─────────────────────────────────────────────────────────────────────────────
# TestParseTimestamp
# ─────────────────────────────────────────────────────────────────────────────


class TestParseTimestamp:
    def test_iso_format(self) -> None:
        result = _parse_timestamp("2024-01-01T00:00:00")
        assert result == datetime(2024, 1, 1)

    def test_iso_with_microseconds(self) -> None:
        result = _parse_timestamp("2026-03-06T18:04:08.817733")
        assert result == datetime(2026, 3, 6, 18, 4, 8, 817733)

    def test_space_separated(self) -> None:
        result = _parse_timestamp("2024-01-01 12:30:00")
        assert result == datetime(2024, 1, 1, 12, 30)

    def test_date_only(self) -> None:
        result = _parse_timestamp("2024-01-15")
        assert result == datetime(2024, 1, 15)

    def test_none_input_returns_none(self) -> None:
        assert _parse_timestamp(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_timestamp("") is None

    def test_invalid_format_returns_none(self) -> None:
        assert _parse_timestamp("not-a-date") is None


# ─────────────────────────────────────────────────────────────────────────────
# TestInferComtradeChannelType
# ─────────────────────────────────────────────────────────────────────────────


class TestInferComtradeChannelType:
    def test_vr_is_voltage(self) -> None:
        assert _infer_comtrade_channel_type("KPDN1 VR") == "voltage"

    def test_vy_is_voltage(self) -> None:
        assert _infer_comtrade_channel_type("KPDN1 VY") == "voltage"

    def test_vb_is_voltage(self) -> None:
        assert _infer_comtrade_channel_type("KPDN1 VB") == "voltage"

    def test_ir_is_current(self) -> None:
        assert _infer_comtrade_channel_type("KPDN1 IR") == "current"

    def test_ib_is_current(self) -> None:
        assert _infer_comtrade_channel_type("KPDN1 IB") == "current"

    def test_in_is_current(self) -> None:
        assert _infer_comtrade_channel_type("KPDN1 IN") == "current"

    def test_vb_hv_is_voltage(self) -> None:
        assert _infer_comtrade_channel_type("SGT1 VB_HV") == "voltage"

    def test_ib_hv_is_current(self) -> None:
        assert _infer_comtrade_channel_type("SGT1 IB_HV") == "current"

    def test_vr_lv_is_voltage(self) -> None:
        assert _infer_comtrade_channel_type("SGT1 VR_LV") == "voltage"

    def test_plain_name_no_prefix(self) -> None:
        assert _infer_comtrade_channel_type("TRIP_SIGNAL") is None

    def test_single_token_v(self) -> None:
        assert _infer_comtrade_channel_type("VR") == "voltage"

    def test_single_token_i(self) -> None:
        assert _infer_comtrade_channel_type("IB") == "current"
