"""Integration tests: YAML manifest -> MultiSourceSession -> SignalMetadata.

Verifies that the full pipeline from manifest parsing through SourceRecord
construction produces correctly annotated SignalMetadata without mutating
the underlying DisturbanceRecord objects.

All provider I/O is replaced by MagicMock so no real files are read.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.data.manifest_loader import build_session_from_manifest
from app.models import (
    AnalogChannel,
    DisturbanceRecord,
    RecordingMetadata,
    SamplingInformation,
    TimingInformation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared manifest templates
# ─────────────────────────────────────────────────────────────────────────────


_CSV_MANIFEST = """\
event_id: integ_csv
sources:
  - source_id: csv_ops
    type: csv
    paths:
      csv: samples/csv/test.csv
    start_time: "2026-03-06T17:25:00"
    columns:
      - name: Time.1
        display_group: other
        confidence: 0.0
        inferred_from: unknown
        requires_user_confirmation: true
      - name: System Demand
        signal_type: active_power
        unit: MW
        display_group: power
        confidence: 0.85
        inferred_from: name_keyword
        requires_user_confirmation: false
      - name: Tie-Line
        signal_type: active_power
        unit: MW
        display_group: power
        confidence: 0.70
        inferred_from: name_keyword
        requires_user_confirmation: true
      - name: Frequency
        signal_type: frequency
        unit: Hz
        display_group: frequency
        confidence: 0.95
        inferred_from: name_exact
        requires_user_confirmation: false
"""

_COMTRADE_MANIFEST = """\
event_id: integ_comtrade
sources:
  - source_id: comtrade_main
    type: comtrade
    paths:
      cfg: samples/comtrade/test.cfg
    voltage_reference: phase_ground
    start_time: "2026-03-06T18:04:08"
"""

_MULTI_SOURCE_MANIFEST = """\
event_id: integ_multi
sources:
  - source_id: comtrade_main
    type: comtrade
    paths:
      cfg: samples/comtrade/test.cfg
    voltage_reference: phase_ground
    start_time: "2026-03-06T18:04:08"
  - source_id: csv_ops
    type: csv
    paths:
      csv: samples/csv/test.csv
    start_time: "2026-03-06T17:25:00"
    columns:
      - name: Frequency
        signal_type: frequency
        unit: Hz
        display_group: frequency
        confidence: 0.95
        inferred_from: name_exact
        requires_user_confirmation: false
      - name: System Demand
        signal_type: active_power
        unit: MW
        display_group: power
        confidence: 0.85
        inferred_from: name_keyword
        requires_user_confirmation: false
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_record(channel_names: list[str], source_file: str = "test.cfg") -> DisturbanceRecord:
    """Minimal DisturbanceRecord with the given analog channel names."""
    channels = [
        AnalogChannel(
            name=n,
            unit="kV" if ("V" in n.split()[-1].split("_")[0] and n.split()[-1].split("_")[0][0] == "V") else "kA",
            index=i,
        )
        for i, n in enumerate(channel_names)
    ]
    n = 10
    data: dict = {"time": np.linspace(0, 1, n)}
    for ch in channels:
        data[ch.name] = np.zeros(n)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST",
            recorder_name="test",
            source_file=source_file,
            provider_type="comtrade",
            nominal_frequency=50.0,
        ),
        waveform_data=pd.DataFrame(data),
        analog_channels=channels,
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[5000.0],
            samples_per_rate=[n],
        ),
        timing_info=TimingInformation(
            start_time=datetime(2026, 3, 6),
            trigger_time=datetime(2026, 3, 6, 0, 0, 1),
        ),
    )


def _mock_pm(*records: DisturbanceRecord) -> MagicMock:
    """Mock provider that returns records in order of calls."""
    pm = MagicMock()
    pm.load.side_effect = list(records)
    return pm


def _write_manifest(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# TestCsvSignalMetadataFromManifest
# ─────────────────────────────────────────────────────────────────────────────


class TestCsvSignalMetadataFromManifest:
    """Manifest columns section → SignalMetadata fields preserved correctly."""

    @pytest.fixture
    def session(self, tmp_path: Path):
        csv_channels = ["Time.1", "System Demand", "Tie-Line", "Frequency"]
        record = _make_record(csv_channels, source_file="test.csv")
        pm = _mock_pm(record)
        manifest = _write_manifest(tmp_path, _CSV_MANIFEST)
        return build_session_from_manifest(manifest, root=tmp_path, provider_manager=pm)

    def test_session_has_one_source(self, session) -> None:
        assert session.source_count() == 1

    def test_source_id_correct(self, session) -> None:
        assert session.source_ids() == ["csv_ops"]

    def test_frequency_signal_type(self, session) -> None:
        meta = session.sources[0].signal_metadata["Frequency"]
        assert meta.signal_type == "frequency"

    def test_frequency_unit(self, session) -> None:
        meta = session.sources[0].signal_metadata["Frequency"]
        assert meta.unit == "Hz"

    def test_frequency_display_group(self, session) -> None:
        meta = session.sources[0].signal_metadata["Frequency"]
        assert meta.display_group == "frequency"

    def test_frequency_confidence(self, session) -> None:
        meta = session.sources[0].signal_metadata["Frequency"]
        assert meta.confidence == pytest.approx(0.95)

    def test_frequency_inferred_from(self, session) -> None:
        meta = session.sources[0].signal_metadata["Frequency"]
        assert meta.inferred_from == "name_exact"

    def test_frequency_no_confirmation_required(self, session) -> None:
        meta = session.sources[0].signal_metadata["Frequency"]
        assert meta.requires_user_confirmation is False

    def test_system_demand_signal_type(self, session) -> None:
        meta = session.sources[0].signal_metadata["System Demand"]
        assert meta.signal_type == "active_power"

    def test_system_demand_unit(self, session) -> None:
        meta = session.sources[0].signal_metadata["System Demand"]
        assert meta.unit == "MW"

    def test_system_demand_no_confirmation_required(self, session) -> None:
        meta = session.sources[0].signal_metadata["System Demand"]
        assert meta.requires_user_confirmation is False

    def test_tie_line_low_confidence(self, session) -> None:
        meta = session.sources[0].signal_metadata["Tie-Line"]
        assert meta.confidence == pytest.approx(0.70)

    def test_tie_line_requires_confirmation(self, session) -> None:
        meta = session.sources[0].signal_metadata["Tie-Line"]
        assert meta.requires_user_confirmation is True

    def test_time_artifact_no_signal_type(self, session) -> None:
        meta = session.sources[0].signal_metadata["Time.1"]
        assert meta.signal_type is None

    def test_time_artifact_zero_confidence(self, session) -> None:
        meta = session.sources[0].signal_metadata["Time.1"]
        assert meta.confidence == pytest.approx(0.0)

    def test_time_artifact_requires_confirmation(self, session) -> None:
        meta = session.sources[0].signal_metadata["Time.1"]
        assert meta.requires_user_confirmation is True


# ─────────────────────────────────────────────────────────────────────────────
# TestComtradeVoltageReference
# ─────────────────────────────────────────────────────────────────────────────


class TestComtradeVoltageReference:
    """voltage_reference in manifest -> phase_reference on voltage channels."""

    @pytest.fixture
    def session(self, tmp_path: Path):
        record = _make_record(["KPDN1 VR", "KPDN1 VY", "KPDN1 IB", "KPDN1 IN"])
        pm = _mock_pm(record)
        manifest = _write_manifest(tmp_path, _COMTRADE_MANIFEST)
        return build_session_from_manifest(manifest, root=tmp_path, provider_manager=pm)

    def test_voltage_channel_gets_phase_reference(self, session) -> None:
        meta = session.sources[0].signal_metadata["KPDN1 VR"]
        assert meta.phase_reference == "phase_ground"

    def test_all_voltage_channels_annotated(self, session) -> None:
        src = session.sources[0]
        assert src.signal_metadata["KPDN1 VR"].phase_reference == "phase_ground"
        assert src.signal_metadata["KPDN1 VY"].phase_reference == "phase_ground"

    def test_current_channel_no_phase_reference(self, session) -> None:
        meta = session.sources[0].signal_metadata["KPDN1 IB"]
        assert meta.phase_reference is None

    def test_neutral_current_no_phase_reference(self, session) -> None:
        meta = session.sources[0].signal_metadata["KPDN1 IN"]
        assert meta.phase_reference is None

    def test_voltage_electrical_type(self, session) -> None:
        meta = session.sources[0].signal_metadata["KPDN1 VR"]
        assert meta.electrical_type == "voltage"

    def test_current_electrical_type(self, session) -> None:
        meta = session.sources[0].signal_metadata["KPDN1 IB"]
        assert meta.electrical_type == "current"


# ─────────────────────────────────────────────────────────────────────────────
# TestOriginalRecordNotMutated
# ─────────────────────────────────────────────────────────────────────────────


class TestOriginalRecordNotMutated:
    """Loading a manifest must not modify the underlying DisturbanceRecord."""

    def test_record_identity_preserved(self, tmp_path: Path) -> None:
        original = _make_record(["KPDN1 VR", "KPDN1 IB"])
        original_id = id(original)
        pm = _mock_pm(original)
        manifest = _write_manifest(tmp_path, _COMTRADE_MANIFEST)
        session = build_session_from_manifest(manifest, root=tmp_path, provider_manager=pm)
        assert id(session.sources[0].record) == original_id

    def test_original_channels_unchanged(self, tmp_path: Path) -> None:
        original = _make_record(["KPDN1 VR", "KPDN1 IB"])
        original_ch_names = [ch.name for ch in original.analog_channels]
        pm = _mock_pm(original)
        manifest = _write_manifest(tmp_path, _COMTRADE_MANIFEST)
        build_session_from_manifest(manifest, root=tmp_path, provider_manager=pm)
        assert [ch.name for ch in original.analog_channels] == original_ch_names

    def test_original_waveform_data_unchanged(self, tmp_path: Path) -> None:
        original = _make_record(["KPDN1 VR"])
        original_shape = original.waveform_data.shape
        pm = _mock_pm(original)
        manifest = _write_manifest(tmp_path, _COMTRADE_MANIFEST)
        build_session_from_manifest(manifest, root=tmp_path, provider_manager=pm)
        assert original.waveform_data.shape == original_shape


# ─────────────────────────────────────────────────────────────────────────────
# TestMultiSourceSession
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiSourceSession:
    """Multi-source manifest produces session with correct source bookkeeping."""

    @pytest.fixture
    def session(self, tmp_path: Path):
        comtrade_record = _make_record(["KPDN1 VR", "KPDN1 IB"])
        csv_record = _make_record(["Frequency", "System Demand"], source_file="test.csv")
        pm = _mock_pm(comtrade_record, csv_record)
        manifest = _write_manifest(tmp_path, _MULTI_SOURCE_MANIFEST)
        return build_session_from_manifest(manifest, root=tmp_path, provider_manager=pm)

    def test_source_count(self, session) -> None:
        assert session.source_count() == 2

    def test_source_ids(self, session) -> None:
        assert session.source_ids() == ["comtrade_main", "csv_ops"]

    def test_session_not_empty(self, session) -> None:
        assert not session.is_empty()

    def test_get_source_by_id(self, session) -> None:
        src = session.get_source("csv_ops")
        assert src is not None
        assert src.source_id == "csv_ops"

    def test_comtrade_source_has_voltage_annotation(self, session) -> None:
        src = session.get_source("comtrade_main")
        assert src is not None
        assert src.signal_metadata["KPDN1 VR"].phase_reference == "phase_ground"

    def test_csv_source_has_signal_types(self, session) -> None:
        src = session.get_source("csv_ops")
        assert src is not None
        assert src.signal_metadata["Frequency"].signal_type == "frequency"
        assert src.signal_metadata["System Demand"].signal_type == "active_power"

    def test_provider_called_once_per_source(self, tmp_path: Path) -> None:
        comtrade_record = _make_record(["KPDN1 VR"])
        csv_record = _make_record(["Frequency"], source_file="test.csv")
        pm = _mock_pm(comtrade_record, csv_record)
        manifest = _write_manifest(tmp_path, _MULTI_SOURCE_MANIFEST)
        build_session_from_manifest(manifest, root=tmp_path, provider_manager=pm)
        assert pm.load.call_count == 2

    def test_start_times_parsed(self, session) -> None:
        comtrade_src = session.get_source("comtrade_main")
        csv_src = session.get_source("csv_ops")
        assert comtrade_src is not None
        assert csv_src is not None
        assert comtrade_src.original_start_time == datetime(2026, 3, 6, 18, 4, 8)
        assert csv_src.original_start_time == datetime(2026, 3, 6, 17, 25, 0)

    def test_sampling_rates_from_record(self, session) -> None:
        src = session.get_source("comtrade_main")
        assert src is not None
        assert src.sampling_rates == [5000.0]
