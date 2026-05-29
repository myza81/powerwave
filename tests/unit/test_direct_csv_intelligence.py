"""Tests for Phase D4.4 — direct CSV/Excel intelligence integration.

Covers:
  - build_signal_metadata: display_group assignment for pulu columns
  - detect_timestamp_ambiguity: ambiguous format detection
  - display group aliases (voltage_rms → voltage_raw, etc.)
  - _log_direct_open_mapping: smoke test
  - _DirectOpenResult dataclass shape
  - Main window _on_record_loaded dispatch (provider_type branch)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.data.direct_load_intelligence import (
    _DISPLAY_GROUP_ALIASES,
    build_signal_metadata,
    detect_timestamp_ambiguity,
)
from app.data.signal_metadata import SignalMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_PULU_CSV = Path("samples/csv/pulu_20260306.csv")


def _make_record(channels: list[str], values: dict[str, list] | None = None):
    """Build a minimal DisturbanceRecord mock for intelligence tests."""
    from app.models import AnalogChannel, DisturbanceRecord, RecordingMetadata
    from app.models import SamplingInformation, TimingInformation

    t = np.linspace(0, 10, 30)
    data: dict = {"time": t}
    analog = []
    for i, name in enumerate(channels):
        v = values.get(name, [1.0] * 30) if values else [1.0] * 30
        data[name] = np.array(v[:30] if len(v) >= 30 else v + [v[-1]] * (30 - len(v)))
        analog.append(AnalogChannel(name=name, unit="MW", index=i))

    waveform_data = pd.DataFrame(data)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST",
            recorder_name="CSV",
            source_file="test.csv",
            provider_type="csv",
            nominal_frequency=50.0,
        ),
        waveform_data=waveform_data,
        analog_channels=analog,
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[1.0], samples_per_rate=[30]),
        timing_info=TimingInformation(
            start_time=datetime(2026, 3, 6, 17, 25, 0),
            trigger_time=datetime(2026, 3, 6, 17, 25, 0),
        ),
        disturbance_info=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Display-group alias table
# ─────────────────────────────────────────────────────────────────────────────


class TestDisplayGroupAliases:
    def test_voltage_rms_maps_to_voltage_raw(self) -> None:
        assert _DISPLAY_GROUP_ALIASES["voltage_rms"] == "voltage_raw"

    def test_current_rms_maps_to_current_raw(self) -> None:
        assert _DISPLAY_GROUP_ALIASES["current_rms"] == "current_raw"

    def test_voltage_maps_to_voltage_raw(self) -> None:
        assert _DISPLAY_GROUP_ALIASES["voltage"] == "voltage_raw"

    def test_current_maps_to_current_raw(self) -> None:
        assert _DISPLAY_GROUP_ALIASES["current"] == "current_raw"


# ─────────────────────────────────────────────────────────────────────────────
# build_signal_metadata — pulu channel names
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildSignalMetadataPuluColumns:
    """Verify pulu_20260306 column names map to the expected display groups."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from app.data.intelligence import IntelligenceManager
        self.mgr = IntelligenceManager()

    def _metadata(self, col: str, val=None) -> SignalMetadata:
        record = _make_record([col], {col: val or [50.0] * 30})
        result = build_signal_metadata(record, self.mgr, "pulu", "csv")
        return result[col]

    def test_system_demand_is_power_group(self) -> None:
        meta = self._metadata("System Demand", [18738.0] * 30)
        assert meta.display_group == "power"

    def test_tie_line_is_power_group(self) -> None:
        meta = self._metadata("Tie-Line", [108.0] * 30)
        assert meta.display_group == "power"

    def test_frequency_is_frequency_group(self) -> None:
        meta = self._metadata("Frequency", [50.02] * 30)
        assert meta.display_group == "frequency"

    def test_system_demand_signal_type_is_active_power(self) -> None:
        meta = self._metadata("System Demand", [18738.0] * 30)
        assert meta.signal_type == "active_power"

    def test_frequency_signal_type_is_frequency(self) -> None:
        meta = self._metadata("Frequency", [50.02] * 30)
        assert meta.signal_type == "frequency"

    def test_metadata_source_set(self) -> None:
        meta = self._metadata("Frequency", [50.02] * 30)
        assert meta.source == "pulu"

    def test_metadata_has_confidence(self) -> None:
        meta = self._metadata("Frequency", [50.02] * 30)
        assert meta.confidence is not None
        assert meta.confidence > 0.0

    def test_tie_line_requires_user_confirmation(self) -> None:
        meta = self._metadata("Tie-Line", [108.0] * 30)
        # Tie-Line is below CONFIRMATION_THRESHOLD (0.80)
        assert meta.requires_user_confirmation is True

    def test_system_demand_high_confidence_no_confirmation(self) -> None:
        meta = self._metadata("System Demand", [18738.0] * 30)
        assert meta.requires_user_confirmation is False

    def test_frequency_high_confidence_no_confirmation(self) -> None:
        meta = self._metadata("Frequency", [50.02] * 30)
        assert meta.requires_user_confirmation is False


class TestBuildSignalMetadataAliases:
    """Ensure voltage_rms / current_rms are aliased to voltage_raw / current_raw."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from app.data.intelligence import IntelligenceManager
        self.mgr = IntelligenceManager()

    def test_voltage_column_maps_to_voltage_raw(self) -> None:
        record = _make_record(["Bus Voltage"], {"Bus Voltage": [1.0] * 30})
        result = build_signal_metadata(record, self.mgr, "src", "csv")
        assert result["Bus Voltage"].display_group == "voltage_raw"

    def test_current_column_maps_to_current_raw(self) -> None:
        record = _make_record(["Line Current"], {"Line Current": [100.0] * 30})
        result = build_signal_metadata(record, self.mgr, "src", "csv")
        assert result["Line Current"].display_group == "current_raw"


class TestBuildSignalMetadataReturnShape:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from app.data.intelligence import IntelligenceManager
        self.mgr = IntelligenceManager()

    def test_all_channels_present(self) -> None:
        record = _make_record(["System Demand", "Tie-Line", "Frequency"])
        result = build_signal_metadata(record, self.mgr, "pulu", "csv")
        assert set(result.keys()) == {"System Demand", "Tie-Line", "Frequency"}

    def test_all_values_are_signal_metadata(self) -> None:
        record = _make_record(["System Demand"])
        result = build_signal_metadata(record, self.mgr, "pulu", "csv")
        assert all(isinstance(v, SignalMetadata) for v in result.values())

    def test_empty_record_returns_empty_dict(self) -> None:
        record = _make_record([])
        result = build_signal_metadata(record, self.mgr, "pulu", "csv")
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# detect_timestamp_ambiguity — pulu_20260306.csv
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not _PULU_CSV.exists(),
    reason="pulu_20260306.csv not present",
)
class TestDetectTimestampAmbiguityPulu:
    @pytest.fixture(autouse=True)
    def _record(self):
        from app.data.intelligence import IntelligenceManager
        from app.providers import ProviderManager, CsvProvider
        pm = ProviderManager()
        pm.register_provider(CsvProvider(IntelligenceManager()))
        self.record = pm.load(_PULU_CSV)

    def test_pulu_timestamp_is_ambiguous(self) -> None:
        is_ambiguous, matrices = detect_timestamp_ambiguity(_PULU_CSV, self.record)
        assert is_ambiguous is True

    def test_pulu_ambiguity_returns_matrices(self) -> None:
        is_ambiguous, matrices = detect_timestamp_ambiguity(_PULU_CSV, self.record)
        assert len(matrices) > 0

    def test_pulu_matrix_has_multiple_interpretations(self) -> None:
        _, matrices = detect_timestamp_ambiguity(_PULU_CSV, self.record)
        for matrix in matrices.values():
            assert len(matrix.interpretations) >= 2


class TestDetectTimestampAmbiguityEdgeCases:
    @pytest.fixture(autouse=True)
    def _record(self):
        from app.models import DisturbanceRecord, RecordingMetadata
        from app.models import SamplingInformation, TimingInformation
        self.dummy_record = DisturbanceRecord(
            metadata=RecordingMetadata(
                station_name="X", recorder_name="X",
                source_file="x.csv", provider_type="csv", nominal_frequency=50.0,
            ),
            waveform_data=pd.DataFrame({"time": [0.0, 1.0]}),
            analog_channels=[],
            digital_channels=[],
            sampling_info=SamplingInformation(sampling_rates=[1.0], samples_per_rate=[2]),
            timing_info=TimingInformation(
                start_time=datetime(2000, 1, 1),
                trigger_time=datetime(2000, 1, 1),
            ),
            disturbance_info=None,
        )

    def test_nonexistent_file_returns_not_ambiguous(self) -> None:
        is_ambiguous, matrices = detect_timestamp_ambiguity(
            Path("nonexistent.csv"), self.dummy_record
        )
        assert is_ambiguous is False
        assert matrices == {}

    def test_unsupported_suffix_returns_not_ambiguous(self, tmp_path) -> None:
        p = tmp_path / "data.comtrade"
        p.write_text("dummy")
        is_ambiguous, _ = detect_timestamp_ambiguity(p, self.dummy_record)
        assert is_ambiguous is False

    def test_csv_without_time_column_returns_not_ambiguous(self, tmp_path) -> None:
        p = tmp_path / "data.csv"
        p.write_text("val1,val2\n1.0,2.0\n3.0,4.0\n")
        is_ambiguous, _ = detect_timestamp_ambiguity(p, self.dummy_record)
        assert is_ambiguous is False

    def test_iso_timestamp_not_ambiguous(self, tmp_path) -> None:
        p = tmp_path / "iso.csv"
        rows = ["time,val\n"]
        rows += [f"2026-03-06 17:2{i}:00,{i}\n" for i in range(10)]
        p.write_text("".join(rows))
        is_ambiguous, _ = detect_timestamp_ambiguity(p, self.dummy_record)
        assert is_ambiguous is False


# ─────────────────────────────────────────────────────────────────────────────
# _log_direct_open_mapping — smoke test
# ─────────────────────────────────────────────────────────────────────────────


def test_log_direct_open_mapping_writes_to_stderr(capsys) -> None:
    from app.ui.main_window.main_window import _log_direct_open_mapping

    meta = MagicMock()
    meta.display_group = "power"
    _log_direct_open_mapping("test.csv", {"SystemLoad": meta})
    captured = capsys.readouterr()
    assert "test.csv" in captured.err
    assert "power" in captured.err


# ─────────────────────────────────────────────────────────────────────────────
# _DirectOpenResult dataclass
# ─────────────────────────────────────────────────────────────────────────────


class TestDirectOpenResult:
    def test_fields_accessible(self) -> None:
        from app.ui.main_window.main_window import _DirectOpenResult
        record = MagicMock()
        result = _DirectOpenResult(
            record=record,
            path=Path("test.csv"),
            provider_type="csv",
            signal_metadata={},
            ts_ambiguous=False,
            ts_matrices={},
        )
        assert result.provider_type == "csv"
        assert result.ts_ambiguous is False

    def test_comtrade_provider_type(self) -> None:
        from app.ui.main_window.main_window import _DirectOpenResult
        result = _DirectOpenResult(
            record=MagicMock(),
            path=Path("test.cfg"),
            provider_type="comtrade",
            signal_metadata={},
            ts_ambiguous=False,
            ts_matrices={},
        )
        assert result.provider_type == "comtrade"


# ─────────────────────────────────────────────────────────────────────────────
# Attribute inspection helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_intelligent_load_worker_class_exists() -> None:
    from app.ui.main_window.main_window import _IntelligentLoadWorker
    assert _IntelligentLoadWorker is not None
