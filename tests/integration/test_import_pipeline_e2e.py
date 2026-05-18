"""End-to-end integration tests for the Import Wizard backend pipeline.

Each test writes a temporary synthetic file, runs run_import_pipeline(), and
asserts properties of the resulting ImportPipelineResult and DisturbanceRecord.

All tests are:
- Backend-only (no Qt, no GUI)
- Deterministic (fixed seeds / fixed data)
- Fast (small synthetic files, ~10–30 rows)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationSeverity
from app.import_wizard.import_pipeline import (
    ImportPipelineOptions,
    run_import_pipeline,
)
from app.models import DisturbanceRecord

openpyxl = pytest.importorskip("openpyxl", reason="openpyxl required for Excel tests")


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _write_standard_csv(path: Path, n: int = 20) -> Path:
    """Write a standard test CSV with timestamp, analog, and digital channels."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="20ms").strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        ),
        "voltage_a": rng.uniform(225.0, 235.0, n),
        "current_a": rng.uniform(0.9, 1.1, n),
        "mw": rng.uniform(99.0, 101.0, n),
        "mvar": rng.uniform(-5.0, 5.0, n),
        "cb1": [i % 2 for i in range(n)],
    })
    df.to_csv(path, index=False)
    return path


def _write_standard_xlsx(path: Path, n: int = 20) -> Path:
    """Write a standard test XLSX with same columns as the CSV fixture."""
    rng = np.random.default_rng(43)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="20ms").strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        ),
        "voltage_a": rng.uniform(225.0, 235.0, n),
        "current_a": rng.uniform(0.9, 1.1, n),
        "mw": rng.uniform(99.0, 101.0, n),
    })
    df.to_excel(path, index=False)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CSV end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestCSVPipelineE2E:
    @pytest.fixture
    def csv_path(self, tmp_path: Path) -> Path:
        return _write_standard_csv(tmp_path / "recording.csv")

    def test_success(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.success

    def test_record_produced(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.record is not None
        assert isinstance(result.record, DisturbanceRecord)

    def test_record_passes_validate(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.record is not None
        assert result.record.validate() == []

    def test_waveform_data_has_time_column(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert "time" in result.record.waveform_data.columns  # type: ignore[union-attr]

    def test_time_column_starts_at_zero(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        first_t = result.record.waveform_data["time"].iloc[0]  # type: ignore[union-attr]
        assert abs(first_t) < 1e-6

    def test_analog_channels_detected(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.record is not None
        assert len(result.record.analog_channels) >= 1

    def test_digital_channel_detected(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.record is not None
        assert len(result.record.digital_channels) >= 1

    def test_row_count_matches_file(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.diagnostics.normalized_row_count == 20

    def test_no_validation_errors(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        errors = [m for m in result.validation_messages if m.severity == ValidationSeverity.ERROR]
        assert errors == []

    def test_session_has_timestamp_column(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.session.selected_timestamp_column == "timestamp"

    def test_profile_preserved_in_result(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.profile is not None
        assert len(result.profile.timestamp_candidates) >= 1

    def test_candidate_preserved_in_result(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.selected_candidate is not None
        assert result.selected_candidate.column_name == "timestamp"

    def test_repair_plan_preserved(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.repair_plan is not None
        assert result.repair_plan.repair_validated is True

    def test_normalization_result_preserved(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.normalization_result is not None
        assert result.normalization_result.success

    def test_dataset_preserved(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.dataset is not None
        assert result.dataset.is_valid

    def test_bridge_result_preserved(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.bridge_result is not None
        assert result.bridge_result.success

    def test_diagnostics_elapsed_positive(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.diagnostics.elapsed_seconds is not None
        assert result.diagnostics.elapsed_seconds > 0.0

    def test_diagnostics_analog_count(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.diagnostics.analog_channel_count >= 1

    def test_diagnostics_digital_count(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.diagnostics.digital_channel_count >= 1

    def test_source_csv_not_mutated(self, csv_path: Path):
        original = pd.read_csv(csv_path)
        original_shape = original.shape
        run_import_pipeline(str(csv_path))
        reloaded = pd.read_csv(csv_path)
        assert reloaded.shape == original_shape

    def test_nominal_frequency_option_passed(self, csv_path: Path):
        opts = ImportPipelineOptions(nominal_frequency=60.0)
        result = run_import_pipeline(str(csv_path), options=opts)
        assert result.record is not None
        assert result.record.metadata.nominal_frequency == 60.0

    def test_station_name_option_passed(self, csv_path: Path):
        opts = ImportPipelineOptions(station_name="Test Substation")
        result = run_import_pipeline(str(csv_path), options=opts)
        assert result.record is not None
        assert result.record.metadata.station_name == "Test Substation"

    def test_metadata_source_file_set(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.record is not None
        assert result.record.metadata.source_file != ""

    def test_channel_names_in_waveform(self, csv_path: Path):
        result = run_import_pipeline(str(csv_path))
        assert result.record is not None
        for ch in result.record.analog_channels:
            assert ch.name in result.record.waveform_data.columns
        for ch in result.record.digital_channels:
            assert ch.name in result.record.waveform_data.columns


# ─────────────────────────────────────────────────────────────────────────────
# Excel end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestExcelPipelineE2E:
    @pytest.fixture
    def xlsx_path(self, tmp_path: Path) -> Path:
        return _write_standard_xlsx(tmp_path / "recording.xlsx")

    def test_success(self, xlsx_path: Path):
        result = run_import_pipeline(str(xlsx_path))
        assert result.success, f"Expected success but got: {[m.message for m in result.validation_messages if m.severity == ValidationSeverity.ERROR]}"

    def test_record_produced(self, xlsx_path: Path):
        result = run_import_pipeline(str(xlsx_path))
        assert result.record is not None

    def test_record_passes_validate(self, xlsx_path: Path):
        result = run_import_pipeline(str(xlsx_path))
        assert result.record is not None
        assert result.record.validate() == []

    def test_provider_type_excel(self, xlsx_path: Path):
        result = run_import_pipeline(str(xlsx_path))
        assert result.diagnostics.provider_type == "excel"

    def test_provider_type_in_metadata(self, xlsx_path: Path):
        result = run_import_pipeline(str(xlsx_path))
        assert result.record is not None
        assert result.record.metadata.provider_type == "normalized_excel"

    def test_analog_channels_present(self, xlsx_path: Path):
        result = run_import_pipeline(str(xlsx_path))
        assert result.record is not None
        assert len(result.record.analog_channels) >= 1

    def test_row_count_correct(self, xlsx_path: Path):
        result = run_import_pipeline(str(xlsx_path))
        assert result.diagnostics.normalized_row_count == 20

    def test_waveform_has_time_column(self, xlsx_path: Path):
        result = run_import_pipeline(str(xlsx_path))
        assert "time" in result.record.waveform_data.columns  # type: ignore[union-attr]


# ─────────────────────────────────────────────────────────────────────────────
# Provider type handling
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderTypeHandling:
    def test_csv_inferred_from_extension(self, tmp_path: Path):
        p = _write_standard_csv(tmp_path / "data.csv")
        result = run_import_pipeline(str(p))
        assert result.diagnostics.provider_type == "csv"

    def test_excel_inferred_from_xlsx_extension(self, tmp_path: Path):
        p = _write_standard_xlsx(tmp_path / "data.xlsx")
        result = run_import_pipeline(str(p))
        assert result.diagnostics.provider_type == "excel"

    def test_explicit_csv_provider_used(self, tmp_path: Path):
        p = _write_standard_csv(tmp_path / "data.csv")
        result = run_import_pipeline(str(p), provider_type="csv")
        assert result.diagnostics.provider_type == "csv"

    def test_csv_provider_type_in_metadata(self, tmp_path: Path):
        p = _write_standard_csv(tmp_path / "data.csv")
        result = run_import_pipeline(str(p))
        assert result.record is not None
        assert result.record.metadata.provider_type == "normalized_csv"


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_unknown_column_preserved_as_analog_with_warning(self, tmp_path: Path):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "xyzzy": [float(i) for i in range(10)],
        })
        p = tmp_path / "unknown.csv"
        df.to_csv(p, index=False)
        result = run_import_pipeline(str(p))

        warning_codes = [m.code for m in result.validation_messages if m.severity == ValidationSeverity.WARNING]
        assert "PIPELINE_UNKNOWN_COLUMN" in warning_codes

        if result.success and result.record:
            analog_names = [ch.name for ch in result.record.analog_channels]
            assert len(analog_names) >= 1

    def test_digital_column_converted_correctly(self, tmp_path: Path):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "voltage_a": [230.0 + i for i in range(10)],
            "cb1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        p = tmp_path / "digital.csv"
        df.to_csv(p, index=False)
        result = run_import_pipeline(str(p))

        assert result.success
        assert result.record is not None
        assert len(result.record.digital_channels) >= 1
        cb_data = result.record.waveform_data[result.record.digital_channels[0].name]
        assert set(cb_data.unique()).issubset({0, 1})

    def test_no_data_columns_fails_gracefully(self, tmp_path: Path):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        })
        p = tmp_path / "ts_only.csv"
        df.to_csv(p, index=False)
        result = run_import_pipeline(str(p))
        # Should not raise; result.success may be False
        assert result is not None
        assert isinstance(result.success, bool)

    def test_partial_nat_timestamps_handled(self, tmp_path: Path):
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1s").strftime(
            "%Y-%m-%d %H:%M:%S"
        ).tolist()
        timestamps[3] = ""  # introduce one blank timestamp
        df = pd.DataFrame({
            "timestamp": timestamps,
            "mw": [float(i) for i in range(10)],
        })
        p = tmp_path / "partial_nat.csv"
        df.to_csv(p, index=False)
        result = run_import_pipeline(str(p))
        # Pipeline should not crash; it may or may not succeed depending on parse
        assert result is not None
        assert not any(
            isinstance(e, Exception) for e in [result.record, result.dataset]
            if e is not None
        )

    def test_unsupported_extension_no_crash(self, tmp_path: Path):
        p = tmp_path / "data.docx"
        p.write_bytes(b"PK\x03\x04")  # DOCX magic bytes
        result = run_import_pipeline(str(p))
        assert not result.success
        assert result.diagnostics is not None

    def test_final_record_validate_succeeds_on_success(self, tmp_path: Path):
        p = _write_standard_csv(tmp_path / "val.csv")
        result = run_import_pipeline(str(p))
        if result.success:
            assert result.record is not None
            assert result.record.validate() == []

    def test_sample_count_matches_timestamp_count(self, tmp_path: Path):
        n = 15
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "mw": [float(i) for i in range(n)],
        })
        p = tmp_path / "count.csv"
        df.to_csv(p, index=False)
        result = run_import_pipeline(str(p))
        if result.success:
            assert result.diagnostics.normalized_row_count == n
            assert len(result.record.waveform_data) == n  # type: ignore[union-attr]

    def test_semicolon_delimited_csv(self, tmp_path: Path):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "voltage_a": [230.0 + i for i in range(10)],
        })
        p = tmp_path / "semicolon.csv"
        df.to_csv(p, index=False, sep=";")
        result = run_import_pipeline(str(p))
        assert result is not None

    def test_large_ish_file_completes(self, tmp_path: Path):
        n = 5000
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="20ms").strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            ),
            "voltage_a": rng.uniform(225.0, 235.0, n),
            "current_a": rng.uniform(0.9, 1.1, n),
            "mw": rng.uniform(99.0, 101.0, n),
        })
        p = tmp_path / "large.csv"
        df.to_csv(p, index=False)
        result = run_import_pipeline(str(p))
        assert result is not None
        assert result.success
        assert result.diagnostics.normalized_row_count == n

    def test_iso8601_timestamp_format(self, tmp_path: Path):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s").strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "mw": [float(i) for i in range(10)],
        })
        p = tmp_path / "iso.csv"
        df.to_csv(p, index=False)
        result = run_import_pipeline(str(p))
        assert result.success

    def test_dmy_timestamp_format(self, tmp_path: Path):
        df = pd.DataFrame({
            "timestamp": [
                f"{d:02d}/01/2024 12:{m:02d}:00" for d, m in enumerate(range(10), start=1)
            ],
            "mw": [float(i) for i in range(10)],
        })
        p = tmp_path / "dmy.csv"
        df.to_csv(p, index=False)
        result = run_import_pipeline(str(p))
        # Should at minimum not crash and detect a timestamp candidate
        assert result is not None
        assert result.selected_candidate is not None
