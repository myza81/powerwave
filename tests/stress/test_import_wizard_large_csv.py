"""Stress coverage for deterministic large-ish Import Wizard CSV inputs."""
from __future__ import annotations

from app.import_wizard import ExportWriteOptions, write_normalized_export
from app.import_wizard.import_pipeline import run_import_pipeline
from tools.generate_import_stress_samples import StressSampleConfig, generate_import_stress_csv


def test_generated_small_csv_import_visualization_handoff(runtime_tmp_path) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "small.csv",
        StressSampleConfig(row_count=1_000, digital_text_values=True),
    )

    result = run_import_pipeline(str(csv_path))

    assert result.success
    assert result.record is not None
    assert result.diagnostics.normalized_row_count == 1_000
    assert "time" in result.record.waveform_data.columns
    assert len(result.record.analog_channels) >= 1
    assert len(result.record.digital_channels) >= 1


def test_generated_medium_csv_import_export_and_sidecar(runtime_tmp_path) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "medium.csv",
        StressSampleConfig(row_count=25_000, analog_column_count=8, digital_column_count=2),
    )

    result = run_import_pipeline(str(csv_path))
    out = runtime_tmp_path / "medium_normalized.csv"
    export_result = write_normalized_export(
        result.dataset,
        out,
        options=ExportWriteOptions(include_metadata_sidecar=True, overwrite=True),
    )

    assert result.success
    assert result.dataset is not None
    assert result.diagnostics.normalized_row_count == 25_000
    assert export_result.success
    assert export_result.rows_written == 25_000
    assert export_result.metadata_path is not None


def test_generated_csv_with_metadata_and_timestamp_gaps_is_recoverable(runtime_tmp_path) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "gappy.csv",
        StressSampleConfig(
            row_count=1_000,
            metadata_rows=3,
            missing_timestamp_ratio=0.02,
            malformed_timestamp_ratio=0.01,
            duplicate_timestamp_ratio=0.03,
            digital_text_values=True,
        ),
    )

    result = run_import_pipeline(str(csv_path))
    codes = {message.code for message in result.validation_messages}

    assert result.success
    assert result.diagnostics.normalized_row_count < 1_000
    assert "TS_DROPPED_ROWS" in codes
    assert "TS_DUPLICATES" in codes
