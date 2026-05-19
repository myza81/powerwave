"""Malformed real-world CSV cases for Import Wizard hardening."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.import_wizard.contracts import ValidationSeverity
from app.import_wizard.import_pipeline import run_import_pipeline
from tools.generate_import_stress_samples import StressSampleConfig, generate_import_stress_csv


@pytest.mark.parametrize("delimiter", [";", "\t", "|"])
def test_delimiter_variants_import_successfully(runtime_tmp_path, delimiter: str) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "delimited.csv",
        StressSampleConfig(row_count=250, delimiter=delimiter, digital_text_values=True),
    )

    result = run_import_pipeline(str(csv_path))

    assert result.success
    assert result.profile.delimiter == delimiter
    assert result.diagnostics.normalized_row_count == 250


def test_metadata_rows_before_header_are_skipped(runtime_tmp_path) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "metadata.csv",
        StressSampleConfig(row_count=100, metadata_rows=5),
    )

    result = run_import_pipeline(str(csv_path))

    assert result.success
    assert result.profile.raw_preview.header_row_index == 5


def test_blank_and_malformed_timestamps_emit_diagnostics(runtime_tmp_path) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "bad_timestamps.csv",
        StressSampleConfig(
            row_count=300,
            missing_timestamp_ratio=0.05,
            malformed_timestamp_ratio=0.05,
        ),
    )

    result = run_import_pipeline(str(csv_path))
    codes = {message.code for message in result.validation_messages}

    assert result.success
    assert result.diagnostics.normalized_row_count < 300
    assert "TS_DROPPED_ROWS" in codes


def test_duplicate_timestamp_rows_warn_without_crash(runtime_tmp_path) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "duplicates.csv",
        StressSampleConfig(row_count=300, duplicate_timestamp_ratio=0.05),
    )

    result = run_import_pipeline(str(csv_path))
    codes = {message.code for message in result.validation_messages}

    assert result.success
    assert "TS_DUPLICATES" in codes


def test_non_monotonic_timestamp_rows_warn_without_crash(runtime_tmp_path) -> None:
    csv_path = runtime_tmp_path / "non_monotonic.csv"
    csv_path.write_text(
        "Timestamp,Voltage A (kV),CB Status\n"
        "2026-01-01 00:00:00.000,132.0,OPEN\n"
        "2026-01-01 00:00:00.040,132.1,CLOSED\n"
        "2026-01-01 00:00:00.020,132.2,OPEN\n"
        "2026-01-01 00:00:00.060,132.3,CLOSED\n",
        encoding="utf-8",
    )

    result = run_import_pipeline(str(csv_path))
    codes = {message.code for message in result.validation_messages}

    assert result.success
    assert "TS_NON_MONOTONIC" in codes


def test_mixed_timestamp_formats_drop_bad_rows_gracefully(runtime_tmp_path) -> None:
    csv_path = runtime_tmp_path / "mixed_formats.csv"
    csv_path.write_text(
        "Timestamp,MW Total\n"
        "2026-01-01 00:00:00,95.0\n"
        "2026-01-01 00:00:01,95.1\n"
        "01/01/2026 00:00:02,95.2\n"
        "2026-01-01 00:00:03,95.3\n",
        encoding="utf-8",
    )

    result = run_import_pipeline(str(csv_path))
    codes = {message.code for message in result.validation_messages}

    assert result.success
    assert result.diagnostics.normalized_row_count == 3
    assert "TS_DROPPED_ROWS" in codes


def test_inconsistent_row_lengths_fail_gracefully(runtime_tmp_path) -> None:
    csv_path = runtime_tmp_path / "ragged.csv"
    csv_path.write_text(
        "Timestamp,Voltage A (kV),MW Total\n"
        "2026-01-01 00:00:00.000,132.0,95.0\n"
        "2026-01-01 00:00:00.020,132.1,95.1,EXTRA_FIELD\n"
        "2026-01-01 00:00:00.040,132.2,95.2\n",
        encoding="utf-8",
    )

    result = run_import_pipeline(str(csv_path))
    error_codes = {m.code for m in result.validation_messages if m.severity == ValidationSeverity.ERROR}

    assert result is not None
    assert "PIPELINE_LOAD_FAILED" in error_codes or result.success


def test_text_noise_digital_status_and_unknown_columns_are_routed(runtime_tmp_path) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "status_text.csv",
        StressSampleConfig(
            row_count=200,
            digital_text_values=True,
            unknown_column_count=2,
        ),
    )

    result = run_import_pipeline(str(csv_path))
    warning_codes = {m.code for m in result.validation_messages if m.severity == ValidationSeverity.WARNING}

    assert result.success
    assert result.record is not None
    assert result.record.digital_channels
    assert "PIPELINE_UNKNOWN_COLUMN" in warning_codes
    for channel in result.record.digital_channels:
        values = set(result.record.waveform_data[channel.name].unique())
        assert values.issubset({0, 1})


def test_unrecoverable_timestamp_file_fails_without_exception(runtime_tmp_path) -> None:
    csv_path = Path(runtime_tmp_path / "unrecoverable.csv")
    csv_path.write_text(
        "Timestamp,MW Total\n"
        "not-a-time,95.0\n"
        "also-bad,95.1\n"
        "still-bad,95.2\n",
        encoding="utf-8",
    )

    result = run_import_pipeline(str(csv_path))
    codes = {message.code for message in result.validation_messages}

    assert not result.success
    assert "PIPELINE_NO_TIMESTAMP" in codes or "TS_ALL_NAT" in codes
