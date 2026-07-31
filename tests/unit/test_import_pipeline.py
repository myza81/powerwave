"""Unit tests for import_pipeline.py — helpers and failure paths.

Tests here are backend-only (no Qt) and use only synthetic in-memory data or
small temporary files.  Every test is deterministic and fast.
"""
from __future__ import annotations

import os
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.file_profiler import FileProfileResult
from app.import_wizard.import_pipeline import (
    ImportPipelineOptions,
    ImportPipelineResult,
    PipelineDiagnostics,
    _build_normalization_plan,
    _build_repair_plan,
    _check_supported_extension,
    _load_full_dataframe,
    run_import_pipeline,
)
from app.import_wizard.models import (
    ColumnMappingCandidate,
    RawPreviewModel,
    TimestampCandidate,
)
from app.import_wizard.timestamp_contracts import TimestampRepairStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ts_candidate(
    col_name: str = "timestamp",
    fmt: str | None = "%Y-%m-%d %H:%M:%S",
    confidence: float = 0.95,
) -> TimestampCandidate:
    return TimestampCandidate(
        column_name=col_name,
        column_index=0,
        confidence=confidence,
        detected_format=fmt,
    )


def _make_mapping(
    source_name: str,
    suggested_name: str | None = None,
    ptype: ParameterType = ParameterType.MW,
    unit: str | None = "MW",
    excluded: bool = False,
    user_name_override: str | None = None,
) -> ColumnMappingCandidate:
    return ColumnMappingCandidate(
        source_name=source_name,
        source_index=0,
        suggested_name=suggested_name or source_name,
        parameter_type=ptype,
        unit=unit,
        confidence=0.8,
        user_name_override=user_name_override,
        excluded=excluded,
    )


def _make_profile(
    mappings: list[ColumnMappingCandidate],
    candidates: list[TimestampCandidate] | None = None,
    provider_type: str = "csv",
    delimiter: str | None = ",",
) -> FileProfileResult:
    return FileProfileResult(
        raw_preview=RawPreviewModel(column_names=[], preview_rows=[]),
        timestamp_candidates=candidates or [],
        column_mappings=mappings,
        provider_type=provider_type,
        delimiter=delimiter,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _check_supported_extension
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSupportedExtension:
    def test_csv_is_supported(self):
        assert _check_supported_extension("recording.csv") is None

    def test_txt_is_supported(self):
        assert _check_supported_extension("data.txt") is None

    def test_tsv_is_supported(self):
        assert _check_supported_extension("data.tsv") is None

    def test_dat_is_supported(self):
        assert _check_supported_extension("data.dat") is None

    def test_xlsx_is_supported(self):
        assert _check_supported_extension("recording.xlsx") is None

    def test_xls_is_supported(self):
        assert _check_supported_extension("recording.xls") is None

    def test_xlsm_is_supported(self):
        assert _check_supported_extension("recording.xlsm") is None

    def test_xlsb_is_supported(self):
        assert _check_supported_extension("recording.xlsb") is None

    def test_pdf_is_not_supported(self):
        result = _check_supported_extension("report.pdf")
        assert result is not None
        assert ".pdf" in result

    def test_bin_is_not_supported(self):
        result = _check_supported_extension("data.bin")
        assert result is not None

    def test_empty_extension_is_not_blocked(self):
        # Files without extension are passed through (profiler handles them)
        assert _check_supported_extension("data") is None

    def test_case_insensitive_csv(self):
        assert _check_supported_extension("DATA.CSV") is None

    def test_case_insensitive_xlsx(self):
        assert _check_supported_extension("RECORDING.XLSX") is None


# ─────────────────────────────────────────────────────────────────────────────
# _build_repair_plan
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildRepairPlan:
    def test_strptime_format_uses_parse_detected(self):
        c = _make_ts_candidate(fmt="%Y-%m-%d %H:%M:%S")
        plan = _build_repair_plan(c)
        assert plan.strategy == TimestampRepairStrategy.PARSE_DETECTED_FORMAT

    def test_strptime_format_preserved_in_plan(self):
        c = _make_ts_candidate(fmt="%d/%m/%Y %H:%M:%S")
        plan = _build_repair_plan(c)
        assert plan.detected_format == "%d/%m/%Y %H:%M:%S"

    def test_excel_serial_uses_excel_strategy(self):
        c = _make_ts_candidate(fmt="excel_serial")
        plan = _build_repair_plan(c)
        assert plan.strategy == TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION

    def test_epoch_seconds_uses_parse_detected(self):
        c = _make_ts_candidate(fmt="epoch_seconds")
        plan = _build_repair_plan(c)
        assert plan.strategy == TimestampRepairStrategy.PARSE_DETECTED_FORMAT
        assert plan.detected_format == "epoch_seconds"

    def test_epoch_milliseconds_uses_parse_detected(self):
        c = _make_ts_candidate(fmt="epoch_milliseconds")
        plan = _build_repair_plan(c)
        assert plan.strategy == TimestampRepairStrategy.PARSE_DETECTED_FORMAT
        assert plan.detected_format == "epoch_milliseconds"

    def test_elapsed_seconds_uses_elapsed_strategy(self):
        c = _make_ts_candidate(fmt="elapsed_seconds")
        plan = _build_repair_plan(c)
        assert plan.strategy == TimestampRepairStrategy.PARSE_ELAPSED_TIME
        assert plan.elapsed_time_unit == "elapsed_seconds"

    def test_none_format_uses_no_repair(self):
        c = _make_ts_candidate(fmt=None)
        plan = _build_repair_plan(c)
        assert plan.strategy == TimestampRepairStrategy.NO_REPAIR

    def test_repair_validated_always_true(self):
        for fmt in ["%Y-%m-%d %H:%M:%S", "excel_serial", "epoch_seconds", None]:
            c = _make_ts_candidate(fmt=fmt)
            plan = _build_repair_plan(c)
            assert plan.repair_validated is True

    def test_repair_notes_set(self):
        c = _make_ts_candidate(fmt="%Y-%m-%d")
        plan = _build_repair_plan(c)
        assert plan.repair_notes is not None and len(plan.repair_notes) > 0


# ─────────────────────────────────────────────────────────────────────────────
# _build_normalization_plan
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildNormalizationPlan:
    def _candidate(self, col="timestamp"):
        from app.import_wizard.timestamp_contracts import TimestampRepairPlan
        plan = TimestampRepairPlan(
            strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
            detected_format="%Y-%m-%d %H:%M:%S",
            repair_validated=True,
        )
        return _make_ts_candidate(col_name=col), plan

    def test_timestamp_column_excluded(self):
        candidate, repair = self._candidate("ts")
        profile = _make_profile([_make_mapping("ts", ptype=ParameterType.MW)])
        msgs: list[ValidationMessage] = []
        plan = _build_normalization_plan(profile, candidate, repair, msgs)
        assert "ts" not in plan.selected_columns
        assert "ts" in plan.excluded_columns

    def test_data_columns_selected(self):
        candidate, repair = self._candidate("ts")
        profile = _make_profile([
            _make_mapping("ts", ptype=ParameterType.MW),
            _make_mapping("mw", ptype=ParameterType.MW),
            _make_mapping("va", ptype=ParameterType.VOLTAGE),
        ])
        msgs: list[ValidationMessage] = []
        plan = _build_normalization_plan(profile, candidate, repair, msgs)
        assert "mw" in plan.selected_columns
        assert "va" in plan.selected_columns

    def test_excluded_mapping_excluded(self):
        candidate, repair = self._candidate("ts")
        profile = _make_profile([
            _make_mapping("mw", excluded=True),
        ])
        msgs: list[ValidationMessage] = []
        plan = _build_normalization_plan(profile, candidate, repair, msgs)
        assert "mw" not in plan.selected_columns
        assert "mw" in plan.excluded_columns

    def test_timestamp_typed_column_excluded(self):
        candidate, repair = self._candidate("ts")
        profile = _make_profile([
            _make_mapping("ts2", ptype=ParameterType.TIMESTAMP),
        ])
        msgs: list[ValidationMessage] = []
        plan = _build_normalization_plan(profile, candidate, repair, msgs)
        assert "ts2" not in plan.selected_columns
        assert "ts2" in plan.excluded_columns

    def test_unknown_column_emits_warning(self):
        candidate, repair = self._candidate("ts")
        profile = _make_profile([
            _make_mapping("mystery", ptype=ParameterType.UNKNOWN),
        ])
        msgs: list[ValidationMessage] = []
        _build_normalization_plan(profile, candidate, repair, msgs)
        assert any(m.code == "PIPELINE_UNKNOWN_COLUMN" for m in msgs)

    def test_unknown_column_still_selected(self):
        candidate, repair = self._candidate("ts")
        profile = _make_profile([
            _make_mapping("mystery", ptype=ParameterType.UNKNOWN),
        ])
        msgs: list[ValidationMessage] = []
        plan = _build_normalization_plan(profile, candidate, repair, msgs)
        assert "mystery" in plan.selected_columns

    def test_unit_preserved(self):
        candidate, repair = self._candidate("ts")
        profile = _make_profile([
            _make_mapping("mw", ptype=ParameterType.MW, unit="MW"),
        ])
        msgs: list[ValidationMessage] = []
        plan = _build_normalization_plan(profile, candidate, repair, msgs)
        assert plan.column_units.get("mw") == "MW"

    def test_type_preserved(self):
        candidate, repair = self._candidate("ts")
        profile = _make_profile([
            _make_mapping("va", ptype=ParameterType.VOLTAGE),
        ])
        msgs: list[ValidationMessage] = []
        plan = _build_normalization_plan(profile, candidate, repair, msgs)
        assert plan.column_types.get("va") == ParameterType.VOLTAGE

    def test_rename_applied(self):
        candidate, repair = self._candidate("ts")
        profile = _make_profile([
            _make_mapping("MW_Total", suggested_name="mw", ptype=ParameterType.MW),
        ])
        msgs: list[ValidationMessage] = []
        plan = _build_normalization_plan(profile, candidate, repair, msgs)
        assert "MW_Total" in plan.selected_columns
        assert plan.column_renames.get("MW_Total") == "mw"

    def test_repair_plan_attached(self):
        candidate, repair = self._candidate("ts")
        profile = _make_profile([_make_mapping("mw")])
        msgs: list[ValidationMessage] = []
        plan = _build_normalization_plan(profile, candidate, repair, msgs)
        assert plan.timestamp_plan is repair

    def test_empty_column_mappings(self):
        candidate, repair = self._candidate("ts")
        profile = _make_profile([])
        msgs: list[ValidationMessage] = []
        plan = _build_normalization_plan(profile, candidate, repair, msgs)
        assert plan.selected_columns == []


class TestElapsedExcelPipeline:
    def test_elapsed_excel_preserves_duration_axis(self, tmp_path: Path):
        path = tmp_path / "tieline.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        assert ws is not None
        ws.title = "Loss of 2000MW"
        ws.append([r"D:\Study_frequency\Loss of 2000MW.out", None, None, None])
        ws.append(["Time", "1 - KAWA FREQ", "2 - TIE LINE 1", "3 - TIE LINE 2"])
        ws.append([-0.002, 0.0, -0.673291, -0.673291])
        ws.append([0.008, 0.00000542871, -0.671219, -0.671219])
        ws.append([0.018, 0.00000524838, -0.669146, -0.669146])
        ws.append([0.028, 0.00000449035, -0.666713, -0.666713])
        wb.save(path)

        result = run_import_pipeline(str(path))

        assert result.success
        assert result.repair_plan is not None
        assert result.repair_plan.strategy == TimestampRepairStrategy.PARSE_ELAPSED_TIME
        assert result.dataset is not None
        assert result.dataset.time_axis_mode == "relative_elapsed"
        assert result.record is not None
        assert result.record.timing_info.timing_reference == "relative_elapsed"
        assert result.record.waveform_data["time"].tolist()[:4] == pytest.approx([
            -0.002,
            0.008,
            0.018,
            0.028,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# _load_full_dataframe
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadFullDataframe:
    def test_csv_loads_all_rows(self, tmp_path: Path):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="1s").astype(str),
            "mw": range(30),
        })
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)

        profile = FileProfileResult(
            raw_preview=RawPreviewModel(column_names=["timestamp", "mw"], preview_rows=[],
                                        header_row_index=0),
            delimiter=",",
        )
        loaded = _load_full_dataframe(str(p), "csv", profile)
        assert len(loaded) == 30

    def test_csv_uses_detected_delimiter(self, tmp_path: Path):
        df = pd.DataFrame({"timestamp": ["2024-01-01"], "mw": [100.0]})
        p = tmp_path / "semicolon.csv"
        df.to_csv(p, index=False, sep=";")

        profile = FileProfileResult(
            raw_preview=RawPreviewModel(column_names=[], preview_rows=[]),
            delimiter=";",
        )
        loaded = _load_full_dataframe(str(p), "csv", profile)
        assert "mw" in loaded.columns

    def test_csv_header_row_index_respected(self, tmp_path: Path):
        # Write a file with one metadata row before the header
        p = tmp_path / "meta.csv"
        p.write_text("# metadata\ntimestamp,mw\n2024-01-01,100\n2024-01-02,200\n")

        profile = FileProfileResult(
            raw_preview=RawPreviewModel(column_names=[], preview_rows=[], header_row_index=1),
            delimiter=",",
        )
        loaded = _load_full_dataframe(str(p), "csv", profile)
        assert "timestamp" in loaded.columns
        assert len(loaded) == 2

    def test_csv_columns_match_file(self, tmp_path: Path):
        df = pd.DataFrame({"ts": ["2024-01-01"], "voltage_a": [230.0], "current_a": [1.0]})
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)

        profile = FileProfileResult(
            raw_preview=RawPreviewModel(column_names=[], preview_rows=[]),
            delimiter=",",
        )
        loaded = _load_full_dataframe(str(p), "csv", profile)
        assert set(loaded.columns) == {"ts", "voltage_a", "current_a"}


# ─────────────────────────────────────────────────────────────────────────────
# run_import_pipeline — failure paths (no real file needed for some)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunImportPipelineFailures:
    def test_file_not_found_returns_failure(self):
        result = run_import_pipeline("/nonexistent/path/file.csv")
        assert not result.success
        assert any(m.severity == ValidationSeverity.ERROR for m in result.validation_messages)

    def test_unsupported_extension_returns_failure(self, tmp_path: Path):
        p = tmp_path / "data.pdf"
        p.write_bytes(b"%PDF-1.4 fake pdf")
        result = run_import_pipeline(str(p))
        assert not result.success
        assert any("PIPELINE_UNSUPPORTED_TYPE" in m.code for m in result.validation_messages)

    def test_unsupported_extension_error_message_mentions_ext(self, tmp_path: Path):
        p = tmp_path / "data.xyz"
        p.write_text("nothing useful")
        result = run_import_pipeline(str(p))
        msgs = [m.message for m in result.validation_messages if m.severity == ValidationSeverity.ERROR]
        assert any(".xyz" in m for m in msgs)

    def test_no_timestamp_candidate_returns_failure(self, tmp_path: Path):
        # A CSV with no timestamp-like column
        df = pd.DataFrame({"alpha": [1.0, 2.0], "beta": [3.0, 4.0]})
        p = tmp_path / "no_ts.csv"
        df.to_csv(p, index=False)
        result = run_import_pipeline(str(p))
        assert not result.success
        error_codes = [m.code for m in result.validation_messages if m.severity == ValidationSeverity.ERROR]
        assert "PIPELINE_NO_TIMESTAMP" in error_codes

    def test_result_always_has_profile(self, tmp_path: Path):
        p = tmp_path / "data.pdf"
        p.write_bytes(b"not a valid file")
        result = run_import_pipeline(str(p))
        assert result.profile is not None

    def test_result_always_has_session(self, tmp_path: Path):
        p = tmp_path / "data.pdf"
        p.write_bytes(b"not a valid file")
        result = run_import_pipeline(str(p))
        assert result.session is not None

    def test_result_always_has_diagnostics(self, tmp_path: Path):
        p = tmp_path / "data.pdf"
        p.write_bytes(b"not a valid file")
        result = run_import_pipeline(str(p))
        assert result.diagnostics is not None

    def test_min_confidence_rejection(self, tmp_path: Path):
        # Use a CSV with a timestamp column whose confidence might be moderate
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1s").astype(str),
            "mw": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        opts = ImportPipelineOptions(min_timestamp_confidence=2.0)  # impossible threshold
        result = run_import_pipeline(str(p), options=opts)
        assert not result.success
        assert any(m.code == "PIPELINE_LOW_TS_CONFIDENCE" for m in result.validation_messages)


# ─────────────────────────────────────────────────────────────────────────────
# PipelineDiagnostics — field population
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineDiagnostics:
    def _run_ok_pipeline(self, tmp_path: Path) -> ImportPipelineResult:
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "voltage_a": [230.0 + i for i in range(10)],
            "mw": [100.0 + i for i in range(10)],
        })
        p = tmp_path / "diag_test.csv"
        df.to_csv(p, index=False)
        return run_import_pipeline(str(p))

    def test_source_file_path_correct(self, tmp_path: Path):
        result = self._run_ok_pipeline(tmp_path)
        assert "diag_test.csv" in result.diagnostics.source_file_path

    def test_provider_type_csv(self, tmp_path: Path):
        result = self._run_ok_pipeline(tmp_path)
        assert result.diagnostics.provider_type == "csv"

    def test_timestamp_column_set(self, tmp_path: Path):
        result = self._run_ok_pipeline(tmp_path)
        assert result.diagnostics.selected_timestamp_column == "timestamp"

    def test_normalized_row_count_correct(self, tmp_path: Path):
        result = self._run_ok_pipeline(tmp_path)
        assert result.diagnostics.normalized_row_count == 10

    def test_elapsed_seconds_positive(self, tmp_path: Path):
        result = self._run_ok_pipeline(tmp_path)
        assert result.diagnostics.elapsed_seconds is not None
        assert result.diagnostics.elapsed_seconds >= 0.0

    def test_analog_channel_count_positive(self, tmp_path: Path):
        result = self._run_ok_pipeline(tmp_path)
        assert result.diagnostics.analog_channel_count >= 1

    def test_error_count_zero_on_success(self, tmp_path: Path):
        result = self._run_ok_pipeline(tmp_path)
        assert result.diagnostics.error_count == 0
