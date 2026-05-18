"""Tests for export_planner.py — ExportPlan and plan_export."""
from __future__ import annotations

import os

import pandas as pd
from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.export_planner import ExportPlan, _stem_from_source, plan_export
from app.import_wizard.normalized_dataset import (
    AssemblyDiagnostics,
    NormalizedDataset,
    ParameterMetadata,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_param(canonical: str = "mw", ptype: ParameterType = ParameterType.MW) -> ParameterMetadata:
    return ParameterMetadata(
        canonical_name=canonical,
        parameter_type=ptype,
        unit="MW",
        source_name="MW",
        source_index=0,
    )


def _make_dataset(
    n: int = 5,
    source_path: str | None = None,
    source_file_name: str | None = None,
    is_valid: bool = True,
    params: list[ParameterMetadata] | None = None,
    diag: AssemblyDiagnostics | None = None,
) -> NormalizedDataset:
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1s"))
    df = pd.DataFrame({"timestamp": ts, "mw": range(n)})
    return NormalizedDataset(
        data=df,
        timestamp_column="timestamp",
        parameters=params if params is not None else [_make_param()],
        excluded_columns=[],
        validation_messages=[],
        diagnostics=diag or AssemblyDiagnostics(total_rows=n, normalized_rows=n),
        source_path=source_path,
        source_file_name=source_file_name,
        timestamp_repair_strategy="no_repair",
        is_valid=is_valid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _stem_from_source
# ─────────────────────────────────────────────────────────────────────────────

class TestStemFromSource:
    def test_none_returns_dataset(self):
        assert _stem_from_source(None) == "dataset"

    def test_strips_extension(self):
        assert _stem_from_source("recording.csv") == "recording"

    def test_strips_path(self):
        stem = _stem_from_source("/data/recordings/event_20240101.csv")
        assert stem == "event_20240101"

    def test_sanitises_special_chars(self):
        stem = _stem_from_source("my file (copy).csv")
        assert " " not in stem
        assert "(" not in stem

    def test_empty_stem_fallback(self):
        assert _stem_from_source("") == "dataset"

    def test_preserves_underscores_hyphens(self):
        stem = _stem_from_source("pulu_2024-01-01.csv")
        assert stem == "pulu_2024-01-01"


# ─────────────────────────────────────────────────────────────────────────────
# plan_export — basic output
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanExportBasic:
    def test_returns_export_plan(self):
        ds = _make_dataset()
        result = plan_export(ds)
        assert isinstance(result, ExportPlan)

    def test_csv_path_not_none(self):
        ds = _make_dataset()
        result = plan_export(ds)
        assert result.suggested_csv_path is not None

    def test_parquet_path_not_none(self):
        ds = _make_dataset()
        result = plan_export(ds)
        assert result.suggested_parquet_path is not None

    def test_feather_path_not_none(self):
        ds = _make_dataset()
        result = plan_export(ds)
        assert result.suggested_feather_path is not None

    def test_csv_extension_correct(self):
        ds = _make_dataset()
        result = plan_export(ds)
        assert result.suggested_csv_path.endswith(".csv")  # type: ignore[union-attr]

    def test_parquet_extension_correct(self):
        ds = _make_dataset()
        result = plan_export(ds)
        assert result.suggested_parquet_path.endswith(".parquet")  # type: ignore[union-attr]

    def test_feather_extension_correct(self):
        ds = _make_dataset()
        result = plan_export(ds)
        assert result.suggested_feather_path.endswith(".feather")  # type: ignore[union-attr]

    def test_normalized_suffix_in_filename(self):
        ds = _make_dataset(source_file_name="recording.csv")
        result = plan_export(ds)
        base = os.path.basename(result.suggested_csv_path)  # type: ignore[arg-type]
        assert "normalized" in base

    def test_row_count_correct(self):
        ds = _make_dataset(n=10)
        result = plan_export(ds)
        assert result.row_count == 10

    def test_column_count_correct(self):
        ds = _make_dataset(params=[_make_param("mw"), _make_param("voltage", ParameterType.VOLTAGE)])
        result = plan_export(ds)
        assert result.column_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# plan_export — source file naming
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanExportNaming:
    def test_source_stem_in_csv_path(self):
        ds = _make_dataset(source_file_name="pulu_20260306.csv")
        result = plan_export(ds)
        assert "pulu_20260306" in result.suggested_csv_path  # type: ignore[operator]

    def test_no_source_uses_dataset_stem(self):
        ds = _make_dataset()
        result = plan_export(ds)
        assert "dataset" in os.path.basename(result.suggested_csv_path)  # type: ignore[arg-type]

    def test_custom_output_dir_used(self):
        ds = _make_dataset(source_file_name="data.csv")
        result = plan_export(ds, output_dir="/custom/dir")
        assert result.suggested_csv_path.startswith("/custom/dir")  # type: ignore[union-attr]

    def test_source_path_dir_used_when_no_output_dir(self):
        ds = _make_dataset(
            source_path="/recordings/event.csv",
            source_file_name="event.csv",
        )
        result = plan_export(ds)
        expected_dir = os.path.dirname(os.path.abspath("/recordings/event.csv"))
        assert result.suggested_csv_path.startswith(expected_dir)  # type: ignore[union-attr]

    def test_dot_dir_used_when_no_source_path(self):
        ds = _make_dataset()
        result = plan_export(ds)
        # Path should be relative or start with the CWD
        assert result.suggested_csv_path is not None
        assert "dataset_normalized.csv" in result.suggested_csv_path  # type: ignore[operator]


# ─────────────────────────────────────────────────────────────────────────────
# plan_export — readiness
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanExportReadiness:
    def test_is_export_ready_true_for_valid_dataset(self):
        ds = _make_dataset(is_valid=True)
        result = plan_export(ds)
        assert result.is_export_ready

    def test_is_export_ready_false_for_invalid(self):
        ds = _make_dataset(is_valid=False)
        result = plan_export(ds)
        assert not result.is_export_ready

    def test_readiness_issues_empty_when_ready(self):
        ds = _make_dataset(is_valid=True)
        result = plan_export(ds)
        assert result.readiness_issues == []

    def test_readiness_issues_for_invalid_dataset(self):
        ds = _make_dataset(is_valid=False)
        result = plan_export(ds)
        assert len(result.readiness_issues) > 0

    def test_readiness_issues_for_no_params(self):
        ds = _make_dataset(params=[], is_valid=False)
        result = plan_export(ds)
        assert any("No data columns" in issue for issue in result.readiness_issues)

    def test_readiness_issues_for_duplicates(self):
        diag = AssemblyDiagnostics(
            total_rows=5,
            normalized_rows=5,
            duplicate_timestamp_count=2,
        )
        ds = _make_dataset(diag=diag, is_valid=True)
        result = plan_export(ds)
        assert any("duplicate" in issue.lower() for issue in result.readiness_issues)
        # Has duplicate warnings but is still export ready (duplicate is a warning, not blocker)
        assert result.is_export_ready


# ─────────────────────────────────────────────────────────────────────────────
# plan_export — export summary
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanExportSummary:
    def test_summary_contains_row_count(self):
        ds = _make_dataset(n=42)
        result = plan_export(ds)
        assert "42" in result.export_summary

    def test_summary_contains_channel_count(self):
        ds = _make_dataset(params=[_make_param("mw"), _make_param("v", ParameterType.VOLTAGE)])
        result = plan_export(ds)
        assert "2" in result.export_summary

    def test_summary_contains_strategy(self):
        ds = _make_dataset()
        result = plan_export(ds)
        assert "no_repair" in result.export_summary

    def test_summary_contains_source_filename(self):
        ds = _make_dataset(source_file_name="event_20240101.csv")
        result = plan_export(ds)
        assert "event_20240101.csv" in result.export_summary

    def test_summary_not_empty(self):
        ds = _make_dataset()
        result = plan_export(ds)
        assert result.export_summary != ""
