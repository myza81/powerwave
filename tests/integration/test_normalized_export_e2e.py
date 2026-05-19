"""End-to-end integration tests for the normalized export writer (Phase 8.55K).

These tests run the full pipeline:
  CSV → profile → assemble NormalizedDataset → write_normalized_export()

No Qt.  Uses real temp CSV files and the full import backend.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.import_wizard import (
    ExportWriteOptions,
    write_normalized_export,
    write_from_export_plan,
    plan_export,
    run_import_pipeline,
)
from app.import_wizard.export_metadata import load_metadata_sidecar, EXPORT_SCHEMA_VERSION
from app.import_wizard.export_writer import _PARQUET_AVAILABLE, _FEATHER_AVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _write_csv(tmp_path: Path, name: str = "source.csv") -> Path:
    p = tmp_path / name
    p.write_text(
        "timestamp,va,ia,mw,mvar,trip\n"
        "2026-01-01 00:00:00.000,230.1,1.00,100.0,5.0,0\n"
        "2026-01-01 00:00:00.020,230.2,1.01,100.1,5.1,0\n"
        "2026-01-01 00:00:00.040,229.9,0.99,99.9,4.9,1\n",
        encoding="utf-8",
    )
    return p


def _get_dataset(csv_path: Path):
    """Run pipeline and return the NormalizedDataset (or fail loudly)."""
    result = run_import_pipeline(str(csv_path))
    assert result.dataset is not None, (
        f"Pipeline produced no NormalizedDataset. Errors: "
        f"{[m.message for m in result.validation_messages if m.severity.value == 'error']}"
    )
    return result.dataset


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFullPipelineCSVExport:
    def test_pipeline_to_csv_export(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        out = tmp_path / "out.csv"

        result = write_normalized_export(dataset, out)

        assert result.success is True
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 3
        assert "timestamp" in df.columns

    def test_canonical_names_from_pipeline(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        out = tmp_path / "out.csv"

        write_normalized_export(dataset, out)

        df = pd.read_csv(out)
        canonical = {p.canonical_name for p in dataset.parameters}
        for name in canonical:
            assert name in df.columns, f"Expected canonical column '{name}' in export"

    def test_source_traceability_in_sidecar(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        out = tmp_path / "out.csv"

        result = write_normalized_export(dataset, out)

        assert result.metadata_path is not None
        sidecar = load_metadata_sidecar(result.metadata_path)
        assert sidecar["schema_version"] == EXPORT_SCHEMA_VERSION
        assert sidecar["source_file_name"] == "source.csv"
        assert sidecar["row_count"] == 3

    def test_parameter_audit_in_sidecar(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        out = tmp_path / "out.csv"

        result = write_normalized_export(dataset, out)

        assert result.metadata_path is not None
        sidecar = load_metadata_sidecar(result.metadata_path)
        param_names = {p["canonical_name"] for p in sidecar["parameters"]}
        expected = {p.canonical_name for p in dataset.parameters}
        assert param_names == expected

    def test_timestamp_repair_strategy_preserved(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        out = tmp_path / "out.csv"

        result = write_normalized_export(dataset, out)

        assert result.metadata_path is not None
        sidecar = load_metadata_sidecar(result.metadata_path)
        assert sidecar["timestamp_repair_strategy"] == dataset.timestamp_repair_strategy
        assert sidecar["timestamp_column"] == dataset.timestamp_column

    def test_export_does_not_mutate_dataset(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        df_before = dataset.data.copy()
        out = tmp_path / "out.csv"

        write_normalized_export(dataset, out)

        pd.testing.assert_frame_equal(dataset.data, df_before)

    def test_overwrite_false_default_blocks_second_write(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        out = tmp_path / "out.csv"

        write_normalized_export(dataset, out, options=ExportWriteOptions(overwrite=True))
        result = write_normalized_export(dataset, out)

        assert result.success is False
        assert any(m.code == "EXPORT_FILE_EXISTS" for m in result.errors())

    def test_overwrite_true_replaces_file(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        out = tmp_path / "out.csv"
        opts = ExportWriteOptions(overwrite=True)

        write_normalized_export(dataset, out, options=opts)
        result = write_normalized_export(dataset, out, options=opts)

        assert result.success is True


class TestExportPlanIntegration:
    def test_write_from_export_plan_uses_suggested_path(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        plan = plan_export(dataset, output_dir=str(tmp_path))

        result = write_from_export_plan(dataset, plan, format="csv", options=ExportWriteOptions(overwrite=True))

        assert result.success is True
        assert result.output_path == plan.suggested_csv_path

    def test_plan_suggested_path_matches_written_file(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        plan = plan_export(dataset, output_dir=str(tmp_path))

        result = write_from_export_plan(dataset, plan, format="csv", options=ExportWriteOptions(overwrite=True))

        assert result.output_path is not None
        assert Path(result.output_path).exists()
        df = pd.read_csv(result.output_path)
        assert len(df) == 3


@pytest.mark.skipif(not _PARQUET_AVAILABLE, reason="pyarrow/fastparquet not installed")
class TestParquetExport:
    def test_parquet_round_trip(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        out = tmp_path / "out.parquet"

        result = write_normalized_export(dataset, out)

        assert result.success is True
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == 3
        canonical = {p.canonical_name for p in dataset.parameters}
        for name in canonical:
            assert name in df.columns

    def test_parquet_sidecar_created(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        out = tmp_path / "out.parquet"

        result = write_normalized_export(dataset, out)

        assert result.metadata_path is not None
        sidecar = load_metadata_sidecar(result.metadata_path)
        assert sidecar["export_format"] == "parquet"


@pytest.mark.skipif(not _FEATHER_AVAILABLE, reason="pyarrow not installed")
class TestFeatherExport:
    def test_feather_round_trip(self, tmp_path) -> None:
        csv_path = _write_csv(tmp_path)
        dataset = _get_dataset(csv_path)
        out = tmp_path / "out.feather"

        result = write_normalized_export(dataset, out)

        assert result.success is True
        assert out.exists()
        import pyarrow.feather as feather  # type: ignore[import-untyped]
        df = feather.read_feather(out)
        assert len(df) == 3
