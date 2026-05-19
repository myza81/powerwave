"""Unit tests for the normalized export writer (Phase 8.55K).

Coverage
--------
1.  CSV export success
2.  Metadata sidecar creation
3.  overwrite=False blocks existing file
4.  overwrite=True replaces existing file
5.  Unsupported format failure
6.  Missing-dependency behavior for Parquet/Feather
7.  Empty dataset failure
8.  Timestamp column preservation in exported CSV
9.  Canonical column names preserved in exported CSV
10. Parameter metadata written to sidecar
11. Timestamp repair strategy written to sidecar
12. Export does not mutate dataset
13. ExportPlan integration via write_from_export_plan()
14. Invalid output path / directory creation failure
15. Metadata sidecar write failure — error recorded, data still written
16. float_precision option (CSV)
17. timestamp_format option (CSV)
18. include_index=True writes index column
19. Parquet export (skip when pyarrow absent)
20. Feather export (skip when pyarrow absent)
21. metadata_sidecar_path naming convention
22. load_metadata_sidecar round-trip
23. ExportPlan not-ready blocks write_from_export_plan
24. write_from_export_plan unknown format
25. ExportWriteResult helpers (errors / warnings)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.export_metadata import (
    EXPORT_SCHEMA_VERSION,
    load_metadata_sidecar,
    metadata_sidecar_path,
)
from app.import_wizard.export_planner import plan_export
from app.import_wizard.export_writer import (
    ExportWriteOptions,
    ExportWriteResult,
    _PARQUET_AVAILABLE,
    _FEATHER_AVAILABLE,
    write_from_export_plan,
    write_normalized_export,
)
from app.import_wizard.normalized_dataset import (
    AssemblyDiagnostics,
    NormalizedDataset,
    ParameterMetadata,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_dataset(
    *,
    rows: int = 5,
    source_name: str = "test.csv",
    repair_strategy: str = "parse_detected_format",
    valid: bool = True,
) -> NormalizedDataset:
    ts = pd.date_range("2026-01-01", periods=rows, freq="20ms")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "va": [230.0 + i * 0.1 for i in range(rows)],
            "ia": [1.0 + i * 0.01 for i in range(rows)],
        }
    )
    params = [
        ParameterMetadata(
            canonical_name="va",
            parameter_type=ParameterType.VOLTAGE,
            unit="V",
            source_name="VA",
            source_index=1,
            user_overridden=False,
            confidence=0.9,
        ),
        ParameterMetadata(
            canonical_name="ia",
            parameter_type=ParameterType.CURRENT,
            unit="A",
            source_name="IA",
            source_index=2,
            user_overridden=True,
            confidence=0.85,
        ),
    ]
    diag = AssemblyDiagnostics(
        total_rows=rows,
        normalized_rows=rows,
    )
    return NormalizedDataset(
        data=df,
        timestamp_column="timestamp",
        parameters=params,
        excluded_columns=["trip"],
        validation_messages=[],
        diagnostics=diag,
        source_path=f"/data/{source_name}",
        source_file_name=source_name,
        timestamp_repair_strategy=repair_strategy,
        is_valid=valid,
    )


def _empty_dataset() -> NormalizedDataset:
    df = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), "va": []})
    params = [
        ParameterMetadata(
            canonical_name="va",
            parameter_type=ParameterType.VOLTAGE,
            unit="V",
            source_name="VA",
            source_index=1,
        )
    ]
    return NormalizedDataset(
        data=df,
        timestamp_column="timestamp",
        parameters=params,
        excluded_columns=[],
        validation_messages=[],
        diagnostics=AssemblyDiagnostics(),
        is_valid=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. CSV export success
# ─────────────────────────────────────────────────────────────────────────────


def test_csv_export_success(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.csv"

    result = write_normalized_export(dataset, out)

    assert result.success is True
    assert result.format_used == "csv"
    assert result.rows_written == 5
    assert result.output_path is not None
    assert Path(result.output_path).exists()
    assert not result.errors()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metadata sidecar creation
# ─────────────────────────────────────────────────────────────────────────────


def test_metadata_sidecar_created(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.csv"

    result = write_normalized_export(dataset, out)

    assert result.metadata_path is not None
    sidecar = Path(result.metadata_path)
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["schema_version"] == EXPORT_SCHEMA_VERSION
    assert data["export_format"] == "csv"
    assert data["row_count"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# 3. overwrite=False blocks existing file
# ─────────────────────────────────────────────────────────────────────────────


def test_overwrite_false_blocks_existing(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.csv"
    opts = ExportWriteOptions(overwrite=True)

    write_normalized_export(dataset, out, options=opts)
    result = write_normalized_export(dataset, out, options=ExportWriteOptions(overwrite=False))

    assert result.success is False
    codes = [m.code for m in result.errors()]
    assert "EXPORT_FILE_EXISTS" in codes


# ─────────────────────────────────────────────────────────────────────────────
# 4. overwrite=True replaces existing file
# ─────────────────────────────────────────────────────────────────────────────


def test_overwrite_true_replaces_file(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.csv"
    opts = ExportWriteOptions(overwrite=True)

    write_normalized_export(dataset, out, options=opts)
    mtime_first = out.stat().st_mtime

    import time; time.sleep(0.02)
    result = write_normalized_export(dataset, out, options=opts)

    assert result.success is True
    assert out.stat().st_mtime >= mtime_first


# ─────────────────────────────────────────────────────────────────────────────
# 5. Unsupported format failure
# ─────────────────────────────────────────────────────────────────────────────


def test_unsupported_format_returns_error(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.xlsx"

    result = write_normalized_export(dataset, out, format="xlsx")

    assert result.success is False
    codes = [m.code for m in result.errors()]
    assert "EXPORT_UNSUPPORTED_FORMAT" in codes


def test_unsupported_extension_returns_error(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.xlsx"

    result = write_normalized_export(dataset, out)

    assert result.success is False
    assert any(m.code == "EXPORT_UNSUPPORTED_FORMAT" for m in result.validation_messages)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Missing dependency behavior
# ─────────────────────────────────────────────────────────────────────────────


def test_parquet_missing_dependency_returns_error(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.parquet"

    with patch("app.import_wizard.export_writer._PARQUET_AVAILABLE", False):
        result = write_normalized_export(dataset, out)

    assert result.success is False
    codes = [m.code for m in result.errors()]
    assert "EXPORT_DEPENDENCY_MISSING" in codes


def test_feather_missing_dependency_returns_error(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.feather"

    with patch("app.import_wizard.export_writer._FEATHER_AVAILABLE", False):
        result = write_normalized_export(dataset, out)

    assert result.success is False
    codes = [m.code for m in result.errors()]
    assert "EXPORT_DEPENDENCY_MISSING" in codes


# ─────────────────────────────────────────────────────────────────────────────
# 7. Empty dataset failure
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_dataset_returns_error(tmp_path) -> None:
    dataset = _empty_dataset()
    out = tmp_path / "export.csv"

    result = write_normalized_export(dataset, out)

    assert result.success is False
    codes = [m.code for m in result.errors()]
    assert "EXPORT_EMPTY_DATASET" in codes


# ─────────────────────────────────────────────────────────────────────────────
# 8. Timestamp column preserved in CSV
# ─────────────────────────────────────────────────────────────────────────────


def test_timestamp_column_in_csv(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.csv"

    write_normalized_export(dataset, out)

    df = pd.read_csv(out)
    assert "timestamp" in df.columns
    assert len(df) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 9. Canonical column names preserved
# ─────────────────────────────────────────────────────────────────────────────


def test_canonical_columns_in_csv(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.csv"

    write_normalized_export(dataset, out)

    df = pd.read_csv(out)
    assert "va" in df.columns
    assert "ia" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# 10. Parameter metadata in sidecar
# ─────────────────────────────────────────────────────────────────────────────


def test_parameter_metadata_in_sidecar(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.csv"

    result = write_normalized_export(dataset, out)

    assert result.metadata_path is not None
    sidecar = load_metadata_sidecar(result.metadata_path)
    params = sidecar["parameters"]
    assert len(params) == 2
    names = {p["canonical_name"] for p in params}
    assert "va" in names
    assert "ia" in names
    va_param = next(p for p in params if p["canonical_name"] == "va")
    assert va_param["unit"] == "V"
    assert va_param["parameter_type"] == "voltage"
    assert va_param["source_name"] == "VA"
    ia_param = next(p for p in params if p["canonical_name"] == "ia")
    assert ia_param["user_overridden"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 11. Timestamp repair strategy in sidecar
# ─────────────────────────────────────────────────────────────────────────────


def test_timestamp_repair_strategy_in_sidecar(tmp_path) -> None:
    dataset = _make_dataset(repair_strategy="reconstruct_from_interval")
    out = tmp_path / "export.csv"

    result = write_normalized_export(dataset, out)

    assert result.metadata_path is not None
    sidecar = load_metadata_sidecar(result.metadata_path)
    assert sidecar["timestamp_repair_strategy"] == "reconstruct_from_interval"
    assert sidecar["timestamp_column"] == "timestamp"


# ─────────────────────────────────────────────────────────────────────────────
# 12. Export does not mutate dataset
# ─────────────────────────────────────────────────────────────────────────────


def test_export_does_not_mutate_dataset(tmp_path) -> None:
    dataset = _make_dataset()
    df_before = dataset.data.copy()
    ts_before = dataset.data["timestamp"].tolist()
    params_before = [p.canonical_name for p in dataset.parameters]
    out = tmp_path / "export.csv"

    write_normalized_export(dataset, out)

    # DataFrame content unchanged
    pd.testing.assert_frame_equal(dataset.data, df_before)
    # Timestamp values unchanged (still datetime objects, not strings)
    assert dataset.data["timestamp"].tolist() == ts_before
    # Parameters list unchanged
    assert [p.canonical_name for p in dataset.parameters] == params_before


# ─────────────────────────────────────────────────────────────────────────────
# 13. ExportPlan integration
# ─────────────────────────────────────────────────────────────────────────────


def test_write_from_export_plan_csv(tmp_path) -> None:
    dataset = _make_dataset(valid=True)
    plan = plan_export(dataset, output_dir=str(tmp_path))

    result = write_from_export_plan(dataset, plan, format="csv", options=ExportWriteOptions(overwrite=True))

    assert result.success is True
    assert result.format_used == "csv"
    assert result.output_path is not None
    assert Path(result.output_path).exists()


# ─────────────────────────────────────────────────────────────────────────────
# 14. Invalid output path / directory creation failure
# ─────────────────────────────────────────────────────────────────────────────


def test_directory_creation_failure_returns_error(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "sub" / "export.csv"

    with patch("app.import_wizard.export_writer.Path.mkdir", side_effect=OSError("permission denied")):
        result = write_normalized_export(dataset, out)

    assert result.success is False
    codes = [m.code for m in result.errors()]
    assert "EXPORT_DIRECTORY_ERROR" in codes


# ─────────────────────────────────────────────────────────────────────────────
# 15. Metadata write failure — data still written, error recorded
# ─────────────────────────────────────────────────────────────────────────────


def test_sidecar_write_failure_recorded_but_data_written(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.csv"

    with patch(
        "app.import_wizard.export_writer.write_metadata_sidecar",
        side_effect=OSError("disk full"),
    ):
        result = write_normalized_export(dataset, out)

    assert result.success is True
    assert result.output_path is not None
    assert Path(result.output_path).exists()
    assert result.metadata_path is None
    assert any(m.code == "EXPORT_SIDECAR_WRITE_ERROR" for m in result.validation_messages)


# ─────────────────────────────────────────────────────────────────────────────
# 16. float_precision option (CSV)
# ─────────────────────────────────────────────────────────────────────────────


def test_float_precision_option(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.csv"
    opts = ExportWriteOptions(float_precision=2)

    write_normalized_export(dataset, out, options=opts)

    df = pd.read_csv(out)
    # Numeric columns must have at most 2 decimal places
    for col in ["va", "ia"]:
        for raw in df[col].astype(str).tolist():
            if "." in raw:
                decimals = str(raw).split(".")[1]
                assert len(decimals) <= 2, f"Precision exceeded in {col}: {raw}"


# ─────────────────────────────────────────────────────────────────────────────
# 17. timestamp_format option (CSV)
# ─────────────────────────────────────────────────────────────────────────────


def test_timestamp_format_option(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.csv"
    opts = ExportWriteOptions(timestamp_format="%Y/%m/%d")

    write_normalized_export(dataset, out, options=opts)

    df = pd.read_csv(out)
    # All timestamp values should match YYYY/MM/DD format
    import re
    for val in df["timestamp"]:
        assert re.match(r"\d{4}/\d{2}/\d{2}", str(val)), f"Unexpected format: {val}"


# ─────────────────────────────────────────────────────────────────────────────
# 18. include_index=True writes index column
# ─────────────────────────────────────────────────────────────────────────────


def test_include_index_writes_extra_column(tmp_path) -> None:
    dataset = _make_dataset()
    out_no_idx = tmp_path / "no_idx.csv"
    out_idx = tmp_path / "with_idx.csv"

    write_normalized_export(dataset, out_no_idx, options=ExportWriteOptions(include_index=False))
    result_idx = write_normalized_export(
        dataset, out_idx, options=ExportWriteOptions(include_index=True)
    )

    cols_no_idx = len(pd.read_csv(out_no_idx).columns)
    cols_idx = len(pd.read_csv(out_idx).columns)
    assert cols_idx == cols_no_idx + 1
    assert result_idx.columns_written == cols_idx


# ─────────────────────────────────────────────────────────────────────────────
# 19. Parquet export (skip when unavailable)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _PARQUET_AVAILABLE, reason="pyarrow/fastparquet not installed")
def test_parquet_export_success(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.parquet"

    result = write_normalized_export(dataset, out)

    assert result.success is True
    assert result.format_used == "parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert "va" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# 20. Feather export (skip when unavailable)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _FEATHER_AVAILABLE, reason="pyarrow not installed")
def test_feather_export_success(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.feather"

    result = write_normalized_export(dataset, out)

    assert result.success is True
    assert result.format_used == "feather"
    assert out.exists()


# ─────────────────────────────────────────────────────────────────────────────
# 21. metadata_sidecar_path naming convention
# ─────────────────────────────────────────────────────────────────────────────


def test_metadata_sidecar_path_naming() -> None:
    assert metadata_sidecar_path("/data/sample.csv") == Path("/data/sample.normalized.json")
    assert metadata_sidecar_path("/data/run_normalized.parquet") == Path(
        "/data/run_normalized.normalized.json"
    )
    assert metadata_sidecar_path("relative/file.feather") == Path(
        "relative/file.normalized.json"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 22. load_metadata_sidecar round-trip
# ─────────────────────────────────────────────────────────────────────────────


def test_load_metadata_sidecar_roundtrip(tmp_path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "export.csv"
    write_normalized_export(dataset, out)

    sidecar_file = metadata_sidecar_path(out)
    data = load_metadata_sidecar(sidecar_file)

    assert isinstance(data, dict)
    assert data["schema_version"] == EXPORT_SCHEMA_VERSION
    assert data["source_file_name"] == "test.csv"
    assert data["column_count"] == 2
    assert len(data["parameters"]) == 2
    assert data["excluded_columns"] == ["trip"]


# ─────────────────────────────────────────────────────────────────────────────
# 23. ExportPlan not-ready blocks write_from_export_plan
# ─────────────────────────────────────────────────────────────────────────────


def test_write_from_export_plan_not_ready_blocks(tmp_path) -> None:
    dataset = _empty_dataset()
    plan = plan_export(dataset, output_dir=str(tmp_path))

    result = write_from_export_plan(dataset, plan)

    assert result.success is False
    codes = [m.code for m in result.errors()]
    assert "EXPORT_NOT_READY" in codes


# ─────────────────────────────────────────────────────────────────────────────
# 24. write_from_export_plan unknown format
# ─────────────────────────────────────────────────────────────────────────────


def test_write_from_export_plan_unknown_format(tmp_path) -> None:
    dataset = _make_dataset(valid=True)
    plan = plan_export(dataset, output_dir=str(tmp_path))

    result = write_from_export_plan(dataset, plan, format="xlsx")

    assert result.success is False
    codes = [m.code for m in result.errors()]
    assert "EXPORT_UNSUPPORTED_FORMAT" in codes


# ─────────────────────────────────────────────────────────────────────────────
# 25. ExportWriteResult helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_export_write_result_helpers() -> None:
    err_msg = ValidationMessage(ValidationSeverity.ERROR, "ERR", "an error")
    warn_msg = ValidationMessage(ValidationSeverity.WARNING, "WARN", "a warning")
    result = ExportWriteResult(
        success=False,
        output_path=None,
        format_used=None,
        rows_written=0,
        columns_written=0,
        metadata_path=None,
        validation_messages=[err_msg, warn_msg],
        diagnostics_summary="",
    )
    assert result.errors() == [err_msg]
    assert result.warnings() == [warn_msg]
