"""Tests for normalized_dataset.py — model classes only."""
from __future__ import annotations

import pandas as pd
import pytest

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.normalized_dataset import (
    AssemblyDiagnostics,
    NormalizedDataset,
    ParameterMetadata,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n: int = 5) -> pd.DataFrame:
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1s"))
    return pd.DataFrame({"timestamp": ts, "mw": range(n)})


def _make_param(
    canonical: str = "mw",
    ptype: ParameterType = ParameterType.MW,
    unit: str | None = "MW",
    source_name: str = "MW",
    source_index: int = 1,
) -> ParameterMetadata:
    return ParameterMetadata(
        canonical_name=canonical,
        parameter_type=ptype,
        unit=unit,
        source_name=source_name,
        source_index=source_index,
    )


def _make_dataset(
    df: pd.DataFrame | None = None,
    params: list[ParameterMetadata] | None = None,
    messages: list[ValidationMessage] | None = None,
    is_valid: bool = True,
) -> NormalizedDataset:
    return NormalizedDataset(
        data=df if df is not None else _make_df(),
        timestamp_column="timestamp",
        parameters=params if params is not None else [_make_param()],
        excluded_columns=["Current"],
        validation_messages=messages or [],
        diagnostics=AssemblyDiagnostics(total_rows=5, normalized_rows=5),
        is_valid=is_valid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ParameterMetadata
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterMetadata:
    def test_required_fields_set(self):
        pm = ParameterMetadata(
            canonical_name="voltage",
            parameter_type=ParameterType.VOLTAGE,
            unit="kV",
            source_name="Voltage_A",
            source_index=2,
        )
        assert pm.canonical_name == "voltage"
        assert pm.parameter_type == ParameterType.VOLTAGE
        assert pm.unit == "kV"
        assert pm.source_name == "Voltage_A"
        assert pm.source_index == 2

    def test_defaults(self):
        pm = ParameterMetadata(
            canonical_name="mw",
            parameter_type=ParameterType.MW,
            unit=None,
            source_name="MW",
            source_index=0,
        )
        assert pm.user_overridden is False
        assert pm.confidence == 0.0
        assert pm.notes == []

    def test_user_overridden_flag(self):
        pm = ParameterMetadata(
            canonical_name="active_power",
            parameter_type=ParameterType.MW,
            unit="MW",
            source_name="P",
            source_index=0,
            user_overridden=True,
        )
        assert pm.user_overridden is True

    def test_notes_mutable(self):
        pm = _make_param()
        pm.notes.append("check units")
        assert len(pm.notes) == 1

    def test_unit_none_allowed(self):
        pm = _make_param(unit=None)
        assert pm.unit is None


# ─────────────────────────────────────────────────────────────────────────────
# AssemblyDiagnostics
# ─────────────────────────────────────────────────────────────────────────────

class TestAssemblyDiagnostics:
    def test_all_defaults_zero(self):
        d = AssemblyDiagnostics()
        assert d.total_rows == 0
        assert d.normalized_rows == 0
        assert d.dropped_rows == 0
        assert d.duplicate_timestamp_count == 0
        assert d.invalid_value_count == 0
        assert d.included_columns == []
        assert d.excluded_columns == []
        assert d.parameter_counts == {}

    def test_fields_assignable(self):
        d = AssemblyDiagnostics(
            total_rows=100,
            normalized_rows=98,
            dropped_rows=2,
            duplicate_timestamp_count=1,
            invalid_value_count=5,
            included_columns=["timestamp", "mw"],
            excluded_columns=["raw_col"],
            parameter_counts={"mw": 1},
        )
        assert d.total_rows == 100
        assert d.normalized_rows == 98
        assert d.dropped_rows == 2
        assert d.parameter_counts["mw"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# NormalizedDataset
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizedDataset:
    def test_creation(self):
        ds = _make_dataset()
        assert isinstance(ds, NormalizedDataset)
        assert isinstance(ds.data, pd.DataFrame)

    def test_is_export_ready_true(self):
        ds = _make_dataset(is_valid=True)
        assert ds.is_export_ready()

    def test_is_export_ready_false_when_not_valid(self):
        ds = _make_dataset(is_valid=False)
        assert not ds.is_export_ready()

    def test_is_export_ready_false_when_no_params(self):
        ds = _make_dataset(params=[], is_valid=True)
        assert not ds.is_export_ready()

    def test_is_export_ready_false_when_empty_df(self):
        empty_df = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns]"), "mw": []})
        ds = _make_dataset(df=empty_df, is_valid=True)
        assert not ds.is_export_ready()

    def test_parameter_by_canonical_found(self):
        pm = _make_param(canonical="mw")
        ds = _make_dataset(params=[pm])
        found = ds.parameter_by_canonical("mw")
        assert found is pm

    def test_parameter_by_canonical_not_found(self):
        ds = _make_dataset()
        assert ds.parameter_by_canonical("does_not_exist") is None

    def test_parameters_by_type_match(self):
        pm_mw = _make_param(canonical="mw", ptype=ParameterType.MW)
        pm_v = _make_param(canonical="voltage", ptype=ParameterType.VOLTAGE, source_name="V", source_index=2)
        ds = _make_dataset(params=[pm_mw, pm_v])
        mw_params = ds.parameters_by_type(ParameterType.MW)
        assert len(mw_params) == 1
        assert mw_params[0].canonical_name == "mw"

    def test_parameters_by_type_no_match(self):
        ds = _make_dataset()
        result = ds.parameters_by_type(ParameterType.CURRENT)
        assert result == []

    def test_has_errors_true(self):
        msg = ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="TEST",
            message="test",
        )
        ds = _make_dataset(messages=[msg])
        assert ds.has_errors()

    def test_has_errors_false_with_warning_only(self):
        msg = ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TEST",
            message="test",
        )
        ds = _make_dataset(messages=[msg])
        assert not ds.has_errors()

    def test_has_errors_false_when_clean(self):
        ds = _make_dataset(messages=[])
        assert not ds.has_errors()

    def test_valid_data_drops_nat(self):
        ts = pd.Series([
            pd.Timestamp("2024-01-01"),
            pd.NaT,
            pd.Timestamp("2024-01-03"),
        ])
        df = pd.DataFrame({"timestamp": ts, "mw": [1.0, 2.0, 3.0]})
        ds = _make_dataset(df=df)
        result = ds.valid_data()
        assert len(result) == 2
        assert result["timestamp"].isna().sum() == 0

    def test_valid_data_returns_reset_index(self):
        ts = pd.Series([pd.Timestamp("2024-01-01"), pd.NaT])
        df = pd.DataFrame({"timestamp": ts, "mw": [1.0, 2.0]})
        ds = _make_dataset(df=df)
        result = ds.valid_data()
        assert list(result.index) == [0]

    def test_optional_fields_default_none(self):
        ds = _make_dataset()
        assert ds.source_path is None
        assert ds.source_file_name is None

    def test_timestamp_repair_strategy_default(self):
        ds = _make_dataset()
        assert ds.timestamp_repair_strategy == "unknown"
