"""Tests for data_assembler.py and dataset_validation.py."""
from __future__ import annotations

import pandas as pd
import pytest

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.data_assembler import (
    _disambiguate_names,
    _normalize_unit,
    _sanitize_name,
    assemble_normalized_dataset,
)
from app.import_wizard.dataset_validation import validate_normalized_dataset
from app.import_wizard.models import ColumnMappingCandidate
from app.import_wizard.normalization_plan import NormalizationPlan
from app.import_wizard.normalized_dataset import (
    AssemblyDiagnostics,
    NormalizedDataset,
    ParameterMetadata,
)
from app.import_wizard.repair_diagnostics import RepairDiagnostics
from app.import_wizard.timestamp_contracts import TimestampRepairPlan, TimestampRepairStrategy
from app.import_wizard.timestamp_normalizer import TimestampNormalizationResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _raw_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "MW": [100.0 + i for i in range(n)],
        "Voltage": [11.0 + i * 0.1 for i in range(n)],
        "Current": [50.0 + i for i in range(n)],
    })


def _ts_result(n: int = 5, success: bool = True) -> TimestampNormalizationResult:
    normalized = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1s"))
    diag = RepairDiagnostics(strategy_used="no_repair", total_rows=n, valid_rows=n)
    return TimestampNormalizationResult(normalized=normalized, diagnostics=diag, success=success)


def _plan(
    selected: list[str] | None = None,
    excluded: list[str] | None = None,
    renames: dict[str, str] | None = None,
    units: dict[str, str] | None = None,
    types: dict[str, ParameterType] | None = None,
) -> NormalizationPlan:
    ts_plan = TimestampRepairPlan(
        strategy=TimestampRepairStrategy.NO_REPAIR,
        repair_validated=True,
    )
    return NormalizationPlan(
        timestamp_plan=ts_plan,
        selected_columns=selected if selected is not None else ["MW", "Voltage"],
        excluded_columns=excluded or [],
        column_renames=renames or {},
        column_units=units or {},
        column_types=types or {},
    )


def _candidates(names: list[str] | None = None) -> list[ColumnMappingCandidate]:
    _info = {
        "MW": (ParameterType.MW, "MW", "mw"),
        "Voltage": (ParameterType.VOLTAGE, "V", "voltage"),
        "Current": (ParameterType.CURRENT, "A", "current"),
    }
    cols = names or ["MW", "Voltage"]
    result = []
    for i, name in enumerate(cols):
        ptype, unit, sname = _info.get(name, (ParameterType.UNKNOWN, None, name.lower()))
        result.append(ColumnMappingCandidate(
            source_name=name,
            source_index=i,
            suggested_name=sname,
            parameter_type=ptype,
            unit=unit,
            confidence=0.80,
        ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# _sanitize_name
# ─────────────────────────────────────────────────────────────────────────────

class TestSanitizeName:
    def test_lowercase(self):
        assert _sanitize_name("MW") == "mw"

    def test_spaces_to_underscores(self):
        assert _sanitize_name("Active Power") == "active_power"

    def test_special_chars_removed(self):
        assert _sanitize_name("V (kV)") == "v_kv"

    def test_empty_string_fallback(self):
        assert _sanitize_name("") == "column"

    def test_only_underscores_fallback(self):
        assert _sanitize_name("___") == "column"

    def test_numeric_preserved(self):
        assert _sanitize_name("ch1") == "ch1"

    def test_leading_trailing_stripped(self):
        assert _sanitize_name("  MW  ") == "mw"


# ─────────────────────────────────────────────────────────────────────────────
# _disambiguate_names
# ─────────────────────────────────────────────────────────────────────────────

class TestDisambiguateNames:
    def test_no_duplicates_unchanged(self):
        assert _disambiguate_names(["a", "b", "c"]) == ["a", "b", "c"]

    def test_duplicates_get_suffixes(self):
        result = _disambiguate_names(["voltage", "voltage"])
        assert result == ["voltage_1", "voltage_2"]

    def test_triple_duplicate(self):
        result = _disambiguate_names(["mw", "mw", "mw"])
        assert result == ["mw_1", "mw_2", "mw_3"]

    def test_mixed_unique_and_duplicate(self):
        result = _disambiguate_names(["mw", "voltage", "mw"])
        assert result[1] == "voltage"  # unique, unchanged
        assert result[0].startswith("mw_")
        assert result[2].startswith("mw_")

    def test_empty_list(self):
        assert _disambiguate_names([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_unit
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeUnit:
    def test_none_passthrough(self):
        assert _normalize_unit(None) is None

    def test_empty_string_returns_none(self):
        assert _normalize_unit("  ") is None

    def test_mw_uppercase(self):
        assert _normalize_unit("mw") == "MW"

    def test_hz_uppercase(self):
        assert _normalize_unit("hz") == "Hz"

    def test_kv_canonical(self):
        assert _normalize_unit("kv") == "kV"

    def test_a_uppercase(self):
        assert _normalize_unit("a") == "A"

    def test_mvar_canonical(self):
        assert _normalize_unit("mvar") == "Mvar"

    def test_hz_per_s_canonical(self):
        assert _normalize_unit("hz/s") == "Hz/s"

    def test_unknown_unit_stripped(self):
        assert _normalize_unit("  Ohm  ") == "Ohm"

    def test_strip_whitespace(self):
        assert _normalize_unit("  MW  ") == "MW"


# ─────────────────────────────────────────────────────────────────────────────
# assemble_normalized_dataset — basic output
# ─────────────────────────────────────────────────────────────────────────────

class TestAssemblyBasic:
    def test_returns_normalized_dataset(self):
        result = assemble_normalized_dataset(_raw_df(), _plan(), _ts_result())
        assert isinstance(result, NormalizedDataset)

    def test_timestamp_column_first(self):
        result = assemble_normalized_dataset(_raw_df(), _plan(), _ts_result())
        assert result.data.columns[0] == "timestamp"

    def test_timestamp_column_is_datetime(self):
        result = assemble_normalized_dataset(_raw_df(), _plan(), _ts_result())
        assert pd.api.types.is_datetime64_any_dtype(result.data["timestamp"])

    def test_selected_columns_present(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(selected=["MW", "Voltage"]), _ts_result(),
            column_mappings=_candidates()
        )
        cols = list(result.data.columns)
        assert "mw" in cols
        assert "voltage" in cols

    def test_excluded_column_absent(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(selected=["MW", "Voltage"], excluded=["Current"]),
            _ts_result(), column_mappings=_candidates()
        )
        assert "current" not in result.data.columns
        assert "Current" not in result.data.columns

    def test_row_count_correct(self):
        result = assemble_normalized_dataset(_raw_df(5), _plan(), _ts_result(5))
        assert len(result.data) == 5

    def test_is_valid_true_for_clean_data(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(), _ts_result(), column_mappings=_candidates()
        )
        assert result.is_valid

    def test_integer_range_index(self):
        result = assemble_normalized_dataset(_raw_df(), _plan(), _ts_result())
        assert list(result.data.index) == list(range(len(result.data)))


# ─────────────────────────────────────────────────────────────────────────────
# assemble_normalized_dataset — NaT row handling
# ─────────────────────────────────────────────────────────────────────────────

class TestNatRowHandling:
    def _ts_with_nat(self, n: int = 5) -> TimestampNormalizationResult:
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1s"), dtype="datetime64[ns]")
        ts.iloc[2] = pd.NaT
        diag = RepairDiagnostics(strategy_used="no_repair", total_rows=n, valid_rows=n - 1)
        return TimestampNormalizationResult(normalized=ts, diagnostics=diag, success=True)

    def test_nat_rows_dropped_by_default(self):
        result = assemble_normalized_dataset(_raw_df(5), _plan(), self._ts_with_nat(5))
        assert len(result.data) == 4
        assert result.data["timestamp"].isna().sum() == 0

    def test_nat_rows_preserved_when_disabled(self):
        result = assemble_normalized_dataset(
            _raw_df(5), _plan(), self._ts_with_nat(5), drop_nat_rows=False
        )
        assert len(result.data) == 5
        assert result.data["timestamp"].isna().sum() == 1

    def test_dropped_rows_counted_in_diagnostics(self):
        result = assemble_normalized_dataset(_raw_df(5), _plan(), self._ts_with_nat(5))
        assert result.diagnostics.dropped_rows == 1


# ─────────────────────────────────────────────────────────────────────────────
# assemble_normalized_dataset — canonical naming
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalNaming:
    def test_plan_rename_applied(self):
        result = assemble_normalized_dataset(
            _raw_df(),
            _plan(selected=["MW"], renames={"MW": "active_power"}),
            _ts_result(),
        )
        assert "active_power" in result.data.columns
        assert "MW" not in result.data.columns

    def test_user_override_wins_over_suggested(self):
        cands = _candidates(["MW", "Voltage"])
        cands[0].user_name_override = "p_mw"
        result = assemble_normalized_dataset(
            _raw_df(),
            _plan(selected=["MW", "Voltage"]),
            _ts_result(),
            column_mappings=cands,
        )
        assert "p_mw" in result.data.columns

    def test_suggested_name_used_when_no_override(self):
        result = assemble_normalized_dataset(
            _raw_df(),
            _plan(selected=["MW"]),
            _ts_result(),
            column_mappings=_candidates(["MW"]),
        )
        assert "mw" in result.data.columns

    def test_sanitize_fallback_when_no_candidate(self):
        df = pd.DataFrame({"My Column": [1.0, 2.0, 3.0]})
        ts = _ts_result(3)
        plan = _plan(selected=["My Column"])
        result = assemble_normalized_dataset(df, plan, ts)
        assert "my_column" in result.data.columns

    def test_duplicate_names_disambiguated(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        ts = _ts_result(3)
        cands = [
            ColumnMappingCandidate(
                source_name="A", source_index=0,
                suggested_name="channel", parameter_type=ParameterType.UNKNOWN,
                confidence=0.1,
            ),
            ColumnMappingCandidate(
                source_name="B", source_index=1,
                suggested_name="channel", parameter_type=ParameterType.UNKNOWN,
                confidence=0.1,
            ),
        ]
        plan = _plan(selected=["A", "B"])
        result = assemble_normalized_dataset(df, plan, ts, column_mappings=cands)
        data_cols = [c for c in result.data.columns if c != "timestamp"]
        assert len(set(data_cols)) == len(data_cols), "Canonical names must be unique"


# ─────────────────────────────────────────────────────────────────────────────
# assemble_normalized_dataset — immutability
# ─────────────────────────────────────────────────────────────────────────────

class TestImmutability:
    def test_raw_dataframe_not_mutated(self):
        df = _raw_df(5)
        original_cols = list(df.columns)
        original_vals = df["MW"].tolist()
        assemble_normalized_dataset(df, _plan(), _ts_result())
        assert list(df.columns) == original_cols
        assert df["MW"].tolist() == original_vals

    def test_timestamp_series_not_mutated(self):
        ts_result = _ts_result(5)
        original_first = ts_result.normalized.iloc[0]
        assemble_normalized_dataset(_raw_df(5), _plan(), ts_result)
        assert ts_result.normalized.iloc[0] == original_first


# ─────────────────────────────────────────────────────────────────────────────
# assemble_normalized_dataset — ParameterMetadata
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterMetadataAssembly:
    def test_source_name_preserved(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(selected=["MW"]), _ts_result(),
            column_mappings=_candidates(["MW"]),
        )
        pm = result.parameters[0]
        assert pm.source_name == "MW"

    def test_source_index_preserved(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(selected=["MW"]), _ts_result(),
            column_mappings=_candidates(["MW"]),
        )
        pm = result.parameters[0]
        assert pm.source_index == 0

    def test_user_overridden_flag_set_when_plan_renames(self):
        result = assemble_normalized_dataset(
            _raw_df(),
            _plan(selected=["MW"], renames={"MW": "active_power"}),
            _ts_result(),
        )
        pm = result.parameters[0]
        assert pm.user_overridden is True

    def test_user_overridden_flag_false_when_auto(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(selected=["MW"]), _ts_result(),
            column_mappings=_candidates(["MW"]),
        )
        pm = result.parameters[0]
        assert pm.user_overridden is False

    def test_confidence_from_candidate(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(selected=["MW"]), _ts_result(),
            column_mappings=_candidates(["MW"]),
        )
        assert result.parameters[0].confidence == pytest.approx(0.80)

    def test_confidence_zero_without_candidate(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(selected=["MW"]), _ts_result(),
        )
        assert result.parameters[0].confidence == 0.0

    def test_unit_normalized(self):
        cands = _candidates(["MW"])
        cands[0].unit = "mw"
        result = assemble_normalized_dataset(
            _raw_df(), _plan(selected=["MW"]), _ts_result(),
            column_mappings=cands,
        )
        assert result.parameters[0].unit == "MW"

    def test_type_from_candidate(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(selected=["Voltage"]), _ts_result(),
            column_mappings=_candidates(["Voltage"]),
        )
        assert result.parameters[0].parameter_type == ParameterType.VOLTAGE

    def test_type_from_plan_overrides_candidate(self):
        result = assemble_normalized_dataset(
            _raw_df(),
            _plan(
                selected=["MW"],
                types={"mw": ParameterType.UNKNOWN},
            ),
            _ts_result(),
            column_mappings=_candidates(["MW"]),
        )
        assert result.parameters[0].parameter_type == ParameterType.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# assemble_normalized_dataset — diagnostics
# ─────────────────────────────────────────────────────────────────────────────

class TestAssemblyDiagnosticsOutput:
    def test_normalized_rows_correct(self):
        result = assemble_normalized_dataset(_raw_df(5), _plan(), _ts_result(5))
        assert result.diagnostics.normalized_rows == 5

    def test_total_rows_correct(self):
        result = assemble_normalized_dataset(_raw_df(10), _plan(), _ts_result(10))
        assert result.diagnostics.total_rows == 10

    def test_included_columns_has_timestamp_first(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(selected=["MW"]), _ts_result(),
            column_mappings=_candidates(["MW"]),
        )
        assert result.diagnostics.included_columns[0] == "timestamp"
        assert "mw" in result.diagnostics.included_columns

    def test_excluded_columns_tracked(self):
        result = assemble_normalized_dataset(
            _raw_df(),
            _plan(selected=["MW"], excluded=["Voltage"]),
            _ts_result(),
        )
        assert "Voltage" in result.diagnostics.excluded_columns

    def test_parameter_counts_by_type(self):
        result = assemble_normalized_dataset(
            _raw_df(),
            _plan(selected=["MW", "Voltage"]),
            _ts_result(),
            column_mappings=_candidates(["MW", "Voltage"]),
        )
        counts = result.diagnostics.parameter_counts
        assert counts.get("mw", 0) == 1
        assert counts.get("voltage", 0) == 1

    def test_invalid_value_count(self):
        df = pd.DataFrame({"MW": [1.0, float("nan"), 3.0]})
        result = assemble_normalized_dataset(
            df, _plan(selected=["MW"]), _ts_result(3)
        )
        assert result.diagnostics.invalid_value_count == 1

    def test_duplicate_timestamp_count(self):
        ts_dup = pd.Series([
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-01"),  # duplicate
            pd.Timestamp("2024-01-02"),
        ])
        diag = RepairDiagnostics(strategy_used="no_repair", total_rows=3, valid_rows=3)
        ts_result = TimestampNormalizationResult(
            normalized=ts_dup, diagnostics=diag, success=True
        )
        df = pd.DataFrame({"MW": [1.0, 2.0, 3.0]})
        result = assemble_normalized_dataset(df, _plan(selected=["MW"]), ts_result)
        assert result.diagnostics.duplicate_timestamp_count >= 1


# ─────────────────────────────────────────────────────────────────────────────
# assemble_normalized_dataset — error conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestAssemblyErrors:
    def test_no_selected_columns_gives_error(self):
        result = assemble_normalized_dataset(_raw_df(), _plan(selected=[]), _ts_result())
        codes = {m.code for m in result.validation_messages}
        assert "ND_NO_COLUMNS" in codes
        assert not result.is_valid

    def test_missing_column_gives_warning(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(selected=["MW", "DoesNotExist"]), _ts_result()
        )
        codes = {m.code for m in result.validation_messages}
        assert "ND_MISSING_COLUMNS" in codes

    def test_ts_repair_failed_gives_warning(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(), _ts_result(success=False)
        )
        codes = {m.code for m in result.validation_messages}
        assert "ND_TS_REPAIR_FAILED" in codes

    def test_ts_length_mismatch_gives_error(self):
        # 5-row DataFrame but only 3 timestamps
        ts_short = pd.Series(pd.date_range("2024-01-01", periods=3, freq="1s"))
        diag = RepairDiagnostics(strategy_used="no_repair", total_rows=3, valid_rows=3)
        ts_result_short = TimestampNormalizationResult(
            normalized=ts_short, diagnostics=diag, success=True
        )
        result = assemble_normalized_dataset(_raw_df(5), _plan(), ts_result_short)
        codes = {m.code for m in result.validation_messages}
        assert "ND_TS_LENGTH_MISMATCH" in codes

    def test_ts_name_conflict_handled(self):
        df = pd.DataFrame({"MW": [1.0, 2.0, 3.0]})
        cands = [ColumnMappingCandidate(
            source_name="MW", source_index=0,
            suggested_name="timestamp",  # would collide!
            parameter_type=ParameterType.MW, confidence=0.5,
        )]
        result = assemble_normalized_dataset(
            df, _plan(selected=["MW"]), _ts_result(3), column_mappings=cands
        )
        assert "timestamp" in result.data.columns
        assert "data_timestamp" in result.data.columns
        codes = {m.code for m in result.validation_messages}
        assert "ND_TIMESTAMP_NAME_CONFLICT" in codes


# ─────────────────────────────────────────────────────────────────────────────
# assemble_normalized_dataset — source traceability
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceability:
    def test_source_path_stored(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(), _ts_result(),
            source_path="/data/recordings/event_20240101.csv",
        )
        assert result.source_path == "/data/recordings/event_20240101.csv"

    def test_source_file_name_extracted(self):
        result = assemble_normalized_dataset(
            _raw_df(), _plan(), _ts_result(),
            source_path="/data/recordings/event_20240101.csv",
        )
        assert result.source_file_name == "event_20240101.csv"

    def test_source_path_none_by_default(self):
        result = assemble_normalized_dataset(_raw_df(), _plan(), _ts_result())
        assert result.source_path is None
        assert result.source_file_name is None

    def test_repair_strategy_from_plan(self):
        result = assemble_normalized_dataset(_raw_df(), _plan(), _ts_result())
        assert result.timestamp_repair_strategy == "no_repair"


# ─────────────────────────────────────────────────────────────────────────────
# validate_normalized_dataset
# ─────────────────────────────────────────────────────────────────────────────

def _make_valid_dataset(n: int = 5) -> NormalizedDataset:
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1s"))
    df = pd.DataFrame({
        "timestamp": ts,
        "mw": [float(i) for i in range(n)],
    })
    params = [ParameterMetadata(
        canonical_name="mw",
        parameter_type=ParameterType.MW,
        unit="MW",
        source_name="MW",
        source_index=0,
    )]
    return NormalizedDataset(
        data=df,
        timestamp_column="timestamp",
        parameters=params,
        excluded_columns=[],
        validation_messages=[],
        diagnostics=AssemblyDiagnostics(total_rows=n, normalized_rows=n),
        is_valid=True,
    )


class TestValidateNormalizedDataset:
    def test_clean_dataset_no_issues(self):
        ds = _make_valid_dataset()
        msgs = validate_normalized_dataset(ds)
        assert msgs == []

    def test_missing_timestamp_col(self):
        df = pd.DataFrame({"mw": [1.0, 2.0, 3.0]})
        ds = NormalizedDataset(
            data=df,
            timestamp_column="timestamp",  # not in df
            parameters=[],
            excluded_columns=[],
            validation_messages=[],
            diagnostics=AssemblyDiagnostics(),
        )
        msgs = validate_normalized_dataset(ds)
        codes = {m.code for m in msgs}
        assert "NV_NO_TIMESTAMP_COL" in codes
        assert all(m.severity == ValidationSeverity.ERROR for m in msgs)

    def test_empty_dataset(self):
        df = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns]"), "mw": []})
        ds = NormalizedDataset(
            data=df,
            timestamp_column="timestamp",
            parameters=[],
            excluded_columns=[],
            validation_messages=[],
            diagnostics=AssemblyDiagnostics(),
        )
        msgs = validate_normalized_dataset(ds)
        codes = {m.code for m in msgs}
        assert "NV_EMPTY_DATASET" in codes

    def test_non_monotonic_timestamps(self):
        df = pd.DataFrame({
            "timestamp": pd.Series([
                pd.Timestamp("2024-01-01 00:00:02"),
                pd.Timestamp("2024-01-01 00:00:01"),  # backward
            ]),
            "mw": [1.0, 2.0],
        })
        ds = _make_valid_dataset()
        ds.data = df
        msgs = validate_normalized_dataset(ds)
        codes = {m.code for m in msgs}
        assert "NV_NON_MONOTONIC" in codes
        assert any(m.severity == ValidationSeverity.WARNING for m in msgs)

    def test_duplicate_timestamps(self):
        df = pd.DataFrame({
            "timestamp": pd.Series([
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),  # duplicate
                pd.Timestamp("2024-01-02"),
            ]),
            "mw": [1.0, 2.0, 3.0],
        })
        ds = _make_valid_dataset()
        ds.data = df
        msgs = validate_normalized_dataset(ds)
        codes = {m.code for m in msgs}
        assert "NV_DUPLICATE_TIMESTAMPS" in codes

    def test_no_data_columns(self):
        df = pd.DataFrame({
            "timestamp": pd.Series(pd.date_range("2024-01-01", periods=3, freq="1s")),
        })
        ds = NormalizedDataset(
            data=df,
            timestamp_column="timestamp",
            parameters=[],
            excluded_columns=[],
            validation_messages=[],
            diagnostics=AssemblyDiagnostics(),
        )
        msgs = validate_normalized_dataset(ds)
        codes = {m.code for m in msgs}
        assert "NV_NO_DATA_COLUMNS" in codes

    def test_all_unknown_types_warning(self):
        ds = _make_valid_dataset()
        ds.parameters[0] = ParameterMetadata(
            canonical_name="mw",
            parameter_type=ParameterType.UNKNOWN,
            unit=None,
            source_name="MW",
            source_index=0,
        )
        msgs = validate_normalized_dataset(ds)
        codes = {m.code for m in msgs}
        assert "NV_ALL_UNKNOWN_TYPES" in codes

    def test_all_nat_gives_error(self):
        df = pd.DataFrame({
            "timestamp": pd.Series([pd.NaT, pd.NaT, pd.NaT]),
            "mw": [1.0, 2.0, 3.0],
        })
        ds = _make_valid_dataset()
        ds.data = df
        msgs = validate_normalized_dataset(ds)
        codes = {m.code for m in msgs}
        assert "NV_ALL_NAT" in codes
        assert any(m.severity == ValidationSeverity.ERROR for m in msgs)

    def test_messages_have_valid_severity(self):
        ds = _make_valid_dataset()
        msgs = validate_normalized_dataset(ds)
        for m in msgs:
            assert isinstance(m.severity, ValidationSeverity)
