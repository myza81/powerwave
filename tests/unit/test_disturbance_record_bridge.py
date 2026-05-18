"""Tests for disturbance_record_bridge and disturbance_record_mapping.

No Qt.  All datasets are synthetic.  Tests are deterministic and fast.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.disturbance_record_bridge import (
    BridgeResult,
    ConversionOptions,
    build_disturbance_record,
    convert_normalized_dataset_to_record,
)
from app.import_wizard.disturbance_record_mapping import (
    _coerce_digital,
    build_time_array,
    build_waveform_data,
    infer_nominal_frequency,
    infer_phase,
    map_to_analog_channel,
    map_to_digital_channel,
    partition_parameters,
)
from app.import_wizard.normalized_dataset import (
    AssemblyDiagnostics,
    NormalizedDataset,
    ParameterMetadata,
)
from app.models import DisturbanceRecord


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_param(
    canonical: str = "mw",
    ptype: ParameterType = ParameterType.MW,
    unit: str | None = None,
    source_name: str | None = None,
) -> ParameterMetadata:
    return ParameterMetadata(
        canonical_name=canonical,
        parameter_type=ptype,
        unit=unit,
        source_name=source_name or canonical,
        source_index=0,
    )


def _make_dataset(
    n: int = 10,
    params: list[ParameterMetadata] | None = None,
    source_path: str | None = None,
    source_file_name: str | None = "recording.csv",
    is_valid: bool = True,
    extra_cols: dict | None = None,
    freq_hz: str = "1s",
) -> NormalizedDataset:
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq=freq_hz))
    df_cols: dict = {"timestamp": ts}
    if params:
        for pm in params:
            if pm.parameter_type == ParameterType.DIGITAL:
                df_cols[pm.canonical_name] = [i % 2 for i in range(n)]
            else:
                df_cols[pm.canonical_name] = [float(i) for i in range(n)]
    if extra_cols:
        df_cols.update(extra_cols)
    df = pd.DataFrame(df_cols)
    return NormalizedDataset(
        data=df,
        timestamp_column="timestamp",
        parameters=params or [],
        excluded_columns=[],
        validation_messages=[],
        diagnostics=AssemblyDiagnostics(total_rows=n, normalized_rows=n),
        source_path=source_path,
        source_file_name=source_file_name,
        timestamp_repair_strategy="no_repair",
        is_valid=is_valid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# partition_parameters
# ─────────────────────────────────────────────────────────────────────────────

class TestPartitionParameters:
    def test_empty_list(self):
        a, d = partition_parameters([])
        assert a == [] and d == []

    def test_digital_goes_to_digital(self):
        p = _make_param("cb1", ParameterType.DIGITAL)
        a, d = partition_parameters([p])
        assert d == [p] and a == []

    def test_voltage_goes_to_analog(self):
        p = _make_param("va", ParameterType.VOLTAGE)
        a, d = partition_parameters([p])
        assert a == [p] and d == []

    def test_unknown_goes_to_analog(self):
        p = _make_param("x", ParameterType.UNKNOWN)
        a, d = partition_parameters([p])
        assert a == [p] and d == []

    def test_mixed_partition(self):
        params = [
            _make_param("va", ParameterType.VOLTAGE),
            _make_param("cb1", ParameterType.DIGITAL),
            _make_param("mw", ParameterType.MW),
            _make_param("cb2", ParameterType.DIGITAL),
        ]
        a, d = partition_parameters(params)
        assert len(a) == 2 and len(d) == 2

    def test_all_analog_types(self):
        types = [
            ParameterType.VOLTAGE, ParameterType.CURRENT, ParameterType.MW,
            ParameterType.MVAR, ParameterType.FREQUENCY, ParameterType.ROCOF,
            ParameterType.UNKNOWN, ParameterType.TIMESTAMP,
        ]
        params = [_make_param(f"ch{i}", t) for i, t in enumerate(types)]
        a, d = partition_parameters(params)
        assert len(a) == len(types) and d == []


# ─────────────────────────────────────────────────────────────────────────────
# infer_phase
# ─────────────────────────────────────────────────────────────────────────────

class TestInferPhase:
    def test_underscore_a(self):
        assert infer_phase("voltage_a") == "A"

    def test_underscore_b(self):
        assert infer_phase("current_b") == "B"

    def test_dash_c(self):
        assert infer_phase("va-c") == "C"

    def test_space_a(self):
        assert infer_phase("voltage a") == "A"

    def test_uppercase_input_normalised(self):
        assert infer_phase("VA_A") == "A"

    def test_no_phase_returns_none(self):
        assert infer_phase("mw") is None

    def test_no_separator_before_a(self):
        assert infer_phase("voltage_va") is None

    def test_frequency_no_phase(self):
        assert infer_phase("frequency") is None


# ─────────────────────────────────────────────────────────────────────────────
# map_to_analog_channel
# ─────────────────────────────────────────────────────────────────────────────

class TestMapToAnalogChannel:
    def test_name_preserved(self):
        ch = map_to_analog_channel(_make_param("voltage_a", ParameterType.VOLTAGE), 0)
        assert ch.name == "voltage_a"

    def test_index_preserved(self):
        ch = map_to_analog_channel(_make_param("mw"), 3)
        assert ch.index == 3

    def test_voltage_unit_kv(self):
        ch = map_to_analog_channel(_make_param("va", ParameterType.VOLTAGE), 0)
        assert ch.unit == "kV"

    def test_current_unit_a(self):
        ch = map_to_analog_channel(_make_param("ia", ParameterType.CURRENT), 0)
        assert ch.unit == "A"

    def test_mw_unit(self):
        ch = map_to_analog_channel(_make_param("mw", ParameterType.MW), 0)
        assert ch.unit == "MW"

    def test_mvar_unit(self):
        ch = map_to_analog_channel(_make_param("mvar", ParameterType.MVAR), 0)
        assert ch.unit == "Mvar"

    def test_frequency_unit_hz(self):
        ch = map_to_analog_channel(_make_param("freq", ParameterType.FREQUENCY), 0)
        assert ch.unit == "Hz"

    def test_rocof_unit(self):
        ch = map_to_analog_channel(_make_param("rocof", ParameterType.ROCOF), 0)
        assert ch.unit == "Hz/s"

    def test_explicit_unit_overrides_type_default(self):
        p = _make_param("va", ParameterType.VOLTAGE, unit="V")
        ch = map_to_analog_channel(p, 0)
        assert ch.unit == "V"

    def test_phase_inferred(self):
        ch = map_to_analog_channel(_make_param("voltage_b", ParameterType.VOLTAGE), 0)
        assert ch.phase == "B"

    def test_phase_none_when_absent(self):
        ch = map_to_analog_channel(_make_param("mw", ParameterType.MW), 0)
        assert ch.phase is None

    def test_description_when_source_differs(self):
        p = _make_param("mw", source_name="MW_Import")
        ch = map_to_analog_channel(p, 0)
        assert ch.description is not None and "MW_Import" in ch.description

    def test_no_description_when_source_same(self):
        ch = map_to_analog_channel(_make_param("mw"), 0)
        assert ch.description is None


# ─────────────────────────────────────────────────────────────────────────────
# map_to_digital_channel
# ─────────────────────────────────────────────────────────────────────────────

class TestMapToDigitalChannel:
    def test_name_preserved(self):
        ch = map_to_digital_channel(_make_param("cb1", ParameterType.DIGITAL), 0)
        assert ch.name == "cb1"

    def test_index_preserved(self):
        ch = map_to_digital_channel(_make_param("cb1", ParameterType.DIGITAL), 5)
        assert ch.index == 5

    def test_normal_state_zero(self):
        ch = map_to_digital_channel(_make_param("cb1", ParameterType.DIGITAL), 0)
        assert ch.normal_state == 0

    def test_description_source_differs(self):
        p = _make_param("cb1", ParameterType.DIGITAL, source_name="CB1_STATUS")
        ch = map_to_digital_channel(p, 0)
        assert ch.description is not None and "CB1_STATUS" in ch.description


# ─────────────────────────────────────────────────────────────────────────────
# build_time_array
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildTimeArray:
    def test_first_element_is_zero(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1s"))
        arr, _, _ = build_time_array(ts)
        assert arr[0] == pytest.approx(0.0)

    def test_length_preserved(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="1s"))
        arr, _, _ = build_time_array(ts)
        assert len(arr) == 10

    def test_dtype_float64(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1s"))
        arr, _, _ = build_time_array(ts)
        assert arr.dtype == np.float64

    def test_regular_1hz_sample_rate(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=100, freq="1s"))
        _, _, sr = build_time_array(ts)
        assert sr == pytest.approx(1.0)

    def test_regular_50hz_sample_rate(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=100, freq="20ms"))
        _, _, sr = build_time_array(ts)
        assert sr == pytest.approx(50.0)

    def test_start_time_correct(self):
        ts = pd.Series(pd.date_range("2024-06-15 08:30:00", periods=5, freq="1s"))
        _, start, _ = build_time_array(ts)
        assert isinstance(start, datetime)
        assert start.year == 2024 and start.month == 6 and start.day == 15

    def test_nat_interpolated_no_nan(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1s"))
        ts.iloc[2] = pd.NaT
        arr, _, _ = build_time_array(ts)
        assert not np.any(np.isnan(arr))

    def test_nat_interpolated_monotonic(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1s"))
        ts.iloc[2] = pd.NaT
        arr, _, _ = build_time_array(ts)
        assert np.all(np.diff(arr) > 0)

    def test_all_nat_returns_index_array(self):
        ts = pd.Series([pd.NaT] * 5)
        arr, start, sr = build_time_array(ts)
        assert len(arr) == 5
        assert sr == 0.0
        assert start.year == 2000

    def test_start_time_tz_stripped(self):
        ts = pd.Series(
            pd.date_range("2024-01-01", periods=5, freq="1s", tz="UTC")
        )
        _, start, _ = build_time_array(ts)
        assert start.tzinfo is None

    def test_seconds_values_correct(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=4, freq="1s"))
        arr, _, _ = build_time_array(ts)
        np.testing.assert_allclose(arr, [0.0, 1.0, 2.0, 3.0])


# ─────────────────────────────────────────────────────────────────────────────
# infer_nominal_frequency
# ─────────────────────────────────────────────────────────────────────────────

class TestInferNominalFrequency:
    def _make_freq_dataset(self, values: list[float]) -> tuple[list[ParameterMetadata], pd.DataFrame]:
        p = _make_param("frequency", ParameterType.FREQUENCY)
        df = pd.DataFrame({"frequency": values})
        return [p], df

    def test_50hz_detected(self):
        params, df = self._make_freq_dataset([49.9, 50.0, 50.1, 50.0])
        assert infer_nominal_frequency(params, df) == 50.0

    def test_60hz_detected(self):
        params, df = self._make_freq_dataset([59.8, 60.0, 60.1, 60.0])
        assert infer_nominal_frequency(params, df) == 60.0

    def test_no_freq_param_returns_default(self):
        params = [_make_param("mw", ParameterType.MW)]
        df = pd.DataFrame({"mw": [100.0]})
        assert infer_nominal_frequency(params, df, default=50.0) == 50.0

    def test_missing_column_returns_default(self):
        p = _make_param("frequency", ParameterType.FREQUENCY)
        df = pd.DataFrame({"other": [50.0]})
        assert infer_nominal_frequency([p], df, default=50.0) == 50.0

    def test_all_nan_returns_default(self):
        p = _make_param("frequency", ParameterType.FREQUENCY)
        df = pd.DataFrame({"frequency": [float("nan"), float("nan")]})
        assert infer_nominal_frequency([p], df, default=50.0) == 50.0

    def test_custom_default_respected(self):
        params = []
        df = pd.DataFrame()
        assert infer_nominal_frequency(params, df, default=60.0) == 60.0


# ─────────────────────────────────────────────────────────────────────────────
# _coerce_digital
# ─────────────────────────────────────────────────────────────────────────────

class TestCoerceDigital:
    def _coerce(self, values, warnings=None):
        w = warnings if warnings is not None else []
        return _coerce_digital(pd.Series(values), "cb1", w)

    def test_numeric_ones_zeros(self):
        result = self._coerce([0, 1, 0, 1])
        np.testing.assert_array_equal(result, [0, 1, 0, 1])

    def test_bool_series(self):
        result = _coerce_digital(pd.Series([True, False, True]), "cb1", [])
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_string_true_false(self):
        result = self._coerce(["true", "false", "True", "False"])
        np.testing.assert_array_equal(result, [1, 0, 1, 0])

    def test_string_open_close(self):
        result = self._coerce(["close", "open", "closed"])
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_string_trip_normal(self):
        result = self._coerce(["trip", "normal", "tripped"])
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_string_yes_no(self):
        result = self._coerce(["yes", "no"])
        np.testing.assert_array_equal(result, [1, 0])

    def test_string_on_off(self):
        result = self._coerce(["on", "off"])
        np.testing.assert_array_equal(result, [1, 0])

    def test_string_active_inactive(self):
        result = self._coerce(["active", "inactive"])
        np.testing.assert_array_equal(result, [1, 0])

    def test_string_high_low(self):
        result = self._coerce(["high", "low"])
        np.testing.assert_array_equal(result, [1, 0])

    def test_unknown_string_maps_to_zero(self):
        result = self._coerce(["mystery"])
        assert result[0] == 0

    def test_unknown_string_emits_warning(self):
        w: list[str] = []
        self._coerce(["mystery"], warnings=w)
        assert any("unrecognised" in msg for msg in w)

    def test_none_maps_to_zero(self):
        result = self._coerce([None, 1])
        assert result[0] == 0

    def test_output_dtype_int8(self):
        result = self._coerce([0, 1])
        assert result.dtype == np.int8


# ─────────────────────────────────────────────────────────────────────────────
# build_waveform_data
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildWaveformData:
    def _make_time(self, n: int) -> np.ndarray:
        return np.arange(n, dtype=np.float64)

    def test_time_column_first(self):
        t = self._make_time(5)
        df = build_waveform_data(t, [], [], pd.DataFrame(), warnings=[])
        assert list(df.columns)[0] == "time"

    def test_analog_column_present(self):
        n = 5
        t = self._make_time(n)
        p = _make_param("mw")
        data = pd.DataFrame({"mw": range(n)})
        df = build_waveform_data(t, [p], [], data, warnings=[])
        assert "mw" in df.columns

    def test_digital_column_present(self):
        n = 5
        t = self._make_time(n)
        p = _make_param("cb1", ParameterType.DIGITAL)
        data = pd.DataFrame({"cb1": [0, 1, 0, 1, 0]})
        df = build_waveform_data(t, [], [p], data, warnings=[])
        assert "cb1" in df.columns

    def test_missing_analog_fills_nan(self):
        n = 5
        t = self._make_time(n)
        p = _make_param("missing_col")
        df = build_waveform_data(t, [p], [], pd.DataFrame(), warnings=[])
        assert df["missing_col"].isna().all()

    def test_missing_digital_fills_zeros(self):
        n = 5
        t = self._make_time(n)
        p = _make_param("missing_cb", ParameterType.DIGITAL)
        df = build_waveform_data(t, [], [p], pd.DataFrame(), warnings=[])
        assert (df["missing_cb"] == 0).all()

    def test_missing_column_emits_warning(self):
        n = 5
        t = self._make_time(n)
        p = _make_param("ghost")
        w: list[str] = []
        build_waveform_data(t, [p], [], pd.DataFrame(), warnings=w)
        assert any("ghost" in msg for msg in w)

    def test_analog_dtype_float64(self):
        n = 5
        t = self._make_time(n)
        p = _make_param("mw")
        data = pd.DataFrame({"mw": range(n)})
        df = build_waveform_data(t, [p], [], data, warnings=[])
        assert df["mw"].dtype == np.float64

    def test_digital_dtype_int8_or_compatible(self):
        n = 5
        t = self._make_time(n)
        p = _make_param("cb1", ParameterType.DIGITAL)
        data = pd.DataFrame({"cb1": [0, 1, 0, 1, 0]})
        df = build_waveform_data(t, [], [p], data, warnings=[])
        assert df["cb1"].dtype in (np.int8, np.int16, np.int32, np.int64)

    def test_row_count_matches_time_array(self):
        n = 7
        t = self._make_time(n)
        df = build_waveform_data(t, [], [], pd.DataFrame(), warnings=[])
        assert len(df) == n


# ─────────────────────────────────────────────────────────────────────────────
# build_disturbance_record — integration
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildDisturbanceRecord:
    def test_returns_bridge_result(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert isinstance(result, BridgeResult)

    def test_success_true_for_valid_dataset(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert result.success

    def test_record_is_disturbance_record(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert isinstance(result.record, DisturbanceRecord)

    def test_waveform_data_has_time_column(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert "time" in result.record.waveform_data.columns

    def test_analog_channel_in_waveform(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert "mw" in result.record.waveform_data.columns

    def test_digital_channel_in_waveform(self):
        ds = _make_dataset(params=[_make_param("cb1", ParameterType.DIGITAL)])
        result = build_disturbance_record(ds)
        assert "cb1" in result.record.waveform_data.columns

    def test_analog_channel_descriptor_present(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert len(result.record.analog_channels) == 1
        assert result.record.analog_channels[0].name == "mw"

    def test_digital_channel_descriptor_present(self):
        ds = _make_dataset(params=[_make_param("cb1", ParameterType.DIGITAL)])
        result = build_disturbance_record(ds)
        assert len(result.record.digital_channels) == 1
        assert result.record.digital_channels[0].name == "cb1"

    def test_no_validation_errors_on_success(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert result.validation_errors == []

    def test_empty_dataset_produces_failure(self):
        ds = _make_dataset(n=0, params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert not result.success
        assert len(result.validation_errors) > 0

    def test_station_name_from_file_stem(self):
        ds = _make_dataset(source_file_name="event_20240101.csv", params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert result.record.metadata.station_name == "event_20240101"

    def test_station_name_explicit_option(self):
        ds = _make_dataset(source_file_name="event.csv", params=[_make_param("mw")])
        opts = ConversionOptions(station_name="Substation Alpha")
        result = build_disturbance_record(ds, options=opts)
        assert result.record.metadata.station_name == "Substation Alpha"

    def test_provider_type_csv(self):
        ds = _make_dataset(source_file_name="event.csv", params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert result.record.metadata.provider_type == "normalized_csv"

    def test_provider_type_excel(self):
        ds = _make_dataset(source_file_name="event.xlsx", params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert result.record.metadata.provider_type == "normalized_excel"

    def test_provider_type_xlsm(self):
        ds = _make_dataset(source_file_name="event.xlsm", params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert result.record.metadata.provider_type == "normalized_excel"

    def test_provider_type_explicit_override(self):
        ds = _make_dataset(source_file_name="event.csv", params=[_make_param("mw")])
        opts = ConversionOptions(provider_type="custom_provider")
        result = build_disturbance_record(ds, options=opts)
        assert result.record.metadata.provider_type == "custom_provider"

    def test_nominal_frequency_50_default(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert result.record.metadata.nominal_frequency == 50.0

    def test_nominal_frequency_inferred_60hz(self):
        p_freq = _make_param("frequency", ParameterType.FREQUENCY)
        p_mw = _make_param("mw")
        ds = _make_dataset(params=[p_freq, p_mw])
        ds.data["frequency"] = [60.0] * len(ds.data)
        result = build_disturbance_record(ds)
        assert result.record.metadata.nominal_frequency == 60.0

    def test_start_time_in_timing_info(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert isinstance(result.record.timing_info.start_time, datetime)

    def test_sample_rate_set(self):
        ds = _make_dataset(params=[_make_param("mw")], freq_hz="1s")
        result = build_disturbance_record(ds)
        assert result.record.sampling_info.sampling_rates[0] == pytest.approx(1.0)

    def test_time_column_starts_at_zero(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        first_t = result.record.waveform_data["time"].iloc[0]
        assert first_t == pytest.approx(0.0)

    def test_source_not_mutated(self):
        ds = _make_dataset(params=[_make_param("mw")])
        original_shape = ds.data.shape
        build_disturbance_record(ds)
        assert ds.data.shape == original_shape

    def test_unknown_param_emits_warning(self):
        ds = _make_dataset(params=[_make_param("mystery", ParameterType.UNKNOWN)])
        result = build_disturbance_record(ds)
        assert any("UNKNOWN" in w for w in result.warnings)

    def test_timestamp_param_emits_warning(self):
        ds = _make_dataset(params=[_make_param("ts_col", ParameterType.TIMESTAMP)])
        ds.data["ts_col"] = 0.0
        result = build_disturbance_record(ds)
        assert any("TIMESTAMP" in w for w in result.warnings)

    def test_multiple_analog_channels_order_preserved(self):
        params = [
            _make_param("voltage_a", ParameterType.VOLTAGE),
            _make_param("current_a", ParameterType.CURRENT),
            _make_param("mw"),
        ]
        ds = _make_dataset(params=params)
        result = build_disturbance_record(ds)
        names = [ch.name for ch in result.record.analog_channels]
        assert names == ["voltage_a", "current_a", "mw"]

    def test_validate_passes_on_success(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert result.record.validate() == []

    def test_missing_timestamp_column_uses_fallback(self):
        ds = _make_dataset(params=[_make_param("mw")])
        ds.data.rename(columns={"timestamp": "ts_renamed"}, inplace=True)
        result = build_disturbance_record(ds)
        assert any("not found" in w or "Timestamp" in w for w in result.warnings)

    def test_recorder_name_in_metadata(self):
        ds = _make_dataset(params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert result.record.metadata.recorder_name == "Import Wizard"

    def test_custom_recorder_name(self):
        ds = _make_dataset(params=[_make_param("mw")])
        opts = ConversionOptions(recorder_name="TestRecorder")
        result = build_disturbance_record(ds, options=opts)
        assert result.record.metadata.recorder_name == "TestRecorder"

    def test_no_source_file_station_fallback(self):
        ds = _make_dataset(params=[_make_param("mw")], source_file_name=None)
        result = build_disturbance_record(ds)
        assert result.record.metadata.station_name == "Unknown Import"

    def test_row_count_matches_dataset(self):
        n = 42
        ds = _make_dataset(n=n, params=[_make_param("mw")])
        result = build_disturbance_record(ds)
        assert len(result.record.waveform_data) == n


# ─────────────────────────────────────────────────────────────────────────────
# convert_normalized_dataset_to_record
# ─────────────────────────────────────────────────────────────────────────────

class TestConvertNormalizedDatasetToRecord:
    def test_returns_disturbance_record(self):
        ds = _make_dataset(params=[_make_param("mw")])
        rec = convert_normalized_dataset_to_record(ds)
        assert isinstance(rec, DisturbanceRecord)

    def test_raises_on_empty_dataset(self):
        ds = _make_dataset(n=0, params=[_make_param("mw")])
        with pytest.raises(ValueError):
            convert_normalized_dataset_to_record(ds)

    def test_nominal_frequency_passed_through(self):
        ds = _make_dataset(params=[_make_param("mw")])
        rec = convert_normalized_dataset_to_record(ds, nominal_frequency=60.0)
        assert rec.metadata.nominal_frequency == 60.0

    def test_station_name_passed_through(self):
        ds = _make_dataset(params=[_make_param("mw")])
        rec = convert_normalized_dataset_to_record(ds, station_name="Alpha")
        assert rec.metadata.station_name == "Alpha"

    def test_record_passes_validate(self):
        ds = _make_dataset(params=[_make_param("mw")])
        rec = convert_normalized_dataset_to_record(ds)
        assert rec.validate() == []

    def test_channels_populated(self):
        ds = _make_dataset(params=[
            _make_param("mw"),
            _make_param("cb1", ParameterType.DIGITAL),
        ])
        rec = convert_normalized_dataset_to_record(ds)
        assert len(rec.analog_channels) == 1
        assert len(rec.digital_channels) == 1
