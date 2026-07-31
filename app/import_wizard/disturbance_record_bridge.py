"""NormalizedDataset → DisturbanceRecord conversion bridge.

Entry point: build_disturbance_record()

This module is the only layer that consumes a NormalizedDataset and produces
a DisturbanceRecord that is immediately compatible with the Powerwave
visualization and analytics engines.

Design principles
-----------------
- Backend-only (no Qt).
- Always returns a BridgeResult; never raises.
- Never mutates the source NormalizedDataset.
- Full traceability: source file path and repair strategy are embedded in metadata.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.disturbance_record_mapping import (
    build_time_array,
    build_waveform_data,
    infer_nominal_frequency,
    map_to_analog_channel,
    map_to_digital_channel,
    partition_parameters,
)
from app.import_wizard.normalized_dataset import NormalizedDataset
from app.models import (
    DisturbanceRecord,
    RecordingMetadata,
    SamplingInformation,
    TimingInformation,
)

_EPOCH_FALLBACK = datetime(2000, 1, 1, 0, 0, 0)

_EXCEL_SUFFIXES: frozenset[str] = frozenset({".xlsx", ".xls", ".xlsm", ".xlsb"})


def _detect_provider_type(source_file_name: str | None) -> str:
    """Return 'normalized_excel' or 'normalized_csv' based on file extension."""
    if source_file_name:
        ext = os.path.splitext(source_file_name)[1].lower()
        if ext in _EXCEL_SUFFIXES:
            return "normalized_excel"
    return "normalized_csv"


@dataclass(slots=True)
class ConversionOptions:
    """Configuration for NormalizedDataset → DisturbanceRecord conversion.

    Fields
    ------
    nominal_frequency       Default power system frequency in Hz (50 or 60).
                            Used when no FREQUENCY channel is present or when
                            infer_nominal_frequency is False.
    station_name            Station name for RecordingMetadata.  When None,
                            the stem of the source file name is used.
    recorder_name           Recorder or instrument name embedded in metadata.
    provider_type           Provider label embedded in metadata.
    default_unit            Fallback unit string when no unit can be determined.
    infer_nominal_frequency When True, auto-detect 50/60 Hz from the data of
                            any FREQUENCY-typed channel; overrides the value of
                            nominal_frequency when a frequency channel is found.
    """

    nominal_frequency: float = 50.0
    station_name: str | None = None
    recorder_name: str = "Import Wizard"
    provider_type: str | None = None  # None → auto-detect from source file extension
    default_unit: str = "unknown"
    infer_nominal_frequency: bool = True


@dataclass
class BridgeResult:
    """Result of a NormalizedDataset → DisturbanceRecord conversion.

    Fields
    ------
    record            The produced DisturbanceRecord.  Always present; callers
                      must check ``success`` and ``validation_errors`` before
                      treating it as fully valid.
    success           True when record.validate() returns no errors.
    validation_errors Errors from DisturbanceRecord.validate().
    warnings          Non-fatal issues encountered during conversion (missing
                      columns, inferred fallbacks, TIMESTAMP-typed parameters).
    """

    record: DisturbanceRecord
    success: bool = False
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_disturbance_record(
    dataset: NormalizedDataset,
    *,
    options: ConversionOptions | None = None,
) -> BridgeResult:
    """Convert a NormalizedDataset into a DisturbanceRecord.

    Parameters
    ----------
    dataset   The fully assembled NormalizedDataset produced by the Import Wizard
              assembly pipeline.  Never mutated.
    options   Optional conversion settings.  Defaults are used when None.

    Returns
    -------
    BridgeResult — always returns, never raises.  Check ``success`` and
    ``validation_errors`` for any issues.
    """
    opts = options or ConversionOptions()
    warnings_list: list[str] = []

    # ── Guard: empty dataset ──────────────────────────────────────────────────
    if dataset.data.empty:
        warnings_list.append(
            "NormalizedDataset is empty; producing a stub DisturbanceRecord."
        )
        return BridgeResult(
            record=_build_stub_record(dataset, opts),
            success=False,
            validation_errors=["waveform_data must not be empty"],
            warnings=warnings_list,
        )

    # ── Convert ───────────────────────────────────────────────────────────────
    try:
        return _convert(dataset, opts, warnings_list)
    except Exception as exc:
        warnings_list.append(f"Unexpected conversion error: {exc}")
        return BridgeResult(
            record=_build_stub_record(dataset, opts),
            success=False,
            validation_errors=[f"Conversion raised an unexpected error: {exc}"],
            warnings=warnings_list,
        )


def convert_normalized_dataset_to_record(
    dataset: NormalizedDataset,
    *,
    nominal_frequency: float = 50.0,
    station_name: str | None = None,
) -> DisturbanceRecord:
    """High-level convenience wrapper: NormalizedDataset → DisturbanceRecord.

    Raises ValueError when the conversion fails (BridgeResult.success is False).
    Use build_disturbance_record() directly for full BridgeResult details.
    """
    opts = ConversionOptions(
        nominal_frequency=nominal_frequency,
        station_name=station_name,
    )
    result = build_disturbance_record(dataset, options=opts)
    if not result.success:
        raise ValueError(
            "NormalizedDataset → DisturbanceRecord conversion failed: "
            + "; ".join(result.validation_errors)
        )
    return result.record


# ─────────────────────────────────────────────────────────────────────────────
# Internal implementation
# ─────────────────────────────────────────────────────────────────────────────

def _convert(
    dataset: NormalizedDataset,
    opts: ConversionOptions,
    warnings_list: list[str],
) -> BridgeResult:
    """Core conversion — assumes dataset is non-empty."""

    # ── Time array ─────────────────────────────────────────────────────────────
    if (
        dataset.time_axis_mode in {"relative_elapsed", "synthetic_elapsed", "sample_index"}
        and dataset.time_axis_seconds is not None
    ):
        time_array, start_time, sample_rate = _build_elapsed_time_array(
            dataset.time_axis_seconds
        )
    else:
        ts_col = dataset.timestamp_column
        if ts_col in dataset.data.columns:
            ts_series = dataset.data[ts_col]
        else:
            warnings_list.append(
                f"Timestamp column '{ts_col}' not found; using index-based time."
            )
            ts_series = pd.Series(
                pd.date_range(_EPOCH_FALLBACK, periods=len(dataset.data), freq="1s")
            )
        time_array, start_time, sample_rate = build_time_array(ts_series)

    # ── Parameter partitioning ─────────────────────────────────────────────────
    analog_params, digital_params = partition_parameters(dataset.parameters)

    if not analog_params and not digital_params:
        warnings_list.append(
            "No parameters available; waveform_data will contain only the time column."
        )

    # Warn about any TIMESTAMP or UNKNOWN-typed parameters that landed in analog
    for p in analog_params:
        if p.parameter_type == ParameterType.TIMESTAMP:
            warnings_list.append(
                f"Parameter '{p.canonical_name}' has type TIMESTAMP and will be treated "
                "as an analog channel — consider excluding it from the selection."
            )
        elif p.parameter_type == ParameterType.UNKNOWN:
            warnings_list.append(
                f"Parameter '{p.canonical_name}' has type UNKNOWN and will be treated "
                "as an analog channel with unit 'unknown'."
            )

    # ── Nominal frequency ──────────────────────────────────────────────────────
    nominal_freq = opts.nominal_frequency
    if opts.infer_nominal_frequency:
        nominal_freq = infer_nominal_frequency(
            analog_params, dataset.data, default=opts.nominal_frequency
        )

    # ── waveform_data ──────────────────────────────────────────────────────────
    waveform_data = build_waveform_data(
        time_array=time_array,
        analog_params=analog_params,
        digital_params=digital_params,
        data=dataset.data,
        warnings=warnings_list,
    )

    # ── Channel descriptors ────────────────────────────────────────────────────
    analog_channels = [
        map_to_analog_channel(pm, i, default_unit=opts.default_unit)
        for i, pm in enumerate(analog_params)
    ]
    digital_channels = [
        map_to_digital_channel(pm, i)
        for i, pm in enumerate(digital_params)
    ]

    # ── Metadata ───────────────────────────────────────────────────────────────
    station_name = _resolve_station_name(dataset, opts)

    metadata = RecordingMetadata(
        station_name=station_name,
        recorder_name=opts.recorder_name,
        source_file=dataset.source_path or dataset.source_file_name or "",
        provider_type=_resolve_provider_type(dataset, opts),
        nominal_frequency=nominal_freq,
    )

    # ── Timing ─────────────────────────────────────────────────────────────────
    timing_info = TimingInformation(
        start_time=start_time,
        trigger_time=start_time,  # no trigger info in CSV/Excel imports
        timing_reference=dataset.time_axis_mode,
        time_axis_unit=dataset.time_axis_unit,
    )

    # ── Sampling ───────────────────────────────────────────────────────────────
    n_samples = len(waveform_data)
    sampling_info = SamplingInformation(
        sampling_rates=[max(sample_rate, 0.0)],
        samples_per_rate=[n_samples],
    )

    # ── Assemble & validate ────────────────────────────────────────────────────
    record = DisturbanceRecord(
        metadata=metadata,
        waveform_data=waveform_data,
        analog_channels=analog_channels,
        digital_channels=digital_channels,
        sampling_info=sampling_info,
        timing_info=timing_info,
        disturbance_info=None,
    )

    validation_errors = record.validate()

    return BridgeResult(
        record=record,
        success=len(validation_errors) == 0,
        validation_errors=validation_errors,
        warnings=warnings_list,
    )


def _build_elapsed_time_array(
    seconds_series: pd.Series,
) -> tuple[np.ndarray, datetime, float]:
    """Build DisturbanceRecord timing from an authoritative elapsed-seconds axis."""
    numeric: pd.Series = pd.to_numeric(seconds_series, errors="coerce")  # type: ignore[assignment]
    numeric = numeric.interpolate(method="linear", limit_direction="both")
    time_array = numeric.to_numpy(dtype=np.float64)

    valid = time_array[np.isfinite(time_array)]
    if len(valid) < 2:
        return time_array, _EPOCH_FALLBACK, 0.0

    diffs = np.diff(valid)
    pos_diffs = diffs[diffs > 0]
    if len(pos_diffs) == 0:
        sample_rate = 0.0
    else:
        median_dt = float(np.median(pos_diffs))
        sample_rate = round(1.0 / median_dt, 6) if median_dt > 0 else 0.0
    return time_array, _EPOCH_FALLBACK, sample_rate


def _resolve_station_name(
    dataset: NormalizedDataset,
    opts: ConversionOptions,
) -> str:
    """Determine the station name for RecordingMetadata.

    Priority: explicit option → file stem (no extension) → file name → fallback.
    """
    if opts.station_name is not None:
        return opts.station_name
    if dataset.source_file_name:
        stem, _ = os.path.splitext(dataset.source_file_name)
        return stem or dataset.source_file_name
    return "Unknown Import"


def _resolve_provider_type(
    dataset: NormalizedDataset,
    opts: ConversionOptions,
) -> str:
    """Return a concrete provider_type string, never None."""
    if opts.provider_type is not None:
        return opts.provider_type
    return _detect_provider_type(dataset.source_file_name)


def _build_stub_record(
    dataset: NormalizedDataset,
    opts: ConversionOptions,
) -> DisturbanceRecord:
    """Return a minimal-structure stub DisturbanceRecord for error paths.

    The stub is structurally sound (no attribute errors) but will fail
    DisturbanceRecord.validate() — callers are expected to check success=False.
    """
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name=_resolve_station_name(dataset, opts),
            recorder_name=opts.recorder_name,
            source_file=dataset.source_path or dataset.source_file_name or "",
            provider_type=_resolve_provider_type(dataset, opts),
            nominal_frequency=opts.nominal_frequency,
        ),
        waveform_data=pd.DataFrame({"time": pd.Series([], dtype="float64")}),
        analog_channels=[],
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[0.0],
            samples_per_rate=[0],
        ),
        timing_info=TimingInformation(
            start_time=_EPOCH_FALLBACK,
            trigger_time=_EPOCH_FALLBACK,
        ),
        disturbance_info=None,
    )
