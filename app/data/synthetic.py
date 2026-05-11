"""Synthetic disturbance data generators for testing and development.

Produces DisturbanceRecord instances with realistic fault profiles.
Three generators are provided:
  make_high_rate_record   — raw three-phase voltage/current at e.g. 6400 Hz
  make_low_rate_record    — RMS / system channels at e.g. 100 Hz
  make_mixed_disturbance_record — merged single record with both groups

Mixed records interpolate low-rate channels onto the high-rate time axis.
Original sampling metadata is preserved in the returned SignalMetadata dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.data.signal_metadata import SignalMetadata
from app.models import (
    AnalogChannel,
    DisturbanceInformation,
    DisturbanceRecord,
    RecordingMetadata,
    SamplingInformation,
    TimingInformation,
)

_BASE_START = datetime(2024, 1, 1, 0, 0, 0)

_DG_VOLTAGE_RAW = "voltage_raw"
_DG_CURRENT_RAW = "current_raw"
_DG_POWER = "power"
_DG_FREQUENCY = "frequency"
_DG_ROCOF = "rocof"


@dataclass(frozen=True)
class SyntheticDisturbanceResult:
    """Container returned by all synthetic generators."""

    record: DisturbanceRecord
    signal_metadata: dict[str, SignalMetadata]


# ─────────────────────────────────────────────────────────────────────────────
# High-rate generator (raw waveform channels)
# ─────────────────────────────────────────────────────────────────────────────


def make_high_rate_record(
    duration_s: float = 5.0,
    sampling_rate_hz: float = 6400.0,
    fault_start_s: float = 1.0,
    fault_end_s: float = 1.5,
    nominal_freq_hz: float = 50.0,
    nominal_voltage_pu: float = 1.0,
    fault_voltage_pu: float = 0.3,
    include_current: bool = True,
) -> SyntheticDisturbanceResult:
    """Generate a high-rate DisturbanceRecord with raw three-phase waveforms.

    Fault profile: step voltage sag to fault_voltage_pu during [fault_start_s, fault_end_s).
    Channels: VA_raw, VB_raw, VC_raw (always); IA_raw, IB_raw, IC_raw (if include_current).
    """
    t = np.arange(0.0, duration_s, 1.0 / sampling_rate_hz, dtype=np.float64)
    n = len(t)
    omega = 2.0 * np.pi * nominal_freq_hz
    phi_load = np.pi / 6  # 30° lag → pf ≈ 0.866

    fault_mask = (t >= fault_start_s) & (t < fault_end_s)
    v_factor = np.where(fault_mask, fault_voltage_pu, nominal_voltage_pu)
    i_factor = np.where(fault_mask, 3.0, 0.5)  # fault / load current in pu

    data: dict[str, np.ndarray] = {"time": t}
    channels: list[AnalogChannel] = []
    meta: dict[str, SignalMetadata] = {}
    idx = 0

    phase_offsets = [("A", 0.0), ("B", -2.0 * np.pi / 3.0), ("C", 2.0 * np.pi / 3.0)]

    for phase, angle in phase_offsets:
        name = f"V{phase}_raw"
        data[name] = v_factor * np.cos(omega * t + angle)
        channels.append(AnalogChannel(name=name, unit="pu", index=idx, phase=phase))
        meta[name] = SignalMetadata(
            name=name,
            unit="pu",
            source="synthetic_high_rate",
            signal_type="voltage_raw",
            sampling_rate_hz=sampling_rate_hz,
            display_group=_DG_VOLTAGE_RAW,
            electrical_type="voltage",
            phase_reference="phase_ground",   # VA = A-G, VB = B-G, VC = C-G
        )
        idx += 1

    if include_current:
        for phase, angle in phase_offsets:
            name = f"I{phase}_raw"
            data[name] = i_factor * np.cos(omega * t + angle - phi_load)
            channels.append(AnalogChannel(name=name, unit="pu", index=idx, phase=phase))
            meta[name] = SignalMetadata(
                name=name,
                unit="pu",
                source="synthetic_high_rate",
                signal_type="current_raw",
                sampling_rate_hz=sampling_rate_hz,
                display_group=_DG_CURRENT_RAW,
                electrical_type="current",
            )
            idx += 1

    waveform_df = pd.DataFrame(data)
    trigger_time = _BASE_START + timedelta(seconds=fault_start_s)

    record = DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="Synthetic",
            recorder_name="synthetic",
            source_file="synthetic_high_rate.cfg",
            provider_type="synthetic",
            nominal_frequency=nominal_freq_hz,
        ),
        waveform_data=waveform_df,
        analog_channels=channels,
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[sampling_rate_hz],
            samples_per_rate=[n],
            nominal_frequency=nominal_freq_hz,
        ),
        timing_info=TimingInformation(
            start_time=_BASE_START,
            trigger_time=trigger_time,
        ),
        disturbance_info=DisturbanceInformation(
            event_type="THREE_PHASE_FAULT",
            tags=["synthetic", "voltage_sag"],
        ),
    )
    return SyntheticDisturbanceResult(record=record, signal_metadata=meta)


# ─────────────────────────────────────────────────────────────────────────────
# Low-rate generator (RMS / system channels)
# ─────────────────────────────────────────────────────────────────────────────


def make_low_rate_record(
    duration_s: float = 5.0,
    sampling_rate_hz: float = 100.0,
    fault_start_s: float = 1.0,
    fault_end_s: float = 1.5,
    nominal_freq_hz: float = 50.0,
    time_offset_s: float = 0.01,
    include_rocof: bool = False,
) -> SyntheticDisturbanceResult:
    """Generate a low-rate DisturbanceRecord with RMS/system channels.

    Fault profile: MW dip, MVar spike, frequency dip during [fault_start_s, fault_end_s).
    time_offset_s introduces a deliberate timestamp offset vs. the high-rate record.
    """
    t = np.arange(time_offset_s, duration_s, 1.0 / sampling_rate_hz, dtype=np.float64)
    n = len(t)

    fault_mask = (t >= fault_start_s) & (t < fault_end_s)

    mw = np.where(fault_mask, 0.1, 1.0)
    mvar = np.where(fault_mask, 0.5, 0.0)
    freq = np.where(fault_mask, 49.8, 50.0)

    data: dict[str, np.ndarray] = {"time": t}
    channels: list[AnalogChannel] = []
    meta: dict[str, SignalMetadata] = {}
    idx = 0

    channel_specs = [
        ("MW", mw, "MW", _DG_POWER, "power"),
        ("MVar", mvar, "MVar", _DG_POWER, "power"),
        ("Frequency", freq, "Hz", _DG_FREQUENCY, "frequency"),
    ]
    for name, values, unit, group, etype in channel_specs:
        data[name] = values
        channels.append(AnalogChannel(name=name, unit=unit, index=idx))
        meta[name] = SignalMetadata(
            name=name,
            unit=unit,
            source="synthetic_low_rate",
            signal_type=group,
            sampling_rate_hz=sampling_rate_hz,
            time_offset_seconds=time_offset_s,
            display_group=group,
            electrical_type=etype,
        )
        idx += 1

    if include_rocof:
        rocof = np.gradient(freq, t)
        data["ROCOF"] = rocof
        channels.append(AnalogChannel(name="ROCOF", unit="Hz/s", index=idx))
        meta["ROCOF"] = SignalMetadata(
            name="ROCOF",
            unit="Hz/s",
            source="synthetic_low_rate",
            signal_type="rocof",
            sampling_rate_hz=sampling_rate_hz,
            time_offset_seconds=time_offset_s,
            display_group=_DG_ROCOF,
            electrical_type="rocof",
        )

    waveform_df = pd.DataFrame(data)
    trigger_time = _BASE_START + timedelta(seconds=fault_start_s)

    record = DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="Synthetic",
            recorder_name="synthetic",
            source_file="synthetic_low_rate.csv",
            provider_type="synthetic",
            nominal_frequency=nominal_freq_hz,
        ),
        waveform_data=waveform_df,
        analog_channels=channels,
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[sampling_rate_hz],
            samples_per_rate=[n],
            nominal_frequency=nominal_freq_hz,
        ),
        timing_info=TimingInformation(
            start_time=_BASE_START,
            trigger_time=trigger_time,
        ),
    )
    return SyntheticDisturbanceResult(record=record, signal_metadata=meta)


# ─────────────────────────────────────────────────────────────────────────────
# Mixed generator (merged single record)
# ─────────────────────────────────────────────────────────────────────────────


def make_mixed_disturbance_record(
    duration_s: float = 5.0,
    high_rate_hz: float = 6400.0,
    low_rate_hz: float = 100.0,
    fault_start_s: float = 1.0,
    fault_end_s: float = 1.5,
    nominal_freq_hz: float = 50.0,
    time_offset_s: float = 0.01,
) -> SyntheticDisturbanceResult:
    """Generate a single merged DisturbanceRecord with both high-rate and low-rate channels.

    Low-rate channels are interpolated onto the high-rate time axis using np.interp
    so that the merged waveform_data has a single consistent time column.

    Original sampling rates and the low-rate time offset are preserved in the
    returned SignalMetadata dict and in sampling_info.sampling_rates.
    """
    high = make_high_rate_record(
        duration_s=duration_s,
        sampling_rate_hz=high_rate_hz,
        fault_start_s=fault_start_s,
        fault_end_s=fault_end_s,
        nominal_freq_hz=nominal_freq_hz,
    )
    low = make_low_rate_record(
        duration_s=duration_s,
        sampling_rate_hz=low_rate_hz,
        fault_start_s=fault_start_s,
        fault_end_s=fault_end_s,
        nominal_freq_hz=nominal_freq_hz,
        time_offset_s=time_offset_s,
    )

    t_high = high.record.waveform_data["time"].to_numpy(dtype=np.float64)
    t_low = low.record.waveform_data["time"].to_numpy(dtype=np.float64)
    low_df = low.record.waveform_data

    merged_data = high.record.waveform_data.copy()
    for col in low_df.columns:
        if col == "time":
            continue
        low_values = low_df[col].to_numpy(dtype=np.float64)
        # np.interp clamps/extrapolates at boundaries (acceptable for display fixture)
        merged_data[col] = np.interp(t_high, t_low, low_values)

    # Re-index merged channel list (indices must be contiguous)
    raw_channels = list(high.record.analog_channels) + list(low.record.analog_channels)
    merged_channels = [
        AnalogChannel(
            name=ch.name,
            unit=ch.unit,
            index=i,
            phase=ch.phase,
            description=ch.description,
            scale=ch.scale,
            offset=ch.offset,
            primary_ratio=ch.primary_ratio,
            secondary_ratio=ch.secondary_ratio,
        )
        for i, ch in enumerate(raw_channels)
    ]

    merged_meta = {**high.signal_metadata, **low.signal_metadata}
    n_high = len(t_high)
    n_low = len(t_low)
    trigger_time = _BASE_START + timedelta(seconds=fault_start_s)

    record = DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="Synthetic Mixed",
            recorder_name="synthetic",
            source_file="synthetic_mixed.cfg",
            provider_type="synthetic",
            nominal_frequency=nominal_freq_hz,
        ),
        waveform_data=merged_data,
        analog_channels=merged_channels,
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[high_rate_hz, low_rate_hz],
            samples_per_rate=[n_high, n_low],
            nominal_frequency=nominal_freq_hz,
        ),
        timing_info=TimingInformation(
            start_time=_BASE_START,
            trigger_time=trigger_time,
        ),
        disturbance_info=DisturbanceInformation(
            event_type="THREE_PHASE_FAULT",
            tags=["synthetic", "mixed_rate", "voltage_sag"],
        ),
    )
    return SyntheticDisturbanceResult(record=record, signal_metadata=merged_meta)
