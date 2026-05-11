from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.models.channels import AnalogChannel, DigitalChannel
from app.models.metadata import RecordingMetadata
from app.models.timing import DisturbanceInformation, SamplingInformation, TimingInformation


@dataclass(slots=True)
class DisturbanceRecord:
    """Unified internal waveform contract for Powerwave.

    This is the single normalized representation consumed by all downstream
    systems: analytics, visualization, synchronization, and export.

    All providers (COMTRADE, CSV, Excel, future formats) must produce a
    DisturbanceRecord. No provider-specific structures may leak outside the
    provider layer.

    waveform_data is stored by reference — never copied on construction.
    """

    metadata: RecordingMetadata
    waveform_data: pd.DataFrame
    analog_channels: list[AnalogChannel]
    digital_channels: list[DigitalChannel]
    sampling_info: SamplingInformation
    timing_info: TimingInformation
    disturbance_info: DisturbanceInformation | None = None

    # ------------------------------------------------------------------ #
    # Channel accessors                                                    #
    # ------------------------------------------------------------------ #

    def analog_channel_names(self) -> list[str]:
        """Return names of all declared analog channels in declaration order."""
        return [ch.name for ch in self.analog_channels]

    def digital_channel_names(self) -> list[str]:
        """Return names of all declared digital channels in declaration order."""
        return [ch.name for ch in self.digital_channels]

    def channel_names(self) -> list[str]:
        """Return all channel names: analog first, then digital."""
        return self.analog_channel_names() + self.digital_channel_names()

    def has_channel(self, name: str) -> bool:
        """Return True if name is declared as an analog or digital channel."""
        return (
            any(ch.name == name for ch in self.analog_channels)
            or any(ch.name == name for ch in self.digital_channels)
        )

    # ------------------------------------------------------------------ #
    # Waveform metrics                                                     #
    # ------------------------------------------------------------------ #

    def sample_count(self) -> int:
        """Return the number of rows in waveform_data."""
        return len(self.waveform_data)

    def duration_seconds(self) -> float:
        """Return the recording duration in seconds.

        Priority:
        1. 'time' column in waveform_data (last - first, in seconds).
        2. Computed from sampling_info (sum of samples / rate per section).
        3. Returns 0.0 if neither source is available.
        """
        if self.waveform_data.empty:
            return 0.0

        if "time" in self.waveform_data.columns:
            return float(self.waveform_data["time"].iloc[-1]) - float(
                self.waveform_data["time"].iloc[0]
            )

        rates = self.sampling_info.sampling_rates
        counts = self.sampling_info.samples_per_rate
        if rates and all(r > 0 for r in rates):
            return sum(n / r for n, r in zip(counts, rates))

        return 0.0

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate(self) -> list[str]:
        """Run lightweight consistency checks and return a list of error strings.

        An empty list means the record is valid. Validation never raises;
        callers decide how to handle reported errors.
        """
        errors: list[str] = []

        # --- waveform_data type and emptiness ---
        if not isinstance(self.waveform_data, pd.DataFrame):
            errors.append("waveform_data must be a pandas DataFrame")
            return errors  # cannot safely continue without a DataFrame

        if self.waveform_data.empty:
            errors.append("waveform_data must not be empty")

        # --- channel presence (only meaningful when data exists) ---
        if not self.waveform_data.empty:
            df_cols = set(self.waveform_data.columns)
            for ch in self.analog_channels:
                if ch.name not in df_cols:
                    errors.append(
                        f"analog channel '{ch.name}' not found in waveform_data columns"
                    )
            for ch in self.digital_channels:
                if ch.name not in df_cols:
                    errors.append(
                        f"digital channel '{ch.name}' not found in waveform_data columns"
                    )

        # --- sampling_info consistency ---
        if not self.sampling_info.sampling_rates:
            errors.append("sampling_info: sampling_rates must not be empty")
        elif len(self.sampling_info.sampling_rates) != len(
            self.sampling_info.samples_per_rate
        ):
            errors.append(
                "sampling_info: sampling_rates and samples_per_rate must have equal length"
            )

        # --- timing_info ordering ---
        if self.timing_info.trigger_time < self.timing_info.start_time:
            errors.append(
                "timing_info: trigger_time cannot be before start_time"
            )

        return errors
