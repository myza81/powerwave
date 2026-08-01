from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.models import (
    AnalogChannel,
    DigitalChannel,
    DisturbanceRecord,
    RecordingMetadata,
    SamplingInformation,
    TimingInformation,
)
from app.providers.base.base_provider import BaseProvider
from app.providers.base.exceptions import ProviderLoadError
from app.data.timestamp_disambiguation import parse_ambiguous_dates
from app.data.column_classifier import (
    classify_csv_column,
    numeric_column_disposition,
    DISPOSITION_ANALOG,
    DISPOSITION_DIGITAL,
    DISPOSITION_IGNORED,
    DISPOSITION_REVIEW,
)
from app.data.intelligence import IntelligenceManager

_SUPPORTED_SUFFIXES = {".csv"}

_TIME_COL_NAMES = {"time", "t", "seconds", "sec", "timestamp", "datetime"}

_DIGITAL_NAME_KEYWORDS = {
    "trip", "pickup", "breaker", "status", "cb", "relay",
    "alarm", "open", "close", "signal", "flag", "state",
}

_EPOCH_FALLBACK = datetime(2000, 1, 1, 0, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Column classification helpers
# ─────────────────────────────────────────────────────────────────────────────


def _detect_time_column(columns: list[str]) -> str | None:
    """Return the first column whose name matches a recognised time-column label."""
    for col in columns:
        if _is_time_like_column(col):
            return col
    return None


def _is_time_like_column(col_name: str) -> bool:
    """Return True for timestamp columns, including pandas duplicate names.

    pandas.read_csv disambiguates duplicate headers as ``Time.1``, ``Time.2``,
    etc. Those columns are timestamp artifacts, not waveform channels.
    """
    name = col_name.lower().strip()
    base, suffix = name.rsplit(".", 1) if "." in name else (name, "")
    return name in _TIME_COL_NAMES or (suffix.isdigit() and base in _TIME_COL_NAMES)


def _infer_unit(col_name: str) -> str:
    """Infer a best-guess engineering unit from a column name."""
    n = col_name.lower()
    if any(kw in n for kw in ("volt", "kv")) or n in (
        "v", "va", "vb", "vc", "vr", "vy", "vab", "vbc", "vca", "vn",
    ):
        return "kV"
    if any(kw in n for kw in ("curr", "amp")) or n in (
        "ia", "ib", "ic", "ir", "iy", "in",
    ):
        return "A"
    if any(kw in n for kw in ("freq", "hz")):
        return "Hz"
    if "mvar" in n:
        return "MVar"
    if "mw" in n:
        return "MW"
    return "unknown"


def _is_digital_column(series: pd.Series, col_name: str) -> bool:
    """Return True if the column should be classified as a digital (binary) channel.

    Conservative: values must be exclusively 0/1 AND the column name must
    contain a recognised status keyword. Boolean-typed columns are always digital.
    """
    if pd.api.types.is_bool_dtype(series):
        return True
    valid = series.dropna()
    if valid.empty:
        return False
    try:
        numeric_vals: pd.Series = valid.astype(float)
    except (ValueError, TypeError):
        return False
    unique_vals = set(numeric_vals.unique())
    if not unique_vals.issubset({0.0, 1.0}):
        return False
    name_lower = col_name.lower()
    return any(kw in name_lower for kw in _DIGITAL_NAME_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# Time array construction
# ─────────────────────────────────────────────────────────────────────────────


def _estimate_rate(time_array: np.ndarray) -> float:
    """Estimate sampling rate (Hz) from the median inter-sample interval."""
    if len(time_array) < 2:
        return 0.0
    diffs = np.diff(time_array)
    median_dt = float(np.median(diffs))
    if median_dt <= 0.0:
        return 0.0
    return round(1.0 / median_dt, 6)


def _build_time_array(
    df: pd.DataFrame,
    time_col: str | None,
    path: Path,
) -> tuple[np.ndarray, datetime, float]:
    """Build the time array (seconds from start), start_time, and sample rate.

    Returns:
        time_array:  float64 ndarray, seconds elapsed from first sample
        start_time:  tz-naive datetime for TimingInformation
        sample_rate: Hz, 0.0 when not determinable
    """
    if time_col is None:
        n = len(df)
        return np.arange(n, dtype=np.float64), _EPOCH_FALLBACK, 0.0

    col: pd.Series = df[time_col]

    # Try datetime parsing when dtype is non-numeric
    if not pd.api.types.is_numeric_dtype(col):
        try:
            # Ambiguous D/M-vs-M/D dates default to day-first (Powerwave policy);
            # unambiguous values are parsed correctly regardless. CsvProvider has
            # no user-override mechanism, so this is always the automatic default.
            dt_series, ambiguity = parse_ambiguous_dates(col)  # type: ignore[assignment]
            if ambiguity.is_ambiguous:
                warnings.warn(
                    f"Column '{time_col}' in '{path}' contains an ambiguous date "
                    f"(e.g. '{ambiguity.sample_value}'); Powerwave assumed "
                    "DD/MM/YYYY by default."
                )
            t0_stamp = dt_series.iloc[0]
            time_array = (dt_series - t0_stamp).dt.total_seconds().to_numpy(
                dtype=np.float64
            )
            start_time = t0_stamp.to_pydatetime().replace(tzinfo=None)
            return time_array, start_time, _estimate_rate(time_array)
        except Exception:
            pass  # fall through to numeric handling

    # Numeric time column — treat as seconds
    try:
        num_series: pd.Series = col.astype(float)
    except (ValueError, TypeError) as exc:
        raise ProviderLoadError(
            f"Cannot parse time column '{time_col}' in '{path}' as numeric or datetime"
        ) from exc

    time_array = num_series.to_numpy(dtype=np.float64)
    return time_array, _EPOCH_FALLBACK, _estimate_rate(time_array)


# ─────────────────────────────────────────────────────────────────────────────
# Provider
# ─────────────────────────────────────────────────────────────────────────────


class CsvProvider(BaseProvider):
    """Generic CSV waveform ingestion provider.

    Accepts .csv files and normalises their contents into a DisturbanceRecord.

    Supported structures:
      time, VA, VB, VC, IA, IB, IC, FREQ
      timestamp, VA, VB, VC, IA, IB, IC
      VA, VB, VC, IA, IB, IC   (no time column)

    Columns are classified as analog (numeric) or digital (binary 0/1 with a
    status-type name) using robust disposition heuristics. Units are inferred
    from column names or intelligence matching.
    """

    provider_name: str = "csv"

    def __init__(self, intelligence_manager: IntelligenceManager | None = None) -> None:
        """Initialize with an optional IntelligenceManager for D4.3 rules."""
        self._mgr = intelligence_manager or IntelligenceManager()

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in _SUPPORTED_SUFFIXES

    def load(self, path: Path) -> DisturbanceRecord:
        """Parse *path* as a CSV file and return a normalised DisturbanceRecord."""
        if not path.exists():
            raise ProviderLoadError(f"CSV file not found: '{path}'")

        try:
            df: pd.DataFrame = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise ProviderLoadError(f"CSV file '{path}' is empty") from exc
        except Exception as exc:
            raise ProviderLoadError(
                f"Cannot parse CSV file '{path}': {exc}"
            ) from exc

        if df.empty:
            raise ProviderLoadError(f"CSV file '{path}' contains no data rows")

        time_candidates = [c for c in df.columns if _is_time_like_column(c)]
        time_col = None
        ignored_cols = set()

        if len(time_candidates) == 1:
            time_col = time_candidates[0]
        elif len(time_candidates) > 1:
            # Phase D4.3: resolve duplicate time columns
            candidates = self._mgr.score_timestamp_candidates(
                time_candidates, df, source_pattern=self.provider_name
            )
            if candidates:
                time_col = candidates[0].column_name
                for c in candidates[1:]:
                    ignored_cols.add(c.column_name)

        if time_col is not None:
            try:
                time_array, start_time, sample_rate = _build_time_array(df, time_col, path)
            except ProviderLoadError:
                raise
            except Exception as exc:
                raise ProviderLoadError(
                    f"Failed to build time array from '{path}': {exc}"
                ) from exc
        else:
            time_array, start_time, sample_rate = _build_time_array(df, None, path)

        waveform_cols = [c for c in df.columns if c != time_col]

        analog_channels: list[AnalogChannel] = []
        digital_channels: list[DigitalChannel] = []
        col_data: dict[str, np.ndarray] = {"time": time_array}
        analog_idx = 0
        digital_idx = 0

        for col_name in waveform_cols:
            series: pd.Series = df[col_name]
            
            try:
                numeric = pd.to_numeric(series, errors="coerce")
                valid_vals = numeric.dropna().tolist()
            except (ValueError, TypeError):
                valid_vals = []
                numeric = series  # Will be all NaN if not numeric

            # 1. Base classification using IntelligenceManager (synonyms, keywords)
            cls, _ = self._mgr.classify_column(col_name, valid_vals)
            
            # 2. Determine disposition
            is_dig = _is_digital_column(series, col_name)
            ignore_reason = "duplicate_timestamp_artifact" if col_name in ignored_cols else None
            
            disp, reason = numeric_column_disposition(
                cls, is_digital=is_dig, ignore_reason=ignore_reason
            )

            # 3. Apply disposition
            if disp == DISPOSITION_IGNORED:
                continue

            if disp == DISPOSITION_DIGITAL:
                vals = series.fillna(0).astype(float).astype(np.int8).to_numpy()
                digital_channels.append(
                    DigitalChannel(name=col_name, index=digital_idx, normal_state=0)
                )
                col_data[col_name] = vals
                digital_idx += 1
            else:
                # Analog or Review (both handled as analog channels)
                if numeric.isna().all():
                    warnings.warn(
                        f"Column '{col_name}' in '{path}' cannot be parsed as "
                        "numeric — skipping"
                    )
                    continue
                vals = numeric.to_numpy(dtype=np.float64)

                # An unconfirmed classification (confidence below the shared
                # threshold) must not populate the engineering unit either --
                # the same confidence discipline already applied to
                # parameter_type below. Fall back to name-only inference (or
                # its own "unknown" default) instead of a low-confidence
                # value-profile-derived unit.
                unit = cls.unit if (cls.unit and not cls.requires_user_confirmation) else _infer_unit(col_name)

                # Populate parameter_type only when the shared classifier's
                # result is confident enough not to require user confirmation
                # (existing CONFIRMATION_THRESHOLD, not a new duplicate one).
                parameter_type = cls.signal_type if not cls.requires_user_confirmation else None

                analog_channels.append(
                    AnalogChannel(
                        name=col_name,
                        unit=unit,
                        index=analog_idx,
                        parameter_type=parameter_type,
                    )
                )
                col_data[col_name] = vals
                analog_idx += 1

        if not analog_channels and not digital_channels:
            raise ProviderLoadError(
                f"CSV file '{path}' contains no usable waveform columns"
            )

        waveform_data = pd.DataFrame(col_data)
        total_samples = len(waveform_data)

        return DisturbanceRecord(
            metadata=RecordingMetadata(
                station_name="Unknown",
                recorder_name="CSV",
                source_file=str(path),
                provider_type="csv",
                nominal_frequency=50.0,
            ),
            waveform_data=waveform_data,
            analog_channels=analog_channels,
            digital_channels=digital_channels,
            sampling_info=SamplingInformation(
                sampling_rates=[max(sample_rate, 0.0)],
                samples_per_rate=[total_samples],
            ),
            timing_info=TimingInformation(
                start_time=start_time,
                trigger_time=start_time,
            ),
            disturbance_info=None,
        )
