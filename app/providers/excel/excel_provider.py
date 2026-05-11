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

_SUPPORTED_SUFFIXES = {".xlsx", ".xls"}

_TIME_COL_NAMES = {"time", "t", "seconds", "sec", "timestamp", "datetime"}

_DIGITAL_NAME_KEYWORDS = {
    "trip", "pickup", "breaker", "status", "cb", "relay",
    "alarm", "open", "close", "signal", "flag", "state",
}

_EPOCH_FALLBACK = datetime(2000, 1, 1, 0, 0, 0)

_MIN_NUMERIC_ROWS = 2  # sheet must have at least this many data rows to be usable


# ─────────────────────────────────────────────────────────────────────────────
# Column classification helpers (mirrors CsvProvider)
# ─────────────────────────────────────────────────────────────────────────────


def _detect_time_column(columns: list[str]) -> str | None:
    """Return the first column whose name matches a recognised time-column label."""
    for col in columns:
        if col.lower().strip() in _TIME_COL_NAMES:
            return col
    return None


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
# Time array construction (mirrors CsvProvider)
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
            dt_series: pd.Series = pd.to_datetime(col, utc=False)  # type: ignore[assignment]
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
# Sheet selection
# ─────────────────────────────────────────────────────────────────────────────


def _score_sheet(df: pd.DataFrame) -> int:
    """Score a sheet by (rows × numeric-like columns) — higher is more data-rich."""
    if df.empty or len(df) < _MIN_NUMERIC_ROWS:
        return 0
    numeric_cols = 0
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols += 1
        else:
            coerced: pd.Series = pd.to_numeric(s.dropna(), errors="coerce")  # type: ignore[assignment]
            if not coerced.empty and bool(coerced.notna().any()):
                numeric_cols += 1
    return len(df) * numeric_cols


def _select_sheet(path: Path) -> str:
    """Return the name of the most data-rich sheet in an xlsx workbook.

    Falls back to the first sheet when no sheet has usable data.
    """
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception as exc:
        raise ProviderLoadError(f"Cannot open Excel file '{path}': {exc}") from exc

    sheet_names: list[str] = [str(s) for s in xl.sheet_names]
    if not sheet_names:
        raise ProviderLoadError(f"Excel file '{path}' contains no sheets")

    if len(sheet_names) == 1:
        return sheet_names[0]

    best_name = sheet_names[0]
    best_score = -1
    for name in sheet_names:
        try:
            df: pd.DataFrame = xl.parse(name, nrows=200)
            score = _score_sheet(df)
            if score > best_score:
                best_score = score
                best_name = name
        except Exception:
            continue

    return best_name


# ─────────────────────────────────────────────────────────────────────────────
# Provider
# ─────────────────────────────────────────────────────────────────────────────


class ExcelProvider(BaseProvider):
    """Excel waveform ingestion provider.

    Accepts .xlsx (fully supported via openpyxl) and .xls (requires xlrd).

    The most data-rich sheet is selected automatically when the workbook
    contains multiple sheets. Columns are classified as analog or digital
    using the same conservative heuristics as CsvProvider.
    """

    provider_name: str = "excel"

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in _SUPPORTED_SUFFIXES

    def load(self, path: Path) -> DisturbanceRecord:
        """Parse *path* as an Excel file and return a normalised DisturbanceRecord."""
        if not path.exists():
            raise ProviderLoadError(f"Excel file not found: '{path}'")

        suffix = path.suffix.lower()

        if suffix == ".xls":
            raise ProviderLoadError(
                f"Cannot load '{path}': legacy .xls format requires the 'xlrd' package, "
                "which is not installed. Install xlrd>=1.2 or convert the file to .xlsx."
            )

        # .xlsx — use openpyxl engine
        sheet_name = _select_sheet(path)

        try:
            df: pd.DataFrame = pd.read_excel(
                path, sheet_name=sheet_name, engine="openpyxl"
            )
        except Exception as exc:
            raise ProviderLoadError(
                f"Cannot read sheet '{sheet_name}' from '{path}': {exc}"
            ) from exc

        if df.empty:
            raise ProviderLoadError(
                f"Excel file '{path}' sheet '{sheet_name}' contains no data rows"
            )

        # Normalise column names to strings (Excel may yield integer column headers)
        df.columns = [f"{c}".strip() for c in df.columns]

        time_col = _detect_time_column(list(df.columns))

        try:
            time_array, start_time, sample_rate = _build_time_array(df, time_col, path)
        except ProviderLoadError:
            raise
        except Exception as exc:
            raise ProviderLoadError(
                f"Failed to build time array from '{path}': {exc}"
            ) from exc

        waveform_cols = [c for c in df.columns if c != time_col]

        analog_channels: list[AnalogChannel] = []
        digital_channels: list[DigitalChannel] = []
        col_data: dict[str, np.ndarray] = {"time": time_array}
        analog_idx = 0
        digital_idx = 0

        for col_name in waveform_cols:
            series: pd.Series = df[col_name]

            if _is_digital_column(series, col_name):
                vals = series.fillna(0).astype(float).astype(np.int8).to_numpy()
                digital_channels.append(
                    DigitalChannel(name=col_name, index=digital_idx, normal_state=0)
                )
                col_data[col_name] = vals
                digital_idx += 1
            else:
                numeric: pd.Series = pd.to_numeric(series, errors="coerce")  # type: ignore[assignment]
                if numeric.isna().all():
                    warnings.warn(
                        f"Column '{col_name}' in sheet '{sheet_name}' of '{path}' "
                        "cannot be parsed as numeric — skipping"
                    )
                    continue
                vals = numeric.to_numpy(dtype=np.float64)
                analog_channels.append(
                    AnalogChannel(
                        name=col_name,
                        unit=_infer_unit(col_name),
                        index=analog_idx,
                    )
                )
                col_data[col_name] = vals
                analog_idx += 1

        if not analog_channels and not digital_channels:
            raise ProviderLoadError(
                f"Excel file '{path}' sheet '{sheet_name}' contains no usable waveform columns"
            )

        waveform_data = pd.DataFrame(col_data)
        total_samples = len(waveform_data)

        return DisturbanceRecord(
            metadata=RecordingMetadata(
                station_name="Unknown",
                recorder_name="Excel",
                source_file=str(path),
                provider_type="excel",
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
