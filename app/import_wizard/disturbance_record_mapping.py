"""Pure mapping utilities for NormalizedDataset → DisturbanceRecord conversion.

Stateless functions only.  No Qt, no side effects, no logging.

Functions
---------
partition_parameters        Split analog vs. digital ParameterMetadata lists.
infer_phase                 Detect A/B/C from a canonical channel name.
map_to_analog_channel       ParameterMetadata → AnalogChannel descriptor.
map_to_digital_channel      ParameterMetadata → DigitalChannel descriptor.
build_time_array            datetime64 Series → (float64 seconds, start_time, Hz).
infer_nominal_frequency     Auto-detect 50 or 60 Hz from a FREQUENCY channel.
build_waveform_data         Assemble the waveform_data DataFrame for a DisturbanceRecord.
"""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.normalized_dataset import ParameterMetadata
from app.models import AnalogChannel, DigitalChannel


# ─── Parameter partitioning ───────────────────────────────────────────────────

_DIGITAL_TYPES: frozenset[ParameterType] = frozenset({ParameterType.DIGITAL})


def partition_parameters(
    parameters: list[ParameterMetadata],
) -> tuple[list[ParameterMetadata], list[ParameterMetadata]]:
    """Split parameters into (analog_params, digital_params).

    DIGITAL → digital channel.
    All other types (VOLTAGE, CURRENT, MW, MVAR, FREQUENCY, ROCOF, UNKNOWN,
    TIMESTAMP) → analog channel.  TIMESTAMP-typed parameters should be excluded
    upstream; the bridge will emit a warning when they appear.
    """
    analog: list[ParameterMetadata] = []
    digital: list[ParameterMetadata] = []
    for pm in parameters:
        if pm.parameter_type in _DIGITAL_TYPES:
            digital.append(pm)
        else:
            analog.append(pm)
    return analog, digital


# ─── Phase inference ──────────────────────────────────────────────────────────

_PHASE_RE = re.compile(r"[_\-\s]([abc])$", re.IGNORECASE)


def infer_phase(name: str) -> str | None:
    """Infer the power-system phase (A, B, or C) from a channel name.

    Matches trailing ``_a``, ``_b``, ``_c``, ``-a``, ``-b``, ``-c``,
    `` a``, `` b``, `` c`` patterns (case-insensitive).  Returns the
    uppercase letter or None when no phase is identifiable.

    Examples
    --------
    >>> infer_phase("voltage_a")   # "A"
    >>> infer_phase("current_b")   # "B"
    >>> infer_phase("mw")          # None
    >>> infer_phase("voltage_va")  # None  (no separator before "a")
    """
    m = _PHASE_RE.search(name)
    return m.group(1).upper() if m else None


# ─── Unit resolution ──────────────────────────────────────────────────────────

_TYPE_DEFAULT_UNIT: dict[ParameterType, str | None] = {
    ParameterType.VOLTAGE: "kV",
    ParameterType.CURRENT: "A",
    ParameterType.MW: "MW",
    ParameterType.MVAR: "Mvar",
    ParameterType.FREQUENCY: "Hz",
    ParameterType.ROCOF: "Hz/s",
    ParameterType.UNKNOWN: "unknown",
    ParameterType.TIMESTAMP: "s",
    ParameterType.DIGITAL: None,
}


def _resolve_unit(pm: ParameterMetadata, default_unit: str) -> str:
    """Return the best unit string for a parameter.

    Priority: metadata unit → type default → caller's default_unit.
    """
    if pm.unit:
        return pm.unit
    type_default = _TYPE_DEFAULT_UNIT.get(pm.parameter_type)
    if type_default:
        return type_default
    return default_unit


# ─── Channel mapping ──────────────────────────────────────────────────────────

def map_to_analog_channel(
    pm: ParameterMetadata,
    index: int,
    *,
    default_unit: str = "unknown",
) -> AnalogChannel:
    """Create an AnalogChannel descriptor from a ParameterMetadata.

    Phase is inferred from the canonical name (e.g. "voltage_a" → "A").
    A source-traceability description is added when the canonical name
    differs from the original source column name.
    """
    description: str | None = None
    if pm.source_name != pm.canonical_name:
        description = f"source: {pm.source_name}"

    return AnalogChannel(
        name=pm.canonical_name,
        unit=_resolve_unit(pm, default_unit),
        index=index,
        phase=infer_phase(pm.canonical_name),
        description=description,
    )


def map_to_digital_channel(
    pm: ParameterMetadata,
    index: int,
) -> DigitalChannel:
    """Create a DigitalChannel descriptor from a ParameterMetadata."""
    description: str | None = None
    if pm.source_name != pm.canonical_name:
        description = f"source: {pm.source_name}"

    return DigitalChannel(
        name=pm.canonical_name,
        index=index,
        normal_state=0,
        description=description,
    )


# ─── Time array construction ──────────────────────────────────────────────────

_EPOCH_FALLBACK = datetime(2000, 1, 1, 0, 0, 0)


def build_time_array(
    timestamp_series: pd.Series,
) -> tuple[np.ndarray, datetime, float]:
    """Convert a datetime64 Series into a float64 time array.

    Handles NaT values via linear interpolation so the output array has no NaN.

    Returns
    -------
    time_array   Float64 ndarray of seconds elapsed from the first valid timestamp.
    start_time   Timezone-naive Python datetime of the first valid sample.
    sample_rate  Dominant sampling rate in Hz; 0.0 when not determinable.
    """
    valid = timestamp_series.dropna()

    if valid.empty:
        n = len(timestamp_series)
        return np.arange(n, dtype=np.float64), _EPOCH_FALLBACK, 0.0

    t0 = valid.iloc[0]

    # Build relative seconds (NaT → NaN via timedelta arithmetic)
    relative = (timestamp_series - t0).dt.total_seconds()

    # Fill NaN positions via linear interpolation so the output is fully numeric.
    # limit_direction="both" handles leading/trailing NaT rows.
    time_array = (
        relative.interpolate(method="linear", limit_direction="both")
        .to_numpy(dtype=np.float64)
    )

    # Sample rate from positive diffs of valid timestamps only
    valid_seconds = (valid - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    pos_diffs = np.diff(valid_seconds)
    pos_diffs = pos_diffs[pos_diffs > 0]
    if len(pos_diffs) > 0:
        median_dt = float(np.median(pos_diffs))
        sample_rate = round(1.0 / median_dt, 6) if median_dt > 0 else 0.0
    else:
        sample_rate = 0.0

    # Convert origin to tz-naive Python datetime
    if hasattr(t0, "to_pydatetime"):
        start_dt: datetime = t0.to_pydatetime().replace(tzinfo=None)
    else:
        start_dt = _EPOCH_FALLBACK

    return time_array, start_dt, sample_rate


# ─── Nominal frequency inference ─────────────────────────────────────────────

def infer_nominal_frequency(
    analog_params: list[ParameterMetadata],
    data: pd.DataFrame,
    *,
    default: float = 50.0,
) -> float:
    """Infer 50 or 60 Hz from the mean value of a FREQUENCY channel.

    Returns ``default`` when no FREQUENCY channel is present or parseable.
    """
    freq_params = [
        p for p in analog_params if p.parameter_type == ParameterType.FREQUENCY
    ]
    if not freq_params:
        return default

    col = freq_params[0].canonical_name
    if col not in data.columns:
        return default

    numeric: pd.Series = pd.to_numeric(data[col], errors="coerce")  # type: ignore[assignment]
    mean_hz = numeric.dropna().mean()
    if not np.isfinite(mean_hz):
        return default

    return 60.0 if abs(mean_hz - 60.0) < abs(mean_hz - 50.0) else 50.0


# ─── waveform_data assembly ───────────────────────────────────────────────────

def build_waveform_data(
    time_array: np.ndarray,
    analog_params: list[ParameterMetadata],
    digital_params: list[ParameterMetadata],
    data: pd.DataFrame,
    *,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble the ``waveform_data`` DataFrame for a DisturbanceRecord.

    Output structure
    ----------------
    - ``"time"`` column: float64 seconds from start (always first).
    - One float64 column per analog channel (NaN for unparseable values).
    - One int8 column per digital channel (0-filled for unparseable values).

    Missing columns and non-numeric values are handled gracefully; a warning
    string is appended to *warnings* for each non-fatal issue.

    Parameters
    ----------
    time_array      Float64 seconds-from-start array aligned to data rows.
    analog_params   ParameterMetadata for analog channels (in output order).
    digital_params  ParameterMetadata for digital channels (in output order).
    data            NormalizedDataset.data DataFrame — never mutated.
    warnings        Mutable list to collect non-fatal warning strings.
    """
    warn = warnings if warnings is not None else []
    n = len(time_array)
    col_data: dict[str, np.ndarray] = {"time": time_array}

    for pm in analog_params:
        name = pm.canonical_name
        if name not in data.columns:
            warn.append(
                f"Analog channel '{name}' not found in dataset; filling with NaN."
            )
            col_data[name] = np.full(n, np.nan, dtype=np.float64)
        else:
            numeric_col: pd.Series = pd.to_numeric(data[name], errors="coerce")  # type: ignore[assignment]
            col_data[name] = numeric_col.to_numpy(dtype=np.float64)

    for pm in digital_params:
        name = pm.canonical_name
        if name not in data.columns:
            warn.append(
                f"Digital channel '{name}' not found in dataset; filling with zeros."
            )
            col_data[name] = np.zeros(n, dtype=np.int8)
        else:
            col_data[name] = _coerce_digital(data[name], name, warn)

    return pd.DataFrame(col_data)


# Textual representations that map to logic-HIGH (1)
_DIGITAL_HIGH_STRINGS: frozenset[str] = frozenset({
    "1", "true", "yes", "on", "close", "closed", "trip", "tripped", "active", "high",
})


def _coerce_digital(series: pd.Series, name: str, warnings: list[str]) -> np.ndarray:
    """Coerce a Series to a binary int8 array for a DigitalChannel.

    Supported input representations:
    - Numeric 0/1 (or bool True/False)
    - Strings: "0"/"1", "true"/"false", "open"/"close", "trip"/"normal",
      "on"/"off", "yes"/"no", "active"/"inactive", "high"/"low"

    Unknown string values map to 0 with a warning.
    """
    # Fast path: boolean or already-numeric 0/1
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(np.int8).to_numpy()

    numeric: pd.Series = pd.to_numeric(series, errors="coerce")  # type: ignore[assignment]
    if numeric.notna().all():
        return numeric.fillna(0).astype(float).astype(np.int8).to_numpy()

    # Mixed or string column — map textually
    result = np.zeros(len(series), dtype=np.int8)
    had_unknown = False
    for i, val in enumerate(series):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            result[i] = 0
        else:
            s = str(val).strip().lower()
            try:
                result[i] = int(float(s)) & 1
            except (ValueError, TypeError):
                if s in _DIGITAL_HIGH_STRINGS:
                    result[i] = 1
                else:
                    result[i] = 0
                    if s not in {"0", "false", "no", "off", "open", "normal", "inactive", "low"}:
                        had_unknown = True
    if had_unknown:
        warnings.append(
            f"Digital channel '{name}' contains unrecognised string values; "
            "they were mapped to 0."
        )
    return result
