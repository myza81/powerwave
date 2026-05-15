"""app/data/direct_load_intelligence.py

Intelligence adapter for direct CSV/Excel file opens (Phase D4.4).

When a user opens a CSV or Excel file via File→Open, this module:
  1. Classifies each analog channel via IntelligenceManager
  2. Builds SignalMetadata with display_group assignments for grouped display
  3. Detects whether the timestamp column interpretation is ambiguous
  4. Applies the operator-selected timestamp format to rebase start_time
  5. Emits structured runtime diagnostics to stderr

All functions are pure Python with no Qt dependency — safe for background threads.

Display-group aliases:
  column_classifier returns groups like "voltage_rms", "current_rms" that differ
  from the channel_grouper constants "voltage_raw", "current_raw". The alias map
  normalises them at the adapter boundary so downstream grouping is consistent.
"""
from __future__ import annotations

import dataclasses
import sys
from datetime import datetime
from pathlib import Path

from app.data.signal_metadata import SignalMetadata
from app.models import DisturbanceRecord
from app.models.timing import TimingInformation

# column_classifier names → channel_grouper names
_DISPLAY_GROUP_ALIASES: dict[str, str] = {
    "voltage_rms": "voltage_raw",
    "current_rms": "current_raw",
    "voltage":     "voltage_raw",
    "current":     "current_raw",
}

# Recognised time-column name stems (matches CsvProvider's _TIME_COL_NAMES)
_TIME_COL_STEMS = frozenset({"time", "t", "seconds", "sec", "timestamp", "datetime"})

# CsvProvider / ExcelProvider fall back to this when no real timestamp exists
_EPOCH_FALLBACK = datetime(2000, 1, 1, 0, 0, 0)


def build_signal_metadata(
    record: DisturbanceRecord,
    intelligence_manager,       # IntelligenceManager — no hard import to avoid circulars
    source_id: str,
    provider_type: str = "csv",  # noqa: ARG001 — reserved for future provider-specific rules
) -> dict[str, SignalMetadata]:
    """Classify each analog channel and return a SignalMetadata dict.

    Display groups from column_classifier are aliased to channel_grouper
    constants (e.g. ``"voltage_rms"`` → ``"voltage_raw"``).

    Args:
        record:               Loaded DisturbanceRecord.
        intelligence_manager: IntelligenceManager instance.
        source_id:            Source identifier (typically the file stem).
        provider_type:        "csv" or "excel" (reserved for future rules).

    Returns:
        Dict mapping channel name → SignalMetadata with display_group set.
    """
    result: dict[str, SignalMetadata] = {}
    for ch in record.analog_channels:
        try:
            vals = record.waveform_data[ch.name].dropna().astype(float).tolist()
        except (KeyError, ValueError, TypeError):
            vals = []
        cls, _ = intelligence_manager.classify_column(
            ch.name, vals if vals else None
        )
        raw_group = cls.display_group or "other"
        display_group = _DISPLAY_GROUP_ALIASES.get(raw_group, raw_group)
        result[ch.name] = SignalMetadata(
            name=ch.name,
            unit=ch.unit,
            source=source_id,
            signal_type=cls.signal_type,
            display_group=display_group,
            confidence=cls.confidence,
            inferred_from=cls.inferred_from,
            requires_user_confirmation=cls.requires_user_confirmation,
        )
    return result


def detect_timestamp_ambiguity(
    path: Path,
    record: DisturbanceRecord,
) -> tuple[bool, dict]:
    """Detect whether the time column in *path* has an ambiguous date interpretation.

    Reads the first 20 rows of the file (fast) and runs the TimestampInterpretation
    matrix on the raw time-column values.  Uses the record's start_time as a
    COMTRADE-style reference when it is not the epoch fallback.

    Args:
        path:   Path to the original CSV or Excel file.
        record: Freshly loaded DisturbanceRecord (for timing reference).

    Returns:
        (is_ambiguous, {col_name: TimestampInterpretationMatrix})
        Returns (False, {}) when the file cannot be read or has no time column.
    """
    from app.data.timestamp_interpreter import build_interpretation_matrix
    import pandas as pd

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            sample_df: pd.DataFrame = pd.read_csv(path, nrows=20)
        elif suffix in (".xlsx", ".xls"):
            sample_df = pd.read_excel(path, nrows=20)
        else:
            return False, {}
    except Exception:
        return False, {}

    # Identify the primary time column (first match, same logic as CsvProvider)
    time_col: str | None = None
    for col in sample_df.columns:
        norm = col.lower().strip()
        stem = norm.rsplit(".", 1)[0] if "." in norm else norm
        if stem in _TIME_COL_STEMS:
            time_col = col
            break

    if time_col is None:
        return False, {}

    raw_values = [str(v) for v in sample_df[time_col].dropna().tolist()]
    if not raw_values:
        return False, {}

    comtrade_start: datetime | None = None
    st = record.timing_info.start_time
    if st is not None and st != _EPOCH_FALLBACK:
        comtrade_start = st

    matrix = build_interpretation_matrix(
        column_name=time_col,
        raw_values=raw_values,
        comtrade_start=comtrade_start,
    )
    return matrix.is_ambiguous, {time_col: matrix}


def apply_selected_timestamp_format(
    record: DisturbanceRecord,
    ts_matrices: dict,
    selected_formats: dict[str, str],
) -> DisturbanceRecord:
    """Rebase record.timing_info.start_time using the operator-selected format.

    Looks up the first matching (col_name, format_string) pair in *selected_formats*
    against *ts_matrices*, extracts ``parsed_samples[0]`` from the matched
    interpretation, and returns a new DisturbanceRecord whose timing_info reflects
    that corrected start_time.

    The waveform_data ``time`` column (elapsed seconds) is never mutated.
    trigger_time is shifted by the same offset as start_time.

    Args:
        record:           Original DisturbanceRecord.
        ts_matrices:      {col_name: TimestampInterpretationMatrix} from
                          detect_timestamp_ambiguity().
        selected_formats: {col_name: format_string} — operator's choices
                          (caller should pass dlg.selected_timestamp_formats[source_id]).

    Returns:
        Rebased DisturbanceRecord (new object, original unchanged) or the
        original record unchanged when no rebasing is possible.
    """
    for col_name, fmt_string in selected_formats.items():
        matrix = ts_matrices.get(col_name)
        if matrix is None:
            continue
        interp = next(
            (i for i in matrix.interpretations if i.format_string == fmt_string),
            matrix.recommended,
        )
        if interp is None or not interp.parsed_samples:
            continue
        new_start = interp.parsed_samples[0]
        return _rebase_record_start_time(record, new_start)

    return record


def _rebase_record_start_time(
    record: DisturbanceRecord,
    new_start_time: datetime,
) -> DisturbanceRecord:
    """Return a new DisturbanceRecord with corrected absolute timestamps.

    Only timing_info.start_time and timing_info.trigger_time are updated.
    All waveform data and channel metadata are shared by reference.
    """
    trigger_offset = record.timing_info.trigger_time - record.timing_info.start_time
    new_timing = TimingInformation(
        start_time=new_start_time,
        trigger_time=new_start_time + trigger_offset,
        time_multiplier=record.timing_info.time_multiplier,
        timezone=record.timing_info.timezone,
    )
    return DisturbanceRecord(
        metadata=record.metadata,
        waveform_data=record.waveform_data,
        analog_channels=record.analog_channels,
        digital_channels=record.digital_channels,
        sampling_info=record.sampling_info,
        timing_info=new_timing,
        disturbance_info=record.disturbance_info,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runtime diagnostics (Part 3)
# ─────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class DirectOpenDiagnostics:
    """Lightweight summary of a direct CSV/Excel open operation for stderr logging."""

    source_path: str
    provider_type: str
    timestamp_column: str | None
    timestamp_interpretation: str | None  # format_label of operator-selected interp
    axis_mode: str
    display_groups: dict[str, str | None]  # channel_name → display_group
    review_columns: list[str]             # channels that required user confirmation


def build_direct_open_diagnostics(
    source_path: str,
    provider_type: str,
    signal_metadata: dict[str, SignalMetadata],
    ts_matrices: dict,
    selected_formats: dict[str, str],
    axis_mode: str,
) -> DirectOpenDiagnostics:
    """Build a DirectOpenDiagnostics from the assembled open-operation state.

    Args:
        source_path:      File path string for display.
        provider_type:    "csv" or "excel".
        signal_metadata:  Channel name → SignalMetadata mapping.
        ts_matrices:      {col_name: TimestampInterpretationMatrix}.
        selected_formats: {col_name: format_string} — operator's choices.
        axis_mode:        "relative_seconds" or "absolute_datetime".
    """
    ts_column: str | None = None
    ts_interp_label: str | None = None

    for col_name, fmt_string in selected_formats.items():
        matrix = ts_matrices.get(col_name)
        if matrix is None:
            continue
        ts_column = col_name
        interp = next(
            (i for i in matrix.interpretations if i.format_string == fmt_string),
            matrix.recommended,
        )
        if interp is not None:
            ts_interp_label = interp.format_label
        break

    if ts_column is None and ts_matrices:
        ts_column = next(iter(ts_matrices))

    display_groups: dict[str, str | None] = {
        name: m.display_group for name, m in signal_metadata.items()
    }
    review_columns = [
        name for name, m in signal_metadata.items() if m.requires_user_confirmation
    ]

    return DirectOpenDiagnostics(
        source_path=source_path,
        provider_type=provider_type,
        timestamp_column=ts_column,
        timestamp_interpretation=ts_interp_label,
        axis_mode=axis_mode,
        display_groups=display_groups,
        review_columns=review_columns,
    )


def log_direct_open_diagnostics(diag: DirectOpenDiagnostics) -> None:
    """Write a structured diagnostic summary to stderr."""
    groups_str = ", ".join(
        f"{name}={grp}" for name, grp in diag.display_groups.items()
    )
    review_str = ", ".join(diag.review_columns) if diag.review_columns else "none"
    ts_str = (
        f"{diag.timestamp_column} → {diag.timestamp_interpretation}"
        if diag.timestamp_column and diag.timestamp_interpretation
        else diag.timestamp_column or "none"
    )
    print(
        f"[D4.4] {diag.source_path}"
        f" | provider={diag.provider_type}"
        f" | axis={diag.axis_mode}"
        f" | timestamp={ts_str}"
        f" | groups=[{groups_str}]"
        f" | review=[{review_str}]",
        file=sys.stderr,
    )
