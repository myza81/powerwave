"""CSV / Excel time-series metadata inspector.

Reads only a sample of rows — safe for very large files.
Detects timestamp columns, infers format, and reports date ambiguity explicitly.

Usage (from project root):
    python tools/inspect_csv_timeseries.py samples/csv/event.csv
    python tools/inspect_csv_timeseries.py samples/csv/event.csv --json
    python tools/inspect_csv_timeseries.py samples/csv/event.csv --sample-rows 20
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import warnings

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_TIMESTAMP_KEYWORDS = frozenset({
    "time", "t", "seconds", "sec", "timestamp", "datetime",
    "date", "ts", "epoch", "unix",
})

_DEFAULT_SAMPLE_ROWS = 50


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TimestampAmbiguity:
    """Describes date format ambiguity for a single sample value."""

    sample_value: str
    us_interpretation: str       # M/D/YYYY reading
    eu_interpretation: str       # D/M/YYYY reading
    is_ambiguous: bool           # True when both readings differ


@dataclass
class CsvTimestampInfo:
    """Timestamp analysis for one column."""

    column_name: str
    detected_type: str           # "datetime_string", "numeric_seconds", "unix_epoch", "unknown"
    sample_values: list[str]
    start_time: str | None       # ISO 8601 or None
    end_time: str | None
    nominal_interval_seconds: float | None
    has_timezone: bool
    ambiguity_detected: bool
    ambiguous_samples: list[TimestampAmbiguity]
    suggested_interpretation: str  # e.g. "M/D/YYYY h:mm:ss A" or "ISO 8601"
    notes: list[str]


@dataclass
class CsvMetadata:
    """Lightweight metadata extracted from a CSV/Excel file sample."""

    file_path: str
    file_type: str               # "csv" or "excel"
    total_columns: int
    sample_rows_read: int
    timestamp_column: str | None
    timestamp_info: CsvTimestampInfo | None
    data_columns: list[str]
    warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp column detection
# ─────────────────────────────────────────────────────────────────────────────


def _find_timestamp_column(df: pd.DataFrame) -> str | None:
    """Return the name of the most likely timestamp column, or None."""
    for col in df.columns:
        if col.strip().lower() in _TIMESTAMP_KEYWORDS:
            return col
    # Secondary pass: partial match
    for col in df.columns:
        col_lower = col.strip().lower()
        if any(kw in col_lower for kw in _TIMESTAMP_KEYWORDS):
            return col
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp type detection and ambiguity analysis
# ─────────────────────────────────────────────────────────────────────────────


def _is_numeric_column(series: pd.Series) -> bool:
    """Return True if the series can be interpreted as purely numeric."""
    try:
        pd.to_numeric(series.dropna(), errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def _check_single_ambiguity(value: str) -> TimestampAmbiguity:
    """Check whether a single date string is M/D/YYYY vs D/M/YYYY ambiguous."""
    # Suppress pandas format-inference warnings — cross-format comparison is intentional
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            us_dt = pd.to_datetime(value, dayfirst=False)
            us_str = us_dt.isoformat()
        except Exception:  # noqa: BLE001
            us_str = "PARSE_ERROR"

        try:
            eu_dt = pd.to_datetime(value, dayfirst=True)
            eu_str = eu_dt.isoformat()
        except Exception:  # noqa: BLE001
            eu_str = "PARSE_ERROR"

    is_ambiguous = (
        us_str != "PARSE_ERROR"
        and eu_str != "PARSE_ERROR"
        and us_str != eu_str
    )
    return TimestampAmbiguity(
        sample_value=str(value),
        us_interpretation=us_str,
        eu_interpretation=eu_str,
        is_ambiguous=is_ambiguous,
    )


def _infer_timestamp_format(series: pd.Series) -> tuple[str, bool, list[TimestampAmbiguity]]:
    """Return (detected_type, has_timezone, ambiguous_samples)."""
    sample = series.dropna().astype(str)
    if sample.empty:
        return "unknown", False, []

    first = sample.iloc[0].strip()

    # Numeric (seconds or epoch)
    if _is_numeric_column(series):
        numeric_vals = pd.to_numeric(series.dropna(), errors="coerce")
        median_val = float(numeric_vals.median())
        # Heuristic: Unix epoch > 1e9 (year ~2001+), otherwise relative seconds
        if median_val > 1e9:
            return "unix_epoch", False, []
        return "numeric_seconds", False, []

    # Has timezone marker?
    has_tz = any(
        tz_marker in v
        for v in sample.head(5)
        for tz_marker in ("+", "UTC", "Z", "GMT")
    )

    # ISO 8601 quick check (starts with 4-digit year)
    if len(first) >= 4 and first[:4].isdigit() and "-" in first:
        return "datetime_string_iso8601", has_tz, []

    # Check for D/M or M/D format ambiguity across unique date patterns
    # Sample up to 10 unique values to catch the ambiguity
    unique_samples = sample.unique()[:10]
    ambiguous: list[TimestampAmbiguity] = []
    for val in unique_samples:
        result = _check_single_ambiguity(str(val))
        if result.is_ambiguous:
            ambiguous.append(result)

    return "datetime_string", has_tz, ambiguous


def _guess_format_label(series: pd.Series, detected_type: str) -> str:
    """Return a human-readable format hint."""
    if detected_type == "numeric_seconds":
        return "Numeric (relative seconds)"
    if detected_type == "unix_epoch":
        return "Unix epoch (integer seconds)"
    if detected_type == "datetime_string_iso8601":
        return "ISO 8601 (YYYY-MM-DD...)"
    if detected_type == "unknown":
        return "Unknown"

    # Try to detect the pattern from the first non-null value
    sample = series.dropna().astype(str)
    if sample.empty:
        return "Unknown"
    first = sample.iloc[0].strip()

    # Common patterns
    if "AM" in first or "PM" in first:
        if "/" in first:
            return "M/D/YYYY h:mm:ss A  OR  D/M/YYYY h:mm:ss A"
        return "h:mm:ss A (date separate)"
    if first.count("/") == 2:
        parts = first.split("/")
        p0, p1 = int("".join(c for c in parts[0] if c.isdigit()) or "0"), \
                  int("".join(c for c in parts[1] if c.isdigit()) or "0")
        if p0 > 12:
            return "D/M/YYYY ..."  # Day must be first
        if p1 > 12:
            return "M/D/YYYY ..."  # Month first
        return "M/D/YYYY ...  OR  D/M/YYYY ...  (AMBIGUOUS)"
    return "datetime_string"


def _compute_interval(series: pd.Series, detected_type: str) -> float | None:
    """Compute median inter-sample interval in seconds."""
    try:
        if detected_type in ("numeric_seconds", "unix_epoch"):
            vals = pd.to_numeric(series.dropna(), errors="coerce").dropna().sort_values()
            if len(vals) < 2:
                return None
            diffs = vals.diff().dropna()
            return float(diffs.median())
        # Datetime string
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dt_series = pd.to_datetime(series.dropna(), errors="coerce")
        dt_series = dt_series.dropna().sort_values()
        if len(dt_series) < 2:
            return None
        diffs = dt_series.diff().dropna()
        median_td = diffs.median()
        return float(median_td.total_seconds())
    except Exception:  # noqa: BLE001
        return None


def _parse_start_end(series: pd.Series, detected_type: str) -> tuple[str | None, str | None]:
    """Return (start_time_iso, end_time_iso)."""
    try:
        if detected_type in ("numeric_seconds", "unix_epoch"):
            vals = pd.to_numeric(series.dropna(), errors="coerce").dropna()
            if vals.empty:
                return None, None
            if detected_type == "unix_epoch":
                start = pd.Timestamp(vals.min(), unit="s").isoformat()
                end = pd.Timestamp(vals.max(), unit="s").isoformat()
            else:
                return str(float(vals.min())), str(float(vals.max()))
            return start, end
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dt_series = pd.to_datetime(series.dropna(), errors="coerce")
        dt_series = dt_series.dropna()
        if dt_series.empty:
            return None, None
        return dt_series.min().isoformat(), dt_series.max().isoformat()
    except Exception:  # noqa: BLE001
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────


def inspect_file(file_path: Path, sample_rows: int = _DEFAULT_SAMPLE_ROWS) -> CsvMetadata:
    """Inspect a CSV or Excel file and return lightweight metadata.

    Reads at most *sample_rows* rows. The full file is never loaded into memory.
    """
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    warnings: list[str] = []

    # Load sample
    try:
        if suffix in {".csv", ".tsv", ".txt"}:
            # Try common separators
            for sep in (",", ";", "\t"):
                try:
                    df = pd.read_csv(file_path, nrows=sample_rows, sep=sep)
                    if len(df.columns) > 1:
                        break
                except Exception:  # noqa: BLE001
                    continue
            else:
                df = pd.read_csv(file_path, nrows=sample_rows)
            file_type = "csv"
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path, nrows=sample_rows)
            file_type = "excel"
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Cannot read file: {exc}") from exc

    ts_col = _find_timestamp_column(df)
    data_cols = [c for c in df.columns if c != ts_col]

    ts_info: CsvTimestampInfo | None = None
    if ts_col is not None:
        ts_series = df[ts_col]
        detected_type, has_tz, ambiguous = _infer_timestamp_format(ts_series)
        format_label = _guess_format_label(ts_series, detected_type)
        interval = _compute_interval(ts_series, detected_type)
        start_iso, end_iso = _parse_start_end(ts_series, detected_type)
        sample_vals = ts_series.dropna().astype(str).head(3).tolist()

        notes: list[str] = []
        if ambiguous:
            notes.append(
                f"WARNING: {len(ambiguous)} sample(s) are ambiguous - "
                "M/D/YYYY and D/M/YYYY produce different dates. "
                "Verify the intended timezone and locale."
            )
        if has_tz:
            notes.append("Timezone information detected in timestamp values.")

        ts_info = CsvTimestampInfo(
            column_name=ts_col,
            detected_type=detected_type,
            sample_values=sample_vals,
            start_time=start_iso,
            end_time=end_iso,
            nominal_interval_seconds=interval,
            has_timezone=has_tz,
            ambiguity_detected=bool(ambiguous),
            ambiguous_samples=ambiguous,
            suggested_interpretation=format_label,
            notes=notes,
        )
    else:
        warnings.append("No timestamp column detected. Numeric index will be used.")

    return CsvMetadata(
        file_path=str(file_path),
        file_type=file_type,
        total_columns=len(df.columns),
        sample_rows_read=len(df),
        timestamp_column=ts_col,
        timestamp_info=ts_info,
        data_columns=data_cols,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────


def format_text_summary(meta: CsvMetadata) -> str:
    """Return a human-readable text summary."""
    lines: list[str] = [
        "CSV Time-Series Summary",
        "-" * 40,
        f"File:           {meta.file_path}",
        f"Type:           {meta.file_type.upper()}",
        f"Columns:        {meta.total_columns}",
        f"Sample rows:    {meta.sample_rows_read}",
        "",
    ]
    if meta.timestamp_info:
        ti = meta.timestamp_info
        lines += [
            f"Timestamp Column: {ti.column_name}",
            f"Detected Format:  {ti.suggested_interpretation}",
            f"Ambiguous Date:   {'YES (!)' if ti.ambiguity_detected else 'No'}",
            f"Has Timezone:     {'Yes' if ti.has_timezone else 'No'}",
            f"Start Time:       {ti.start_time or 'unknown'}",
            f"End Time:         {ti.end_time or 'unknown'}",
            f"Nominal Interval: "
            + (f"{ti.nominal_interval_seconds:.3g} s" if ti.nominal_interval_seconds is not None else "unknown"),
            "",
        ]
        if ti.ambiguous_samples:
            lines.append("Ambiguous Samples:")
            for a in ti.ambiguous_samples[:5]:
                lines.append(f'  "{a.sample_value}"')
                lines.append(f"    US (M/D/Y): {a.us_interpretation}")
                lines.append(f"    EU (D/M/Y): {a.eu_interpretation}")
            lines.append("")
        if ti.notes:
            for note in ti.notes:
                lines.append(f"NOTE: {note}")
            lines.append("")

    lines.append(f"Data Columns ({len(meta.data_columns)}):")
    for col in meta.data_columns:
        lines.append(f"  - {col}")

    if meta.warnings:
        lines.append("")
        for w in meta.warnings:
            lines.append(f"WARNING: {w}")

    return "\n".join(lines)


def format_classification_summary(meta: CsvMetadata) -> str:
    """Return a text summary of column classifications (requires app/ on sys.path)."""
    try:
        import sys
        from pathlib import Path
        _root = Path(__file__).parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        import pandas as pd
        from app.data.column_classifier import classify_csv_columns, CONFIRMATION_THRESHOLD
    except ImportError:
        return "Column classification unavailable (app/ not on sys.path)."

    try:
        if meta.file_type == "csv":
            df = pd.read_csv(meta.file_path, nrows=100)
        else:
            df = pd.read_excel(meta.file_path, nrows=100)
    except Exception as exc:  # noqa: BLE001
        return f"Column classification failed: {exc}"

    ts_col = meta.timestamp_column
    classifications = classify_csv_columns(df, timestamp_column=ts_col)

    lines: list[str] = [
        "Column Classification",
        "-" * 40,
        f"Confidence threshold: {CONFIRMATION_THRESHOLD}",
        "",
    ]
    for col in meta.data_columns:
        if col not in classifications:
            continue
        cl = classifications[col]
        conf_str = f"{cl.confidence:.2f}"
        confirm = " [CONFIRM REQUIRED]" if cl.requires_user_confirmation else ""
        st = cl.signal_type or "unknown"
        unit = cl.unit or "?"
        lines.append(f"  {col}")
        lines.append(f"    signal_type:   {st}  ({unit})")
        lines.append(f"    display_group: {cl.display_group}")
        lines.append(f"    confidence:    {conf_str}  [{cl.inferred_from}]{confirm}")

    return "\n".join(lines)


def format_json_summary(meta: CsvMetadata) -> str:
    """Return JSON representation."""
    d = asdict(meta)
    return json.dumps(d, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect a CSV/Excel time-series file (timestamp analysis, ambiguity detection)."
    )
    p.add_argument("path", type=Path, help="Path to a .csv or .xlsx file.")
    p.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")
    p.add_argument(
        "--classify",
        action="store_true",
        dest="classify",
        help="Show column signal classification (requires app/ on sys.path).",
    )
    p.add_argument(
        "--sample-rows",
        type=int,
        default=_DEFAULT_SAMPLE_ROWS,
        metavar="N",
        help=f"Number of rows to sample (default: {_DEFAULT_SAMPLE_ROWS}).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    try:
        meta = inspect_file(args.path, sample_rows=args.sample_rows)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(format_json_summary(meta))
    else:
        print(format_text_summary(meta))
        if args.classify:
            print()
            print(format_classification_summary(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
