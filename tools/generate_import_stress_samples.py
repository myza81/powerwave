"""Deterministic CSV sample generator for Import Wizard stress testing.

Generated files are intended for runtime temp directories and benchmark runs.
The module streams rows directly to disk so medium and large samples do not
need to be held in memory.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StressSampleConfig:
    row_count: int = 1_000
    analog_column_count: int = 6
    unknown_column_count: int = 1
    digital_column_count: int = 2
    timestamp_format: str = "%Y-%m-%d %H:%M:%S.%f"
    sampling_interval_ms: float = 20.0
    delimiter: str = ","
    metadata_rows: int = 0
    malformed_timestamp_ratio: float = 0.0
    duplicate_timestamp_ratio: float = 0.0
    missing_timestamp_ratio: float = 0.0
    digital_text_values: bool = False
    include_header: bool = True
    start_time: datetime = datetime(2026, 1, 1, 0, 0, 0)


PRESETS: dict[str, StressSampleConfig] = {
    "small": StressSampleConfig(row_count=1_000),
    "medium": StressSampleConfig(row_count=100_000),
    "large": StressSampleConfig(row_count=1_000_000),
}


BASE_ANALOG_COLUMNS: tuple[str, ...] = (
    "Voltage A (kV)",
    "Voltage B (kV)",
    "Voltage C (kV)",
    "Current A (A)",
    "Current B (A)",
    "Current C (A)",
    "MW Total",
    "MVar Total",
    "Frequency Hz",
)


def generate_import_stress_csv(path: str | Path, config: StressSampleConfig) -> Path:
    """Write a deterministic stress CSV and return its path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    analog_columns = _analog_columns(config.analog_column_count)
    digital_columns = [f"CB Status {idx + 1}" for idx in range(config.digital_column_count)]
    unknown_columns = [f"Historian Extra {idx + 1}" for idx in range(config.unknown_column_count)]
    header = ["Timestamp", *analog_columns, *digital_columns, *unknown_columns]

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=config.delimiter)
        for idx in range(config.metadata_rows):
            writer.writerow([f"Metadata {idx + 1}", "Powerwave stress sample"])
        if config.include_header:
            writer.writerow(header)
        for row_index in range(config.row_count):
            writer.writerow(_build_row(row_index, config, analog_columns, digital_columns, unknown_columns))

    return out


def _analog_columns(count: int) -> list[str]:
    if count <= len(BASE_ANALOG_COLUMNS):
        return list(BASE_ANALOG_COLUMNS[:count])
    extra = [f"Analog Spare {idx + 1}" for idx in range(count - len(BASE_ANALOG_COLUMNS))]
    return [*BASE_ANALOG_COLUMNS, *extra]


def _build_row(
    row_index: int,
    config: StressSampleConfig,
    analog_columns: list[str],
    digital_columns: list[str],
    unknown_columns: list[str],
) -> list[str]:
    ts = _timestamp_value(row_index, config)
    analog_values = [_analog_value(name, row_index) for name in analog_columns]
    digital_values = [_digital_value(row_index, idx, config.digital_text_values) for idx, _ in enumerate(digital_columns)]
    unknown_values = [_unknown_value(row_index, idx) for idx, _ in enumerate(unknown_columns)]
    return [ts, *analog_values, *digital_values, *unknown_values]


def _timestamp_value(row_index: int, config: StressSampleConfig) -> str:
    interval = timedelta(milliseconds=config.sampling_interval_ms)
    effective_index = row_index
    if row_index > 0 and _hit_ratio(row_index, config.duplicate_timestamp_ratio):
        effective_index -= 1

    if _hit_ratio(row_index, config.missing_timestamp_ratio):
        return ""
    if _hit_ratio(row_index, config.malformed_timestamp_ratio):
        return f"BAD_TS_{row_index}"

    stamp = config.start_time + (interval * effective_index)
    return stamp.strftime(config.timestamp_format)


def _hit_ratio(row_index: int, ratio: float) -> bool:
    if ratio <= 0:
        return False
    if ratio >= 1:
        return True
    period = max(1, round(1.0 / ratio))
    return row_index > 0 and row_index % period == 0


def _analog_value(name: str, row_index: int) -> str:
    wave = math.sin(row_index / 25.0)
    lower = name.lower()
    if "voltage" in lower:
        return f"{132.0 + wave * 0.8:.4f}"
    if "current" in lower:
        return f"{450.0 + wave * 12.0:.4f}"
    if "mvar" in lower:
        return f"{18.0 + wave * 1.5:.4f}"
    if "mw" in lower:
        return f"{95.0 + wave * 3.0:.4f}"
    if "frequency" in lower:
        return f"{50.0 + wave * 0.015:.5f}"
    return f"{row_index % 1000 + wave:.4f}"


def _digital_value(row_index: int, column_index: int, text_values: bool) -> str:
    high = ((row_index // 25) + column_index) % 2 == 1
    if not text_values:
        return "1" if high else "0"
    return "CLOSED" if high else "OPEN"


def _unknown_value(row_index: int, column_index: int) -> str:
    if column_index % 2 == 0:
        return f"{(row_index * (column_index + 1)) % 17}"
    return "operator-note" if row_index % 10 == 0 else ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic Import Wizard stress CSV samples.")
    parser.add_argument("output", type=Path, help="Output CSV path. Use a runtime temp directory for large files.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--analog-columns", type=int, default=None)
    parser.add_argument("--unknown-columns", type=int, default=None)
    parser.add_argument("--digital-columns", type=int, default=None)
    parser.add_argument("--timestamp-format", default=None)
    parser.add_argument("--interval-ms", type=float, default=None)
    parser.add_argument("--delimiter", default=None)
    parser.add_argument("--metadata-rows", type=int, default=None)
    parser.add_argument("--malformed-ratio", type=float, default=None)
    parser.add_argument("--duplicate-ratio", type=float, default=None)
    parser.add_argument("--missing-ratio", type=float, default=None)
    parser.add_argument("--digital-text", action="store_true")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> StressSampleConfig:
    base = PRESETS.get(args.preset or "small", PRESETS["small"])
    return StressSampleConfig(
        row_count=args.rows if args.rows is not None else base.row_count,
        analog_column_count=args.analog_columns if args.analog_columns is not None else base.analog_column_count,
        unknown_column_count=args.unknown_columns if args.unknown_columns is not None else base.unknown_column_count,
        digital_column_count=args.digital_columns if args.digital_columns is not None else base.digital_column_count,
        timestamp_format=args.timestamp_format or base.timestamp_format,
        sampling_interval_ms=args.interval_ms if args.interval_ms is not None else base.sampling_interval_ms,
        delimiter=args.delimiter or base.delimiter,
        metadata_rows=args.metadata_rows if args.metadata_rows is not None else base.metadata_rows,
        malformed_timestamp_ratio=args.malformed_ratio if args.malformed_ratio is not None else base.malformed_timestamp_ratio,
        duplicate_timestamp_ratio=args.duplicate_ratio if args.duplicate_ratio is not None else base.duplicate_timestamp_ratio,
        missing_timestamp_ratio=args.missing_ratio if args.missing_ratio is not None else base.missing_timestamp_ratio,
        digital_text_values=args.digital_text or base.digital_text_values,
    )


def main() -> int:
    args = _parse_args()
    config = _config_from_args(args)
    path = generate_import_stress_csv(args.output, config)
    print(f"Wrote {config.row_count:,} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
