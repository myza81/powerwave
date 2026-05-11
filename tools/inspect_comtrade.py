"""Lightweight COMTRADE .cfg metadata inspector.

Reads only the CFG file — never loads the full waveform DAT array.
Safe for arbitrarily large recordings.

Usage (from project root):
    python tools/inspect_comtrade.py samples/comtrade/event/
    python tools/inspect_comtrade.py samples/comtrade/event/event.cfg
    python tools/inspect_comtrade.py samples/comtrade/event/ --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ComtradeMetadata:
    """Lightweight metadata extracted from a COMTRADE CFG file."""

    cfg_path: str
    station_name: str
    rec_dev_id: str
    rev_yr: str
    n_analog: int
    n_digital: int
    analog_channel_names: list[str]
    digital_channel_names: list[str]
    nominal_frequency_hz: float
    sampling_rates_hz: list[float]
    samples_per_rate: list[int]
    total_samples: int
    start_time: str          # ISO 8601 string
    trigger_time: str        # ISO 8601 string
    duration_seconds: float
    dat_format: str
    timemult: float
    dat_path: str | None     # None if .dat not found
    dat_size_bytes: int | None


# ─────────────────────────────────────────────────────────────────────────────
# CFG parser
# ─────────────────────────────────────────────────────────────────────────────

_TIMESTAMP_FMTS = (
    "%d/%m/%Y,%H:%M:%S.%f",
    "%d/%m/%Y,%H:%M:%S",
)
_VALID_REV_YRS = {"1991", "1999", "2013"}


def _parse_cfg_timestamp(s: str) -> datetime:
    for fmt in _TIMESTAMP_FMTS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse COMTRADE timestamp: {s!r}")


def _locate_cfg(path: Path) -> Path:
    """Accept a .cfg file path or a directory (finds the first .cfg inside)."""
    if path.is_file():
        if path.suffix.lower() not in {".cfg", ".comtrade"}:
            raise FileNotFoundError(f"Not a .cfg file: {path}")
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.cfg")) + sorted(path.glob("*.CFG"))
        if not candidates:
            raise FileNotFoundError(f"No .cfg file found in directory: {path}")
        return candidates[0]
    raise FileNotFoundError(f"Path not found: {path}")


def _locate_dat(cfg_path: Path) -> Path | None:
    """Find the .dat companion file, returning None if absent."""
    for suffix in (".dat", ".DAT"):
        candidate = cfg_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def parse_cfg(cfg_path: Path) -> ComtradeMetadata:
    """Parse a COMTRADE CFG file and return lightweight metadata.

    Never reads the DAT waveform array — safe for huge files.
    """
    cfg_path = cfg_path.resolve()
    try:
        text = cfg_path.read_text(encoding="latin-1")
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read CFG: {exc}") from exc

    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    idx = 0

    def _next(field: str) -> str:
        nonlocal idx
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx >= len(lines):
            raise ValueError(f"CFG truncated: missing field '{field}'")
        val = lines[idx].strip()
        idx += 1
        return val

    # Line 1: station_name, rec_dev_id [, rev_yr]
    line1_parts = _next("station/device/rev_yr").split(",")
    station_name = line1_parts[0].strip()
    rec_dev_id = line1_parts[1].strip() if len(line1_parts) > 1 else ""
    rev_yr_raw = line1_parts[2].strip() if len(line1_parts) > 2 else ""
    rev_yr = rev_yr_raw if rev_yr_raw in _VALID_REV_YRS else "1999"

    # Line 2: TT,##A,##D  (e.g. "9,6A,3D" or legacy "6,3")
    line2 = _next("channel counts")
    parts2 = [p.strip() for p in line2.split(",")]
    try:
        if len(parts2) >= 3:
            # Modern: TT, ##A, ##D
            n_analog = int("".join(c for c in parts2[1] if c.isdigit()))
            n_digital = int("".join(c for c in parts2[2] if c.isdigit()))
        else:
            # Legacy: ##, ##
            n_analog = int(parts2[0])
            n_digital = int(parts2[1]) if len(parts2) > 1 else 0
    except (ValueError, IndexError) as exc:
        raise ValueError(f"CFG line 2 malformed: {line2!r}") from exc

    # Analog channel definitions
    analog_names: list[str] = []
    for _ in range(n_analog):
        ch_line = _next("analog channel")
        ch_parts = ch_line.split(",")
        # Field 2 (index 1) is ch_id
        analog_names.append(ch_parts[1].strip() if len(ch_parts) > 1 else "?")

    # Digital channel definitions
    digital_names: list[str] = []
    for _ in range(n_digital):
        ch_line = _next("digital channel")
        ch_parts = ch_line.split(",")
        digital_names.append(ch_parts[1].strip() if len(ch_parts) > 1 else "?")

    # lf (nominal frequency)
    try:
        nominal_freq = float(_next("nominal frequency (lf)"))
    except ValueError:
        nominal_freq = 50.0

    # nrates
    try:
        n_rates = int(_next("nrates"))
    except ValueError:
        n_rates = 1

    sampling_rates: list[float] = []
    samples_per_rate: list[int] = []
    for _ in range(n_rates):
        rate_line = _next("samp,endsamp")
        rate_parts = rate_line.split(",")
        try:
            sampling_rates.append(float(rate_parts[0].strip()))
            samples_per_rate.append(int(rate_parts[1].strip()) if len(rate_parts) > 1 else 0)
        except ValueError:
            sampling_rates.append(0.0)
            samples_per_rate.append(0)

    total_samples = samples_per_rate[-1] if samples_per_rate else 0

    # start_time  (DD/MM/YYYY,HH:MM:SS[.ffffff])
    start_time_str = _next("start_time")
    try:
        start_dt = _parse_cfg_timestamp(start_time_str)
    except ValueError:
        start_dt = datetime(1970, 1, 1)

    # trigger_time
    trig_time_str = _next("trigger_time")
    try:
        trigger_dt = _parse_cfg_timestamp(trig_time_str)
    except ValueError:
        trigger_dt = start_dt

    duration = (trigger_dt - start_dt).total_seconds()
    # Prefer sample-count-based duration if available
    if sampling_rates and sampling_rates[-1] > 0 and total_samples > 0:
        duration = total_samples / sampling_rates[-1]

    # ft (file type)
    try:
        dat_format = _next("file_type (ft)")
    except ValueError:
        dat_format = "BINARY"

    # timemult
    try:
        timemult = float(_next("timemult"))
    except ValueError:
        timemult = 1.0

    # DAT companion file
    dat_path = _locate_dat(cfg_path)
    dat_size = dat_path.stat().st_size if dat_path else None

    return ComtradeMetadata(
        cfg_path=str(cfg_path),
        station_name=station_name,
        rec_dev_id=rec_dev_id,
        rev_yr=rev_yr,
        n_analog=n_analog,
        n_digital=n_digital,
        analog_channel_names=analog_names,
        digital_channel_names=digital_names,
        nominal_frequency_hz=nominal_freq,
        sampling_rates_hz=sampling_rates,
        samples_per_rate=samples_per_rate,
        total_samples=total_samples,
        start_time=start_dt.isoformat(),
        trigger_time=trigger_dt.isoformat(),
        duration_seconds=round(duration, 6),
        dat_format=dat_format,
        timemult=timemult,
        dat_path=str(dat_path) if dat_path else None,
        dat_size_bytes=dat_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────


def format_text_summary(meta: ComtradeMetadata) -> str:
    """Return a human-readable text summary."""
    lines: list[str] = [
        "COMTRADE Summary",
        "-" * 40,
        f"CFG:            {meta.cfg_path}",
        f"Station:        {meta.station_name}",
        f"Recorder ID:    {meta.rec_dev_id}",
        f"Revision:       {meta.rev_yr}",
        f"File Type (ft): {meta.dat_format}",
        f"Time Multiplier:{meta.timemult}",
        "",
        f"Start Time:     {meta.start_time}",
        f"Trigger Time:   {meta.trigger_time}",
        f"Duration:       {meta.duration_seconds:.3f} s",
        "",
        f"Nominal Freq:   {meta.nominal_frequency_hz:.1f} Hz",
        "Sampling Rates:",
    ]
    for rate, npts in zip(meta.sampling_rates_hz, meta.samples_per_rate):
        lines.append(f"  - {rate:.0f} Hz  ({npts} samples)")
    lines.append(f"  Total samples: {meta.total_samples}")
    lines.append("")
    lines.append(f"Analog Channels ({meta.n_analog}):")
    for ch in meta.analog_channel_names:
        lines.append(f"  - {ch}")
    lines.append("")
    lines.append(f"Digital Channels ({meta.n_digital}):")
    for ch in meta.digital_channel_names:
        lines.append(f"  - {ch}")
    if not meta.digital_channel_names:
        lines.append("  (none)")
    lines.append("")
    if meta.dat_path:
        size_kb = (meta.dat_size_bytes or 0) / 1024
        lines.append(f"DAT file:       {meta.dat_path}  ({size_kb:.1f} KB)")
    else:
        lines.append("DAT file:       NOT FOUND")
    return "\n".join(lines)


def format_json_summary(meta: ComtradeMetadata) -> str:
    """Return JSON representation."""
    return json.dumps(asdict(meta), indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect a COMTRADE .cfg file without loading waveform data."
    )
    p.add_argument(
        "path",
        type=Path,
        help="Path to a .cfg file or directory containing one.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output JSON instead of human-readable text.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    try:
        cfg_path = _locate_cfg(args.path)
        meta = parse_cfg(cfg_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(format_json_summary(meta))
    else:
        print(format_text_summary(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
