"""Event manifest builder — generates a YAML summary of a multi-source disturbance event.

Internally calls inspect_comtrade and inspect_csv_timeseries, then writes a
repository-relative YAML manifest that can be loaded by the D3 multi-source
session workflow.

Usage (from project root):
    python tools/build_event_manifest.py \\
        --event-id pulu_20260306 \\
        --comtrade samples/comtrade/pulu_20260306/ \\
        --csv samples/csv/pulu_20260306.csv \\
        --output samples/manifests/pulu_20260306.yaml

    # Print to stdout instead of file:
    python tools/build_event_manifest.py --event-id pulu_20260306 \\
        --comtrade samples/comtrade/pulu_20260306/
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Manifest data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SourceEntry:
    source_id: str
    source_type: str              # "comtrade", "csv", "excel"
    paths: dict[str, str]         # {"cfg": "...", "dat": "..."} or {"csv": "..."}
    start_time: str | None
    trigger_time: str | None      # None for CSV sources
    sampling_rates_hz: list[float]
    channel_names: list[str]      # analog + data columns
    notes: list[str] = field(default_factory=list)
    # Extended metadata (D4)
    voltage_reference: str | None = None       # "phase_ground" | "phase_phase" | None
    column_classifications: list[dict] = field(default_factory=list)


@dataclass
class AlignmentInfo:
    reference_source: str
    reference_start: str | None
    offsets: dict[str, float]     # source_id → offset in seconds


@dataclass
class EventManifest:
    event_id: str
    generated_at: str             # ISO 8601
    sources: list[SourceEntry]
    alignment: AlignmentInfo


# ─────────────────────────────────────────────────────────────────────────────
# Minimal YAML emitter (no PyYAML dependency)
# ─────────────────────────────────────────────────────────────────────────────


def _yaml_str(value: str) -> str:
    """Quote a string value if it contains YAML-special characters."""
    needs_quotes = any(c in value for c in (':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`', '"', "'", '\n'))
    if needs_quotes or not value:
        return f'"{value}"'
    return value


def _to_yaml(obj: Any, indent: int = 0) -> str:
    """Recursively emit a simple Python object as YAML text."""
    pad = "  " * indent
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, str):
        return _yaml_str(obj)
    if isinstance(obj, list):
        if not obj:
            return "[]"
        lines: list[str] = []
        for item in obj:
            if isinstance(item, dict):
                # Render dict at root indent so extra padding is not baked in,
                # then offset each line by "  " (aligned with content after "- ").
                item_yaml = _to_yaml(item, 0)
                item_lines = item_yaml.splitlines()
                first = item_lines[0] if item_lines else ""
                lines.append(f"{pad}- {first}")
                for rl in item_lines[1:]:
                    lines.append(f"{pad}  {rl}")
            else:
                lines.append(f"{pad}- {_to_yaml(item, indent + 1)}")
        return "\n".join(lines)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            v_yaml = _to_yaml(v, indent + 1)
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines.append(v_yaml)
            else:
                lines.append(f"{pad}{k}: {v_yaml}")
        return "\n".join(lines)
    return str(obj)


def _manifest_to_yaml(manifest: EventManifest) -> str:
    """Serialize an EventManifest to a YAML string."""
    sources_list: list[dict] = []
    for src in manifest.sources:
        entry: dict[str, Any] = {
            "source_id": src.source_id,
            "type": src.source_type,
            "paths": src.paths,
        }
        if src.voltage_reference:
            entry["voltage_reference"] = src.voltage_reference
        if src.start_time:
            entry["start_time"] = src.start_time
        if src.trigger_time:
            entry["trigger_time"] = src.trigger_time
        if src.sampling_rates_hz:
            entry["sampling_rates_hz"] = src.sampling_rates_hz
        if src.channel_names:
            entry["channels"] = src.channel_names
        if src.column_classifications:
            entry["columns"] = src.column_classifications
        if src.notes:
            entry["notes"] = src.notes
        sources_list.append(entry)

    offsets_dict: dict[str, Any] = {k: round(v, 6) for k, v in manifest.alignment.offsets.items()}

    obj: dict[str, Any] = {
        "event_id": manifest.event_id,
        "generated_at": manifest.generated_at,
        "sources": sources_list,
        "alignment": {
            "reference_source": manifest.alignment.reference_source,
            "reference_start": manifest.alignment.reference_start,
            "offsets_seconds": offsets_dict,
        },
    }
    return _to_yaml(obj) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────


def _repo_relative(path: Path, root: Path) -> str:
    """Return path relative to root, using forward slashes."""
    try:
        rel = path.resolve().relative_to(root.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# ─────────────────────────────────────────────────────────────────────────────
# Alignment computation
# ─────────────────────────────────────────────────────────────────────────────


def _compute_alignment(sources: list[SourceEntry]) -> AlignmentInfo:
    """Compute time offsets relative to the earliest source start."""
    times: list[tuple[str, datetime]] = []
    for src in sources:
        if src.start_time:
            try:
                times.append((src.source_id, datetime.fromisoformat(src.start_time)))
            except ValueError:
                pass

    if not times:
        return AlignmentInfo(
            reference_source=sources[0].source_id if sources else "",
            reference_start=None,
            offsets={src.source_id: 0.0 for src in sources},
        )

    ref_id, ref_dt = min(times, key=lambda t: t[1])
    offsets: dict[str, float] = {}
    for src_id, dt in times:
        offsets[src_id] = (dt - ref_dt).total_seconds()
    # Sources with no timestamp get 0.0
    for src in sources:
        if src.source_id not in offsets:
            offsets[src.source_id] = 0.0

    return AlignmentInfo(
        reference_source=ref_id,
        reference_start=ref_dt.isoformat(),
        offsets=offsets,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Source builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_comtrade_entry(
    source_id: str,
    path: Path,
    root: Path,
) -> SourceEntry:
    """Inspect a COMTRADE directory/file and build a SourceEntry."""
    # Import here to avoid circular import if this module is imported from tools/
    from inspect_comtrade import _locate_cfg, parse_cfg

    cfg_path = _locate_cfg(path)
    meta = parse_cfg(cfg_path)

    paths: dict[str, str] = {"cfg": _repo_relative(cfg_path, root)}
    if meta.dat_path:
        paths["dat"] = _repo_relative(Path(meta.dat_path), root)

    notes: list[str] = []
    if meta.dat_path is None:
        notes.append("DAT file not found alongside CFG")

    return SourceEntry(
        source_id=source_id,
        source_type="comtrade",
        paths=paths,
        start_time=meta.start_time,
        trigger_time=meta.trigger_time,
        sampling_rates_hz=meta.sampling_rates_hz,
        channel_names=meta.analog_channel_names + meta.digital_channel_names,
        notes=notes,
    )


def _build_csv_entry(
    source_id: str,
    path: Path,
    root: Path,
) -> SourceEntry:
    """Inspect a CSV/Excel file and build a SourceEntry with column classification."""
    from inspect_csv_timeseries import inspect_file

    meta = inspect_file(path)
    file_type = meta.file_type

    paths: dict[str, str] = {file_type: _repo_relative(path, root)}

    start_time: str | None = None
    notes: list[str] = []

    if meta.timestamp_info:
        ti = meta.timestamp_info
        start_time = ti.start_time
        if ti.ambiguity_detected:
            notes.append(
                f"Date format ambiguous for column '{ti.column_name}'. "
                "Verify M/D/YYYY vs D/M/YYYY interpretation before use."
            )
        notes.extend(ti.notes)

    sampling_rates: list[float] = []
    if meta.timestamp_info and meta.timestamp_info.nominal_interval_seconds:
        interval = meta.timestamp_info.nominal_interval_seconds
        if interval > 0:
            sampling_rates = [round(1.0 / interval, 6)]

    # Column classification — requires app/ on sys.path
    column_classifications: list[dict] = []
    try:
        import pandas as pd
        from app.data.column_classifier import classify_csv_columns
        ts_col = meta.timestamp_column
        df = pd.read_csv(path, nrows=100) if file_type == "csv" else pd.read_excel(path, nrows=100)
        classifications = classify_csv_columns(df, timestamp_column=ts_col)
        for col_name in meta.data_columns:
            if col_name in classifications:
                cl = classifications[col_name]
                col_dict: dict = {
                    "name": col_name,
                    "display_group": cl.display_group,
                    "confidence": round(cl.confidence, 3),
                    "inferred_from": cl.inferred_from,
                    "requires_user_confirmation": cl.requires_user_confirmation,
                }
                if cl.signal_type is not None:
                    col_dict["signal_type"] = cl.signal_type
                if cl.unit is not None:
                    col_dict["unit"] = cl.unit
                if cl.notes:
                    col_dict["notes"] = cl.notes
                column_classifications.append(col_dict)
    except ImportError:
        pass  # app/ not on sys.path — skip classification

    return SourceEntry(
        source_id=source_id,
        source_type=file_type,
        paths=paths,
        start_time=start_time,
        trigger_time=None,
        sampling_rates_hz=sampling_rates,
        channel_names=meta.data_columns,
        notes=notes,
        column_classifications=column_classifications,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def build_manifest(
    event_id: str,
    comtrade_paths: list[Path],
    csv_paths: list[Path],
    excel_paths: list[Path],
    root: Path | None = None,
) -> EventManifest:
    """Build an EventManifest from a collection of source paths.

    Args:
        event_id:       Identifier for this event (e.g. "pulu_20260306").
        comtrade_paths: List of COMTRADE .cfg paths or directories.
        csv_paths:      List of CSV file paths.
        excel_paths:    List of Excel file paths.
        root:           Project root for relative path computation. Defaults
                        to the current working directory.

    Returns:
        EventManifest — call _manifest_to_yaml() to serialise.
    """
    if root is None:
        root = Path.cwd()

    sources: list[SourceEntry] = []

    for i, p in enumerate(comtrade_paths):
        sid = f"comtrade_{i + 1}" if len(comtrade_paths) > 1 else "comtrade_main"
        sources.append(_build_comtrade_entry(sid, p, root))

    for i, p in enumerate(csv_paths):
        sid = f"csv_{i + 1}" if len(csv_paths) > 1 else "csv_ops"
        sources.append(_build_csv_entry(sid, p, root))

    for i, p in enumerate(excel_paths):
        sid = f"excel_{i + 1}" if len(excel_paths) > 1 else "excel_ops"
        sources.append(_build_csv_entry(sid, p, root))

    alignment = _compute_alignment(sources)

    return EventManifest(
        event_id=event_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        sources=sources,
        alignment=alignment,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a YAML event manifest from COMTRADE + CSV/Excel sources."
    )
    p.add_argument("--event-id", required=True, help="Event identifier (e.g. pulu_20260306).")
    p.add_argument(
        "--comtrade",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="COMTRADE directory or .cfg file (repeatable).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="CSV file path (repeatable).",
    )
    p.add_argument(
        "--excel",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Excel file path (repeatable).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output .yaml path. Omit to print to stdout.",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Project root for relative paths (default: current directory).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    root = args.root or Path.cwd()

    # Ensure tools/ and project root are on sys.path for inner imports
    tools_dir = Path(__file__).parent
    project_root = tools_dir.parent
    for p in (str(tools_dir), str(project_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        manifest = build_manifest(
            event_id=args.event_id,
            comtrade_paths=args.comtrade,
            csv_paths=args.csv,
            excel_paths=args.excel,
            root=root,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    yaml_text = _manifest_to_yaml(manifest)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml_text, encoding="utf-8")
        print(f"Manifest written to: {args.output}", file=sys.stderr)
    else:
        print(yaml_text)

    return 0


if __name__ == "__main__":
    # Ensure tools/ and project root are importable
    _tools = Path(__file__).parent
    _root = _tools.parent
    for _p in (str(_tools), str(_root)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    sys.exit(main())
