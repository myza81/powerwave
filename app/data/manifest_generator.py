"""Manifest generator — serialises an EventAnalysisSession to YAML.

The produced YAML is compatible with manifest_loader.build_session_from_manifest.
Paths are written relative to the manifest file's parent directory so the
bundle is portable when moved as a folder.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from app.sessions import EventAnalysisSession


def _rel_path(file_path: str | None, manifest_dir: Path) -> str | None:
    """Return a path relative to manifest_dir, or None if unavailable."""
    if not file_path:
        return None
    try:
        return str(Path(file_path).resolve().relative_to(manifest_dir.resolve()))
    except ValueError:
        return str(Path(file_path).resolve())


def generate_manifest(
    session: "EventAnalysisSession",
    event_id: str,
    manifest_path: Path,
) -> None:
    """Write *session* as a YAML manifest to *manifest_path*.

    Args:
        session:       Active EventAnalysisSession to serialise.
        event_id:      Human-readable event identifier (used as the top-level key).
        manifest_path: Destination .yaml file path.
    """
    manifest_dir = manifest_path.parent
    sources_out = []

    for source in session.list_sources():
        rel = _rel_path(source.origin_path, manifest_dir)
        provider = source.provider_type.lower()

        # Build paths block
        if provider == "comtrade" and rel:
            cfg_path = Path(rel)
            dat_path = cfg_path.with_suffix(".dat")
            paths_block = {"cfg": str(cfg_path), "dat": str(dat_path)}
        elif provider in ("csv", "excel") and rel:
            key = "csv" if provider == "csv" else "excel"
            paths_block = {key: str(rel)}
        else:
            paths_block = {"path": rel} if rel else {}

        # Collect analog channels for this source
        analog_chs = [
            ch for ch in session.list_analog_channels(active_only=False)
            if ch.source_id == source.source_id
        ]
        digital_chs = [
            ch for ch in session.list_digital_channels(active_only=False)
            if ch.source_id == source.source_id
        ]

        # Channel list (names only — loader infers types)
        channel_names = [ch.channel_name for ch in analog_chs + digital_chs]

        # Column metadata block (only emit overrides the user actually made)
        columns_out = []
        for ch in analog_chs:
            col: dict = {"name": ch.channel_name}
            if ch.display_name != ch.channel_name:
                col["display_name"] = ch.display_name
            if ch.color_hex:
                col["color_hex"] = ch.color_hex
            if len(col) > 1:          # has at least one override beyond name
                columns_out.append(col)

        # Start time from the record's timing info
        try:
            start_dt: datetime | None = source.record.timing_info.start_time
            start_str = start_dt.isoformat() if start_dt else None
        except AttributeError:
            start_str = None

        src_dict: dict = {
            "source_id": source.source_id,
            "display_name": source.display_name,
            "type": provider,
            "paths": paths_block,
        }
        if start_str:
            src_dict["start_time"] = start_str
        if channel_names:
            src_dict["channels"] = channel_names
        if columns_out:
            src_dict["columns"] = columns_out

        sources_out.append(src_dict)

    # Alignment block
    sources = session.list_sources()
    offsets = {s.source_id: round(s.time_offset_s, 9) for s in sources}
    ref = next(
        (s.source_id for s in sources if s.time_offset_s == 0.0),
        sources[0].source_id if sources else None,
    )
    alignment_block: dict = {}
    if ref:
        alignment_block["reference_source"] = ref
    if any(v != 0.0 for v in offsets.values()):
        alignment_block["offsets_seconds"] = offsets

    manifest: dict = {
        "event_id": event_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources_out,
    }
    if alignment_block:
        manifest["alignment"] = alignment_block

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        yaml.dump(manifest, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
