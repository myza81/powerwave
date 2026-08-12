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

from app.data.manifest_loader import physical_source_type

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

        # Build paths block, keyed by the source's PHYSICAL format. An
        # Import-Wizard source keeps its own 'normalized_csv'/'normalized_excel'
        # type (that is its reload recipe -- see manifest_loader), but its file
        # still lives under the 'csv'/'excel' key so every reader locates it the
        # same way.
        physical = physical_source_type(provider)
        if physical == "comtrade" and rel:
            cfg_path = Path(rel)
            dat_path = cfg_path.with_suffix(".dat")
            paths_block = {"cfg": str(cfg_path), "dat": str(dat_path)}
        elif physical in ("csv", "excel") and rel:
            paths_block = {physical: str(rel)}
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

    # ── Alignment block ───────────────────────────────────────────────────────
    # Stage 3: the session's coordinate geometry is persisted explicitly so a
    # reload reproduces it exactly instead of re-deriving it.
    #
    # 'absolute_time_origin' is the authoritative reference: it is the
    # wall-clock instant session x = 0 denotes, so restored offsets keep their
    # absolute meaning. 'reference_source' is retained for compatibility with
    # readers that already consume it (app.data.review_summary), but it is NOT
    # the origin and is no longer what defines the coordinate reference —
    # inferring the reference from "whichever source has offset 0.0" is
    # ambiguous (several sources can be at zero, and Set-as-Reference can
    # rebase onto a source of any method).
    #
    # 'offsets_seconds' is now written whenever the session has sources, not
    # only when some offset is non-zero: an all-zero geometry is still a real,
    # deliberate geometry that must survive reload rather than be re-derived.
    #
    # 'methods' / 'confidences' / 'notes' preserve alignment provenance, so a
    # manually corrected or trigger-aligned source is not silently re-derived
    # as absolute-timestamp on reload, and so the UI never claims a method
    # whose supporting metadata was lost.
    sources = session.list_sources()
    alignment_block: dict = {}
    if sources:
        offsets = {s.source_id: round(s.time_offset_s, 9) for s in sources}
        methods = {s.source_id: s.alignment_method for s in sources}
        confidences = {
            s.source_id: float(s.alignment_confidence)
            for s in sources
            if s.alignment_confidence is not None
        }
        notes = {}
        for s in sources:
            note = session.get_alignment_notes(s.source_id)
            if note:
                notes[s.source_id] = note

        origin = getattr(session, "absolute_time_origin", None)
        if origin is not None:
            # Same convention as the per-source start_time above: an ISO 8601
            # string, microseconds intact, no timezone invented for a naive
            # origin. isoformat() omits ".000000" for a whole second, which the
            # loader's format list also accepts.
            alignment_block["absolute_time_origin"] = origin.isoformat()

        ref = next(
            (s.source_id for s in sources if s.time_offset_s == 0.0),
            sources[0].source_id,
        )
        alignment_block["reference_source"] = ref
        alignment_block["offsets_seconds"] = offsets
        alignment_block["methods"] = methods
        if confidences:
            alignment_block["confidences"] = confidences
        if notes:
            alignment_block["notes"] = notes

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
