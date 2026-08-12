"""Manifest-based multi-source session loader for Powerwave.

Loads a YAML event manifest and builds a MultiSourceSession, resolving
repository-relative paths, applying column classification metadata, and
preserving COMTRADE voltage reference semantics.

The manifest is the repeatable event recipe: source paths, timestamp
interpretation, alignment assumptions, and column classification are all
captured there. Original DisturbanceRecord objects are never mutated.

Usage::

    from app.data.manifest_loader import build_session_from_manifest
    from pathlib import Path

    session = build_session_from_manifest(
        Path("samples/manifests/pulu_20260306.yaml")
    )
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from app.data.multi_source_session import (
    MultiSourceSession,
    SessionAlignmentState,
    SourceRecord,
)
from app.data.signal_metadata import SignalMetadata


# ─────────────────────────────────────────────────────────────────────────────
# COMTRADE channel type inference
# ─────────────────────────────────────────────────────────────────────────────


def _infer_comtrade_channel_type(channel_name: str) -> str | None:
    """Return 'voltage' or 'current' inferred from COMTRADE naming conventions.

    Handles patterns like: "KPDN1 VR", "SGT1 VB_HV", "KPDN2 IB", "KPDN1 IN".
    Strips _HV / _LV suffixes then checks the first character of the last word.
    Returns None when the pattern is unrecognised.
    """
    last = channel_name.split()[-1] if " " in channel_name else channel_name
    base = last.split("_")[0].upper()
    if base and base[0] == "V":
        return "voltage"
    if base and base[0] == "I":
        return "current"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Manifest I/O
# ─────────────────────────────────────────────────────────────────────────────


def load_manifest(path: Path) -> dict:
    """Load a YAML manifest file and return the raw parsed dict.

    Raises:
        FileNotFoundError: Manifest file does not exist.
        ValueError:        Manifest is not a YAML mapping or missing required fields.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a YAML mapping: {path}")
    if "event_id" not in data:
        raise ValueError(f"Manifest missing required field 'event_id': {path}")
    if "sources" not in data:
        raise ValueError(f"Manifest missing required field 'sources': {path}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Source-type vocabulary
# ─────────────────────────────────────────────────────────────────────────────
#
# ``sources[].type`` answers one question: HOW IS THIS SOURCE RELOADED?
#
# For 'comtrade' / 'csv' / 'excel' that means the matching provider reads the
# file directly. The Import Wizard is a second, distinct way to read a CSV or
# XLSX: it applies timestamp detection, semantic classification and canonical
# channel renaming, and produces a materially different DisturbanceRecord from
# the same bytes -- e.g. the same column arrives as
# ('mw_dc_total_demand', 'MW', 'mw') from the Wizard but
# ('DC Total Demand', 'MW', 'active_power') from ExcelProvider. Reloading a
# Wizard-imported source through the raw provider would therefore silently
# change channel identity, so 'normalized_csv' / 'normalized_excel' are
# preserved as their own reload recipes rather than flattened to the physical
# format. Both spellings resolve to the same physical format for locating the
# file on disk.
_NORMALIZED_SOURCE_TYPES: dict[str, str] = {
    "normalized_csv": "csv",
    "normalized_excel": "excel",
}

SUPPORTED_SOURCE_TYPES: frozenset[str] = frozenset(
    {"comtrade", "csv", "excel"} | set(_NORMALIZED_SOURCE_TYPES)
)


def physical_source_type(source_type: str) -> str:
    """Return the on-disk format for a manifest source type.

    'normalized_excel' -> 'excel', 'normalized_csv' -> 'csv'; every other
    value is already a physical format and is returned unchanged.
    """
    key = str(source_type or "").lower()
    return _NORMALIZED_SOURCE_TYPES.get(key, key)


def is_normalized_source_type(source_type: str) -> bool:
    """True when the source must be reloaded through the Import Wizard pipeline."""
    return str(source_type or "").lower() in _NORMALIZED_SOURCE_TYPES


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_path(rel_str: str, root: Path) -> Path:
    """Resolve a manifest-relative path against the project root."""
    p = Path(str(rel_str))
    if p.is_absolute():
        return p
    return root / p


def _get_source_file_path(
    src_def: dict,
    source_type: str,
    root: Path,
    source_id: str,
) -> Path:
    """Extract and resolve the primary loadable file path from a source definition.

    Supports both ``paths`` (dict) and ``path`` (string) keys in the manifest.
    For COMTRADE: resolves ``paths.cfg``.
    For CSV/Excel (including their normalized Import-Wizard variants):
    resolves ``paths.csv``, ``paths.excel``, ``paths.path``, or a top-level
    ``path``. ``paths.path`` is accepted because manifests written before the
    normalized types were reloadable stored the file under that generic key.
    """
    paths = src_def.get("paths", {})
    path_str = src_def.get("path")
    physical = physical_source_type(source_type)

    if physical == "comtrade":
        if isinstance(paths, dict) and "cfg" in paths:
            return _resolve_path(str(paths["cfg"]), root)
        if path_str:
            return _resolve_path(str(path_str), root)

    elif physical in ("csv", "excel"):
        if isinstance(paths, dict):
            for key in ("csv", "excel", "xlsx", "path"):
                if key in paths:
                    return _resolve_path(str(paths[key]), root)
        if path_str:
            return _resolve_path(str(path_str), root)

    raise ValueError(
        f"Cannot determine file path for source '{source_id}' "
        f"(type={source_type!r}). Expected 'paths.cfg', 'paths.csv', or 'path'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SignalMetadata construction
# ─────────────────────────────────────────────────────────────────────────────


def _build_comtrade_signal_metadata(
    analog_channels: list,  # list[AnalogChannel]
    voltage_reference: str | None,
    manifest_columns: list[dict],
) -> dict[str, SignalMetadata]:
    """Build SignalMetadata for COMTRADE analog channels.

    Applies name-based type inference (voltage vs current) and uses
    voltage_reference from the manifest for phase annotation.
    Manifest ``columns`` entries take precedence when present.
    """
    col_map: dict[str, dict] = {
        c["name"]: c for c in manifest_columns if isinstance(c, dict) and "name" in c
    }
    result: dict[str, SignalMetadata] = {}
    for ch in analog_channels:
        name = ch.name
        col = col_map.get(name, {})
        ch_type = _infer_comtrade_channel_type(name)

        result[name] = SignalMetadata(
            name=name,
            unit=col.get("unit") or ch.unit,
            source="comtrade",
            electrical_type=col.get("electrical_type") or ch_type,
            phase_reference=col.get("phase_reference") or (
                voltage_reference if ch_type == "voltage" else None
            ),
            confidence=col.get("confidence"),
            inferred_from=col.get("inferred_from"),
            requires_user_confirmation=bool(col.get("requires_user_confirmation", False)),
        )
    return result


def _build_csv_signal_metadata(
    analog_channels: list,  # list[AnalogChannel]
    manifest_columns: list[dict],
) -> dict[str, SignalMetadata]:
    """Build SignalMetadata for CSV/Excel analog channels from manifest columns."""
    col_map: dict[str, dict] = {
        c["name"]: c for c in manifest_columns if isinstance(c, dict) and "name" in c
    }
    result: dict[str, SignalMetadata] = {}
    for ch in analog_channels:
        name = ch.name
        col = col_map.get(name, {})
        result[name] = SignalMetadata(
            name=name,
            unit=col.get("unit") or ch.unit,
            source="csv",
            signal_type=col.get("signal_type"),
            display_group=col.get("display_group"),
            confidence=col.get("confidence"),
            inferred_from=col.get("inferred_from"),
            requires_user_confirmation=bool(col.get("requires_user_confirmation", True)),
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp parsing
# ─────────────────────────────────────────────────────────────────────────────

_TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO 8601 or space-separated datetime string, returning None on failure.

    Accepts a datetime directly as well: an unquoted ISO scalar in hand-written
    YAML is deserialised by PyYAML as a datetime rather than a str, and both
    spellings must round-trip identically. Timezone information is neither
    invented nor stripped — a naive origin stays naive.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Alignment block (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────


def _coerce_float_map(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue          # ignore an unparseable entry rather than failing the load
    return out


def _coerce_str_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if v is not None}


def parse_alignment_state(manifest_data: dict) -> SessionAlignmentState:
    """Read a manifest's ``alignment`` block into a SessionAlignmentState.

    Every field is optional. A manifest with no alignment block, or with only
    the pre-Stage-3 ``offsets_seconds``/``reference_source`` keys, yields a
    state that still loads — see SessionAlignmentState.has_trustworthy_origin
    for how the caller then treats it. Malformed individual entries are skipped
    rather than aborting the load, because alignment is display geometry and a
    bad entry must not make an otherwise valid event unopenable.
    """
    alignment = manifest_data.get("alignment") or {}
    if not isinstance(alignment, dict):
        return SessionAlignmentState()

    reference = alignment.get("reference_source")
    return SessionAlignmentState(
        absolute_time_origin=_parse_timestamp(alignment.get("absolute_time_origin")),
        reference_source=str(reference) if reference is not None else None,
        offsets_seconds=_coerce_float_map(alignment.get("offsets_seconds")),
        methods=_coerce_str_map(alignment.get("methods")),
        confidences=_coerce_float_map(alignment.get("confidences")),
        notes=_coerce_str_map(alignment.get("notes")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Import-Wizard (normalized) source reload
# ─────────────────────────────────────────────────────────────────────────────


def _load_normalized_record(file_path: Path, source_type: str, source_id: str):
    """Reload an Import-Wizard source by re-running the Wizard's own backend.

    A 'normalized_csv'/'normalized_excel' source was produced by
    run_import_pipeline(), not by CsvProvider/ExcelProvider, and the two
    disagree about channel identity for the same file (canonical name and
    parameter_type vs. raw header and classifier type). Re-running the same
    pipeline is what reproduces the record that was saved: the pipeline is
    deterministic for a given file, so timing, time array, channel names,
    units and parameter types all come back identical.

    Limitation (documented, not worked around): only decisions the pipeline
    re-derives automatically are reproduced. A manual override the analyst
    made inside the Wizard UI -- an explicit timestamp format, a column
    exclusion -- is not persisted in the manifest today, so it is re-derived
    by auto-detection instead. Sources whose import needed manual correction
    should be re-imported through the Wizard rather than trusted to reload
    identically.
    """
    from app.import_wizard.import_pipeline import run_import_pipeline

    result = run_import_pipeline(
        str(file_path), provider_type=physical_source_type(source_type)
    )
    if not result.success or result.record is None:
        messages = "; ".join(
            m.message for m in result.validation_messages
            if getattr(m, "severity", None) is not None
            and str(getattr(m.severity, "name", "")).upper() == "ERROR"
        )
        raise ValueError(
            f"Failed to reload Import-Wizard source '{source_id}' from {file_path}"
            + (f": {messages}" if messages else "")
        )
    return result.record


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def build_session_from_manifest(
    manifest_path: Path,
    root: Path | None = None,
    provider_manager=None,
) -> MultiSourceSession:
    """Load a YAML manifest and build a MultiSourceSession.

    Args:
        manifest_path:    Path to the YAML manifest file (absolute or relative
                          to ``root``).
        root:             Project root for resolving manifest-relative source
                          paths. Defaults to ``Path.cwd()``.
        provider_manager: Optional ``ProviderManager`` for loading files.
                          If None, a default manager with all standard providers
                          (ComtradeProvider, CsvProvider, ExcelProvider) is used.

    Returns:
        ``MultiSourceSession`` with all sources loaded and SignalMetadata applied.

    Raises:
        FileNotFoundError: Manifest or a source file cannot be found.
        ValueError:        Manifest missing required fields, unknown source type,
                           or a source fails to load.
    """
    if root is None:
        root = Path.cwd()
    root = root.resolve()

    if provider_manager is None:
        from app.providers import (
            ComtradeProvider,
            CsvProvider,
            ExcelProvider,
            ProviderManager,
        )
        pm = ProviderManager()
        pm.register_provider(ComtradeProvider())
        pm.register_provider(CsvProvider())
        pm.register_provider(ExcelProvider())
    else:
        pm = provider_manager

    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    data = load_manifest(manifest_path)
    session = MultiSourceSession(alignment=parse_alignment_state(data))

    for src_def in data.get("sources", []):
        source_id = str(src_def.get("source_id", "unknown"))
        source_type = str(src_def.get("type", "")).lower()
        manifest_columns: list[dict] = src_def.get("columns", []) or []

        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(
                f"Unknown source type {source_type!r} for source '{source_id}'. "
                f"Expected one of {', '.join(sorted(SUPPORTED_SOURCE_TYPES))}."
            )

        file_path = _get_source_file_path(src_def, source_type, root, source_id)

        try:
            if is_normalized_source_type(source_type):
                record = _load_normalized_record(file_path, source_type, source_id)
            else:
                record = pm.load(file_path)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Failed to load source '{source_id}' from {file_path}: {exc}"
            ) from exc

        if physical_source_type(source_type) == "comtrade":
            voltage_reference = src_def.get("voltage_reference")
            signal_metadata = _build_comtrade_signal_metadata(
                list(record.analog_channels), voltage_reference, manifest_columns
            )
        else:
            signal_metadata = _build_csv_signal_metadata(
                list(record.analog_channels), manifest_columns
            )

        start_time = _parse_timestamp(src_def.get("start_time"))

        source_record = SourceRecord(
            source_id=source_id,
            provider_type=source_type,
            record=record,
            signal_metadata=signal_metadata,
            original_start_time=start_time,
            sampling_rates=list(record.sampling_info.sampling_rates),
        )
        session.add_source(source_record)

    return session
