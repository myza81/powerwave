"""Source fingerprinting for Powerwave's intelligence layer.

A SourceFingerprint is a lightweight, deterministic descriptor of a
recurring data source pattern. Fingerprints scope mapping rules to
specific sources without hardcoding site logic into providers.

Usage::

    fp = build_fingerprint_from_columns(
        ["System Demand", "Frequency", "Tie-Line"],
        source_type="csv",
    )
"""
from __future__ import annotations

import hashlib
import json

from app.data.intelligence.models import SourceFingerprint


def _column_signature(column_names: list[str]) -> str:
    """Return a 16-hex-char SHA-256 prefix of sorted, normalised column names.

    Deterministic: same set of names in any order → same signature.
    Empty or whitespace-only names are ignored.
    """
    normalised = sorted(n.strip().lower() for n in column_names if n.strip())
    payload = json.dumps(normalised, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_fingerprint_from_columns(
    column_names: list[str],
    source_type: str | None = None,
    station: str | None = None,
    source_kind: str | None = None,
    vendor: str | None = None,
) -> SourceFingerprint:
    """Build a SourceFingerprint from column names and optional source metadata.

    column_signature is set only when column_names is non-empty.
    """
    sig = _column_signature(column_names) if column_names else None
    return SourceFingerprint(
        vendor=vendor,
        station=station,
        export_type=source_type,
        source_kind=source_kind,
        column_signature=sig,
    )


def build_fingerprint_from_record(
    record,                             # DisturbanceRecord — avoid circular import
    source_type: str | None = None,
    source_kind: str | None = None,
) -> SourceFingerprint:
    """Build a SourceFingerprint from a DisturbanceRecord's analog channels."""
    names = [ch.name for ch in record.analog_channels]
    station = getattr(record.metadata, "station_name", None) or None
    return build_fingerprint_from_columns(
        names,
        source_type=source_type,
        station=station,
        source_kind=source_kind,
    )


def fingerprints_match(a: SourceFingerprint, b: SourceFingerprint) -> bool:
    """Return True if all non-None fields present in both fingerprints agree.

    Conflict only when both sides carry a non-None value that differs.
    A fingerprint with all-None fields matches everything (wildcard).
    """
    for field in ("vendor", "station", "export_type", "source_kind", "column_signature"):
        av = getattr(a, field)
        bv = getattr(b, field)
        if av is not None and bv is not None and av != bv:
            return False
    return True
