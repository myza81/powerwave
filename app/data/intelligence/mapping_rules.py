"""Persistent column-to-signal-type mapping rule management.

Rules are stored in config/column_mapping_rules.yaml.
This module is stateless: callers (IntelligenceManager) load and cache rules.
The built-in column_classifier.py is unchanged; rules layer on top of it.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from app.data.column_classifier import ColumnClassification
from app.data.intelligence.fingerprints import fingerprints_match
from app.data.intelligence.models import (
    ConfidencePromotion,
    MappingRule,
    SourceFingerprint,
)

DEFAULT_MAPPING_RULES_PATH = Path("config") / "column_mapping_rules.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# Built-in synonym / variant library (Phase D4.3)
# ─────────────────────────────────────────────────────────────────────────────
# Maps normalised variant strings → (signal_type, unit, display_group, confidence)
# These encode domain knowledge about common abbreviations and inconsistent labels
# found in real operational CSV exports.
#
# Lower confidence than "name_exact" rules so operator-confirmed rules take priority.
#
# match_type "fuzzy" in MappingRule triggers lookup here.

_SYNONYM_MAP: dict[str, tuple[str, str, str, float]] = {
    # Frequency variants
    "freq":              ("frequency",    "Hz",   "frequency", 0.90),
    "frequency":         ("frequency",    "Hz",   "frequency", 0.95),
    "sys freq":          ("frequency",    "Hz",   "frequency", 0.90),
    "system frequency":  ("frequency",    "Hz",   "frequency", 0.95),
    "sysfreq":           ("frequency",    "Hz",   "frequency", 0.88),
    "f":                 ("frequency",    "Hz",   "frequency", 0.72),
    "hz":                ("frequency",    "Hz",   "frequency", 0.88),
    # Active power / demand variants
    "system demand":     ("active_power", "MW",   "power",     0.88),
    "sys demand":        ("active_power", "MW",   "power",     0.88),
    "sysdem":            ("active_power", "MW",   "power",     0.85),
    "total demand":      ("active_power", "MW",   "power",     0.88),
    "total load":        ("active_power", "MW",   "power",     0.85),
    "load mw":           ("active_power", "MW",   "power",     0.90),
    "mw":                ("active_power", "MW",   "power",     0.95),
    "active power":      ("active_power", "MW",   "power",     0.90),
    "real power":        ("active_power", "MW",   "power",     0.90),
    "p":                 ("active_power", "MW",   "power",     0.72),
    # Tie-line / interchange
    "tie-line":          ("active_power", "MW",   "power",     0.72),
    "tieline":           ("active_power", "MW",   "power",     0.70),
    "tie line":          ("active_power", "MW",   "power",     0.72),
    "interchange":       ("active_power", "MW",   "power",     0.72),
    "export":            ("active_power", "MW",   "power",     0.68),
    "import":            ("active_power", "MW",   "power",     0.68),
    # Reactive power
    "reactive power":    ("reactive_power", "MVAr", "power",  0.90),
    "mvar":              ("reactive_power", "MVAr", "power",  0.95),
    "q":                 ("reactive_power", "MVAr", "power",  0.72),
    # Voltage
    "voltage":           ("voltage_rms",  "kV",   "voltage_rms", 0.80),
    "kv":                ("voltage_rms",  "kV",   "voltage_rms", 0.90),
    "bus voltage":       ("voltage_rms",  "pu",   "voltage_rms", 0.88),
    # ROCOF
    "rocof":             ("rocof",         "Hz/s", "rocof",     0.95),
    "df/dt":             ("rocof",         "Hz/s", "rocof",     0.92),
    "dfdt":              ("rocof",         "Hz/s", "rocof",     0.92),
}


# ─────────────────────────────────────────────────────────────────────────────
# Load / save
# ─────────────────────────────────────────────────────────────────────────────


def load_mapping_rules(path: Path = DEFAULT_MAPPING_RULES_PATH) -> list[MappingRule]:
    """Load mapping rules from a YAML file.

    Returns an empty list when the file does not exist — graceful no-config
    behaviour is intentional so the system works out-of-the-box.
    Raises ValueError for malformed YAML or missing required fields.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed mapping rules YAML at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Mapping rules file must be a YAML mapping: {path}")

    rules: list[MappingRule] = []
    for i, entry in enumerate(data.get("rules") or []):
        if not isinstance(entry, dict):
            raise ValueError(f"Mapping rule #{i} must be a YAML mapping")
        if "match_pattern" not in entry:
            raise ValueError(f"Mapping rule #{i} missing required field 'match_pattern'")
        if "signal_type" not in entry:
            raise ValueError(f"Mapping rule #{i} missing required field 'signal_type'")
        fp = _parse_fingerprint(entry.get("source_fingerprint"))
        rules.append(MappingRule(
            match_pattern=str(entry["match_pattern"]).strip().lower(),
            match_type=str(entry.get("match_type", "exact")).lower(),
            signal_type=str(entry["signal_type"]),
            unit=entry.get("unit"),
            display_group=str(entry.get("display_group", "other")),
            confidence=float(entry.get("confidence", 0.95)),
            confirmed_by_operator=bool(entry.get("confirmed_by_operator", False)),
            source_fingerprint=fp,
            notes=entry.get("notes"),
        ))
    return rules


def _parse_fingerprint(fp_data: object) -> SourceFingerprint | None:
    if not fp_data or not isinstance(fp_data, dict):
        return None
    return SourceFingerprint(
        vendor=fp_data.get("vendor"),
        station=fp_data.get("station"),
        export_type=fp_data.get("export_type"),
        source_kind=fp_data.get("source_kind"),
        column_signature=fp_data.get("column_signature"),
    )


def save_mapping_rules(
    rules: list[MappingRule],
    path: Path = DEFAULT_MAPPING_RULES_PATH,
) -> None:
    """Persist mapping rules to a YAML file.

    Creates parent directories as needed. Overwrites the existing file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"rules": [_rule_to_dict(r) for r in rules]}
    text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def _rule_to_dict(rule: MappingRule) -> dict:
    d: dict = {
        "match_pattern": rule.match_pattern,
        "match_type": rule.match_type,
        "signal_type": rule.signal_type,
        "display_group": rule.display_group,
        "confidence": rule.confidence,
        "confirmed_by_operator": rule.confirmed_by_operator,
    }
    if rule.unit is not None:
        d["unit"] = rule.unit
    if rule.source_fingerprint is not None:
        fp_dict = {
            k: getattr(rule.source_fingerprint, k)
            for k in ("vendor", "station", "export_type", "source_kind", "column_signature")
            if getattr(rule.source_fingerprint, k) is not None
        }
        if fp_dict:
            d["source_fingerprint"] = fp_dict
    if rule.notes is not None:
        d["notes"] = rule.notes
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Rule matching
# ─────────────────────────────────────────────────────────────────────────────


def find_matching_rule(
    column_name: str,
    rules: list[MappingRule],
    fingerprint: SourceFingerprint | None = None,
) -> MappingRule | None:
    """Return the first rule matching column_name, with fingerprint-scoped rules first.

    Priority:
      1. Fingerprint-specific rules (source_fingerprint matches the given fingerprint)
      2. Global rules (source_fingerprint is None)
    Within each group, list order determines precedence (first match wins).

    Matching is always case-insensitive — column_name is normalised internally.
    Returns None when no rule matches.
    """
    norm = column_name.strip().lower()

    # Pass 1: fingerprint-scoped rules
    for rule in rules:
        if rule.source_fingerprint is not None:
            if fingerprint is not None and fingerprints_match(rule.source_fingerprint, fingerprint):
                if _pattern_matches(norm, rule):
                    return rule

    # Pass 2: global rules
    for rule in rules:
        if rule.source_fingerprint is None:
            if _pattern_matches(norm, rule):
                return rule

    return None


def _pattern_matches(norm: str, rule: MappingRule) -> bool:
    if rule.match_type == "exact":
        return norm == rule.match_pattern
    if rule.match_type == "keyword":
        return rule.match_pattern in norm
    if rule.match_type == "fuzzy":
        # fuzzy: try substring containment in both directions
        return rule.match_pattern in norm or norm in rule.match_pattern
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Rule application
# ─────────────────────────────────────────────────────────────────────────────


def apply_rule_to_classification(
    classification: ColumnClassification,
    rule: MappingRule,
) -> tuple[ColumnClassification, ConfidencePromotion]:
    """Apply a mapping rule to an existing ColumnClassification.

    Returns (updated_classification, promotion_audit).
    The original ColumnClassification is never mutated — ColumnClassification
    is a frozen dataclass; this returns a new instance.

    Unit handling: if rule.unit is None, the classifier's inferred unit is kept.
    """
    audit = ConfidencePromotion(
        original_confidence=classification.confidence,
        promoted_confidence=rule.confidence,
        original_inferred_from=classification.inferred_from,
        promoted_inferred_from="persistent_mapping_rule",
        rule_match_pattern=rule.match_pattern,
    )
    updated = ColumnClassification(
        column_name=classification.column_name,
        signal_type=rule.signal_type,
        unit=rule.unit if rule.unit is not None else classification.unit,
        display_group=rule.display_group,
        confidence=rule.confidence,
        inferred_from="persistent_mapping_rule",
        requires_user_confirmation=not rule.confirmed_by_operator,
        notes=classification.notes,
    )
    return updated, audit


# ─────────────────────────────────────────────────────────────────────────────
# Synonym / variant lookup (Phase D4.3)
# ─────────────────────────────────────────────────────────────────────────────


import re

def classify_by_synonym(
    column_name: str,
) -> tuple[str, str, str, float] | None:
    """Return (signal_type, unit, display_group, confidence) from the built-in
    synonym map, or None if no synonym matches.

    Matching is exact on the normalised (lower-stripped) name first, then
    substring — longest-match substring wins to avoid false positives.
    Substring matching strictly enforces word boundaries to avoid 'p' in 'rpm'.
    """
    norm = column_name.strip().lower()

    # Exact match first
    if norm in _SYNONYM_MAP:
        return _SYNONYM_MAP[norm]

    # Substring: prefer longest key that appears as a whole word in name
    best_key: str | None = None
    for key in _SYNONYM_MAP:
        # Use word boundaries so 'p' doesn't match inside 'rpm',
        # but handle special characters carefully.
        # \b doesn't work well with non-alphanumerics like '-', '/', so we fallback
        # to strict containment for those if they are long enough, or just use regex.
        pattern = rf'(?:\b|_){re.escape(key)}(?:\b|_)' if key.isalnum() else re.escape(key)
        if key in norm and (not key.isalnum() or re.search(pattern, norm)):
            if best_key is None or len(key) > len(best_key):
                best_key = key
                
    if best_key is not None:
        return _SYNONYM_MAP[best_key]

    return None
