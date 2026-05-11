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
