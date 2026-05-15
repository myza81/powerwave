"""Persistent timestamp format interpretation rules for Powerwave.

Rules are stored in config/timestamp_rules.yaml.
Once an operator confirms how an ambiguous date column should be parsed,
that interpretation is stored here and reused for matching sources.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from app.data.intelligence.models import TimestampRule

DEFAULT_TIMESTAMP_RULES_PATH = Path("config") / "timestamp_rules.yaml"


def load_timestamp_rules(
    path: Path = DEFAULT_TIMESTAMP_RULES_PATH,
) -> list[TimestampRule]:
    """Load timestamp rules from a YAML file.

    Returns an empty list when the file does not exist.
    Raises ValueError for malformed YAML or missing required fields.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed timestamp rules YAML at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Timestamp rules file must be a YAML mapping: {path}")

    rules: list[TimestampRule] = []
    for i, entry in enumerate(data.get("rules") or []):
        if not isinstance(entry, dict):
            raise ValueError(f"Timestamp rule #{i} must be a YAML mapping")
        if "source_pattern" not in entry:
            raise ValueError(f"Timestamp rule #{i} missing required field 'source_pattern'")
        if "date_format" not in entry:
            raise ValueError(f"Timestamp rule #{i} missing required field 'date_format'")
        rules.append(TimestampRule(
            source_pattern=str(entry["source_pattern"]),
            date_format=str(entry["date_format"]),
            ambiguous_resolution=str(
                entry.get("ambiguous_resolution", entry["date_format"])
            ),
            confirmed_by_operator=bool(entry.get("confirmed_by_operator", False)),
            timestamp_column=entry.get("timestamp_column"),
            timezone=entry.get("timezone"),
            column_fingerprint=entry.get("column_fingerprint"),
            notes=entry.get("notes"),
        ))
    return rules


def save_timestamp_rules(
    rules: list[TimestampRule],
    path: Path = DEFAULT_TIMESTAMP_RULES_PATH,
) -> None:
    """Persist timestamp rules to a YAML file.

    Creates parent directories as needed. Overwrites the existing file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rule_dicts: list[dict] = []
    for r in rules:
        d: dict = {
            "source_pattern": r.source_pattern,
            "date_format": r.date_format,
            "ambiguous_resolution": r.ambiguous_resolution,
            "confirmed_by_operator": r.confirmed_by_operator,
        }
        if r.timestamp_column is not None:
            d["timestamp_column"] = r.timestamp_column
        if r.timezone is not None:
            d["timezone"] = r.timezone
        if r.column_fingerprint is not None:
            d["column_fingerprint"] = r.column_fingerprint
        if r.notes is not None:
            d["notes"] = r.notes
        rule_dicts.append(d)
    data = {"rules": rule_dicts}
    text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def find_matching_timestamp_rule(
    source_pattern: str,
    rules: list[TimestampRule],
) -> TimestampRule | None:
    """Return the first rule whose source_pattern matches (case-insensitive exact).

    Returns None when no rule matches.
    """
    norm = source_pattern.strip().lower()
    for rule in rules:
        if rule.source_pattern.strip().lower() == norm:
            return rule
    return None


def find_matching_timestamp_rule_for_column(
    source_pattern: str,
    column_name: str,
    rules: list[TimestampRule],
) -> TimestampRule | None:
    """Return the most-specific matching rule for *source_pattern* + *column_name*.

    Priority:
      1. Rules matching both source_pattern AND timestamp_column (most specific)
      2. Rules matching source_pattern with timestamp_column=None (source-level)

    Returns None when no rule matches.
    """
    norm_src = source_pattern.strip().lower()
    norm_col = column_name.strip().lower()

    # Pass 1: column-scoped rules
    for rule in rules:
        if rule.source_pattern.strip().lower() != norm_src:
            continue
        if rule.timestamp_column is not None and rule.timestamp_column.strip().lower() == norm_col:
            return rule

    # Pass 2: source-level rules (timestamp_column=None)
    for rule in rules:
        if rule.source_pattern.strip().lower() == norm_src and rule.timestamp_column is None:
            return rule

    return None
