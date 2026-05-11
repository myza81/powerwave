"""Central orchestration layer for Powerwave's data intelligence system.

IntelligenceManager combines:
  - built-in column classifier (column_classifier.py — unchanged)
  - persistent mapping rules  (config/column_mapping_rules.yaml)
  - timestamp interpretation rules (config/timestamp_rules.yaml)
  - source fingerprinting

All operations are non-destructive: original DisturbanceRecord objects
and ColumnClassification values are never mutated. ConfidencePromotion
provides a full audit trail for every confidence change.

IntelligenceManager is provider-independent. It works without any config
files (defaults to empty rules) so the rest of the system degrades gracefully.

Design note: column_classifier.py is intentionally not modified. Adding
an intelligence_manager parameter there would create a circular import
(column_classifier → intelligence_manager → column_classifier). Instead,
IntelligenceManager wraps the classifier and applies rules on top.

Usage::

    mgr = IntelligenceManager()
    cls, audit = mgr.classify_column("SYSF")
    # audit is None if no rule matched; ConfidencePromotion otherwise
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.data.column_classifier import ColumnClassification, classify_csv_column
from app.data.intelligence.fingerprints import (
    build_fingerprint_from_columns,
    build_fingerprint_from_record,
)
from app.data.intelligence.mapping_rules import (
    DEFAULT_MAPPING_RULES_PATH,
    apply_rule_to_classification,
    find_matching_rule,
    load_mapping_rules,
    save_mapping_rules,
)
from app.data.intelligence.models import (
    ConfidencePromotion,
    MappingRule,
    SourceFingerprint,
    TimestampRule,
)
from app.data.intelligence.timestamp_rules import (
    DEFAULT_TIMESTAMP_RULES_PATH,
    find_matching_timestamp_rule,
    load_timestamp_rules,
    save_timestamp_rules,
)


class IntelligenceManager:
    """Orchestrates column classification, mapping rules, and fingerprinting.

    Instantiate once per application session. Pass custom rule paths for
    testing without touching repository config files::

        mgr = IntelligenceManager(
            mapping_rules_path=Path("config/column_mapping_rules.yaml"),
        )
        cls, audit = mgr.classify_column("SYSF")
    """

    def __init__(
        self,
        mapping_rules_path: Path | None = None,
        timestamp_rules_path: Path | None = None,
    ) -> None:
        mp = Path(mapping_rules_path) if mapping_rules_path else DEFAULT_MAPPING_RULES_PATH
        tp = Path(timestamp_rules_path) if timestamp_rules_path else DEFAULT_TIMESTAMP_RULES_PATH
        self._mapping_rules: list[MappingRule] = load_mapping_rules(mp)
        self._timestamp_rules: list[TimestampRule] = load_timestamp_rules(tp)
        self._mapping_rules_path: Path = mp
        self._timestamp_rules_path: Path = tp

    # ─────────────────────────────────────────────────────────────────────
    # Column classification
    # ─────────────────────────────────────────────────────────────────────

    def classify_column(
        self,
        column_name: str,
        values: Sequence[float] | None = None,
        fingerprint: SourceFingerprint | None = None,
    ) -> tuple[ColumnClassification, ConfidencePromotion | None]:
        """Classify a single column, applying persistent rules when they match.

        The built-in classifier always runs first. If a matching rule exists,
        it overrides the classification and returns a ConfidencePromotion audit.

        Returns:
            (classification, promotion_audit):
              promotion_audit is None when no persistent rule was applied.
        """
        base = classify_csv_column(column_name, values)
        rule = find_matching_rule(column_name, self._mapping_rules, fingerprint)
        if rule is not None:
            promoted, audit = apply_rule_to_classification(base, rule)
            return promoted, audit
        return base, None

    def classify_columns(
        self,
        dataframe,                              # pd.DataFrame — avoid hard import
        timestamp_column: str | None = None,
        fingerprint: SourceFingerprint | None = None,
    ) -> dict[str, tuple[ColumnClassification, ConfidencePromotion | None]]:
        """Classify all non-timestamp DataFrame columns with rule application.

        Returns:
            Mapping of column_name → (ColumnClassification, ConfidencePromotion | None).
            ConfidencePromotion is None for columns where no rule fired.
        """
        result: dict[str, tuple[ColumnClassification, ConfidencePromotion | None]] = {}
        for col in dataframe.columns:
            if col == timestamp_column:
                continue
            try:
                vals: list[float] = dataframe[col].dropna().astype(float).tolist()
            except (ValueError, TypeError):
                vals = []
            result[col] = self.classify_column(col, vals if vals else None, fingerprint)
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Timestamp rules
    # ─────────────────────────────────────────────────────────────────────

    def resolve_timestamp_format(self, source_pattern: str) -> TimestampRule | None:
        """Return the confirmed timestamp rule for source_pattern, or None."""
        return find_matching_timestamp_rule(source_pattern, self._timestamp_rules)

    # ─────────────────────────────────────────────────────────────────────
    # Fingerprinting
    # ─────────────────────────────────────────────────────────────────────

    def build_fingerprint(
        self,
        column_names: list[str],
        source_type: str | None = None,
        station: str | None = None,
        source_kind: str | None = None,
    ) -> SourceFingerprint:
        """Build a SourceFingerprint from column names and optional source metadata."""
        return build_fingerprint_from_columns(
            column_names,
            source_type=source_type,
            station=station,
            source_kind=source_kind,
        )

    def build_fingerprint_from_record(
        self,
        record,                         # DisturbanceRecord
        source_type: str | None = None,
    ) -> SourceFingerprint:
        """Build a SourceFingerprint from a DisturbanceRecord's analog channels."""
        return build_fingerprint_from_record(record, source_type=source_type)

    # ─────────────────────────────────────────────────────────────────────
    # Manifest → rules extraction (explicit, never automatic)
    # ─────────────────────────────────────────────────────────────────────

    def extract_rules_from_manifest(
        self,
        manifest_data: dict,
    ) -> list[MappingRule]:
        """Extract reusable column mapping rules from a manifest's columns sections.

        Rules are produced for columns that have a recognised signal_type and
        inferred_from != 'unknown'. confirmed_by_operator mirrors the inverse
        of requires_user_confirmation from the manifest.

        This does NOT persist anything. Call save_rules_from_manifest() explicitly.
        """
        rules: list[MappingRule] = []
        for src_def in manifest_data.get("sources") or []:
            source_id = str(src_def.get("source_id", "unknown"))
            for col in src_def.get("columns") or []:
                if not isinstance(col, dict):
                    continue
                signal_type = col.get("signal_type")
                inferred_from = col.get("inferred_from", "unknown")
                if not signal_type or inferred_from == "unknown":
                    continue
                requires_confirm = bool(col.get("requires_user_confirmation", True))
                rules.append(MappingRule(
                    match_pattern=str(col["name"]).strip().lower(),
                    match_type="exact",
                    signal_type=signal_type,
                    unit=col.get("unit"),
                    display_group=str(col.get("display_group", "other")),
                    confidence=float(col.get("confidence", 0.80)),
                    confirmed_by_operator=not requires_confirm,
                    notes=f"Extracted from manifest source '{source_id}'",
                ))
        return rules

    def save_rules_from_manifest(
        self,
        manifest_data: dict,
        path: Path | None = None,
    ) -> int:
        """Explicitly persist rules extracted from a manifest.

        New rules are merged with existing rules keyed by match_pattern:
        manifest rules override existing rules with the same pattern.

        Returns the number of rules extracted from the manifest.
        """
        new_rules = self.extract_rules_from_manifest(manifest_data)
        merged: dict[str, MappingRule] = {r.match_pattern: r for r in self._mapping_rules}
        for r in new_rules:
            merged[r.match_pattern] = r
        all_rules = list(merged.values())
        target = path or self._mapping_rules_path
        save_mapping_rules(all_rules, target)
        self._mapping_rules = all_rules
        return len(new_rules)

    def save_timestamp_rule(
        self,
        rule: TimestampRule,
        path: Path | None = None,
    ) -> None:
        """Persist a single timestamp rule, merging with existing rules by source_pattern."""
        merged: dict[str, TimestampRule] = {
            r.source_pattern.strip().lower(): r for r in self._timestamp_rules
        }
        merged[rule.source_pattern.strip().lower()] = rule
        all_rules = list(merged.values())
        target = path or self._timestamp_rules_path
        save_timestamp_rules(all_rules, target)
        self._timestamp_rules = all_rules
