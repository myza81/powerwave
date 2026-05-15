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
    classify_by_synonym,
    find_matching_rule,
    load_mapping_rules,
    save_mapping_rules,
)
from app.data.intelligence.models import (
    ConfidencePromotion,
    MappingRule,
    SourceFingerprint,
    TimestampColumnCandidate,
    TimestampRule,
)
from app.data.intelligence.timestamp_rules import (
    DEFAULT_TIMESTAMP_RULES_PATH,
    find_matching_timestamp_rule,
    find_matching_timestamp_rule_for_column,
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

        Priority (Phase D4.3):
          1. Persistent mapping rule (confirmed_by_operator=True wins)
          2. Persistent mapping rule (any)
          3. Built-in classifier (exact keyword)
          4. Synonym/variant library
          5. Value-profile inference
          6. Unknown — requires operator review

        The built-in classifier always runs first. If a matching persistent rule
        exists, it overrides the classification and returns a ConfidencePromotion.
        If no persistent rule matches and the base classifier yields no result,
        the synonym library is consulted as a fallback.

        Returns:
            (classification, promotion_audit):
              promotion_audit is None when no persistent rule was applied.
        """
        base = classify_csv_column(column_name, values)
        rule = find_matching_rule(column_name, self._mapping_rules, fingerprint)
        if rule is not None:
            promoted, audit = apply_rule_to_classification(base, rule)
            return promoted, audit

        # Synonym/variant fallback (Phase D4.3)
        if base.signal_type is None or base.confidence < 0.50:
            syn = classify_by_synonym(column_name)
            if syn is not None:
                st, unit, dg, conf = syn
                # Only promote if the synonym is more confident than the base
                if conf > base.confidence:
                    from app.data.column_classifier import CONFIRMATION_THRESHOLD
                    promoted = ColumnClassification(
                        column_name=column_name,
                        signal_type=st,
                        unit=unit,
                        display_group=dg,
                        confidence=conf,
                        inferred_from="synonym_match",
                        requires_user_confirmation=conf < CONFIRMATION_THRESHOLD,
                        notes=base.notes,
                    )
                    return promoted, None

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

    def resolve_timestamp_format(
        self, source_pattern: str, column_name: str | None = None
    ) -> TimestampRule | None:
        """Return the confirmed timestamp rule for source_pattern, or None.

        When *column_name* is provided, column-scoped rules take priority
        over source-level rules (Phase D4.3).
        """
        if column_name is not None:
            return find_matching_timestamp_rule_for_column(
                source_pattern, column_name, self._timestamp_rules
            )
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

    def save_timestamp_rule_for_column(
        self,
        source_pattern: str,
        column_name: str,
        date_format: str,
        timezone: str | None = None,
        confirmed_by_operator: bool = True,
        notes: str | None = None,
        path: Path | None = None,
    ) -> TimestampRule:
        """Persist a column-scoped timestamp rule (Phase D4.3).

        Creates a TimestampRule keyed by (source_pattern, column_name) so it
        applies only to that specific column in future matching sessions.

        Returns the persisted TimestampRule.
        """
        rule = TimestampRule(
            source_pattern=source_pattern,
            date_format=date_format,
            ambiguous_resolution=date_format,
            confirmed_by_operator=confirmed_by_operator,
            timestamp_column=column_name,
            timezone=timezone,
            notes=notes,
        )
        # Merge keyed by (source_pattern_lower, column_name_lower)
        existing = {}
        for r in self._timestamp_rules:
            key = (r.source_pattern.strip().lower(),
                   (r.timestamp_column or "").strip().lower())
            existing[key] = r
        key = (source_pattern.strip().lower(), column_name.strip().lower())
        existing[key] = rule
        all_rules = list(existing.values())
        target = path or self._timestamp_rules_path
        save_timestamp_rules(all_rules, target)
        self._timestamp_rules = all_rules
        return rule

    # ─────────────────────────────────────────────────────────────────────────────
    # Duplicate timestamp column scoring (Phase D4.3)
    # ─────────────────────────────────────────────────────────────────────────────

    def score_timestamp_candidates(
        self,
        candidate_columns: list[str],
        dataframe,                          # pd.DataFrame
        source_pattern: str | None = None,
        event_start=None,                   # datetime | None
        event_end=None,                     # datetime | None
    ) -> list[TimestampColumnCandidate]:
        """Score each candidate column for timestamp suitability.

        Used when duplicate time-like columns are present (e.g. 'Time', 'Time.1').
        Returns candidates sorted by total_score descending.

        Scoring factors:
          - parse_success_rate   (can values be read as datetime?)
          - is_monotonic         (series increases over time?)
          - finite_ratio         (how many values are non-null/finite?)
          - missing_ratio        (penalty for nulls)
          - interval_score       (regularity of time steps)
          - overlap_score        (proximity to COMTRADE/manifest event)
          - confirmed_by_rule    (bonus from persistent operator rule)
        """
        import numpy as np
        import pandas as pd

        results: list[TimestampColumnCandidate] = []

        for col in candidate_columns:
            if col not in dataframe.columns:
                results.append(TimestampColumnCandidate(
                    column_name=col,
                    parse_success_rate=0.0,
                    is_monotonic=False,
                    finite_ratio=0.0,
                    missing_ratio=1.0,
                    interval_score=0.0,
                    overlap_score=0.0,
                    confirmed_by_rule=False,
                    total_score=0.0,
                    rejection_reason="column_not_found",
                ))
                continue

            series = dataframe[col]
            n_total = len(series)
            n_missing = int(series.isna().sum())
            missing_ratio = n_missing / n_total if n_total > 0 else 1.0
            valid = series.dropna()

            # Attempt datetime parsing
            try:
                dt_series = pd.to_datetime(valid, utc=False, errors="coerce")
                parsed_mask = dt_series.notna()
                parse_success_rate = float(parsed_mask.sum()) / len(valid) if len(valid) > 0 else 0.0
                parsed_dts = dt_series[parsed_mask]
                finite_ratio = parse_success_rate
            except Exception:
                parse_success_rate = 0.0
                parsed_dts = pd.Series([], dtype="datetime64[ns]")
                finite_ratio = 0.0

            # Monotonic check
            is_monotonic = bool(parsed_dts.is_monotonic_increasing) if len(parsed_dts) > 1 else True

            # Interval consistency
            interval_score = 0.0
            if len(parsed_dts) >= 3:
                try:
                    secs = parsed_dts.diff().dt.total_seconds().dropna().to_numpy()
                    mean_s = float(np.mean(secs))
                    if mean_s > 0:
                        std_s = float(np.std(secs))
                        interval_score = float(np.clip(1.0 - std_s / mean_s, 0.0, 1.0))
                except Exception:
                    interval_score = 0.0

            # Event overlap score
            overlap_score = 0.0
            if event_start is not None and len(parsed_dts) > 0:
                from datetime import timedelta
                window = 7200.0  # ±2 hours
                lo = pd.Timestamp(event_start) - pd.Timedelta(seconds=window)
                hi = pd.Timestamp(event_end or event_start) + pd.Timedelta(seconds=window)
                n_overlap = int(((parsed_dts >= lo) & (parsed_dts <= hi)).sum())
                overlap_score = min(1.0, n_overlap / max(1, len(parsed_dts)))

            # Confirmed-by-rule bonus
            confirmed = False
            if source_pattern is not None:
                rule = find_matching_timestamp_rule_for_column(
                    source_pattern, col, self._timestamp_rules
                )
                confirmed = rule is not None and rule.confirmed_by_operator

            # Weighted total score
            total = (
                parse_success_rate * 0.30
                + (1.0 - missing_ratio) * 0.15
                + finite_ratio * 0.10
                + (0.15 if is_monotonic else 0.0)
                + interval_score * 0.10
                + overlap_score * 0.15
                + (0.05 if confirmed else 0.0)
            )
            results.append(TimestampColumnCandidate(
                column_name=col,
                parse_success_rate=parse_success_rate,
                is_monotonic=is_monotonic,
                finite_ratio=finite_ratio,
                missing_ratio=missing_ratio,
                interval_score=interval_score,
                overlap_score=overlap_score,
                confirmed_by_rule=confirmed,
                total_score=float(np.clip(total, 0.0, 1.0)),
            ))

        results.sort(key=lambda c: c.total_score, reverse=True)
        return results
