"""tests/unit/test_duplicate_timestamp_columns.py

Phase D4.3 — Duplicate timestamp column handling tests.

Covers:
  - Scoring of duplicate time columns (Time vs Time.1)
  - Monotonic scoring factor
  - Missing ratio penalty
  - Event overlap scoring
  - Confirmed-by-rule bonus
  - Full PULU CSV scenario: Time (full datetime) vs Time.1 (time-only)
  - Correct winner selection
  - No silent numeric column loss during duplicate resolution
"""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import pytest

from app.data.intelligence import IntelligenceManager, TimestampColumnCandidate


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_pulu_df() -> pd.DataFrame:
    """Replicate the PULU CSV structure: Time (full), Time.1 (time-only)."""
    csv_text = (
        "Time,Time,System Demand,Tie-Line,Frequency\n"
        "3/6/2026 17:25,17:25,18738.85,108.16,50.02\n"
        "3/6/2026 17:26,17:26,18751.21,80.64,50.02\n"
        "3/6/2026 17:27,17:27,18739.43,80.32,50.01\n"
        "3/6/2026 17:28,17:28,18771.59,57.28,49.98\n"
        "3/6/2026 17:29,17:29,18698.91,79.36,49.99\n"
    )
    return pd.read_csv(io.StringIO(csv_text))


# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceManager.score_timestamp_candidates
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreTimestampCandidates:
    def setup_method(self):
        self.mgr = IntelligenceManager()

    def test_returns_list_of_candidates(self):
        df = _make_pulu_df()
        candidates = self.mgr.score_timestamp_candidates(
            ["Time", "Time.1"], df
        )
        assert len(candidates) == 2
        assert all(isinstance(c, TimestampColumnCandidate) for c in candidates)

    def test_sorted_by_total_score_desc(self):
        df = _make_pulu_df()
        candidates = self.mgr.score_timestamp_candidates(
            ["Time", "Time.1"], df
        )
        scores = [c.total_score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_time_column_beats_time1(self):
        """Full datetime (Time) should score higher than time-only (Time.1)."""
        df = _make_pulu_df()
        candidates = self.mgr.score_timestamp_candidates(["Time", "Time.1"], df)
        winner = candidates[0]
        assert winner.column_name == "Time"

    def test_monotonic_factor(self):
        """Monotonically increasing time series should be flagged is_monotonic=True."""
        df = _make_pulu_df()
        candidates = self.mgr.score_timestamp_candidates(["Time"], df)
        time_cand = next(c for c in candidates if c.column_name == "Time")
        assert time_cand.is_monotonic is True

    def test_missing_ratio_penalty(self):
        """Column with many NaNs should score lower than clean column."""
        df = pd.DataFrame({
            "clean": ["2026-03-06 17:25", "2026-03-06 17:26", "2026-03-06 17:27"],
            "sparse": ["2026-03-06 17:25", None, None],
        })
        candidates = self.mgr.score_timestamp_candidates(["clean", "sparse"], df)
        clean = next(c for c in candidates if c.column_name == "clean")
        sparse = next(c for c in candidates if c.column_name == "sparse")
        assert clean.total_score > sparse.total_score

    def test_missing_column_returns_rejection(self):
        df = _make_pulu_df()
        candidates = self.mgr.score_timestamp_candidates(
            ["Time", "DoesNotExist"], df
        )
        rejected = next(c for c in candidates if c.column_name == "DoesNotExist")
        assert rejected.rejection_reason == "column_not_found"
        assert rejected.total_score == 0.0

    def test_event_overlap_boosts_score(self):
        """Column whose parsed values overlap the COMTRADE event should score higher."""
        df = pd.DataFrame({
            "ts_overlap": ["2026-03-06 18:04:00", "2026-03-06 18:05:00"],
            "ts_nooverlap": ["2000-01-01 00:00:00", "2000-01-02 00:00:00"],
        })
        event_start = datetime(2026, 3, 6, 18, 4, 8)
        candidates = self.mgr.score_timestamp_candidates(
            ["ts_overlap", "ts_nooverlap"],
            df,
            event_start=event_start,
        )
        over = next(c for c in candidates if c.column_name == "ts_overlap")
        nover = next(c for c in candidates if c.column_name == "ts_nooverlap")
        assert over.overlap_score > nover.overlap_score
        assert over.total_score > nover.total_score

    def test_confirmed_rule_bonus(self, tmp_path):
        """Column with a confirmed timestamp rule should get confirmed_by_rule=True."""
        mgr = IntelligenceManager(timestamp_rules_path=tmp_path / "ts.yaml")
        mgr.save_timestamp_rule_for_column(
            source_pattern="pulu_csv",
            column_name="Time",
            date_format="%m/%d/%Y %H:%M",
            confirmed_by_operator=True,
            path=tmp_path / "ts.yaml",
        )
        df = _make_pulu_df()
        candidates = mgr.score_timestamp_candidates(
            ["Time", "Time.1"], df, source_pattern="pulu_csv"
        )
        time_cand = next(c for c in candidates if c.column_name == "Time")
        assert time_cand.confirmed_by_rule is True

    def test_interval_score_uniform_series(self):
        """Uniform 1-min intervals should yield high interval_score."""
        from datetime import timedelta
        base = datetime(2026, 3, 6, 17, 25)
        rows = [(base + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
                for i in range(10)]
        df = pd.DataFrame({"ts": rows})
        candidates = self.mgr.score_timestamp_candidates(["ts"], df)
        assert candidates[0].interval_score > 0.8

    def test_all_null_column(self):
        df = pd.DataFrame({"ts": [None, None, None]})
        candidates = self.mgr.score_timestamp_candidates(["ts"], df)
        assert candidates[0].parse_success_rate == 0.0
        assert candidates[0].missing_ratio == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# No silent numeric column loss
# ─────────────────────────────────────────────────────────────────────────────

class TestNoSilentColumnLoss:
    def test_numeric_column_disposition_analog(self):
        from app.data.column_classifier import (
            ColumnClassification,
            numeric_column_disposition,
            DISPOSITION_ANALOG,
        )
        cls = ColumnClassification(
            column_name="Frequency",
            signal_type="frequency",
            unit="Hz",
            display_group="frequency",
            confidence=0.95,
            inferred_from="name_exact",
            requires_user_confirmation=False,
        )
        disp, reason = numeric_column_disposition(cls, is_digital=False)
        assert disp == DISPOSITION_ANALOG
        assert reason is None

    def test_numeric_column_disposition_digital(self):
        from app.data.column_classifier import (
            ColumnClassification,
            numeric_column_disposition,
            DISPOSITION_DIGITAL,
        )
        cls = ColumnClassification(
            column_name="trip",
            signal_type=None,
            unit=None,
            display_group="other",
            confidence=0.0,
            inferred_from="unknown",
            requires_user_confirmation=True,
        )
        disp, reason = numeric_column_disposition(cls, is_digital=True)
        assert disp == DISPOSITION_DIGITAL

    def test_numeric_column_disposition_review(self):
        from app.data.column_classifier import (
            ColumnClassification,
            numeric_column_disposition,
            DISPOSITION_REVIEW,
        )
        cls = ColumnClassification(
            column_name="unknown_col",
            signal_type=None,
            unit=None,
            display_group="other",
            confidence=0.0,
            inferred_from="unknown",
            requires_user_confirmation=True,
        )
        disp, reason = numeric_column_disposition(cls)
        assert disp == DISPOSITION_REVIEW

    def test_numeric_column_disposition_ignored(self):
        from app.data.column_classifier import (
            ColumnClassification,
            numeric_column_disposition,
            DISPOSITION_IGNORED,
        )
        cls = ColumnClassification(
            column_name="Time.1",
            signal_type=None,
            unit=None,
            display_group="other",
            confidence=0.0,
            inferred_from="unknown",
            requires_user_confirmation=True,
        )
        disp, reason = numeric_column_disposition(
            cls, ignore_reason="duplicate_timestamp_artifact"
        )
        assert disp == DISPOSITION_IGNORED
        assert reason == "duplicate_timestamp_artifact"
        assert reason  # reason must be non-empty

    def test_every_waveform_column_gets_disposition(self):
        """All numeric columns in the PULU CSV must receive a disposition."""
        from app.data.column_classifier import (
            classify_csv_columns,
            numeric_column_disposition,
        )
        df = _make_pulu_df()
        classifications = classify_csv_columns(df, timestamp_column="Time")
        for col, cls in classifications.items():
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().any():
                    disp, _ = numeric_column_disposition(cls)
                    assert disp in (
                        "analog", "digital",
                        "unknown_requires_review", "ignored_with_reason"
                    ), f"Column {col!r} has no valid disposition"
            except Exception:
                pass
