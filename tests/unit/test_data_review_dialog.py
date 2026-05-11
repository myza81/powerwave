"""Unit tests for app/ui/dialogs/data_review_dialog.py

Tests verify:
  - Module and class importability
  - DataReviewDialog is a QDialog subclass
  - Required methods and properties exist
  - Helper functions (_row_colour, _status_text, _fmt_offset) are correct
  - Legend and colour constants are defined

No QApplication is created — class attributes are inspected without instantiation.
"""
from __future__ import annotations

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Import tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDataReviewDialogImport:
    def test_module_importable(self) -> None:
        import app.ui.dialogs.data_review_dialog  # noqa: F401

    def test_dialog_class_importable(self) -> None:
        from app.ui.dialogs.data_review_dialog import DataReviewDialog
        assert DataReviewDialog is not None

    def test_helper_functions_importable(self) -> None:
        from app.ui.dialogs.data_review_dialog import (
            _fmt_offset,
            _row_colour,
            _status_text,
        )
        assert callable(_fmt_offset)
        assert callable(_row_colour)
        assert callable(_status_text)

    def test_colour_constants_defined(self) -> None:
        from app.ui.dialogs.data_review_dialog import (
            _COLOUR_CONFIRMED,
            _COLOUR_UNKNOWN,
            _COLOUR_WARN,
        )
        assert _COLOUR_CONFIRMED is not None
        assert _COLOUR_WARN is not None
        assert _COLOUR_UNKNOWN is not None


# ─────────────────────────────────────────────────────────────────────────────
# DataReviewDialog class hierarchy
# ─────────────────────────────────────────────────────────────────────────────


class TestDataReviewDialogHierarchy:
    def test_is_qdialog_subclass(self) -> None:
        from PyQt6.QtWidgets import QDialog
        from app.ui.dialogs.data_review_dialog import DataReviewDialog
        assert issubclass(DataReviewDialog, QDialog)

    def test_has_setup_ui_method(self) -> None:
        from app.ui.dialogs.data_review_dialog import DataReviewDialog
        assert callable(getattr(DataReviewDialog, "_setup_ui", None))

    def test_has_build_event_section_method(self) -> None:
        from app.ui.dialogs.data_review_dialog import DataReviewDialog
        assert callable(getattr(DataReviewDialog, "_build_event_section", None))

    def test_has_build_timestamp_section_method(self) -> None:
        from app.ui.dialogs.data_review_dialog import DataReviewDialog
        assert callable(getattr(DataReviewDialog, "_build_timestamp_section", None))

    def test_has_build_column_section_method(self) -> None:
        from app.ui.dialogs.data_review_dialog import DataReviewDialog
        assert callable(getattr(DataReviewDialog, "_build_column_section", None))

    def test_has_populate_table_row_method(self) -> None:
        from app.ui.dialogs.data_review_dialog import DataReviewDialog
        assert callable(getattr(DataReviewDialog, "_populate_table_row", None))

    def test_has_populate_summary_row_method(self) -> None:
        from app.ui.dialogs.data_review_dialog import DataReviewDialog
        assert callable(getattr(DataReviewDialog, "_populate_summary_row", None))

    def test_has_build_legend_method(self) -> None:
        from app.ui.dialogs.data_review_dialog import DataReviewDialog
        assert callable(getattr(DataReviewDialog, "_build_legend", None))


# ─────────────────────────────────────────────────────────────────────────────
# Helper function logic — no Qt widgets needed
# ─────────────────────────────────────────────────────────────────────────────


class TestFmtOffset:
    def _fmt(self, s: float) -> str:
        from app.ui.dialogs.data_review_dialog import _fmt_offset
        return _fmt_offset(s)

    def test_zero_is_zero_seconds(self) -> None:
        assert "0.0 s" in self._fmt(0.0)

    def test_positive_sign_present(self) -> None:
        assert "+" in self._fmt(10.0)

    def test_negative_sign_present(self) -> None:
        assert "-" in self._fmt(-10.0)

    def test_seconds_format_for_small_values(self) -> None:
        result = self._fmt(45.0)
        assert "s" in result
        assert "min" not in result

    def test_minutes_format_for_large_values(self) -> None:
        result = self._fmt(3600.0)
        assert "min" in result

    def test_60_seconds_uses_minutes(self) -> None:
        result = self._fmt(61.0)
        assert "min" in result


class TestStatusText:
    def _row(self, confidence: float, requires: bool):
        from app.data.review_summary import ColumnReviewRow
        return ColumnReviewRow(
            column_name="x",
            signal_type=None,
            unit=None,
            display_group="other",
            confidence=confidence,
            inferred_from="unknown",
            requires_user_confirmation=requires,
        )

    def test_confirmed_row_shows_tick(self) -> None:
        from app.ui.dialogs.data_review_dialog import _status_text
        row = self._row(0.95, False)
        assert "✓" in _status_text(row)

    def test_warn_row_shows_warning_icon(self) -> None:
        from app.ui.dialogs.data_review_dialog import _status_text
        row = self._row(0.70, True)
        assert "⚠" in _status_text(row)

    def test_unknown_row_shows_review_required(self) -> None:
        from app.ui.dialogs.data_review_dialog import _status_text
        row = self._row(0.30, True)
        assert "⚠" in _status_text(row)
        assert "required" in _status_text(row).lower()


class TestRowColour:
    def _row(self, confidence: float, requires: bool):
        from app.data.review_summary import ColumnReviewRow
        return ColumnReviewRow(
            column_name="x",
            signal_type=None,
            unit=None,
            display_group="other",
            confidence=confidence,
            inferred_from="unknown",
            requires_user_confirmation=requires,
        )

    def test_confirmed_row_gets_green(self) -> None:
        from app.ui.dialogs.data_review_dialog import _COLOUR_CONFIRMED, _row_colour
        row = self._row(0.95, False)
        assert _row_colour(row) == _COLOUR_CONFIRMED

    def test_warn_row_gets_yellow(self) -> None:
        from app.ui.dialogs.data_review_dialog import _COLOUR_WARN, _row_colour
        row = self._row(0.70, True)
        assert _row_colour(row) == _COLOUR_WARN

    def test_unknown_row_gets_red(self) -> None:
        from app.ui.dialogs.data_review_dialog import _COLOUR_UNKNOWN, _row_colour
        row = self._row(0.30, True)
        assert _row_colour(row) == _COLOUR_UNKNOWN

    def test_boundary_at_0_50_gets_red(self) -> None:
        from app.ui.dialogs.data_review_dialog import _COLOUR_UNKNOWN, _row_colour
        row = self._row(0.49, True)
        assert _row_colour(row) == _COLOUR_UNKNOWN

    def test_boundary_at_0_50_gets_yellow(self) -> None:
        from app.ui.dialogs.data_review_dialog import _COLOUR_WARN, _row_colour
        row = self._row(0.50, True)
        assert _row_colour(row) == _COLOUR_WARN


# ─────────────────────────────────────────────────────────────────────────────
# Table column constants
# ─────────────────────────────────────────────────────────────────────────────


class TestTableConstants:
    def test_n_cols_is_7(self) -> None:
        from app.ui.dialogs.data_review_dialog import _N_COLS
        assert _N_COLS == 7

    def test_header_labels_count_matches_n_cols(self) -> None:
        from app.ui.dialogs.data_review_dialog import _HEADER_LABELS, _N_COLS
        assert len(_HEADER_LABELS) == _N_COLS

    def test_required_column_names_present(self) -> None:
        from app.ui.dialogs.data_review_dialog import _HEADER_LABELS
        lower = [h.lower() for h in _HEADER_LABELS]
        assert any("source" in h for h in lower)
        assert any("conf" in h for h in lower)
        assert any("status" in h for h in lower)
