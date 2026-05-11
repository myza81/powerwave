"""Unit tests for the manifest review workflow integration in PowerwaveMainWindow.

Tests verify:
  - PowerwaveMainWindow has the updated _load_manifest method
  - review_summary is exported from app.data
  - DataReviewDialog is accessible from app.ui.dialogs
  - Workflow plumbing: manifest + review summary + dialog + visualization
  - Existing manifest-related methods still exist (regression)

No QApplication is created — only class-level attribute inspection and
helper function logic are tested.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Regression — existing manifest workflow methods still present
# ─────────────────────────────────────────────────────────────────────────────


class TestExistingWorkflowMethodsPresent:
    def test_open_manifest_dialog_exists(self) -> None:
        from app.ui.main_window.main_window import PowerwaveMainWindow
        assert callable(getattr(PowerwaveMainWindow, "_open_manifest_dialog", None))

    def test_load_manifest_exists(self) -> None:
        from app.ui.main_window.main_window import PowerwaveMainWindow
        assert callable(getattr(PowerwaveMainWindow, "_load_manifest", None))

    def test_on_multi_source_loaded_exists(self) -> None:
        from app.ui.main_window.main_window import PowerwaveMainWindow
        assert callable(getattr(PowerwaveMainWindow, "_on_multi_source_loaded", None))

    def test_on_load_sample_pulu_exists(self) -> None:
        from app.ui.main_window.main_window import PowerwaveMainWindow
        assert callable(getattr(PowerwaveMainWindow, "_on_load_sample_pulu", None))


# ─────────────────────────────────────────────────────────────────────────────
# New review workflow — public surface exists
# ─────────────────────────────────────────────────────────────────────────────


class TestReviewSummaryPublicSurface:
    def test_build_event_review_summary_importable(self) -> None:
        from app.data.review_summary import build_event_review_summary
        assert callable(build_event_review_summary)

    def test_event_review_summary_importable(self) -> None:
        from app.data.review_summary import EventReviewSummary
        assert EventReviewSummary is not None

    def test_source_review_summary_importable(self) -> None:
        from app.data.review_summary import SourceReviewSummary
        assert SourceReviewSummary is not None

    def test_column_review_row_importable(self) -> None:
        from app.data.review_summary import ColumnReviewRow
        assert ColumnReviewRow is not None

    def test_timestamp_review_summary_importable(self) -> None:
        from app.data.review_summary import TimestampReviewSummary
        assert TimestampReviewSummary is not None

    def test_data_review_dialog_importable(self) -> None:
        from app.ui.dialogs.data_review_dialog import DataReviewDialog
        assert DataReviewDialog is not None


# ─────────────────────────────────────────────────────────────────────────────
# Workflow integration logic — tested via helper inspection
# ─────────────────────────────────────────────────────────────────────────────


class TestReviewWorkflowLogic:
    """Verify that the review workflow correctly integrates with build_session_from_manifest."""

    def test_review_summary_builds_from_manifest_loader_output(self) -> None:
        """build_event_review_summary accepts a MultiSourceSession + manifest dict."""
        from unittest.mock import MagicMock, patch
        from app.data.multi_source_session import MultiSourceSession, SourceRecord
        from app.data.review_summary import build_event_review_summary
        from app.data.signal_metadata import SignalMetadata

        # Minimal synthetic session
        src = MagicMock(spec=SourceRecord)
        src.source_id = "ct"
        src.provider_type = "comtrade"
        src.sampling_rates = [5000.0]
        src.original_start_time = None
        src.signal_metadata = {}
        src.record.metadata.source_file = "test.cfg"
        src.record.timing_info.start_time = None
        src.record.timing_info.trigger_time = None
        src.record.analog_channels = []
        src.record.digital_channels = []
        src.record.sample_count.return_value = 0

        session = MultiSourceSession()
        session.add_source(src)

        manifest_data = {"event_id": "ev_test", "sources": [], "alignment": {}}
        result = build_event_review_summary(session, manifest_data=manifest_data)
        assert result.event_id == "ev_test"

    def test_load_manifest_imports_inside_function(self) -> None:
        """Verify _load_manifest imports DataReviewDialog (module-level import test)."""
        import inspect
        from app.ui.main_window.main_window import PowerwaveMainWindow
        source = inspect.getsource(PowerwaveMainWindow._load_manifest)
        assert "DataReviewDialog" in source
        assert "build_event_review_summary" in source
        assert "load_manifest" in source

    def test_cancelled_manifest_does_not_call_visualization(self) -> None:
        """Cancelling the dialog must abort visualization — verified by mock."""
        from unittest.mock import MagicMock, patch
        from app.ui.main_window.main_window import PowerwaveMainWindow

        # Patch QDialog.Rejected return from dlg.exec()
        with (
            patch("app.ui.main_window.main_window.PowerwaveMainWindow.__init__",
                  return_value=None),
        ):
            win = PowerwaveMainWindow.__new__(PowerwaveMainWindow)
            win._on_multi_source_loaded = MagicMock()

            # Simulate _load_manifest internals
            with (
                patch("app.data.manifest_loader.load_manifest", return_value={"event_id": "x", "sources": []}),
                patch("app.data.manifest_loader.build_session_from_manifest",
                      return_value=MagicMock()),
                patch("app.data.review_summary.build_event_review_summary",
                      return_value=MagicMock()),
                patch("app.ui.dialogs.data_review_dialog.DataReviewDialog") as MockDlg,
            ):
                from PyQt6.QtWidgets import QDialog
                mock_dlg_instance = MagicMock()
                mock_dlg_instance.exec.return_value = QDialog.DialogCode.Rejected
                MockDlg.return_value = mock_dlg_instance

                win.statusBar = MagicMock()
                win.statusBar.return_value = MagicMock()

                win._load_manifest(Path("fake.yaml"))

                win._on_multi_source_loaded.assert_not_called()

    def test_accepted_manifest_calls_visualization(self) -> None:
        """Accepting the dialog must proceed to _on_multi_source_loaded."""
        from unittest.mock import MagicMock, patch
        from app.ui.main_window.main_window import PowerwaveMainWindow

        with (
            patch("app.ui.main_window.main_window.PowerwaveMainWindow.__init__",
                  return_value=None),
        ):
            win = PowerwaveMainWindow.__new__(PowerwaveMainWindow)
            win._on_multi_source_loaded = MagicMock()

            with (
                patch("app.data.manifest_loader.load_manifest",
                      return_value={"event_id": "x", "sources": []}),
                patch("app.data.manifest_loader.build_session_from_manifest",
                      return_value=MagicMock()),
                patch("app.data.review_summary.build_event_review_summary",
                      return_value=MagicMock()),
                patch("app.ui.dialogs.data_review_dialog.DataReviewDialog") as MockDlg,
            ):
                from PyQt6.QtWidgets import QDialog
                mock_dlg_instance = MagicMock()
                mock_dlg_instance.exec.return_value = QDialog.DialogCode.Accepted
                MockDlg.return_value = mock_dlg_instance

                win.statusBar = MagicMock()
                win.statusBar.return_value = MagicMock()

                win._load_manifest(Path("fake.yaml"))

                win._on_multi_source_loaded.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Regression — existing non-manifest workflows untouched
# ─────────────────────────────────────────────────────────────────────────────


class TestExistingWorkflowsPreserved:
    def test_open_file_dialog_exists(self) -> None:
        from app.ui.main_window.main_window import PowerwaveMainWindow
        assert callable(getattr(PowerwaveMainWindow, "_open_file_dialog", None))

    def test_load_file_exists(self) -> None:
        from app.ui.main_window.main_window import PowerwaveMainWindow
        assert callable(getattr(PowerwaveMainWindow, "_load_file", None))

    def test_on_record_loaded_exists(self) -> None:
        from app.ui.main_window.main_window import PowerwaveMainWindow
        assert callable(getattr(PowerwaveMainWindow, "_on_record_loaded", None))

    def test_on_load_synthetic_mixed_exists(self) -> None:
        from app.ui.main_window.main_window import PowerwaveMainWindow
        assert callable(getattr(PowerwaveMainWindow, "_on_load_synthetic_mixed", None))

    def test_build_provider_manager_still_works(self) -> None:
        from app.ui.main_window.main_window import _build_provider_manager
        mgr = _build_provider_manager()
        assert len(mgr.available_providers()) == 3

    def test_format_load_status_still_works(self) -> None:
        from unittest.mock import MagicMock
        from app.ui.main_window.main_window import _format_load_status
        record = MagicMock()
        record.analog_channels = [MagicMock()]
        record.digital_channels = []
        record.sampling_info.sampling_rates = [5000.0]
        record.metadata.source_file = "test.cfg"
        result = _format_load_status(record)
        assert "test.cfg" in result
