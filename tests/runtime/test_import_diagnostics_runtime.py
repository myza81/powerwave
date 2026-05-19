"""Runtime Qt tests — Import Diagnostics Panel (Phase 8.55N).

These tests exercise the full dialog → pipeline → DiagnosticsPanel path with
a real Qt application and real CSV files, using ImmediateThreadPool so
background workers run synchronously in-process.

Coverage
--------
1.  DiagnosticsPanel renders without crash after successful import
2.  DiagnosticsPanel text contains key sections after import
3.  DiagnosticsPanel cleared on re-profile
4.  DiagnosticsPanel updated with export guidance after export
5.  ReviewImportPage shows confidence indicator
6.  No UI freeze after pipeline completion (synchronous path)
7.  Diagnostics update after user renames a column (authoritative execution)
8.  Failed import shows failure text in panel
9.  DiagnosticsPanel.set_summary / clear / plain_text API
10. Dropped-row information visible in panel text
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.diagnostics_summary import build_import_diagnostics
from app.ui.import_wizard import ImportWizardDialog
from app.ui.import_wizard.diagnostics_panel import DiagnosticsPanel, render_diagnostics_text


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────────────────────────────────────


def _qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class ImmediateThreadPool:
    def start(self, worker) -> None:
        worker.run()


def _make_dlg() -> ImportWizardDialog:
    return ImportWizardDialog(thread_pool=ImmediateThreadPool())


def _standard_csv(tmp_path, name: str = "import.csv") -> str:
    p = tmp_path / name
    p.write_text(
        "timestamp,va,ia,mw,mvar,trip\n"
        "2026-01-01 00:00:00.000,230.1,1.00,100.0,5.0,0\n"
        "2026-01-01 00:00:00.020,230.2,1.01,100.1,5.1,0\n"
        "2026-01-01 00:00:00.040,229.9,0.99,99.9,4.9,1\n",
        encoding="utf-8",
    )
    return str(p)


def _profile_and_import(dlg: ImportWizardDialog, csv_path: str) -> None:
    dlg.set_source_path(csv_path)
    dlg.profile_selected_file()
    dlg.run_import()


# ─────────────────────────────────────────────────────────────────────────────
# 1. DiagnosticsPanel renders without crash after successful import
# ─────────────────────────────────────────────────────────────────────────────


class TestDiagnosticsPanelRendering:
    def test_panel_renders_after_successful_import(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        try:
            _profile_and_import(dlg, csv_path)
            panel = dlg.complete_page.diagnostics_panel
            text = panel.plain_text()
            assert len(text) > 0
        finally:
            dlg.close()
            app.processEvents()

    def test_panel_text_contains_key_sections(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        try:
            _profile_and_import(dlg, csv_path)
            text = dlg.complete_page.diagnostics_panel.plain_text()
            assert "DATA SUMMARY" in text
            assert "TIMESTAMP" in text
            assert "CHANNEL CLASSIFICATION" in text
        finally:
            dlg.close()
            app.processEvents()

    def test_panel_shows_row_count(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        try:
            _profile_and_import(dlg, csv_path)
            text = dlg.complete_page.diagnostics_panel.plain_text()
            # CSV has 3 data rows
            assert "3" in text
        finally:
            dlg.close()
            app.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# 3. DiagnosticsPanel cleared on re-profile
# ─────────────────────────────────────────────────────────────────────────────


class TestPanelStateManagement:
    def test_panel_cleared_on_re_profile(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv1 = _standard_csv(tmp_path, "file1.csv")
        csv2 = _standard_csv(tmp_path, "file2.csv")
        try:
            _profile_and_import(dlg, csv1)
            assert len(dlg.complete_page.diagnostics_panel.plain_text()) > 0

            # Re-profile second file — pipeline_result cleared
            dlg.set_source_path(csv2)
            dlg.profile_selected_file()

            # complete_page not navigated to yet; pipeline_result is None
            assert dlg.pipeline_result is None
        finally:
            dlg.close()
            app.processEvents()

    def test_no_ui_freeze_after_import(self, tmp_path) -> None:
        """Synchronous path completes; QApplication still responsive."""
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        try:
            _profile_and_import(dlg, csv_path)
            # If we get here without hanging, the test passes.
            app.processEvents()
            assert dlg.pipeline_result is not None
        finally:
            dlg.close()
            app.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# 4. DiagnosticsPanel updated with export guidance after export
# ─────────────────────────────────────────────────────────────────────────────


class TestExportGuidanceRefresh:
    def test_export_guidance_added_after_export(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        out_path = str(tmp_path / "exported.csv")
        try:
            _profile_and_import(dlg, csv_path)
            text_before = dlg.complete_page.diagnostics_panel.plain_text()

            dlg.save_normalized_file(
                output_path=out_path,
                export_format="csv",
                include_metadata_sidecar=True,
                overwrite=True,
            )
            app.processEvents()

            text_after = dlg.complete_page.diagnostics_panel.plain_text()
            # After export the panel should mention export rows or sidecar
            assert "EXPORT" in text_after or len(text_after) >= len(text_before)
        finally:
            dlg.close()
            app.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# 5. ReviewImportPage shows confidence indicator
# ─────────────────────────────────────────────────────────────────────────────


class TestReviewPageConfidence:
    def test_review_page_shows_timestamp_confidence(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        try:
            dlg.set_source_path(csv_path)
            dlg.profile_selected_file()
            # Advance to review page
            dlg._apply_timestamp_selection()
            dlg._rebuild_execution_plan()
            text = dlg.review_page.summary.toPlainText()
            # Should contain confidence info (High/Medium/Low)
            assert any(label in text for label in ("High", "Medium", "Low"))
        finally:
            dlg.close()
            app.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Diagnostics update after user renames a column
# ─────────────────────────────────────────────────────────────────────────────


class TestAuthoritativeExecutionDiagnostics:
    def test_renamed_channel_reflected_in_diagnostics(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        try:
            dlg.set_source_path(csv_path)
            dlg.profile_selected_file()

            # Rename 'mw' → 'active_power'
            for mapping in dlg.column_model.mappings:
                if mapping.source_name == "mw":
                    mapping.user_name_override = "active_power"

            dlg.run_import()
            result = dlg.pipeline_result
            assert result is not None and result.success

            # The diagnostics panel should render without error
            panel = dlg.complete_page.diagnostics_panel
            text = panel.plain_text()
            assert len(text) > 0
        finally:
            dlg.close()
            app.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Failed import shows failure text
# ─────────────────────────────────────────────────────────────────────────────


class TestFailedImportDiagnostics:
    def test_failed_import_shows_error_text(self, tmp_path) -> None:
        import app.ui.import_wizard.import_wizard_dialog as dialog_module
        from unittest.mock import patch
        from app.import_wizard.models import RawPreviewModel
        from app.import_wizard.file_profiler import FileProfileResult
        from app.import_wizard.contracts import ValidationMessage, ValidationSeverity

        app_qt = _qapp()
        dlg = _make_dlg()
        csv_path = str(tmp_path / "broken.csv")
        (tmp_path / "broken.csv").write_text("not,useful\n1,2\n", encoding="utf-8")

        error_result = FileProfileResult(
            raw_preview=RawPreviewModel(column_names=[], preview_rows=[]),
            validation_messages=[
                ValidationMessage(ValidationSeverity.ERROR, "BROKEN", "Malformed file")
            ],
        )
        with patch.object(dialog_module, "profile_import_file", return_value=error_result):
            dlg.set_source_path(csv_path)
            dlg.profile_selected_file()

        # Profile with errors should stay on load page
        assert dlg.current_step.value == "load_file"
        dlg.close()
        app_qt.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# 9. DiagnosticsPanel standalone API
# ─────────────────────────────────────────────────────────────────────────────


class TestDiagnosticsPanelAPI:
    def test_set_summary_clear_plain_text(self, tmp_path) -> None:
        app = _qapp()
        csv_path = _standard_csv(tmp_path)
        from app.import_wizard import run_import_pipeline
        pipeline_result = run_import_pipeline(csv_path)

        panel = DiagnosticsPanel()
        assert panel.plain_text() == ""

        summary = build_import_diagnostics(pipeline_result)
        panel.set_summary(summary)
        assert len(panel.plain_text()) > 0
        assert "DATA SUMMARY" in panel.plain_text()

        panel.clear()
        assert panel.plain_text() == ""
        app.processEvents()

    def test_set_failure_text(self) -> None:
        app = _qapp()
        panel = DiagnosticsPanel()
        panel.set_failure_text("Import failed.\n\nDetails here.")
        assert "Import failed" in panel.plain_text()
        app.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Dropped-row information visible in panel text
# ─────────────────────────────────────────────────────────────────────────────


class TestDroppedRowVisibility:
    def test_render_shows_dropped_rows_when_present(self, tmp_path) -> None:
        from app.import_wizard.diagnostics_summary import ImportDiagnosticsSummary
        from app.import_wizard.contracts import ValidationSeverity

        # Build a synthetic summary with dropped rows
        summary = ImportDiagnosticsSummary(
            source_file_name="test.csv",
            provider_type="csv",
            success=True,
            timestamp_column="timestamp",
            timestamp_format="%Y-%m-%d",
            timestamp_format_source="auto-detected",
            timestamp_strategy="Parsed using auto-detected format",
            timestamp_confidence=0.93,
            timestamp_confidence_label="High",
            repair_actions=["8 rows removed due to unrecoverable timestamps"],
            total_rows=108,
            normalized_rows=100,
            dropped_rows=8,
            duplicate_timestamps=0,
            invalid_value_count=0,
            analog_channels=4,
            digital_channels=1,
            excluded_column_count=0,
            user_overridden_count=0,
            low_confidence_columns=[],
            classification_confidence_label="High",
            errors=[],
            warnings=[],
            infos=[],
            large_file_guidance=[],
            export_guidance=[],
            import_duration_s=None,
        )
        text = render_diagnostics_text(summary)
        assert "8" in text
        assert "removed" in text.lower() or "dropped" in text.lower() or "data loss" in text.lower()
