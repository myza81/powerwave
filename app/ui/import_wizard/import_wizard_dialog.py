"""Qt Import Wizard dialog — plan-aware execution.

The dialog is a thin UI orchestrator.  All heavy work is delegated to the
existing backend services in app.import_wizard.

GUI authority (Phase 8.55H)
----------------------------
User decisions made inside the wizard are now the authoritative execution
contract.  Before dispatching to the backend pipeline the dialog:

1. Snapshots the live column mapping model state.
2. Calls build_execution_plan() to validate and convert GUI state into
   an executable NormalizationPlan.
3. Rejects execution with clear error messages when the plan is invalid.
4. Passes the validated plan to run_import_pipeline_with_plan() so the
   backend uses the exact user choices — not a regenerated auto-plan.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
)

from app.import_wizard import (
    ExportPlan,
    ExportWriteOptions,
    ExportWriteResult,
    FileProfileResult,
    ImportPipelineOptions,
    ImportPipelineResult,
    ImportWizardSession,
    NormalizationPlan,
    TimestampRepairPlan,
    TimestampRepairStrategy,
    ValidationMessage,
    ValidationSeverity,
    WizardStep,
    plan_export,
    populate_session,
    profile_import_file,
    validate_user_timestamp_format,
    write_from_export_plan,
    write_normalized_export,
)
from app.import_wizard.import_pipeline import run_import_pipeline_with_plan
from app.import_wizard.pipeline_plan_builder import PlanBuildResult, build_execution_plan
from app.import_wizard.truncation_detector import detect_timestamp_truncation
from app.ui.import_wizard.column_mapping_model import ColumnMappingTableModel
from app.ui.import_wizard.preview_table_model import PreviewTableModel
from app.ui.import_wizard.timestamp_candidate_model import TimestampCandidateTableModel
from app.ui.import_wizard.workflow_state import evaluate_workflow_actions
from app.ui.import_wizard.wizard_pages import (
    ColumnMappingPage,
    ImportCompletePage,
    ImportRunningPage,
    LoadFilePage,
    RawPreviewPage,
    ReviewImportPage,
    TimestampSelectPage,
)


class _WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class _ProfileWorker(QRunnable):
    def __init__(self, path: str, sheet_name: str | None = None) -> None:
        super().__init__()
        self.path = path
        self.sheet_name = sheet_name
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            self.signals.finished.emit(profile_import_file(self.path, sheet_name=self.sheet_name))
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


class _PlanAwarePipelineWorker(QRunnable):
    """Background worker that executes the plan-aware import pipeline.

    Carries the exact NormalizationPlan and column mappings captured from
    the GUI at the moment the user clicked "Run Import".  The backend pipeline
    uses these plans verbatim — no auto-regeneration occurs.
    """

    def __init__(
        self,
        path: str,
        session: ImportWizardSession,
        normalization_plan: NormalizationPlan,
        column_mappings: list,
        options: ImportPipelineOptions | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.session = session
        self.normalization_plan = normalization_plan
        self.column_mappings = column_mappings
        self.options = options
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            result = run_import_pipeline_with_plan(
                self.path,
                self.session,
                self.normalization_plan,
                self.column_mappings,
                self.options,
            )
            self.signals.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


class _ExportWorker(QRunnable):
    """Background worker for normalized dataset export."""

    def __init__(
        self,
        dataset,
        output_path: str,
        export_format: str,
        options: ExportWriteOptions,
        export_plan: ExportPlan | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.output_path = output_path
        self.export_format = export_format
        self.options = options
        self.export_plan = export_plan
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            result = None
            if self.export_plan is not None:
                suggested = {
                    "csv": self.export_plan.suggested_csv_path,
                    "parquet": self.export_plan.suggested_parquet_path,
                    "feather": self.export_plan.suggested_feather_path,
                }.get(self.export_format)
                if suggested is not None and Path(self.output_path).resolve() == Path(suggested).resolve():
                    result = write_from_export_plan(
                        self.dataset,
                        self.export_plan,
                        format=self.export_format,
                        options=self.options,
                    )
            if result is None:
                result = write_normalized_export(
                    self.dataset,
                    self.output_path,
                    format=self.export_format,
                    options=self.options,
                )
            self.signals.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


_PAGE_STEPS: list[WizardStep] = [
    WizardStep.LOAD_FILE,
    WizardStep.RAW_PREVIEW,
    WizardStep.TIMESTAMP_SELECT,
    WizardStep.COLUMN_MAPPING,
    WizardStep.NORMALIZATION_REVIEW,
    WizardStep.SAVE_NORMALIZED,
    WizardStep.RENDER_WAVEFORM,
]

_STEP_LABELS: dict[WizardStep, str] = {
    WizardStep.LOAD_FILE: "Load File",
    WizardStep.RAW_PREVIEW: "Raw Preview",
    WizardStep.TIMESTAMP_SELECT: "Timestamp",
    WizardStep.COLUMN_MAPPING: "Column Mapping",
    WizardStep.NORMALIZATION_REVIEW: "Review",
    WizardStep.SAVE_NORMALIZED: "Import Running",
    WizardStep.RENDER_WAVEFORM: "Complete",
}


class ImportWizardDialog(QDialog):
    """Operational CSV/Excel import wizard with plan-aware authoritative execution."""

    import_completed = pyqtSignal(object)  # DisturbanceRecord

    def __init__(self, parent=None, *, thread_pool=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Wizard")
        self.resize(1120, 720)
        # Accept any object with a .start(worker) interface so tests can inject
        # an ImmediateThreadPool without inheriting from QThreadPool.
        self._thread_pool = thread_pool if thread_pool is not None else QThreadPool.globalInstance()
        self._session: ImportWizardSession | None = None
        self._profile: FileProfileResult | None = None
        self._normalization_plan: NormalizationPlan | None = None
        self._plan_build_result: PlanBuildResult | None = None
        self._pipeline_result: ImportPipelineResult | None = None
        self._export_result: ExportWriteResult | None = None
        self._export_running = False
        self._import_running = False
        self._settings_dirty_since_import = False
        self._normalized_export_saved = False
        self._closing = False
        self._max_reached_idx: int = 0  # furthest step index the user has reached

        self.preview_model = PreviewTableModel(parent=self)
        self.timestamp_model = TimestampCandidateTableModel(parent=self)
        self.column_model = ColumnMappingTableModel(parent=self)

        self._build_ui()
        self._set_step(WizardStep.LOAD_FILE)

    @property
    def session(self) -> ImportWizardSession | None:
        return self._session

    @property
    def pipeline_result(self) -> ImportPipelineResult | None:
        return self._pipeline_result

    @property
    def export_result(self) -> ExportWriteResult | None:
        return self._export_result

    @property
    def plan_build_result(self) -> PlanBuildResult | None:
        return self._plan_build_result

    @property
    def current_step(self) -> WizardStep:
        return _PAGE_STEPS[self.stack.currentIndex()]

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        title = QLabel("CSV/Excel Import Wizard")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(title)

        body = QHBoxLayout()
        self.step_list = QListWidget()
        self.step_list.setFixedWidth(180)
        for step in _PAGE_STEPS:
            item = QListWidgetItem(_STEP_LABELS[step])
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # selectable flag added as steps are reached
            self.step_list.addItem(item)
        self.step_list.itemClicked.connect(self._on_step_list_clicked)
        body.addWidget(self.step_list)

        self.stack = QStackedWidget()
        self.load_page = LoadFilePage()
        self.preview_page = RawPreviewPage(self.preview_model)
        self.timestamp_page = TimestampSelectPage(self.timestamp_model)
        self.mapping_page = ColumnMappingPage(self.column_model)
        self.review_page = ReviewImportPage()
        self.running_page = ImportRunningPage()
        self.complete_page = ImportCompletePage()
        for page in (
            self.load_page,
            self.preview_page,
            self.timestamp_page,
            self.mapping_page,
            self.review_page,
            self.running_page,
            self.complete_page,
        ):
            self.stack.addWidget(page)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)

        nav = QHBoxLayout()
        nav.addStretch(1)
        self.back_button = QPushButton("Back")
        self.next_button = QPushButton("Next")
        self.import_button = QPushButton("Run Import")
        self.open_button = QPushButton("Open Waveform")
        self.cancel_button = QPushButton("Close")
        for button in (
            self.back_button,
            self.next_button,
            self.import_button,
            self.open_button,
            self.cancel_button,
        ):
            nav.addWidget(button)
        root.addLayout(nav)

        self.load_page.browse_button.clicked.connect(self._browse_file)
        self.load_page.path_edit.textEdited.connect(self._on_source_path_edited)
        self.load_page.path_edit.returnPressed.connect(self.profile_selected_file)
        self.timestamp_page.table.clicked.connect(self._on_timestamp_clicked)
        self.timestamp_page.time_axis_mode_combo.currentIndexChanged.connect(self._on_time_axis_mode_changed)
        self.timestamp_page.elapsed_unit_combo.currentIndexChanged.connect(self._on_time_axis_mode_changed)
        self.timestamp_page.synthetic_basis_combo.currentIndexChanged.connect(self._on_time_axis_mode_changed)
        self.timestamp_page.synthetic_value_edit.textChanged.connect(self._on_time_axis_mode_changed)
        self.timestamp_page.override_edit.textChanged.connect(self._on_timestamp_override_changed)
        self.timestamp_page.reset_button.clicked.connect(self._reset_timestamp_override)
        self.timestamp_page.recon_enable_check.toggled.connect(self._on_recon_setting_changed)
        self.timestamp_page.recon_start_edit.textChanged.connect(self._on_recon_setting_changed)
        self.timestamp_page.recon_rate_spin.valueChanged.connect(self._on_recon_setting_changed)
        self.column_model.dataChanged.connect(self._on_column_mapping_changed)
        self.back_button.clicked.connect(self.go_back)
        self.next_button.clicked.connect(self.go_next)
        self.import_button.clicked.connect(self.run_import)
        self.open_button.clicked.connect(self.open_waveform)
        self.complete_page.save_normalized_button.clicked.connect(self.save_normalized_file)
        self.cancel_button.clicked.connect(self.request_close)

        self.workflow_status_label = QLabel("")
        self.workflow_status_label.setWordWrap(True)
        root.addWidget(self.workflow_status_label)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File to Import",
            "",
            "Import Files (*.csv *.txt *.tsv *.dat *.xlsx *.xls *.xlsm *.xlsb);;All Files (*)",
        )
        if path:
            if self._session is not None and Path(path) != Path(self._session.source_path):
                if not self._confirm_discard_current_session():
                    return
                self._reset_session_state_for_new_source()
            self.load_page.set_path(path)
            self.profile_selected_file()

    def set_source_path(self, path: str) -> None:
        """Set source path for tests or external callers."""
        if self._session is not None and Path(path) != Path(self._session.source_path):
            self._reset_session_state_for_new_source()
        self.load_page.set_path(path)

    def _on_source_path_edited(self, _text: str) -> None:
        if self._session is not None or self._profile is not None:
            self._reset_session_state_for_new_source()

    def _reset_session_state_for_new_source(self) -> None:
        """Clear state that belongs to the previously profiled/imported file."""
        self._session = None
        self._profile = None
        self._normalization_plan = None
        self._plan_build_result = None
        self._pipeline_result = None
        self._export_result = None
        self._settings_dirty_since_import = False
        self._normalized_export_saved = False
        self._max_reached_idx = 0
        self.preview_model.set_preview(None)
        self.timestamp_model.set_candidates([])
        self.column_model.set_mappings([])
        self.timestamp_page.override_edit.clear()
        self._reset_reconstruction_ui()
        self.complete_page.set_result(None)
        self.review_page.refresh(None, None)
        self._set_step(WizardStep.LOAD_FILE)

    def _invalidate_execution_state(self, reason: str) -> None:
        """Invalidate import/export results after user-controlled settings change."""
        had_result = self._pipeline_result is not None
        self._normalization_plan = None
        self._plan_build_result = None
        self._pipeline_result = None
        self._export_result = None
        self._normalized_export_saved = False
        if self._session is not None:
            self._session.normalization_plan = None
            self._session.add_message(
                ValidationMessage(
                    ValidationSeverity.INFO,
                    "UI_REIMPORT_REQUIRED",
                    f"{reason} Re-import required before export or waveform handoff.",
                )
            )
        if had_result:
            self._settings_dirty_since_import = True
            self.complete_page.set_result(None, error_text="Import settings changed. Re-import required.")
        self._refresh_pages()
        self._update_buttons()

    def profile_selected_file(self) -> None:
        path = self.load_page.path_edit.text().strip()
        if not path:
            self.load_page.set_messages([
                ValidationMessage(
                    ValidationSeverity.ERROR,
                    "UI_NO_FILE",
                    "Choose a CSV or Excel file before profiling.",
                    suggested_action="Select a supported source file.",
                )
            ])
            return
        self.load_page.set_messages([
            ValidationMessage(ValidationSeverity.INFO, "UI_PROFILE_START", "Profiling file...")
        ])
        worker = _ProfileWorker(path)
        worker.signals.finished.connect(self._on_profile_finished)
        worker.signals.error.connect(self._on_profile_error)
        self._thread_pool.start(worker)

    def _on_profile_finished(self, result: object) -> None:
        if not isinstance(result, FileProfileResult):
            self._on_profile_error("Profiler returned an unexpected result.")
            return
        path = self.load_page.path_edit.text().strip()
        self._profile = result
        self._session = ImportWizardSession(
            source_path=path,
            provider_type=result.provider_type,
            sheet_name=result.sheet_name,
            delimiter=result.delimiter,
        )
        populate_session(self._session, result)
        self.preview_model.set_preview(result.raw_preview)
        self.timestamp_model.set_candidates(result.timestamp_candidates)
        self.column_model.set_mappings(result.column_mappings)
        best = self.timestamp_model.selected_candidate()
        if best is not None:
            self._run_truncation_detection(best)
        was_blocked = self.timestamp_page.override_edit.blockSignals(True)
        self.timestamp_page.override_edit.clear()
        self.timestamp_page.override_edit.blockSignals(was_blocked)
        self.load_page.set_messages(result.validation_messages)
        # Reset any stale plan state from a previous import.
        self._normalization_plan = None
        self._plan_build_result = None
        self._pipeline_result = None
        self._export_result = None
        self._settings_dirty_since_import = False
        self._normalized_export_saved = False
        self._refresh_pages()
        if result.has_errors():
            self._set_step(WizardStep.LOAD_FILE)
            return
        self._set_step(WizardStep.RAW_PREVIEW)

    def _on_profile_error(self, message: str) -> None:
        self.load_page.set_messages([
            ValidationMessage(
                ValidationSeverity.ERROR,
                "UI_PROFILE_FAILED",
                f"Could not profile the selected file: {message}",
                suggested_action="Check that the file exists and is a supported CSV or Excel file.",
            )
        ])

    def _on_timestamp_clicked(self, index) -> None:
        self.timestamp_model.select_row(index.row())
        candidate = self.timestamp_model.selected_candidate()
        if self._session is not None and candidate is not None:
            self._session.selected_timestamp_column = candidate.column_name
            self._session.timestamp_repair_plan = None
            self._run_truncation_detection(candidate)
            self._sync_timestamp_override_plan()
            self._refresh_timestamp_override_feedback()
            self._invalidate_execution_state("Timestamp selection changed.")

    def _on_timestamp_override_changed(self, _text: str) -> None:
        if self._session is None:
            return
        self._sync_timestamp_override_plan()
        self._refresh_timestamp_override_feedback()
        self._invalidate_execution_state("Timestamp format override changed.")

    def _on_time_axis_mode_changed(self, *_args) -> None:
        if self._session is None:
            return
        self._sync_timestamp_override_plan()
        self._refresh_timestamp_override_feedback()
        self._invalidate_execution_state("Time axis mode changed.")

    def _reset_timestamp_override(self) -> None:
        self.timestamp_page.override_edit.clear()

    def _reset_reconstruction_ui(self) -> None:
        """Reset reconstruction panel to default state without emitting signals."""
        for w in (
            self.timestamp_page.recon_enable_check,
            self.timestamp_page.recon_rate_spin,
        ):
            w.blockSignals(True)
        self.timestamp_page.recon_start_edit.blockSignals(True)
        self.timestamp_page.recon_enable_check.setChecked(False)
        self.timestamp_page.recon_start_edit.clear()
        self.timestamp_page.recon_rate_spin.setValue(0.001)
        self.timestamp_page.recon_status_label.setText("No truncation detected.")
        self.timestamp_page.recon_dt_label.setText("")
        self.timestamp_page.recon_sample_label.setText("")
        for w in (
            self.timestamp_page.recon_enable_check,
            self.timestamp_page.recon_rate_spin,
        ):
            w.blockSignals(False)
        self.timestamp_page.recon_start_edit.blockSignals(False)

    def _run_truncation_detection(self, candidate) -> None:
        """Analyse candidate for truncated timestamps and populate the UI panel.

        Signals on the reconstruction widgets are blocked so that auto-population
        does not trigger _on_recon_setting_changed — the calling site handles any
        required plan invalidation.
        """
        fmt = candidate.detected_format
        if fmt in (None, "excel_serial"):
            fmt = None
        analysis = detect_timestamp_truncation(candidate.example_values, fmt=fmt)
        for w in (
            self.timestamp_page.recon_enable_check,
            self.timestamp_page.recon_rate_spin,
        ):
            w.blockSignals(True)
        self.timestamp_page.update_truncation_analysis(analysis)
        for w in (
            self.timestamp_page.recon_enable_check,
            self.timestamp_page.recon_rate_spin,
        ):
            w.blockSignals(False)
        self.timestamp_page.recon_sample_label.setText(
            self._generate_reconstruction_sample(candidate)
        )

    def _on_recon_setting_changed(self) -> None:
        if self._session is None:
            return
        candidate = self.timestamp_model.selected_candidate()
        if candidate is not None:
            self.timestamp_page.recon_sample_label.setText(
                self._generate_reconstruction_sample(candidate)
            )
        self._sync_timestamp_override_plan()
        self._invalidate_execution_state("Timestamp reconstruction setting changed.")

    def _generate_reconstruction_sample(self, candidate) -> str:
        """Return a short preview string of the first 5 reconstructed timestamps."""
        if not candidate or not candidate.example_values:
            return ""
        hz = self.timestamp_page.reconstruction_sample_rate_hz or 50.0
        try:
            import pandas as pd
            from datetime import timedelta
            fmt = candidate.detected_format
            if fmt in (None, "excel_serial"):
                fmt = None
            raw = pd.Series(candidate.example_values[:20], dtype=str)
            parsed = pd.to_datetime(raw, format=fmt if fmt else "mixed", errors="coerce")
            valid = parsed.dropna()
            if valid.empty:
                return ""
            start_text = self.timestamp_page.reconstruction_start_datetime
            if start_text:
                try:
                    first_ts = pd.Timestamp(start_text)
                except Exception:
                    first_ts = valid.iloc[0]
            else:
                first_ts = valid.iloc[0]
            dt_td = timedelta(seconds=1.0 / hz)
            samples = [first_ts + i * dt_td for i in range(5)]
            return "  →  ".join(t.strftime("%H:%M:%S.%f")[:12] for t in samples)
        except Exception:
            return ""

    def _on_column_mapping_changed(self, *_args) -> None:
        if self._session is None:
            return
        self._invalidate_execution_state("Column mapping changed.")

    def go_back(self) -> None:
        idx = self.stack.currentIndex()
        if idx > 0:
            self._set_step(_PAGE_STEPS[idx - 1])

    def go_next(self) -> None:
        step = self.current_step
        if step == WizardStep.LOAD_FILE:
            if self._profile is None:
                self.profile_selected_file()
                return
            if self._profile.has_errors():
                return
        elif step == WizardStep.TIMESTAMP_SELECT:
            mode = self.timestamp_page.selected_time_axis_mode()
            if mode not in {"sample_index", "synthetic_elapsed"} and not self.timestamp_model.selected_candidate():
                self.timestamp_page.set_candidate_details(None, None, "Select a timestamp column before continuing.")
                return
            self._apply_timestamp_selection()
        elif step == WizardStep.COLUMN_MAPPING:
            plan_result = self._rebuild_execution_plan()
            if plan_result is None or plan_result.normalization_plan is None:
                return
            if not self.column_model.visible_mappings or all(m.excluded for m in self.column_model.visible_mappings):
                return
        idx = self.stack.currentIndex()
        if idx + 1 < len(_PAGE_STEPS):
            self._set_step(_PAGE_STEPS[idx + 1])

    def run_import(self) -> None:
        """Execute the import using the authoritative GUI plan.

        Validates the current GUI state before dispatching.  Execution is
        blocked with a clear message when the plan has errors (no timestamp,
        no data columns, duplicate names, etc.).
        """
        if self._session is None:
            QMessageBox.warning(self, "Import Wizard", "Profile a file before importing.")
            return

        # Synchronise session with latest GUI selections.
        self._apply_timestamp_selection()

        # Build and validate the authoritative execution plan from live GUI state.
        plan_result = build_execution_plan(self._session, list(self.column_model.mappings))
        self._plan_build_result = plan_result

        if not plan_result.is_executable:
            error_lines = self._plan_blocking_messages(plan_result)
            QMessageBox.critical(
                self,
                "Import Wizard — Cannot Import",
                "The import plan has the following errors:\n\n"
                + "\n".join(f"• {e}" for e in error_lines),
            )
            return

        normalization_plan = plan_result.normalization_plan
        assert normalization_plan is not None  # guaranteed: is_executable checked above
        self._normalization_plan = normalization_plan
        self._session.normalization_plan = normalization_plan
        self.review_page.refresh(self._session, normalization_plan)

        self._set_step(WizardStep.SAVE_NORMALIZED)
        self._import_running = True
        self.running_page.set_running(True, "Running backend import pipeline...")
        self._update_buttons()

        # Snapshot the column mappings at execution time so any concurrent GUI
        # edits cannot affect the running import.
        column_mappings_snapshot = list(self.column_model.mappings)

        worker = _PlanAwarePipelineWorker(
            self._session.source_path,
            self._session,
            normalization_plan,
            column_mappings_snapshot,
            ImportPipelineOptions(),
        )
        worker.signals.finished.connect(self._on_pipeline_finished)
        worker.signals.error.connect(self._on_pipeline_error)
        self._thread_pool.start(worker)

    def _on_pipeline_finished(self, result: object) -> None:
        self._import_running = False
        if not isinstance(result, ImportPipelineResult):
            self._on_pipeline_error("Pipeline returned an unexpected result.")
            return
        self._pipeline_result = result
        self._export_result = None
        self._settings_dirty_since_import = False
        self._normalized_export_saved = False
        self.running_page.set_running(False, "Import finished.")
        self.complete_page.set_result(result)
        self._set_step(WizardStep.RENDER_WAVEFORM)
        self._update_buttons()

    def _on_pipeline_error(self, message: str) -> None:
        self._import_running = False
        self._settings_dirty_since_import = False
        self.running_page.set_running(False, "Import failed.")
        self.complete_page.set_result(None, error_text=message)
        self._set_step(WizardStep.RENDER_WAVEFORM)
        self._update_buttons()

    def save_normalized_file(
        self,
        output_path: str | None = None,
        *,
        export_format: str | None = None,
        include_metadata_sidecar: bool | None = None,
        overwrite: bool | None = None,
    ) -> None:
        """Write the normalized dataset using the export backend.

        When *output_path* is None, a QFileDialog is shown. Tests pass an
        explicit path to avoid modal UI while exercising the same worker path.
        """
        result = self._pipeline_result
        if (
            result is None
            or not result.success
            or result.dataset is None
            or not result.dataset.is_export_ready()
            or self._settings_dirty_since_import
        ):
            self.complete_page.set_export_result(
                None,
                error_text="No export-ready normalized dataset is available. Re-import if settings changed.",
            )
            return
        if self._export_running:
            return

        dataset = result.dataset
        plan = plan_export(dataset)
        fmt = export_format or self.complete_page.selected_export_format()
        target_path = output_path
        if target_path is None:
            target_path, fmt = self._choose_export_path(plan, fmt)
            if not target_path:
                return
        target_path = self._ensure_export_extension(target_path, fmt)

        options = ExportWriteOptions(
            include_metadata_sidecar=(
                self.complete_page.include_metadata_sidecar.isChecked()
                if include_metadata_sidecar is None
                else include_metadata_sidecar
            ),
            overwrite=(
                self.complete_page.overwrite_existing.isChecked()
                if overwrite is None
                else overwrite
            ),
        )
        self._export_running = True
        self.complete_page.set_export_running(True)
        self._update_buttons()

        worker = _ExportWorker(dataset, target_path, fmt, options, plan)
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.error.connect(self._on_export_error)
        self._thread_pool.start(worker)

    def _choose_export_path(self, plan: ExportPlan, preferred_format: str) -> tuple[str | None, str]:
        suggestions = {
            "csv": plan.suggested_csv_path,
            "parquet": plan.suggested_parquet_path,
            "feather": plan.suggested_feather_path,
        }
        default_path = suggestions.get(preferred_format) or plan.suggested_csv_path or "dataset_normalized.csv"
        filters = (
            "CSV (*.csv);;"
            "Parquet (*.parquet);;"
            "Feather (*.feather);;"
            "All Files (*)"
        )
        selected_filter = {
            "csv": "CSV (*.csv)",
            "parquet": "Parquet (*.parquet)",
            "feather": "Feather (*.feather)",
        }.get(preferred_format, "CSV (*.csv)")
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Save Normalized File",
            default_path,
            filters,
            selected_filter,
        )
        if not path:
            return None, preferred_format
        return path, self._format_from_filter_or_path(selected, path, preferred_format)

    @staticmethod
    def _format_from_filter_or_path(selected_filter: str, path: str, fallback: str) -> str:
        lower_filter = selected_filter.lower()
        if "parquet" in lower_filter:
            return "parquet"
        if "feather" in lower_filter:
            return "feather"
        if "csv" in lower_filter:
            return "csv"
        suffix = Path(path).suffix.lower()
        return {
            ".csv": "csv",
            ".parquet": "parquet",
            ".feather": "feather",
        }.get(suffix, fallback)

    @staticmethod
    def _ensure_export_extension(path: str, export_format: str) -> str:
        suffixes = {"csv": ".csv", "parquet": ".parquet", "feather": ".feather"}
        target = Path(path)
        suffix = suffixes.get(export_format)
        if suffix is not None and target.suffix.lower() != suffix:
            target = target.with_suffix(suffix)
        return str(target)

    def _on_export_finished(self, result: object) -> None:
        self._export_running = False
        if not isinstance(result, ExportWriteResult):
            self._on_export_error("Export returned an unexpected result.")
            return
        self._export_result = result
        self._normalized_export_saved = result.success
        self.complete_page.set_export_result(result)
        self._update_buttons()

    def _on_export_error(self, message: str) -> None:
        self._export_running = False
        self.complete_page.set_export_result(None, error_text=message)
        self._update_buttons()

    def open_waveform(self) -> None:
        result = self._pipeline_result
        if result is None or not result.success or result.record is None or self._settings_dirty_since_import:
            QMessageBox.warning(
                self,
                "Import Wizard",
                "No current imported waveform is available. Re-import after changing settings.",
            )
            return
        self.import_completed.emit(result.record)
        self.accept()

    def _apply_timestamp_selection(self) -> None:
        """Sync the GUI's selected timestamp candidate into the session."""
        candidate = self.timestamp_model.selected_candidate()
        if self._session is None:
            return
        mode = self.timestamp_page.selected_time_axis_mode()
        if mode in {"sample_index", "synthetic_elapsed"}:
            candidate = None
        self._session.time_axis_mode = mode
        self._session.elapsed_time_unit = self.timestamp_page.selected_elapsed_unit()
        value = self.timestamp_page.synthetic_timing_value()
        basis = self.timestamp_page.synthetic_timing_basis()
        self._session.sample_rate_hz = value if basis == "sample_rate" else None
        self._session.sample_interval_seconds = value if basis == "sample_interval" else None
        self._session.selected_timestamp_column = candidate.column_name if candidate is not None else None
        # Mark the candidate as user-selected so best_timestamp_candidate() finds it.
        for c in self._session.timestamp_candidates:
            c.user_selected = candidate is not None and c.column_name == candidate.column_name
        self._sync_timestamp_override_plan()
        self._refresh_timestamp_override_feedback()

    def _sync_timestamp_override_plan(self) -> None:
        """Build the current timestamp repair plan from selection + override."""
        if self._session is None:
            return
        mode = self.timestamp_page.selected_time_axis_mode()
        self._session.time_axis_mode = mode
        self._session.elapsed_time_unit = self.timestamp_page.selected_elapsed_unit()
        value = self.timestamp_page.synthetic_timing_value()
        basis = self.timestamp_page.synthetic_timing_basis()
        self._session.sample_rate_hz = value if basis == "sample_rate" else None
        self._session.sample_interval_seconds = value if basis == "sample_interval" else None

        if mode == "sample_index":
            self._session.timestamp_repair_plan = TimestampRepairPlan(
                strategy=TimestampRepairStrategy.GENERATE_SAMPLE_INDEX,
                repair_validated=True,
                repair_notes="Sample index axis generated from row order.",
            )
            return

        if mode == "synthetic_elapsed":
            valid = value is not None and value > 0
            self._session.timestamp_repair_plan = TimestampRepairPlan(
                strategy=TimestampRepairStrategy.GENERATE_SYNTHETIC_ELAPSED,
                sample_rate_hz=self._session.sample_rate_hz,
                sampling_interval_seconds=self._session.sample_interval_seconds,
                repair_validated=valid,
                repair_notes="Synthetic elapsed time generated from row order.",
            )
            return

        candidate = self.timestamp_model.selected_candidate()
        if candidate is None:
            return

        # Reconstruction takes priority over format-only overrides.
        if self.timestamp_page.reconstruction_enabled:
            hz = self.timestamp_page.reconstruction_sample_rate_hz
            start_dt = self.timestamp_page.reconstruction_start_datetime or None
            self._session.timestamp_repair_plan = TimestampRepairPlan(
                strategy=TimestampRepairStrategy.RECONSTRUCT_HYBRID,
                detected_format=candidate.detected_format,
                sampling_interval_seconds=(1.0 / hz) if hz else None,
                override_sample_rate_hz=hz,
                override_start_datetime=start_dt if start_dt else None,
                repair_validated=True,
                repair_notes="Anchor-based sub-interval reconstruction enabled.",
            )
            return

        if mode == "elapsed":
            self._session.timestamp_repair_plan = TimestampRepairPlan(
                strategy=TimestampRepairStrategy.PARSE_ELAPSED_TIME,
                detected_format=candidate.detected_format,
                elapsed_time_unit=self.timestamp_page.selected_elapsed_unit(),
                repair_validated=True,
                repair_notes="Elapsed-time unit selected by operator.",
            )
            return

        manual_format = self.timestamp_page.override_edit.text().strip()
        if manual_format:
            validation = validate_user_timestamp_format(
                manual_format,
                candidate.example_values,
                affected_column=candidate.column_name,
            )
            self._session.timestamp_repair_plan = TimestampRepairPlan(
                strategy=TimestampRepairStrategy.PARSE_USER_FORMAT,
                detected_format=candidate.detected_format,
                user_format=manual_format,
                repair_validated=validation.is_valid,
                repair_notes=f"User format override: {manual_format}",
            )
            return

        elapsed_formats = {"elapsed_seconds", "elapsed_milliseconds", "elapsed_minutes"}
        if candidate.detected_format in elapsed_formats:
            self._session.timestamp_repair_plan = TimestampRepairPlan(
                strategy=TimestampRepairStrategy.PARSE_ELAPSED_TIME,
                detected_format=candidate.detected_format,
                elapsed_time_unit=candidate.detected_format,
                repair_validated=True,
            )
            return

        strategy = (
            TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION
            if candidate.detected_format == "excel_serial"
            else TimestampRepairStrategy.PARSE_DETECTED_FORMAT
            if candidate.detected_format
            else TimestampRepairStrategy.NO_REPAIR
        )
        self._session.timestamp_repair_plan = TimestampRepairPlan(
            strategy=strategy,
            detected_format=candidate.detected_format,
            repair_validated=True,
        )

    def _refresh_timestamp_override_feedback(self) -> None:
        candidate = self.timestamp_model.selected_candidate()
        mode = self.timestamp_page.selected_time_axis_mode()
        if mode == "sample_index":
            self.timestamp_page.set_candidate_details(None, None, "Using sample index axis from row order. No time metadata is required.")
            return
        if mode == "synthetic_elapsed":
            value = self.timestamp_page.synthetic_timing_value()
            if value is None or value <= 0:
                self.timestamp_page.set_candidate_details(None, None, "Enter a positive sample rate or sample interval to generate synthetic elapsed time.")
            else:
                basis = "sample rate" if self.timestamp_page.synthetic_timing_basis() == "sample_rate" else "sample interval"
                self.timestamp_page.set_candidate_details(None, None, f"Using synthetic elapsed time from {basis}: {value:g}.")
            return
        if candidate is None:
            self.timestamp_page.set_candidate_details(None, None, "No timestamp candidate detected.")
            return

        manual_format = self.timestamp_page.override_edit.text().strip()
        if manual_format:
            validation = validate_user_timestamp_format(
                manual_format,
                candidate.example_values,
                affected_column=candidate.column_name,
            )
            message = validation.validation_messages[0].message if validation.validation_messages else ""
        else:
            if candidate.detected_format in {
                "elapsed_seconds",
                "elapsed_milliseconds",
                "elapsed_minutes",
            }:
                message = "Using detected relative elapsed-time axis."
            else:
                message = (
                    "Manual override is empty; using detected format."
                    if candidate.detected_format
                    else "Manual override is empty; using automatic timestamp parsing."
                )
        self.timestamp_page.set_candidate_details(
            candidate.column_name,
            candidate.detected_format,
            message,
        )

    def _sync_execution_plan(self) -> PlanBuildResult | None:
        """Build and cache the execution plan from live GUI state.

        Called when the user advances to or lands on the review page so the
        review state is never stale, including when using the left step list.
        """
        if self._session is None:
            return None
        if self._session.timestamp_repair_plan is None:
            self._apply_timestamp_selection()
        plan_result = build_execution_plan(self._session, list(self.column_model.mappings))
        self._plan_build_result = plan_result
        self._normalization_plan = plan_result.normalization_plan
        self._session.normalization_plan = plan_result.normalization_plan
        return plan_result

    def _rebuild_execution_plan(self) -> PlanBuildResult | None:
        plan_result = self._sync_execution_plan()
        if plan_result is None:
            return None
        self.review_page.refresh(self._session, plan_result.normalization_plan)
        self._update_buttons()
        return plan_result

    def _on_step_list_clicked(self, item: QListWidgetItem) -> None:
        if self._import_running or self._export_running:
            return
        row = self.step_list.row(item)
        if 0 <= row <= self._max_reached_idx:
            self._set_step(_PAGE_STEPS[row])

    def _update_step_list(self) -> None:
        """Refresh nav item selectability and dim unreachable steps."""
        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            if item is None:
                continue
            if i <= self._max_reached_idx:
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setData(Qt.ItemDataRole.ForegroundRole, None)  # restore default colour
            else:
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(QBrush(QColor("#666666")))

    def _set_step(self, step: WizardStep) -> None:
        if step not in _PAGE_STEPS:
            return
        idx = _PAGE_STEPS.index(step)
        self.stack.setCurrentIndex(idx)
        self.step_list.setCurrentRow(idx)
        if idx > self._max_reached_idx:
            self._max_reached_idx = idx
        if self._session is not None:
            self._session.current_step = step
        if step == WizardStep.NORMALIZATION_REVIEW:
            self._sync_execution_plan()
        self._refresh_pages()
        self._update_buttons()
        self._update_step_list()

    def _refresh_pages(self) -> None:
        self.preview_page.refresh(self._session)
        self.timestamp_page.refresh(self._session)
        self.mapping_page.refresh(self._session)
        self.review_page.refresh(self._session, self._normalization_plan)
        self._refresh_timestamp_override_feedback()

    def _update_buttons(self) -> None:
        step = self.current_step
        profile_has_errors = self._profile.has_errors() if self._profile is not None else False
        mode = self.timestamp_page.selected_time_axis_mode()
        has_candidates = bool(self._session and self._session.timestamp_candidates) or mode in {
            "sample_index",
            "synthetic_elapsed",
        }
        has_included_columns = bool(self.column_model.visible_mappings) and any(
            not mapping.excluded for mapping in self.column_model.visible_mappings
        )
        plan_is_executable = (
            self._normalization_plan is not None
            and self._normalization_plan.is_executable
            and self._plan_build_result is not None
            and self._plan_build_result.is_executable
        )
        import_success = (
            self._pipeline_result is not None
            and self._pipeline_result.success
            and self._pipeline_result.record is not None
        )
        export_ready = (
            self._pipeline_result is not None
            and self._pipeline_result.success
            and self._pipeline_result.dataset is not None
            and self._pipeline_result.dataset.is_export_ready()
        )
        state = evaluate_workflow_actions(
            step=step,
            step_index=self.stack.currentIndex(),
            has_path=bool(self.load_page.path_edit.text().strip()),
            has_profile=self._profile is not None,
            profile_has_errors=profile_has_errors,
            has_timestamp_candidates=has_candidates,
            has_included_columns=has_included_columns,
            plan_is_executable=plan_is_executable,
            import_running=self._import_running,
            export_running=self._export_running,
            import_success=import_success,
            export_ready=export_ready,
            settings_dirty_since_import=self._settings_dirty_since_import,
        )
        self.back_button.setEnabled(state.can_go_back)
        self.next_button.setVisible(step not in (WizardStep.SAVE_NORMALIZED, WizardStep.RENDER_WAVEFORM))
        self.next_button.setEnabled(state.can_go_next)
        self.import_button.setVisible(step == WizardStep.NORMALIZATION_REVIEW)
        self.import_button.setEnabled(state.can_import)
        self.open_button.setVisible(step == WizardStep.RENDER_WAVEFORM)
        self.open_button.setEnabled(state.can_open_waveform)
        self.complete_page.set_export_enabled(state.can_export)
        self.cancel_button.setEnabled(state.can_close)
        status_text = state.status_text
        if step == WizardStep.NORMALIZATION_REVIEW and not state.can_import:
            issues = self._plan_blocking_messages(self._plan_build_result)
            if issues:
                status_text = issues[0]
        self.workflow_status_label.setText(status_text)

    def _plan_blocking_messages(self, plan_result: PlanBuildResult | None) -> list[str]:
        if plan_result is None:
            return ["Execution plan is being prepared. Review will update automatically."]
        errors = [m.message for m in plan_result.errors()]
        if errors:
            return errors
        plan = plan_result.normalization_plan
        if plan is None:
            return ["Execution plan could not be built."]
        issues: list[str] = []
        if plan.timestamp_plan is None:
            issues.append("No validated time-axis repair plan is available.")
        elif not plan.timestamp_plan.repair_validated:
            issues.append("The selected time-axis parsing settings are not valid.")
        if not plan.selected_columns:
            issues.append("No signal/data columns are selected for import.")
        return issues or ["Execution plan is not ready to import."]

    def _has_discard_risk(self) -> bool:
        if self._import_running or self._export_running:
            return True
        if self._settings_dirty_since_import:
            return True
        if self._pipeline_result is not None and self._pipeline_result.success and not self._normalized_export_saved:
            return True
        if self._session is not None:
            if self.timestamp_page.override_edit.text().strip():
                return True
            if any(m.has_user_override or m.excluded for m in self.column_model.mappings):
                return True
        return False

    def _confirm_discard_current_session(self) -> bool:
        if not self._has_discard_risk():
            return True
        response = QMessageBox.question(
            self,
            "Import Wizard",
            "Discard current import session? Unsaved normalized exports and user settings will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return response == QMessageBox.StandardButton.Yes

    def request_close(self) -> None:
        if self._confirm_discard_current_session():
            self._closing = True
            super().reject()

    def reject(self) -> None:
        if self._closing or self._confirm_discard_current_session():
            self._closing = True
            super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._closing = True
        event.accept()
