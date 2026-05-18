"""Qt Import Wizard dialog skeleton.

The dialog is intentionally a thin UI orchestrator. File profiling and import
execution are delegated to the existing backend services in app.import_wizard.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal
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
    populate_session,
    profile_import_file,
    run_import_pipeline,
)
from app.import_wizard.column_mapping import ParameterType
from app.models import DisturbanceRecord
from app.ui.import_wizard.column_mapping_model import ColumnMappingTableModel
from app.ui.import_wizard.preview_table_model import PreviewTableModel
from app.ui.import_wizard.timestamp_candidate_model import TimestampCandidateTableModel
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


class _PipelineWorker(QRunnable):
    def __init__(
        self,
        path: str,
        provider_type: str | None,
        sheet_name: str | None,
        options: ImportPipelineOptions | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.provider_type = provider_type
        self.sheet_name = sheet_name
        self.options = options
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            result = run_import_pipeline(
                self.path,
                provider_type=self.provider_type,
                sheet_name=self.sheet_name,
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
    """Operational CSV/Excel import wizard skeleton."""

    import_completed = pyqtSignal(object)  # DisturbanceRecord

    def __init__(self, parent=None, *, thread_pool: QThreadPool | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Wizard")
        self.resize(980, 680)
        self._thread_pool = thread_pool or QThreadPool.globalInstance()
        self._session: ImportWizardSession | None = None
        self._profile: FileProfileResult | None = None
        self._normalization_plan: NormalizationPlan | None = None
        self._pipeline_result: ImportPipelineResult | None = None
        self._import_running = False

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
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.step_list.addItem(item)
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
        self.load_page.path_edit.returnPressed.connect(self.profile_selected_file)
        self.timestamp_page.table.clicked.connect(self._on_timestamp_clicked)
        self.back_button.clicked.connect(self.go_back)
        self.next_button.clicked.connect(self.go_next)
        self.import_button.clicked.connect(self.run_import)
        self.open_button.clicked.connect(self.open_waveform)
        self.cancel_button.clicked.connect(self.reject)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File to Import",
            "",
            "Import Files (*.csv *.txt *.tsv *.dat *.xlsx *.xls *.xlsm *.xlsb);;All Files (*)",
        )
        if path:
            self.load_page.set_path(path)
            self.profile_selected_file()

    def set_source_path(self, path: str) -> None:
        """Set source path for tests or external callers."""
        self.load_page.set_path(path)

    def profile_selected_file(self) -> None:
        path = self.load_page.path_edit.text().strip()
        if not path:
            self.load_page.set_messages([
                ValidationMessage(ValidationSeverity.ERROR, "UI_NO_FILE", "Choose a file first.")
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
        self.load_page.set_messages(result.validation_messages)
        self._refresh_pages()
        if result.has_errors():
            return
        self._set_step(WizardStep.RAW_PREVIEW)

    def _on_profile_error(self, message: str) -> None:
        self.load_page.set_messages([
            ValidationMessage(ValidationSeverity.ERROR, "UI_PROFILE_FAILED", message)
        ])

    def _on_timestamp_clicked(self, index) -> None:
        self.timestamp_model.select_row(index.row())
        candidate = self.timestamp_model.selected_candidate()
        if self._session is not None and candidate is not None:
            self._session.selected_timestamp_column = candidate.column_name

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
        elif step == WizardStep.TIMESTAMP_SELECT:
            self._apply_timestamp_selection()
        elif step == WizardStep.COLUMN_MAPPING:
            self._build_normalization_plan()
        idx = self.stack.currentIndex()
        if idx + 1 < len(_PAGE_STEPS):
            self._set_step(_PAGE_STEPS[idx + 1])

    def run_import(self) -> None:
        if self._session is None:
            QMessageBox.warning(self, "Import Wizard", "Profile a file before importing.")
            return
        self._apply_timestamp_selection()
        self._build_normalization_plan()
        self._set_step(WizardStep.SAVE_NORMALIZED)
        self._import_running = True
        self.running_page.set_running(True, "Running backend import pipeline...")
        self._update_buttons()
        worker = _PipelineWorker(
            self._session.source_path,
            self._session.provider_type,
            self._session.sheet_name,
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
        self.running_page.set_running(False, "Import finished.")
        self.complete_page.set_result(result)
        self._set_step(WizardStep.RENDER_WAVEFORM)
        self._update_buttons()

    def _on_pipeline_error(self, message: str) -> None:
        self._import_running = False
        self.running_page.set_running(False, "Import failed.")
        self.complete_page.set_result(None, error_text=message)
        self._set_step(WizardStep.RENDER_WAVEFORM)
        self._update_buttons()

    def open_waveform(self) -> None:
        result = self._pipeline_result
        if result is None or not result.success or result.record is None:
            QMessageBox.warning(self, "Import Wizard", "No imported waveform is available.")
            return
        self.import_completed.emit(result.record)
        self.accept()

    def _apply_timestamp_selection(self) -> None:
        candidate = self.timestamp_model.selected_candidate()
        if self._session is None or candidate is None:
            return
        self._session.selected_timestamp_column = candidate.column_name
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

    def _build_normalization_plan(self) -> NormalizationPlan | None:
        if self._session is None:
            return None
        if self._session.timestamp_repair_plan is None:
            self._apply_timestamp_selection()
        selected: list[str] = []
        excluded: list[str] = []
        renames: dict[str, str] = {}
        units: dict[str, str] = {}
        types: dict[str, ParameterType] = {}
        timestamp_name = self._session.selected_timestamp_column
        for mapping in self.column_model.mappings:
            if mapping.source_name == timestamp_name or mapping.effective_type == ParameterType.TIMESTAMP:
                excluded.append(mapping.source_name)
                continue
            if mapping.excluded:
                excluded.append(mapping.source_name)
                continue
            selected.append(mapping.source_name)
            if mapping.effective_name != mapping.source_name:
                renames[mapping.source_name] = mapping.effective_name
            if mapping.effective_unit:
                units[mapping.effective_name] = mapping.effective_unit
            types[mapping.effective_name] = mapping.effective_type
        plan = NormalizationPlan(
            timestamp_plan=self._session.timestamp_repair_plan,
            selected_columns=selected,
            excluded_columns=excluded,
            column_renames=renames,
            column_units=units,
            column_types=types,
            validation_messages=list(self._session.validation_messages),
        )
        self._normalization_plan = plan
        self._session.normalization_plan = plan
        self.review_page.refresh(self._session, plan)
        return plan

    def _set_step(self, step: WizardStep) -> None:
        if step not in _PAGE_STEPS:
            return
        idx = _PAGE_STEPS.index(step)
        self.stack.setCurrentIndex(idx)
        self.step_list.setCurrentRow(idx)
        if self._session is not None:
            self._session.current_step = step
        self._refresh_pages()
        self._update_buttons()

    def _refresh_pages(self) -> None:
        self.preview_page.refresh(self._session)
        self.timestamp_page.refresh(self._session)
        self.mapping_page.refresh(self._session)
        self.review_page.refresh(self._session, self._normalization_plan)

    def _update_buttons(self) -> None:
        step = self.current_step
        self.back_button.setEnabled(self.stack.currentIndex() > 0 and not self._import_running)
        self.next_button.setVisible(step not in (WizardStep.SAVE_NORMALIZED, WizardStep.RENDER_WAVEFORM))
        self.next_button.setEnabled(not self._import_running)
        self.import_button.setVisible(step == WizardStep.NORMALIZATION_REVIEW)
        self.import_button.setEnabled(not self._import_running)
        self.open_button.setVisible(step == WizardStep.RENDER_WAVEFORM)
        can_open = (
            self._pipeline_result is not None
            and self._pipeline_result.success
            and self._pipeline_result.record is not None
        )
        self.open_button.setEnabled(can_open)
        self.cancel_button.setEnabled(not self._import_running)
