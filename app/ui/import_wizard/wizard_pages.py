"""Page widgets used by the Import Wizard dialog."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.import_pipeline import ImportPipelineResult
from app.import_wizard.models import ImportWizardSession
from app.import_wizard.normalization_plan import NormalizationPlan
from app.ui.import_wizard.column_mapping_model import ColumnMappingTableModel
from app.ui.import_wizard.preview_table_model import PreviewTableModel
from app.ui.import_wizard.timestamp_candidate_model import TimestampCandidateTableModel


def _message_line(message: ValidationMessage) -> str:
    column = f" [{message.affected_column}]" if message.affected_column else ""
    action = f" ({message.suggested_action})" if message.suggested_action else ""
    return f"{message.severity.value.upper()} {message.code}{column}: {message.message}{action}"


def format_messages(messages: list[ValidationMessage]) -> str:
    if not messages:
        return "No validation messages."
    return "\n".join(_message_line(message) for message in messages)


class LoadFilePage(QWidget):
    """File selection page."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Choose a CSV or Excel file")
        self.browse_button = QPushButton("Browse")
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.path_edit, 1)
        row_layout.addWidget(self.browse_button)
        form.addRow("File", row)
        self.provider_label = QLabel("Provider: auto")
        form.addRow("Detected", self.provider_label)
        layout.addLayout(form)
        self.message_view = QPlainTextEdit()
        self.message_view.setReadOnly(True)
        self.message_view.setMaximumHeight(120)
        layout.addWidget(self.message_view)
        layout.addStretch(1)

    def set_path(self, path: str) -> None:
        self.path_edit.setText(path)
        suffix = Path(path).suffix.lower()
        provider = "excel" if suffix in (".xlsx", ".xls", ".xlsm", ".xlsb") else "csv"
        self.provider_label.setText(f"Provider: {provider}")

    def set_messages(self, messages: list[ValidationMessage]) -> None:
        self.message_view.setPlainText(format_messages(messages))


class RawPreviewPage(QWidget):
    """Raw preview table page."""

    def __init__(self, model: PreviewTableModel, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary_label = QLabel("No file profiled.")
        layout.addWidget(self.summary_label)
        self.table = QTableView()
        self.table.setModel(model)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        layout.addWidget(self.table, 1)
        self.message_view = QPlainTextEdit()
        self.message_view.setReadOnly(True)
        self.message_view.setMaximumHeight(110)
        layout.addWidget(self.message_view)

    def refresh(self, session: ImportWizardSession | None) -> None:
        preview = session.raw_preview if session else None
        if preview is None:
            self.summary_label.setText("No file profiled.")
            self.message_view.clear()
            return
        sheet = f" | sheet: {preview.sheet_name}" if preview.sheet_name else ""
        self.summary_label.setText(
            f"{len(preview.column_names)} columns | "
            f"{len(preview.preview_rows)} preview rows | "
            f"estimated rows: {preview.row_count_estimate}{sheet}"
        )
        warnings = "\n".join(preview.parse_warnings)
        self.message_view.setPlainText(warnings or format_messages(session.validation_messages))


class TimestampSelectPage(QWidget):
    """Timestamp candidate selection page."""

    def __init__(self, model: TimestampCandidateTableModel, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select the column that represents event time."))
        self.table = QTableView()
        self.table.setModel(model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        layout.addWidget(self.table, 1)
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

    def refresh(self, session: ImportWizardSession | None) -> None:
        count = len(session.timestamp_candidates) if session else 0
        self.message_label.setText(
            "No timestamp candidate detected." if count == 0 else f"{count} candidate(s) detected."
        )


class ColumnMappingPage(QWidget):
    """Column mapping review page."""

    def __init__(self, model: ColumnMappingTableModel, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Review detected mappings. Basic include, name, type, and unit edits are supported."))
        self.table = QTableView()
        self.table.setModel(model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

    def refresh(self, session: ImportWizardSession | None) -> None:
        mappings = session.column_mappings if session else []
        included = sum(1 for mapping in mappings if not mapping.excluded)
        self.message_label.setText(f"{included} included column(s), {len(mappings) - included} excluded.")


class ReviewImportPage(QWidget):
    """Summary before import execution."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        layout.addWidget(self.summary, 1)

    def refresh(self, session: ImportWizardSession | None, plan: NormalizationPlan | None) -> None:
        if session is None:
            self.summary.setPlainText("No active import session.")
            return
        selected = session.selected_timestamp_column or "None"
        candidate = session.best_timestamp_candidate()
        fmt = candidate.detected_format if candidate else "unknown"
        mappings = session.column_mappings
        included = [m for m in mappings if not m.excluded]
        digital = [m for m in included if m.effective_type.value == "digital"]
        excluded = [m.source_name for m in mappings if m.excluded]
        warnings = len(session.warnings())
        errors = len(session.errors())
        executable = plan.is_executable if plan is not None else False
        self.summary.setPlainText(
            "\n".join([
                f"File: {session.source_path}",
                f"Provider: {session.provider_type}",
                f"Timestamp column: {selected}",
                f"Timestamp format: {fmt or 'auto'}",
                f"Analog-like columns: {len(included) - len(digital)}",
                f"Digital columns: {len(digital)}",
                f"Excluded columns: {', '.join(excluded) if excluded else 'None'}",
                f"Warnings: {warnings}",
                f"Errors: {errors}",
                f"Plan executable: {'yes' if executable else 'no'}",
            ])
        )


class ImportRunningPage(QWidget):
    """Progress/status page while the backend pipeline runs."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.status_label = QLabel("Ready to import.")
        layout.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        layout.addStretch(1)

    def set_running(self, running: bool, text: str) -> None:
        self.status_label.setText(text)
        self.progress.setVisible(running)


class ImportCompletePage(QWidget):
    """Final success/failure page."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        layout.addWidget(self.summary, 1)
        self.open_waveform = QCheckBox("Open waveform after closing")
        self.open_waveform.setChecked(True)
        layout.addWidget(self.open_waveform)

    def set_result(self, result: ImportPipelineResult | None, error_text: str | None = None) -> None:
        if error_text:
            self.summary.setPlainText(f"Import failed.\n\n{error_text}")
            return
        if result is None:
            self.summary.setPlainText("Import has not run.")
            return
        diagnostics = result.diagnostics
        lines = [
            "Import complete." if result.success else "Import failed.",
            f"Rows: {diagnostics.normalized_row_count}",
            f"Analog channels: {diagnostics.analog_channel_count}",
            f"Digital channels: {diagnostics.digital_channel_count}",
            f"Warnings: {diagnostics.warning_count}",
            f"Errors: {diagnostics.error_count}",
            "",
            format_messages(result.validation_messages),
        ]
        self.summary.setPlainText("\n".join(lines))
