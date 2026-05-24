"""Page widgets used by the Import Wizard dialog."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.import_wizard.contracts import ValidationMessage
from app.import_wizard.diagnostics_summary import build_import_diagnostics, _confidence_label
from app.import_wizard.export_writer import ExportWriteResult
from app.import_wizard.import_pipeline import ImportPipelineResult
from app.import_wizard.models import ImportWizardSession
from app.import_wizard.normalization_plan import NormalizationPlan
from app.ui.import_wizard.column_mapping_model import ColumnMappingTableModel, ParameterTypeDelegate
from app.ui.import_wizard.diagnostics_panel import DiagnosticsPanel
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
        guidance = QLabel("Preview uses sampled rows for large files; full normalization runs during import.")
        guidance.setWordWrap(True)
        layout.addWidget(guidance)
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
        self.guidance_label = QLabel("Confirm the detected header and sampled rows before continuing.")
        self.guidance_label.setWordWrap(True)
        layout.addWidget(self.guidance_label)
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
            self.guidance_label.setText("Profile a file to inspect sampled rows.")
            self.message_view.clear()
            return
        sheet = f" | sheet: {preview.sheet_name}" if preview.sheet_name else ""
        self.summary_label.setText(
            f"{len(preview.column_names)} columns | "
            f"{len(preview.preview_rows)} preview rows | "
            f"estimated rows: {preview.row_count_estimate}{sheet}"
        )
        self.guidance_label.setText(
            "Preview is sampled for responsiveness. Import uses the full dataset after review."
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
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        # ── Format Override ───────────────────────────────────────────────
        override_group = QGroupBox("Format Override")
        form = QFormLayout(override_group)
        self.selected_column_label = QLabel("None")
        self.detected_format_label = QLabel("auto")
        self.override_status_label = QLabel("Auto-detected")
        self.override_edit = QLineEdit()
        self.override_edit.setPlaceholderText("%Y-%m-%d %H:%M:%S")
        self.reset_button = QPushButton("Reset to detected")
        override_row = QWidget()
        override_layout = QHBoxLayout(override_row)
        override_layout.setContentsMargins(0, 0, 0, 0)
        override_layout.addWidget(self.override_edit, 1)
        override_layout.addWidget(self.reset_button)
        form.addRow("Selected column", self.selected_column_label)
        form.addRow("Detected format", self.detected_format_label)
        form.addRow("Format source", self.override_status_label)
        form.addRow("Manual format", override_row)
        layout.addWidget(override_group)

        # ── Timestamp Reconstruction ──────────────────────────────────────
        self._recon_group = QGroupBox("Timestamp Reconstruction")
        recon_form = QFormLayout(self._recon_group)

        self.recon_status_label = QLabel("No truncation detected.")
        self.recon_status_label.setWordWrap(True)
        recon_form.addRow("Detection", self.recon_status_label)

        self.recon_enable_check = QCheckBox("Enable anchor-based sub-interval reconstruction")
        self.recon_enable_check.setChecked(False)
        self.recon_enable_check.setToolTip(
            "When enabled, duplicate timestamps are treated as 100 ms anchor windows and "
            "sub-interval times are interpolated using the sample rate below."
        )
        recon_form.addRow("", self.recon_enable_check)

        self.recon_start_edit = QLineEdit()
        self.recon_start_edit.setPlaceholderText("YYYY-MM-DD HH:MM:SS.sss  (leave blank to use file's first timestamp)")
        self.recon_start_edit.setEnabled(False)
        recon_form.addRow("Start date/time", self.recon_start_edit)

        self.recon_rate_spin = QDoubleSpinBox()
        self.recon_rate_spin.setRange(0.001, 100_000.0)
        self.recon_rate_spin.setDecimals(3)
        self.recon_rate_spin.setSuffix(" Hz")
        self.recon_rate_spin.setSpecialValueText("auto")
        self.recon_rate_spin.setValue(0.001)  # sentinel = auto
        self.recon_rate_spin.setEnabled(False)
        recon_form.addRow("Sample rate", self.recon_rate_spin)

        self.recon_dt_label = QLabel("")
        recon_form.addRow("Interval", self.recon_dt_label)

        self.recon_sample_label = QLabel("")
        self.recon_sample_label.setWordWrap(True)
        self.recon_sample_label.setStyleSheet("font-family: monospace; color: #888888;")
        recon_form.addRow("Preview", self.recon_sample_label)

        self.recon_enable_check.toggled.connect(self._on_recon_toggled)
        self.recon_rate_spin.valueChanged.connect(self._on_recon_rate_changed)

        layout.addWidget(self._recon_group)

        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

    # ── Reconstruction helpers ────────────────────────────────────────────

    def _on_recon_toggled(self, enabled: bool) -> None:
        self.recon_start_edit.setEnabled(enabled)
        self.recon_rate_spin.setEnabled(enabled)

    def _on_recon_rate_changed(self, value: float) -> None:
        _AUTO = 0.001
        if value <= _AUTO:
            self.recon_dt_label.setText("auto (inferred from data)")
        else:
            dt_ms = 1000.0 / value
            self.recon_dt_label.setText(f"{dt_ms:.3f} ms per sample")

    def update_truncation_analysis(self, analysis) -> None:
        """Populate the reconstruction panel from a TruncationAnalysis result."""
        self.recon_status_label.setText(analysis.notes or "—")
        if analysis.is_truncated:
            self.recon_enable_check.setChecked(True)
            hz = analysis.inferred_sample_rate_hz
            if hz and hz > 0.001:
                was = self.recon_rate_spin.blockSignals(True)
                self.recon_rate_spin.setValue(round(hz, 3))
                self.recon_rate_spin.blockSignals(was)
                self._on_recon_rate_changed(hz)
        else:
            self.recon_enable_check.setChecked(False)

    @property
    def reconstruction_enabled(self) -> bool:
        return self.recon_enable_check.isChecked()

    @property
    def reconstruction_start_datetime(self) -> str:
        return self.recon_start_edit.text().strip()

    @property
    def reconstruction_sample_rate_hz(self) -> float | None:
        """Return the user-specified Hz, or None when set to auto (≤ 0.001)."""
        val = self.recon_rate_spin.value()
        return val if val > 0.001 else None

    # ── Standard page API ─────────────────────────────────────────────────

    def refresh(self, session: ImportWizardSession | None) -> None:
        count = len(session.timestamp_candidates) if session else 0
        self.message_label.setText(
            "No timestamp candidate detected. Choose another file or provide a source with a timestamp column."
            if count == 0
            else f"{count} candidate(s) detected. Manual override becomes authoritative when entered."
        )

    def set_candidate_details(
        self,
        column_name: str | None,
        detected_format: str | None,
        feedback: str,
    ) -> None:
        self.selected_column_label.setText(column_name or "None")
        self.detected_format_label.setText(detected_format or "auto")
        has_override = bool(self.override_edit.text().strip())
        self.override_status_label.setText("User Override" if has_override else "Auto-detected")
        if feedback:
            self.message_label.setText(feedback)


class ColumnMappingPage(QWidget):
    """Column mapping review page."""

    def __init__(self, model: ColumnMappingTableModel, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Review detected mappings. User edits require re-import before export or waveform handoff."))
        self.table = QTableView()
        self.table.setModel(model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setItemDelegateForColumn(3, ParameterTypeDelegate(self.table))
        layout.addWidget(self.table, 1)
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

    def refresh(self, session: ImportWizardSession | None) -> None:
        mappings = session.column_mappings if session else []
        included = sum(1 for mapping in mappings if not mapping.excluded)
        overrides = sum(1 for mapping in mappings if mapping.has_user_override or mapping.excluded)
        self.message_label.setText(
            f"{included} included column(s), {len(mappings) - included} excluded. "
            f"{overrides} user override(s)."
        )


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

        lines = [
            f"File: {session.source_path}",
            f"Provider: {session.provider_type}",
            f"Timestamp column: {selected}",
            f"Timestamp format: {fmt or 'auto'}",
        ]
        if candidate is not None:
            conf_label = _confidence_label(candidate.confidence)
            conf_pct = f"{candidate.confidence * 100:.0f}%"
            lines.append(f"Timestamp confidence: {conf_label} ({conf_pct})")
        lines += [
            "Guidance: verify this plan before importing. Later mapping or timestamp changes require re-import.",
            f"Analog-like columns: {len(included) - len(digital)}",
            f"Digital columns: {len(digital)}",
            f"Excluded columns: {', '.join(excluded) if excluded else 'None'}",
            f"Warnings: {warnings}",
            f"Errors: {errors}",
            f"Plan executable: {'yes' if executable else 'no'}",
        ]
        self.summary.setPlainText("\n".join(lines))


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
    """Final success/failure page with engineering diagnostics panel."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._last_pipeline_result: ImportPipelineResult | None = None

        layout = QVBoxLayout(self)
        self.diagnostics_panel = DiagnosticsPanel()
        layout.addWidget(self.diagnostics_panel, 1)

        export_group = QGroupBox("Save Normalized File")
        export_layout = QFormLayout(export_group)
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItem("CSV (.csv)", "csv")
        self.export_format_combo.addItem("Parquet (.parquet)", "parquet")
        self.export_format_combo.addItem("Feather (.feather)", "feather")
        self.include_metadata_sidecar = QCheckBox("Include metadata sidecar")
        self.include_metadata_sidecar.setChecked(True)
        self.overwrite_existing = QCheckBox("Overwrite existing file")
        self.overwrite_existing.setChecked(False)
        self.save_normalized_button = QPushButton("Save Normalized File...")
        self.export_status = QLabel("Run a successful import before exporting.")
        self.export_status.setWordWrap(True)
        self.export_guidance = QLabel("Metadata sidecar is recommended for auditability.")
        self.export_guidance.setWordWrap(True)
        export_layout.addRow("Format", self.export_format_combo)
        export_layout.addRow("", self.include_metadata_sidecar)
        export_layout.addRow("", self.overwrite_existing)
        export_layout.addRow("", self.save_normalized_button)
        export_layout.addRow("Guidance", self.export_guidance)
        export_layout.addRow("Status", self.export_status)
        layout.addWidget(export_group)

        self.open_waveform = QCheckBox("Open waveform after closing")
        self.open_waveform.setChecked(True)
        layout.addWidget(self.open_waveform)
        self.set_export_enabled(False)

    def set_result(self, result: ImportPipelineResult | None, error_text: str | None = None) -> None:
        self._last_pipeline_result = result
        dataset = result.dataset if result is not None else None
        self.set_export_enabled(
            result is not None
            and result.success
            and dataset is not None
            and dataset.is_export_ready()
        )
        if error_text:
            self.diagnostics_panel.set_failure_text(f"Import failed.\n\n{error_text}")
            self.export_status.setText("Import failed; export is unavailable.")
            return
        if result is None:
            self.diagnostics_panel.clear()
            self.export_status.setText("Run a successful import before exporting.")
            return

        # Build and display rich engineering diagnostics.
        summary = build_import_diagnostics(result)
        self.diagnostics_panel.set_summary(summary)

        if result.success and dataset is not None and dataset.is_export_ready():
            self.export_status.setText("Normalized dataset is ready to export.")
        elif result.success:
            self.export_status.setText("Import succeeded, but no export-ready normalized dataset is available.")
        else:
            self.export_status.setText("Import failed; export is unavailable.")

    def selected_export_format(self) -> str:
        return str(self.export_format_combo.currentData())

    def set_export_enabled(self, enabled: bool) -> None:
        self.save_normalized_button.setEnabled(enabled)
        self.export_format_combo.setEnabled(enabled)
        self.include_metadata_sidecar.setEnabled(enabled)
        self.overwrite_existing.setEnabled(enabled)
        if not enabled and "Writing normalized export" not in self.export_status.text():
            self.save_normalized_button.setToolTip("Run a successful import before saving normalized data.")
        else:
            self.save_normalized_button.setToolTip("Save the normalized dataset using the selected format.")

    def set_export_running(self, running: bool) -> None:
        self.save_normalized_button.setEnabled(not running)
        if running:
            self.export_status.setText("Writing normalized export...")

    def set_export_result(self, result: ExportWriteResult | None, error_text: str | None = None) -> None:
        if error_text:
            self.export_status.setText(f"Export failed: {error_text}")
        elif result is None:
            self.export_status.setText("No export has run.")
        else:
            lines = [
                "Export complete." if result.success else "Export failed.",
                result.diagnostics_summary,
                f"Output: {result.output_path or 'not written'}",
            ]
            if result.metadata_path:
                lines.append(f"Metadata: {result.metadata_path}")
            if result.validation_messages:
                lines.extend(["", format_messages(result.validation_messages)])
            self.export_status.setText("\n".join(lines))

        # Refresh diagnostics panel to include export guidance.
        if self._last_pipeline_result is not None and self._last_pipeline_result.success:
            export_write_result = result if (result is not None and error_text is None) else None
            summary = build_import_diagnostics(self._last_pipeline_result, export_write_result)
            self.diagnostics_panel.set_summary(summary)
