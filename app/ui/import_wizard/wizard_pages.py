"""Page widgets used by the Import Wizard dialog."""
from __future__ import annotations

from html import escape
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from app.import_wizard.contracts import ValidationMessage
from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.diagnostics_summary import build_import_diagnostics, _confidence_label
from app.import_wizard.export_writer import ExportWriteResult
from app.import_wizard.import_pipeline import ImportPipelineResult
from app.import_wizard.models import ImportWizardSession
from app.import_wizard.normalization_plan import NormalizationPlan
from app.ui.import_wizard.column_mapping_model import (
    ColumnMappingTableModel,
    ParameterTypeDelegate,
    UnitDelegate,
)
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
        self.table.setMinimumHeight(92)
        self.table.setMaximumHeight(150)
        layout.addWidget(self.table)

        mode_group = QGroupBox("Time Axis Mode")
        mode_layout = QGridLayout(mode_group)
        mode_layout.setContentsMargins(14, 12, 14, 12)
        mode_layout.setHorizontalSpacing(12)
        mode_layout.setVerticalSpacing(10)
        self.time_axis_mode_combo = QComboBox(self)
        self.time_axis_mode_combo.addItem("Auto-detect", "auto")
        self.time_axis_mode_combo.addItem("Absolute timestamp", "absolute")
        self.time_axis_mode_combo.addItem("Elapsed time values", "elapsed")
        self.time_axis_mode_combo.addItem("Synthetic time from sample rate", "synthetic_elapsed")
        self.time_axis_mode_combo.addItem("Sample index", "sample_index")
        self.time_axis_mode_combo.setVisible(False)
        self._mode_radios: dict[str, QRadioButton] = {}
        self._mode_button_group = QButtonGroup(self)
        self._mode_button_group.setExclusive(True)
        mode_cards = [
            (
                "auto",
                "Auto-detect",
                "Use the app's recommended timestamp interpretation.",
            ),
            (
                "absolute",
                "Absolute timestamp",
                "Calendar/date-time values from a timestamp column.",
            ),
            (
                "elapsed",
                "Elapsed time",
                "Duration values such as seconds, milliseconds, or minutes.",
            ),
            (
                "synthetic_elapsed",
                "Synthetic elapsed time",
                "Generate elapsed time from sample rate or interval.",
            ),
            (
                "sample_index",
                "Sample index",
                "Ignore time metadata and plot by row sequence.",
            ),
        ]
        for index, (mode_key, title, description) in enumerate(mode_cards):
            card = self._build_mode_card(mode_key, title, description)
            mode_layout.addWidget(card, index // 2, index % 2)
        self.elapsed_unit_combo = QComboBox()
        self.elapsed_unit_combo.addItem("Seconds", "elapsed_seconds")
        self.elapsed_unit_combo.addItem("Milliseconds", "elapsed_milliseconds")
        self.elapsed_unit_combo.addItem("Minutes", "elapsed_minutes")
        self.synthetic_basis_combo = QComboBox()
        self.synthetic_basis_combo.addItem("Sample rate (Hz)", "sample_rate")
        self.synthetic_basis_combo.addItem("Sample interval (seconds)", "sample_interval")
        self.synthetic_value_edit = QLineEdit()
        self.synthetic_value_edit.setPlaceholderText("Sample rate, e.g. 100")
        self.elapsed_unit_label = QLabel("Elapsed unit")
        self.synthetic_basis_label = QLabel("Synthetic basis")
        self._mode_settings_widget = QWidget()
        mode_settings_layout = QGridLayout(self._mode_settings_widget)
        mode_settings_layout.setContentsMargins(0, 8, 0, 0)
        mode_settings_layout.setHorizontalSpacing(12)
        mode_settings_layout.setVerticalSpacing(8)
        self.elapsed_unit_label.setMinimumWidth(110)
        self.synthetic_basis_label.setMinimumWidth(110)
        mode_settings_layout.addWidget(self.elapsed_unit_label, 0, 0)
        mode_settings_layout.addWidget(self.elapsed_unit_combo, 0, 1, 1, 2)
        mode_settings_layout.addWidget(self.synthetic_basis_label, 1, 0)
        mode_settings_layout.addWidget(self.synthetic_basis_combo, 1, 1)
        mode_settings_layout.addWidget(self.synthetic_value_edit, 1, 2)
        mode_settings_layout.setColumnStretch(1, 1)
        mode_settings_layout.setColumnStretch(2, 1)
        mode_layout.addWidget(self._mode_settings_widget, 3, 0, 1, 2)
        mode_layout.setColumnStretch(0, 1)
        mode_layout.setColumnStretch(1, 1)
        layout.addWidget(mode_group)

        self.override_group = QGroupBox("Format Override")
        override_layout = QVBoxLayout(self.override_group)
        override_layout.setContentsMargins(14, 12, 14, 12)
        override_layout.setSpacing(8)
        details = QWidget()
        details_layout = QGridLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setHorizontalSpacing(18)
        details_layout.setVerticalSpacing(8)
        self.selected_column_caption = QLabel("Selected column")
        self.detected_format_caption = QLabel("Detected format")
        self.format_source_caption = QLabel("Format source")
        self.manual_format_caption = QLabel("Manual format")
        self.selected_column_label = QLabel("None")
        self.detected_format_label = QLabel("auto")
        self.override_status_label = QLabel("Auto-detected")
        self.override_edit = QLineEdit()
        self.override_edit.setPlaceholderText("%Y-%m-%d %H:%M:%S")
        self.override_edit.setMinimumWidth(320)
        self.reset_button = QPushButton("Reset to detected")
        self.reset_button.setMinimumWidth(138)
        for caption in (
            self.selected_column_caption,
            self.detected_format_caption,
            self.format_source_caption,
            self.manual_format_caption,
        ):
            caption.setMinimumWidth(96)
        details_layout.addWidget(self.selected_column_caption, 0, 0)
        details_layout.addWidget(self.selected_column_label, 0, 1)
        details_layout.addWidget(self.detected_format_caption, 1, 0)
        details_layout.addWidget(self.detected_format_label, 1, 1)
        details_layout.addWidget(self.format_source_caption, 2, 0)
        details_layout.addWidget(self.override_status_label, 2, 1)
        details_layout.addWidget(self.manual_format_caption, 3, 0)
        details_layout.addWidget(self.override_edit, 3, 1)
        details_layout.addWidget(self.reset_button, 3, 2)
        details_layout.setColumnStretch(1, 1)
        override_layout.addWidget(details)
        layout.addWidget(self.override_group)

        # ── Timestamp Reconstruction ──────────────────────────────────────
        self._recon_group = QGroupBox("Advanced Timestamp Repair")
        recon_root_layout = QVBoxLayout(self._recon_group)
        recon_root_layout.setContentsMargins(14, 12, 14, 12)
        recon_root_layout.setSpacing(8)
        self.recon_advanced_check = QCheckBox("Show timestamp reconstruction tools")
        self.recon_advanced_check.setToolTip(
            "Use this only when timestamp values are truncated or repeated and need repair."
        )
        recon_root_layout.addWidget(self.recon_advanced_check)
        self._recon_body = QWidget()
        recon_layout = QGridLayout(self._recon_body)
        recon_layout.setContentsMargins(14, 12, 14, 12)
        recon_layout.setHorizontalSpacing(18)
        recon_layout.setVerticalSpacing(10)

        self.recon_status_label = QLabel("No truncation detected.")
        self.recon_status_label.setWordWrap(True)
        self.recon_status_label.setMinimumHeight(34)
        self.recon_status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        recon_detection_label = QLabel("Detection")
        recon_detection_label.setMinimumWidth(96)
        recon_layout.addWidget(recon_detection_label, 0, 0, Qt.AlignmentFlag.AlignTop)
        recon_layout.addWidget(self.recon_status_label, 0, 1, 1, 3)

        self.recon_enable_check = QCheckBox("Enable anchor-based sub-interval reconstruction")
        self.recon_enable_check.setChecked(False)
        self.recon_enable_check.setToolTip(
            "When enabled, duplicate timestamps are treated as 100 ms anchor windows and "
            "sub-interval times are interpolated using the sample rate below."
        )
        recon_layout.addWidget(self.recon_enable_check, 1, 1, 1, 3)

        self.recon_start_edit = QLineEdit()
        self.recon_start_edit.setPlaceholderText("YYYY-MM-DD HH:MM:SS.sss  (leave blank to use file's first timestamp)")
        self.recon_start_edit.setEnabled(False)
        recon_layout.addWidget(QLabel("Start date/time"), 2, 0)
        recon_layout.addWidget(self.recon_start_edit, 2, 1, 1, 3)

        self.recon_rate_spin = QDoubleSpinBox()
        self.recon_rate_spin.setRange(0.001, 100_000.0)
        self.recon_rate_spin.setDecimals(3)
        self.recon_rate_spin.setSuffix(" Hz")
        self.recon_rate_spin.setSpecialValueText("auto")
        self.recon_rate_spin.setValue(0.001)  # sentinel = auto
        self.recon_rate_spin.setEnabled(False)
        self.recon_rate_spin.setMinimumWidth(150)
        recon_layout.addWidget(QLabel("Sample rate"), 3, 0)
        recon_layout.addWidget(self.recon_rate_spin, 3, 1)

        self.recon_dt_label = QLabel("")
        self.recon_dt_label.setWordWrap(True)
        recon_layout.addWidget(QLabel("Interval"), 3, 2)
        recon_layout.addWidget(self.recon_dt_label, 3, 3)

        self.recon_sample_label = QLabel("")
        self.recon_sample_label.setWordWrap(True)
        self.recon_sample_label.setStyleSheet(
            "font-family: Menlo, Consolas, 'Courier New'; color: #888888;"
        )
        self.recon_sample_label.setMinimumHeight(26)
        recon_layout.addWidget(QLabel("Preview"), 4, 0, Qt.AlignmentFlag.AlignTop)
        recon_layout.addWidget(self.recon_sample_label, 4, 1, 1, 3)
        recon_layout.setColumnStretch(1, 2)
        recon_layout.setColumnStretch(3, 2)
        recon_root_layout.addWidget(self._recon_body)
        self._recon_body.setVisible(False)

        self.recon_enable_check.toggled.connect(self._on_recon_toggled)
        self.recon_rate_spin.valueChanged.connect(self._on_recon_rate_changed)
        self.recon_advanced_check.toggled.connect(self._recon_body.setVisible)

        layout.addWidget(self._recon_group)

        self.plan_group = QGroupBox("Current Plan")
        plan_layout = QVBoxLayout(self.plan_group)
        plan_layout.setContentsMargins(8, 8, 8, 8)
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self._configure_feedback_text_block(self.message_label, min_height=54)
        plan_layout.addWidget(self.message_label)
        layout.addWidget(self.plan_group)
        self.time_axis_mode_combo.currentIndexChanged.connect(self._refresh_mode_visibility)
        self.time_axis_mode_combo.currentIndexChanged.connect(self._sync_mode_cards_from_combo)
        self.elapsed_unit_combo.currentIndexChanged.connect(self._refresh_mode_visibility)
        self.synthetic_basis_combo.currentIndexChanged.connect(self._refresh_mode_visibility)
        self.synthetic_value_edit.textChanged.connect(self._refresh_mode_visibility)
        self.override_edit.textChanged.connect(self._refresh_mode_visibility)
        self._sync_mode_cards_from_combo()
        self._refresh_mode_visibility()

    def _build_mode_card(self, mode_key: str, title: str, description: str) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setStyleSheet(
            "QFrame { border-radius: 6px; }"
            "QLabel { background: transparent; }"
            "QRadioButton { background: transparent; font-weight: 600; }"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(3)
        radio = QRadioButton(title)
        radio.toggled.connect(
            lambda checked, key=mode_key: self._on_mode_card_toggled(key, checked)
        )
        description_label = QLabel(description)
        description_label.setWordWrap(True)
        description_label.setStyleSheet("color: #888888; font-size: 11px;")
        card_layout.addWidget(radio)
        card_layout.addWidget(description_label)
        self._mode_radios[mode_key] = radio
        self._mode_button_group.addButton(radio)
        card.mousePressEvent = lambda _event, btn=radio: btn.setChecked(True)
        return card

    def _on_mode_card_toggled(self, mode_key: str, checked: bool) -> None:
        if not checked:
            return
        index = self.time_axis_mode_combo.findData(mode_key)
        if index >= 0 and index != self.time_axis_mode_combo.currentIndex():
            self.time_axis_mode_combo.setCurrentIndex(index)

    def _sync_mode_cards_from_combo(self) -> None:
        mode = self.selected_time_axis_mode()
        radio = self._mode_radios.get(mode)
        if radio is not None and not radio.isChecked():
            radio.blockSignals(True)
            radio.setChecked(True)
            radio.blockSignals(False)

    def _configure_feedback_text_block(self, label: QLabel, *, min_height: int) -> None:
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        label.setMinimumHeight(min_height)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setFrameShape(QFrame.Shape.StyledPanel)
        label.setContentsMargins(8, 6, 8, 6)

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
            self.recon_advanced_check.setChecked(True)
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
        self._refresh_mode_visibility()
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
        self._refresh_time_axis_details(column_name, detected_format)
        if feedback:
            self.message_label.setText(feedback)

    def selected_time_axis_mode(self) -> str:
        return str(self.time_axis_mode_combo.currentData() or "auto")

    def selected_elapsed_unit(self) -> str:
        return str(self.elapsed_unit_combo.currentData() or "elapsed_seconds")

    def synthetic_timing_basis(self) -> str:
        return str(self.synthetic_basis_combo.currentData() or "sample_rate")

    def synthetic_timing_value(self) -> float | None:
        text = self.synthetic_value_edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _refresh_mode_visibility(self) -> None:
        mode = self.selected_time_axis_mode()
        show_elapsed = mode == "elapsed"
        show_synthetic = mode == "synthetic_elapsed"
        self.elapsed_unit_label.setVisible(show_elapsed)
        self.elapsed_unit_combo.setVisible(show_elapsed)
        self.synthetic_basis_label.setVisible(show_synthetic)
        self.synthetic_basis_combo.setVisible(show_synthetic)
        self.synthetic_value_edit.setVisible(show_synthetic)
        self._mode_settings_widget.setVisible(show_elapsed or show_synthetic)
        if self.synthetic_timing_basis() == "sample_interval":
            self.synthetic_value_edit.setPlaceholderText("Sample interval, e.g. 0.01")
        else:
            self.synthetic_value_edit.setPlaceholderText("Sample rate, e.g. 100")
        self._refresh_time_axis_details(
            self.selected_column_label.text() if self.selected_column_label.text() != "Not used" else None,
            self.detected_format_label.text(),
        )

    def _refresh_time_axis_details(
        self,
        column_name: str | None,
        detected_format: str | None,
    ) -> None:
        mode = self.selected_time_axis_mode()
        elapsed_formats = {"elapsed_seconds", "elapsed_milliseconds", "elapsed_minutes"}
        if mode == "sample_index":
            self.override_group.setTitle("Sample Index Settings")
            self.selected_column_caption.setText("Selected column")
            self.selected_column_label.setText("Not used")
            self.detected_format_caption.setText("Axis basis")
            self.detected_format_label.setText("Sample index")
            self.format_source_caption.setText("Source")
            self.override_status_label.setText("User-selected mode")
            self.manual_format_caption.setVisible(False)
            self.override_edit.setVisible(False)
            self.reset_button.setVisible(False)
            return

        if mode == "synthetic_elapsed":
            self.override_group.setTitle("Synthetic Time Settings")
            self.selected_column_caption.setText("Selected column")
            self.selected_column_label.setText("Not used")
            self.detected_format_caption.setText("Axis basis")
            self.detected_format_label.setText("Synthetic elapsed time")
            self.format_source_caption.setText("Source")
            self.override_status_label.setText("User-selected mode")
            self.manual_format_caption.setVisible(False)
            self.override_edit.setVisible(False)
            self.reset_button.setVisible(False)
            return

        if mode == "elapsed":
            self.override_group.setTitle("Elapsed Time Settings")
            self.selected_column_caption.setText("Selected column")
            self.selected_column_label.setText(column_name or "None")
            self.detected_format_caption.setText("Elapsed unit")
            self.detected_format_label.setText(self.elapsed_unit_combo.currentText())
            self.format_source_caption.setText("Source")
            self.override_status_label.setText("User-selected mode")
            self.manual_format_caption.setVisible(False)
            self.override_edit.setVisible(False)
            self.reset_button.setVisible(False)
            return

        self.override_group.setTitle(
            "Timestamp Format" if mode == "absolute" else "Detected Timestamp Details"
        )
        self.selected_column_caption.setText("Selected column")
        self.selected_column_label.setText(column_name or "None")
        self.detected_format_caption.setText("Detected format")
        self.detected_format_label.setText(detected_format or "auto")
        self.format_source_caption.setText("Format source")
        has_override = bool(self.override_edit.text().strip())
        self.override_status_label.setText("User Override" if has_override else "Auto-detected")
        show_manual_format = mode == "absolute" or (
            mode == "auto" and (detected_format or "") not in elapsed_formats
        )
        self.manual_format_caption.setVisible(show_manual_format)
        self.override_edit.setVisible(show_manual_format)
        self.reset_button.setVisible(show_manual_format)
        self.override_edit.setEnabled(show_manual_format)


class ColumnMappingPage(QWidget):
    """Column mapping review page."""

    def __init__(self, model: ColumnMappingTableModel, parent=None) -> None:
        super().__init__(parent)
        self._model = model
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Review detected signal mappings. The time axis is selected in the previous step."))
        self.time_axis_label = QLabel("Time axis: not selected")
        self.time_axis_label.setWordWrap(True)
        layout.addWidget(self.time_axis_label)
        self.table = QTableView()
        self.table.setModel(model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.setItemDelegateForColumn(3, ParameterTypeDelegate(self.table))
        self.table.setItemDelegateForColumn(4, UnitDelegate(self.table))
        model.modelReset.connect(self._schedule_mapping_select_refresh)
        model.rowsInserted.connect(self._schedule_mapping_select_refresh)
        model.layoutChanged.connect(self._schedule_mapping_select_refresh)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(72)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 72)
        self.table.setColumnWidth(3, 220)
        self.table.setColumnWidth(4, 140)
        self.table.setColumnWidth(5, 104)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.verticalHeader().setMinimumSectionSize(32)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        layout.addWidget(self.table, 1)
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

    def _schedule_mapping_select_refresh(self) -> None:
        QTimer.singleShot(0, self._show_mapping_selects)

    def _show_mapping_selects(self) -> None:
        model = self.table.model()
        if model is None:
            return
        for row in range(model.rowCount()):
            self.table.openPersistentEditor(model.index(row, 3))
            self.table.openPersistentEditor(model.index(row, 4))

    def refresh(self, session: ImportWizardSession | None) -> None:
        mappings = self._model.visible_mappings
        included = sum(1 for mapping in mappings if not mapping.excluded)
        overrides = sum(1 for mapping in mappings if mapping.has_user_override or mapping.excluded)
        time_axis = session.selected_timestamp_column if session else None
        hidden = self._model.hidden_time_axis_count()
        self.time_axis_label.setText(f"Time axis: {time_axis or 'not selected'}")
        hidden_text = f" {hidden} time-axis column hidden." if hidden else ""
        self.message_label.setText(
            f"{included} included column(s), {len(mappings) - included} excluded. "
            f"{overrides} user override(s).{hidden_text}"
        )
        self._schedule_mapping_select_refresh()


class ReviewImportPage(QWidget):
    """Summary before import execution."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary = QTextEdit()
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
        if plan is not None:
            selected_sources = set(plan.selected_columns)
            included = [m for m in mappings if m.source_name in selected_sources]
            plan_messages = plan.validation_messages
        else:
            included = [
                m
                for m in mappings
                if not m.excluded and m.effective_type != ParameterType.TIMESTAMP
            ]
            plan_messages = []
        digital = [m for m in included if m.effective_type.value == "digital"]
        excluded = [
            m.source_name
            for m in mappings
            if m.excluded and m.effective_type != ParameterType.TIMESTAMP
        ]
        warnings = len([m for m in plan_messages if m.severity.value == "warning"])
        errors = len([m for m in plan_messages if m.severity.value == "error"])
        executable = plan.is_executable if plan is not None else False
        blocking_issues: list[str] = []
        if plan is None:
            blocking_issues.append("Execution plan has not been built yet.")
        elif not executable:
            blocking_issues = [
                message.message
                for message in plan.validation_messages
                if message.severity.value == "error"
            ]
            if plan.timestamp_plan is None:
                blocking_issues.append("No validated time-axis repair plan is available.")
            elif not plan.timestamp_plan.repair_validated:
                blocking_issues.append("The selected time-axis parsing settings are not valid.")
            if not plan.selected_columns:
                blocking_issues.append("No signal/data columns are selected for import.")
        warning_issues = [
            message.message
            for message in plan_messages
            if message.severity.value == "warning"
        ]

        lines = [
            f"File: {session.source_path}",
            f"Provider: {session.provider_type}",
            f"Time axis mode: {session.time_axis_mode}",
            f"Timestamp column: {selected}",
            f"Timestamp format: {fmt or 'auto'}",
        ]
        if session.time_axis_mode == "synthetic_elapsed":
            if session.sample_rate_hz:
                lines.append(f"Synthetic sample rate: {session.sample_rate_hz:g} Hz")
            if session.sample_interval_seconds:
                lines.append(f"Synthetic sample interval: {session.sample_interval_seconds:g} s")
        elif session.time_axis_mode == "sample_index":
            lines.append("X-axis: Sample Index (non-time sequence)")
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
        if blocking_issues:
            lines.append("Blocking issues:")
            lines.extend(f"- {issue}" for issue in blocking_issues)
        if warning_issues:
            lines.append("Warning details:")
            lines.extend(f"- {issue}" for issue in warning_issues)
        self.summary.setHtml(self._format_review_html(lines, warning_issues))

    def _format_review_html(self, lines: list[str], warning_issues: list[str]) -> str:
        warning_start = lines.index("Warning details:") if warning_issues else len(lines)
        normal_lines = lines[:warning_start]
        warning_lines = lines[warning_start:]
        html = [
            "<div style='white-space: pre-wrap;'>",
            "<br>".join(escape(line) for line in normal_lines),
        ]
        if warning_lines:
            html.append(
                "<br><span style='color: #C62828; font-weight: 600;'>"
                + "<br>".join(escape(line) for line in warning_lines)
                + "</span>"
            )
        html.append("</div>")
        return "".join(html)


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
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(8)
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
        self.export_guidance = QLabel("Metadata sidecar is recommended for auditability.")
        self._configure_export_text_block(self.export_guidance, min_height=46)
        self._configure_export_text_block(self.export_status, min_height=82)

        controls = QWidget()
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(6)
        controls_layout.addWidget(QLabel("Format"), 0, 0)
        controls_layout.addWidget(self.export_format_combo, 0, 1)
        controls_layout.addWidget(self.include_metadata_sidecar, 1, 1)
        controls_layout.addWidget(self.overwrite_existing, 2, 1)
        controls_layout.addWidget(self.save_normalized_button, 0, 2)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(2, 0)
        export_layout.addWidget(controls)
        export_layout.addWidget(self._export_text_section("Guidance", self.export_guidance))
        export_layout.addWidget(self._export_text_section("Status", self.export_status))
        layout.addWidget(export_group)

        self.open_waveform = QCheckBox("Open waveform after closing")
        self.open_waveform.setChecked(True)
        layout.addWidget(self.open_waveform)
        self.set_export_enabled(False)

    def _configure_export_text_block(self, label: QLabel, *, min_height: int) -> None:
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        label.setMinimumHeight(min_height)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setFrameShape(QFrame.Shape.StyledPanel)
        label.setContentsMargins(8, 6, 8, 6)

    def _export_text_section(self, title: str, content: QLabel) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        heading = QLabel(title)
        heading.setStyleSheet("font-weight: 600;")
        layout.addWidget(heading)
        layout.addWidget(content)
        return section

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
