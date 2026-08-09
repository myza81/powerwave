"""Calculated Signal creation/preview dialog — Phase 3A.

User flow::

    select analog channels (AnalogInputSelectorDialog)
            |
    assign variable aliases A/B/C...
            |
    write expression (aliases only)
            |
    choose reference signal
            |
    review units/alignment/overlap in Preview
            |
    Create -> session.add_calculated_signal() + resolution_service.resolve_one()

This dialog is pure UI: it constructs a CalculatedSignalDefinition from user
input and drives it entirely through the existing backend --
app.calculated_signals (Phase 2A expression validation, Phase 2B unit rules/
alignment/calculation) and app.calculated_signals.resolver.
CalculatedSignalResolutionService (Phase 2C-3 dependency resolution,
Phase 3A's non-mutating preview_definition()). No expression parsing, unit
rule, time-alignment, interpolation, or calculation logic is implemented
here.

Out of scope for Phase 3A (see the task's own scope boundary): Session
Canvas rendering of calculated signals, visibility controls, curve deletion,
post-creation editing, persistence, export, calculated-signal chaining, and
background workers. Preview is metadata/statistics only -- no waveform plot.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.calculated_signals.expression import (
    ExpressionSyntaxError,
    ExpressionValidationError,
    validate_expression,
)
from app.calculated_signals.models import CalculatedSignalDefinition, CalculatedSignalResult, ChannelRef
from app.calculated_signals.resolver import (
    CalculatedSignalResolutionError,
    CalculatedSignalResolutionService,
)
from app.sessions.event_session import EventAnalysisSession
from app.ui.calculated_signals.analog_input_selector import AnalogInputSelectorDialog

_ALIASES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Expression Builder toolbar (UX experiment): functions offered here must
# match app.calculated_signals.expression.validate_expression()'s actual
# supported set exactly -- currently only abs(). This mirrors, rather than
# imports, that module's internal function allow-list (which is private):
# adding a name here without the parser also accepting it would let the
# builder offer something Create then rejects, which is worse than not
# offering it at all.
_SUPPORTED_FUNCTIONS: tuple[str, ...] = ("abs",)

# Operator buttons: (display glyph, parser-compatible text, tooltip).
# Display may use × / ÷ for readability; only the parser-compatible text is
# ever inserted into the expression.
_OPERATOR_BUTTONS: tuple[tuple[str, str, str], ...] = (
    ("+", "+", "Add"),
    ("−", "-", "Subtract"),
    ("×", "*", "Multiply"),
    ("÷", "/", "Divide"),
    ("(", "(", "Open parenthesis"),
    (")", ")", "Close parenthesis"),
)


@dataclass(slots=True)
class _Binding:
    """One row of the dialog's Inputs table. variable + ref (source_id,
    channel_name) is the only identity used when constructing a
    CalculatedSignalDefinition; the *_display_name/unit fields are shown to
    the user only and never feed back into any backend call.
    """

    variable: str
    ref: ChannelRef
    source_display_name: str
    channel_display_name: str
    unit: str | None


class CalculatedSignalDialog(QDialog):
    """Modal dialog for creating one Calculated Signal in *session*."""

    def __init__(self, session: EventAnalysisSession, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calculated Signals")
        self.setMinimumSize(600, 700)

        self._session = session
        self._resolution_service = CalculatedSignalResolutionService(session)
        self._calc_id = str(uuid.uuid4())

        self._bindings: list[_Binding] = []
        self._preview_result: CalculatedSignalResult | None = None
        self._preview_fingerprint: tuple | None = None
        self._preview_current = False
        self._suspend_signals = False

        self._build_ui()
        self._refresh_bindings_table()
        self._refresh_reference_choices()
        self._update_create_enabled()
        self._update_preview_status_label()
        self._update_expression_feedback()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(self._build_name_group())
        layout.addWidget(self._build_inputs_group())
        layout.addWidget(self._build_expression_group())
        layout.addWidget(self._build_reference_group())
        layout.addWidget(self._build_output_unit_group())
        layout.addWidget(self._build_preview_group(), stretch=1)
        layout.addLayout(self._build_bottom_buttons())

    def _build_name_group(self) -> QGroupBox:
        group = QGroupBox("Name")
        row = QHBoxLayout(group)
        row.addWidget(QLabel("Calculated Signal Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Net System Power")
        self._name_edit.textChanged.connect(self._on_name_changed)
        row.addWidget(self._name_edit)
        return group

    def _build_inputs_group(self) -> QGroupBox:
        group = QGroupBox("Inputs / Variable Bindings")
        vbox = QVBoxLayout(group)

        self._bindings_table = QTableWidget(0, 4)
        self._bindings_table.setHorizontalHeaderLabels(["Variable", "Source", "Channel", "Unit"])
        self._bindings_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._bindings_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self._bindings_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._bindings_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._bindings_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._bindings_table.setMaximumHeight(160)
        vbox.addWidget(self._bindings_table)

        buttons_row = QHBoxLayout()
        self._add_input_btn = QPushButton("Add Input")
        self._add_input_btn.clicked.connect(self._on_add_input)
        buttons_row.addWidget(self._add_input_btn)
        self._remove_input_btn = QPushButton("Remove Input")
        self._remove_input_btn.clicked.connect(self._on_remove_input)
        buttons_row.addWidget(self._remove_input_btn)
        buttons_row.addStretch(1)
        vbox.addLayout(buttons_row)

        return group

    def _build_expression_group(self) -> QGroupBox:
        group = QGroupBox("Expression")
        vbox = QVBoxLayout(group)
        self._expression_edit = QLineEdit()
        self._expression_edit.setPlaceholderText("e.g. A - B")
        self._expression_edit.textChanged.connect(self._on_expression_changed)
        vbox.addWidget(self._expression_edit)
        vbox.addLayout(self._build_expression_builder_row())
        return group

    def _build_expression_builder_row(self) -> QHBoxLayout:
        """Compact Expression Builder toolbar (UX experiment): Signal and
        Function dropdowns plus operator/parenthesis buttons, all inserting
        at the expression field's current cursor/selection. Replaces the
        former static "Supported: ..." helper line in the same row -- see
        _update_expression_feedback() for where that hint text now lives
        (the field's own tooltip, so this row costs no extra height).
        """
        row = QHBoxLayout()
        row.setSpacing(4)

        self._signal_menu_button = QToolButton()
        self._signal_menu_button.setText("Signal ▾")
        self._signal_menu_button.setToolTip("Insert a bound input signal")
        self._signal_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._signal_menu = QMenu(self._signal_menu_button)
        self._signal_menu_button.setMenu(self._signal_menu)
        row.addWidget(self._signal_menu_button)

        self._function_menu_button = QToolButton()
        self._function_menu_button.setText("Function ▾")
        self._function_menu_button.setToolTip("Wrap the selection, or insert, a supported function")
        self._function_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        function_menu = QMenu(self._function_menu_button)
        for func_name in _SUPPORTED_FUNCTIONS:
            action = function_menu.addAction(f"{func_name}()")
            action.triggered.connect(
                lambda checked=False, f=func_name: self._insert_function(f)
            )
        self._function_menu_button.setMenu(function_menu)
        row.addWidget(self._function_menu_button)

        for glyph, token, tooltip in _OPERATOR_BUTTONS:
            btn = QToolButton()
            btn.setText(glyph)
            btn.setToolTip(tooltip)
            btn.setMaximumWidth(28)
            btn.clicked.connect(lambda checked=False, t=token: self._insert_text_at_cursor(t))
            row.addWidget(btn)

        row.addStretch(1)
        self._refresh_signal_menu()
        return row

    # ─────────────────────────────────────────────────────────────────────────
    # Expression Builder: cursor-position insertion helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_signal_menu(self) -> None:
        """Rebuild the Signal dropdown from current bindings only (Step 12:
        the menu must always reflect exactly what is bound right now, never
        a stale or renumbered list)."""
        self._signal_menu.clear()
        for b in self._bindings:
            unit_text = b.unit if b.unit else "Unknown"
            label = f"{b.variable} — {b.source_display_name} / {b.channel_display_name} [{unit_text}]"
            action = self._signal_menu.addAction(label)
            action.triggered.connect(
                lambda checked=False, v=b.variable: self._insert_text_at_cursor(v)
            )
        self._signal_menu_button.setEnabled(bool(self._bindings))

    def _insert_text_at_cursor(self, text: str) -> None:
        """Insert *text* at the expression field's current cursor position,
        replacing the current selection if any (QLineEdit.insert()'s own
        behaviour) -- shared by Signal and operator/parenthesis buttons so
        insertion logic is not duplicated per button."""
        edit = self._expression_edit
        edit.insert(text)
        edit.setFocus()

    def _insert_function(self, func_name: str) -> None:
        """Insert a supported function call at the cursor. Wraps the current
        selection, e.g. selecting "A-B" and choosing abs() produces
        "abs(A-B)"; with nothing selected, inserts "func()" with the cursor
        left positioned between the parentheses so the user can keep typing
        immediately."""
        edit = self._expression_edit
        selected = edit.selectedText()
        if selected:
            edit.insert(f"{func_name}({selected})")
        else:
            pos = edit.cursorPosition()
            edit.insert(f"{func_name}()")
            edit.setCursorPosition(pos + len(func_name) + 1)
        edit.setFocus()

    def _update_expression_feedback(self) -> None:
        """Lightweight syntax-only feedback (Step 9): reuses the field's own
        tooltip and a subtle border tint instead of a dedicated status row,
        so this costs no extra dialog height. Never runs the numerical
        preview, never touches session data, never pops up a dialog --
        Preview remains the sole authority for numerical/unit/alignment
        validation. An incomplete expression (e.g. "A+") is expected while
        typing and is shown the same subtle way as any other syntax error.
        """
        edit = self._expression_edit
        text = edit.text().strip()
        if not text:
            edit.setStyleSheet("")
            edit.setToolTip("Supported: +  -  *  /  ( )  abs()")
            return
        allowed_variables = {b.variable for b in self._bindings}
        try:
            validate_expression(text, allowed_variables)
        except (ExpressionSyntaxError, ExpressionValidationError) as exc:
            edit.setStyleSheet("border: 1px solid #b06000;")
            edit.setToolTip(str(exc))
        else:
            edit.setStyleSheet("")
            edit.setToolTip("Valid expression")

    def _build_reference_group(self) -> QGroupBox:
        group = QGroupBox("Reference Signal")
        row = QHBoxLayout(group)
        row.addWidget(QLabel("Reference signal (controls output time base):"))
        self._reference_combo = QComboBox()
        self._reference_combo.currentIndexChanged.connect(self._on_reference_changed)
        row.addWidget(self._reference_combo, stretch=1)
        return group

    def _build_output_unit_group(self) -> QGroupBox:
        group = QGroupBox("Output Unit")
        row = QHBoxLayout(group)
        row.addWidget(QLabel("Output unit:"))
        self._output_unit_edit = QLineEdit()
        self._output_unit_edit.setPlaceholderText("Auto")
        self._output_unit_edit.textChanged.connect(self._on_output_unit_changed)
        row.addWidget(self._output_unit_edit)
        return group

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("Preview")
        vbox = QVBoxLayout(group)

        status_row = QHBoxLayout()
        self._preview_status_label = QLabel()
        status_row.addWidget(self._preview_status_label)
        status_row.addStretch(1)
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview_clicked)
        status_row.addWidget(self._preview_btn)
        vbox.addLayout(status_row)

        self._preview_error_label = QLabel()
        self._preview_error_label.setStyleSheet("color: #b00020; font-weight: bold;")
        self._preview_error_label.setWordWrap(True)
        self._preview_error_label.setVisible(False)
        vbox.addWidget(self._preview_error_label)

        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setStyleSheet("font-family: monospace;")
        vbox.addWidget(self._preview_text, stretch=1)

        return group

    def _build_bottom_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        row.addWidget(self._cancel_btn)
        self._create_btn = QPushButton("Create")
        self._create_btn.clicked.connect(self._on_create_clicked)
        row.addWidget(self._create_btn)
        return row

    # ─────────────────────────────────────────────────────────────────────────
    # Change handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_name_changed(self, _text: str) -> None:
        if self._suspend_signals:
            return
        # Name is not a "meaningful" definition field for calculation
        # purposes (matches EventAnalysisSession's own
        # _MEANINGFUL_DEFINITION_FIELDS), so it does not invalidate the
        # numerical preview -- only Create's structural enablement.
        self._update_create_enabled()

    def _on_expression_changed(self, _text: str) -> None:
        if self._suspend_signals:
            return
        self._mark_preview_outdated()
        self._update_expression_feedback()

    def _on_reference_changed(self, _index: int) -> None:
        if self._suspend_signals:
            return
        self._mark_preview_outdated()

    def _on_output_unit_changed(self, _text: str) -> None:
        if self._suspend_signals:
            return
        self._mark_preview_outdated()

    # ─────────────────────────────────────────────────────────────────────────
    # Bindings management
    # ─────────────────────────────────────────────────────────────────────────

    def _next_alias(self) -> str:
        used = {b.variable for b in self._bindings}
        for letter in _ALIASES:
            if letter not in used:
                return letter
        raise ValueError("No more variable aliases available (26 inputs already bound).")

    def _on_add_input(self) -> None:
        dlg = AnalogInputSelectorDialog(self._session, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dlg.selected_channel()
        if selected is None:
            return

        if any(b.ref == selected.ref for b in self._bindings):
            QMessageBox.information(
                self,
                "Already Bound",
                f"{selected.source_display_name} / {selected.channel_display_name} "
                "is already bound to a variable.",
            )
            return

        try:
            alias = self._next_alias()
        except ValueError as exc:
            QMessageBox.information(self, "Too Many Inputs", str(exc))
            return

        self._bindings.append(
            _Binding(
                variable=alias,
                ref=selected.ref,
                source_display_name=selected.source_display_name,
                channel_display_name=selected.channel_display_name,
                unit=selected.unit,
            )
        )
        self._refresh_bindings_table()
        self._refresh_reference_choices()
        self._mark_preview_outdated()

    def _on_remove_input(self) -> None:
        row = self._bindings_table.currentRow()
        if row < 0 or row >= len(self._bindings):
            return
        del self._bindings[row]
        self._refresh_bindings_table()
        self._refresh_reference_choices()
        self._mark_preview_outdated()

    def _refresh_bindings_table(self) -> None:
        self._suspend_signals = True
        try:
            table = self._bindings_table
            table.setRowCount(len(self._bindings))
            for row, b in enumerate(self._bindings):
                values = (b.variable, b.source_display_name, b.channel_display_name, b.unit or "Unknown")
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    table.setItem(row, col, item)
        finally:
            self._suspend_signals = False
        self._refresh_signal_menu()
        self._update_create_enabled()

    def _refresh_reference_choices(self) -> None:
        self._suspend_signals = True
        try:
            previous = self._reference_combo.currentData()
            self._reference_combo.clear()
            for b in self._bindings:
                label = f"{b.variable} — {b.source_display_name} / {b.channel_display_name}"
                self._reference_combo.addItem(label, userData=b.variable)

            still_valid = any(b.variable == previous for b in self._bindings)
            target = previous if still_valid else (self._bindings[0].variable if self._bindings else None)
            if target is not None:
                for i in range(self._reference_combo.count()):
                    if self._reference_combo.itemData(i) == target:
                        self._reference_combo.setCurrentIndex(i)
                        break
        finally:
            self._suspend_signals = False
        self._update_create_enabled()

    # ─────────────────────────────────────────────────────────────────────────
    # Preview staleness / Create enablement
    # ─────────────────────────────────────────────────────────────────────────

    def _current_fingerprint(self) -> tuple:
        """The subset of dialog state that affects the numerical result --
        deliberately excludes the calculated signal's display name, matching
        EventAnalysisSession's own _MEANINGFUL_DEFINITION_FIELDS policy.
        """
        bindings_key = tuple(
            sorted((b.variable, b.ref.source_id, b.ref.channel_name) for b in self._bindings)
        )
        reference = self._reference_combo.currentData()
        output_unit = self._output_unit_edit.text().strip() or None
        expression = self._expression_edit.text().strip()
        return (expression, bindings_key, reference, output_unit)

    def _mark_preview_outdated(self) -> None:
        self._preview_current = False
        self._update_create_enabled()
        self._update_preview_status_label()

    def _update_preview_status_label(self) -> None:
        if self._preview_result is None:
            self._preview_status_label.setText("Preview: not run yet")
            self._preview_status_label.setStyleSheet("color: gray;")
        elif self._preview_current:
            self._preview_status_label.setText("Preview: current")
            self._preview_status_label.setStyleSheet("color: #1b7a1b;")
        else:
            self._preview_status_label.setText("⚠ Preview out of date — click Preview to refresh")
            self._preview_status_label.setStyleSheet("color: #b06000; font-weight: bold;")

    def _update_create_enabled(self) -> None:
        name_ok = bool(self._name_edit.text().strip())
        bindings_ok = bool(self._bindings)
        expression_ok = bool(self._expression_edit.text().strip())
        reference_ok = self._reference_combo.currentData() is not None
        self._create_btn.setEnabled(name_ok and bindings_ok and expression_ok and reference_ok)

    # ─────────────────────────────────────────────────────────────────────────
    # Definition construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_definition(self) -> CalculatedSignalDefinition:
        name = self._name_edit.text().strip()
        if not name:
            raise ValueError("Calculated signal name must not be empty.")
        if not self._bindings:
            raise ValueError("At least one input must be bound before creating a calculated signal.")
        expression = self._expression_edit.text().strip()
        if not expression:
            raise ValueError("Expression must not be empty.")
        reference = self._reference_combo.currentData()
        if reference is None:
            raise ValueError("A reference signal must be selected.")

        output_unit_text = self._output_unit_edit.text().strip()
        bindings = {b.variable: b.ref for b in self._bindings}

        return CalculatedSignalDefinition(
            calc_id=self._calc_id,
            name=name,
            expression=expression,
            variable_bindings=bindings,
            reference_variable=reference,
            output_unit=output_unit_text or None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Preview
    # ─────────────────────────────────────────────────────────────────────────

    def _clear_preview_error(self) -> None:
        self._preview_error_label.setText("")
        self._preview_error_label.setVisible(False)

    def _show_preview_error(self, message: str) -> None:
        self._preview_error_label.setText(message)
        self._preview_error_label.setVisible(True)

    def _on_preview_clicked(self) -> None:
        self._clear_preview_error()
        try:
            definition = self._build_definition()
        except ValueError as exc:
            self._show_preview_error(str(exc))
            return

        try:
            result = self._resolution_service.preview_definition(definition)
        except CalculatedSignalResolutionError as exc:
            self._show_preview_error(str(exc))
            return

        self._preview_result = result
        self._preview_fingerprint = self._current_fingerprint()
        self._preview_current = True
        self._render_preview_summary(definition, result)
        self._update_preview_status_label()

    def _render_preview_summary(
        self, definition: CalculatedSignalDefinition, result: CalculatedSignalResult
    ) -> None:
        """Render the preview text block from *result* as returned by the
        backend -- sample count, validity, and numerical warnings are read
        directly from `result`, never independently recomputed here.
        """
        lines: list[str] = ["Calculated Signal Preview", ""]

        lines.append("Expression:")
        lines.append(definition.expression)
        lines.append("")

        lines.append("Inputs:")
        for b in self._bindings:
            unit_text = b.unit if b.unit else "Unknown"
            lines.append(f"{b.variable} = {b.source_display_name} / {b.channel_display_name} [{unit_text}]")
        lines.append("")

        lines.append("Reference:")
        lines.append(definition.reference_variable)
        lines.append("")

        total = int(result.time.size)
        valid = int(np.count_nonzero(result.validity_mask))
        duration = float(result.time[-1] - result.time[0]) if total >= 1 else 0.0
        lines.append("Output:")
        lines.append(f"Unit: {result.unit if result.unit else 'Unknown'}")
        lines.append(f"Samples: {total:,}")
        lines.append(f"Duration: {duration:.3f} s")
        lines.append(f"Valid samples: {valid:,} / {total:,}")
        lines.append("")

        lines.append("Alignment:")
        lines.append("Calculation uses the current session alignment offsets.")
        unique_sources = {b.ref.source_id for b in self._bindings}
        if len(unique_sources) <= 1:
            only_source_id = next(iter(unique_sources)) if unique_sources else None
            offset = self._session.get_time_offset(only_source_id) if only_source_id else 0.0
            lines.append(f"All inputs share one source's own time base (offset: {offset:.3f} s).")
        else:
            for b in self._bindings:
                offset = self._session.get_time_offset(b.ref.source_id)
                lines.append(f"{b.variable} offset: {offset:.3f} s")
        lines.append("")

        warnings = list(result.warnings)
        manual_source_names = sorted(
            {
                b.source_display_name
                for b in self._bindings
                if self._is_manually_aligned(b.ref.source_id)
            }
        )
        for source_name in manual_source_names:
            warnings.append(
                f"This calculation uses manually aligned source {source_name!r}. "
                "Verify the time relationship before relying on the result."
            )

        unknown_unit_vars = [b.variable for b in self._bindings if not b.unit]
        if unknown_unit_vars:
            warnings.append(
                "Input " + ", ".join(unknown_unit_vars) + " has no engineering unit. "
                "Numeric constants are interpreted relative to the signal without "
                "verified unit compatibility."
            )

        lines.append("Warnings:")
        if warnings:
            lines.extend(f"• {w}" for w in warnings)
        else:
            lines.append("(none)")

        self._preview_text.setPlainText("\n".join(lines))

    def _is_manually_aligned(self, source_id: str) -> bool:
        source = self._session.get_source(source_id)
        return source is not None and source.alignment_method == "manual"

    # ─────────────────────────────────────────────────────────────────────────
    # Create
    # ─────────────────────────────────────────────────────────────────────────

    def _on_create_clicked(self) -> None:
        self._clear_preview_error()

        if not self._preview_current or self._preview_result is None:
            self._show_preview_error(
                "Run Preview first — Create requires a current, successful preview."
            )
            return

        try:
            definition = self._build_definition()
        except ValueError as exc:
            self._show_preview_error(str(exc))
            return

        if self._current_fingerprint() != self._preview_fingerprint:
            # Defensive: every relevant edit already marks the preview
            # outdated, so this should be unreachable in practice.
            self._mark_preview_outdated()
            self._show_preview_error(
                "Run Preview first — Create requires a current, successful preview."
            )
            return

        try:
            self._session.add_calculated_signal(definition)
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot Create", str(exc))
            return

        try:
            self._resolution_service.resolve_one(definition.calc_id)
        except CalculatedSignalResolutionError as exc:
            # Atomic from the user's perspective: roll back the just-added
            # definition rather than leaving a broken entry in the session.
            self._session.remove_calculated_signal(definition.calc_id)
            QMessageBox.critical(self, "Creation Failed", str(exc))
            return

        self.accept()
