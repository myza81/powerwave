"""app/ui/dialogs/data_review_dialog.py

Data Mapping Review Dialog — Phase D4.2 / D4.3.

Shows operator-facing review of:
  Section 1 — Event summary: sources, reference start, display offsets
  Section 2 — Timestamp interpretation: format, confidence, ambiguity warnings
             Phase D4.3: ranked interpretations with confidence and reason codes
  Section 3 — Column classification table: type, unit, group, confidence, status

Low-confidence and unconfirmed columns are visually highlighted.
The operator can accept (Proceed) or reject (Cancel) before visualization starts.

Phase D4.3 additions:
  - TimestampInterpretationPanel: shows ranked interpretations when ambiguous
  - High-confidence + no ambiguity: auto-apply (no UI shown for that column)
  - High-confidence + ambiguity: confirmation required
  - Low confidence: operator must choose from ranked list

Usage::

    from app.data.review_summary import build_event_review_summary
    from app.ui.dialogs.data_review_dialog import DataReviewDialog

    summary = build_event_review_summary(session, manifest_data)
    dlg = DataReviewDialog(summary, parent=main_window)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        # access dlg.selected_timestamp_formats for confirmed choices
        # proceed to visualization
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.review_summary import (
    ColumnReviewRow,
    EventReviewSummary,
    SourceReviewSummary,
    TimestampReviewSummary,
)

if TYPE_CHECKING:
    from app.data.timestamp_interpreter import TimestampInterpretation, TimestampInterpretationMatrix

# ─── Row colour thresholds ────────────────────────────────────────────────────
# Colours are light-tinted backgrounds compatible with both light and dark themes.
_COLOUR_CONFIRMED = QColor("#d4edda")   # green  — confirmed, no review needed
_COLOUR_WARN      = QColor("#fff3cd")   # yellow — moderate confidence, flagged
_COLOUR_UNKNOWN   = QColor("#f8d7da")   # red    — low confidence / unknown

# Table column indices for the classification section
_COL_SOURCE = 0
_COL_NAME   = 1
_COL_TYPE   = 2
_COL_UNIT   = 3
_COL_GROUP  = 4
_COL_CONF   = 5
_COL_STATUS = 6
_N_COLS     = 7

_HEADER_LABELS = ["Source", "Column", "Signal Type", "Unit", "Group", "Conf.", "Status"]

_DIALOG_DEFAULT_SIZE = (940, 640)
_DIALOG_MINIMUM_SIZE = (820, 560)
_TABLE_ROW_HEIGHT = 22
_TABLE_HEADER_HEIGHT = 24
_TABLE_MIN_HEIGHT = 112
_TABLE_MAX_HEIGHT = 260


# ─── Timestamp interpretation thresholds (Phase D4.3) ────────────────────────
_AUTO_APPLY_THRESHOLD = 0.85    # confidence ≥ this: auto-apply if unambiguous
_CONFIRM_THRESHOLD = 0.60       # confidence ≥ this: show with pre-selection
# Below _CONFIRM_THRESHOLD → operator must explicitly choose


def _row_colour(row: ColumnReviewRow) -> QColor | None:
    """Map a ColumnReviewRow to its background colour."""
    if row.requires_user_confirmation:
        if row.confidence < 0.50:
            return _COLOUR_UNKNOWN
        return _COLOUR_WARN
    return _COLOUR_CONFIRMED


def _status_text(row: ColumnReviewRow) -> str:
    if row.requires_user_confirmation:
        if row.confidence < 0.50:
            return "⚠ Review required"
        return "⚠ Review suggested"
    if row.inferred_from == "persistent_mapping_rule":
        return "✓ Confirmed [rule]"
    return "✓ Confirmed [heuristic]"


def _fmt_offset(seconds: float) -> str:
    """Format a display offset (seconds) as +Xs or +Xm Ys."""
    sign = "+" if seconds >= 0 else "-"
    secs = abs(seconds)
    if secs < 60:
        return f"{sign}{secs:.1f} s"
    minutes = int(secs // 60)
    rem = secs - minutes * 60
    return f"{sign}{minutes} min {rem:.1f} s  ({sign}{secs:.1f} s)"


def _fmt_rates(rates: list[float]) -> str:
    if not rates:
        return "unknown"
    return ", ".join(f"{r:.4g} Hz" for r in rates)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for building sub-widgets
# ─────────────────────────────────────────────────────────────────────────────


def _make_bold_label(text: str) -> QLabel:
    lbl = QLabel(text)
    font = QFont()
    font.setBold(True)
    lbl.setFont(font)
    return lbl


def _make_kv_row(key: str, value: str, warn: bool = False) -> QWidget:
    """Return a horizontal widget with a bold key label and a value label."""
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    key_lbl = _make_bold_label(key)
    key_lbl.setFixedWidth(132)
    key_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    val_lbl = QLabel(value)
    val_lbl.setWordWrap(True)
    if warn:
        val_lbl.setStyleSheet("color: #856404; background: #fff3cd; padding: 1px 4px;")
    h.addWidget(key_lbl)
    h.addWidget(val_lbl, 1)
    return row


def _make_warning_label(text: str) -> QLabel:
    lbl = QLabel(f"⚠  {text}")
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        "color: #856404; background: #fff3cd; "
        "padding: 3px 5px; border: 1px solid #ffc107; border-radius: 3px;"
    )
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────────────


class DataReviewDialog(QDialog):
    """Operator-facing data review dialog shown before visualization.

    Presents event summary, timestamp interpretation, and column classification
    review in three clearly delineated sections. Operator can Proceed or Cancel.

    Phase D4.3 additions:
      - Timestamp interpretation matrices may be passed via *ts_matrices*.
        When present, ambiguous columns show a ranked-interpretation panel.
      - selected_timestamp_formats: dict[source_id → dict[column_name → format_string]]
        Populated after the dialog is accepted; callers read this to apply formats.

    Args:
        summary:     EventReviewSummary from build_event_review_summary().
        ts_matrices: Optional map of {source_id → {col_name → TimestampInterpretationMatrix}}.
        parent:      Optional parent QWidget.
    """

    def __init__(
        self,
        summary: EventReviewSummary,
        ts_matrices: dict | None = None,    # dict[str, dict[str, TimestampInterpretationMatrix]]
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._summary = summary
        self._ts_matrices: dict = ts_matrices or {}
        # Stores operator-selected format strings: {source_id: {col_name: format_str}}
        self.selected_timestamp_formats: dict[str, dict[str, str]] = {}
        # Confirmed column rows populated on accept: {source_id: [ColumnReviewRow, ...]}
        # Rows with requires_user_confirmation=False and a signal_type are included.
        self.confirmed_column_rows: dict[str, list[ColumnReviewRow]] = {}
        # Radio-button groups keyed by (source_id, col_name) for later harvest
        self._radio_groups: dict[tuple[str, str], QButtonGroup] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle(f"Data Review — {self._summary.event_id}")
        self.setMinimumSize(*_DIALOG_MINIMUM_SIZE)
        self.resize(*_DIALOG_DEFAULT_SIZE)

        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Section 1: Event Summary ──────────────────────────────────────────
        root.addWidget(self._build_event_section())

        # ── Section 2: Timestamp Interpretation ──────────────────────────────
        ts_widget = self._build_timestamp_section()
        if ts_widget is not None:
            root.addWidget(ts_widget)

        # ── Section 2b: Ranked Timestamp Interpretation (Phase D4.3) ─────────
        ts_interp_widget = self._build_ts_interpretation_section()
        if ts_interp_widget is not None:
            root.addWidget(ts_interp_widget)

        # ── Section 3: Column Classification ─────────────────────────────────
        root.addWidget(self._build_column_section(), 1)

        # ── Unconfirmed summary bar ───────────────────────────────────────────
        if self._summary.has_unconfirmed_columns():
            n = self._summary.unconfirmed_count()
            warn = _make_warning_label(
                f"{n} column(s) flagged for review. "
                "You can proceed — flagged items will not affect waveform display."
            )
            root.addWidget(warn)

        # ── Buttons ───────────────────────────────────────────────────────────
        buttons = QDialogButtonBox()
        proceed_btn = buttons.addButton(
            "Proceed to Visualization", QDialogButtonBox.ButtonRole.AcceptRole
        )
        cancel_btn = buttons.addButton(
            "Cancel", QDialogButtonBox.ButtonRole.RejectRole
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        # Make Proceed the default
        proceed_btn.setDefault(True)
        root.addWidget(buttons)

    def showEvent(self, event) -> None:  # noqa: N802
        """Center the dialog on first show without constraining manual resize."""
        super().showEvent(event)
        if getattr(self, "_centered_once", False):
            return
        self._centered_once = True

        parent = self.parentWidget()
        if parent is not None and parent.isVisible():
            target = parent.frameGeometry().center()
        else:
            screen = self.screen()
            if screen is None:
                return
            target = screen.availableGeometry().center()

        frame = self.frameGeometry()
        frame.moveCenter(target)
        self.move(frame.topLeft())

    def _on_accept(self) -> None:
        """Harvest all selections before closing."""
        self._harvest_ts_selections()
        self._harvest_confirmed_rows()
        self.accept()

    def _harvest_confirmed_rows(self) -> None:
        """Collect confirmed column rows per source for the caller to persist.

        Only rows with requires_user_confirmation=False and a valid signal_type
        are included — these are the mappings the operator implicitly confirmed
        by clicking Proceed without overriding them.
        """
        for src in self._summary.sources:
            confirmed = [
                r for r in src.column_rows
                if not r.requires_user_confirmation and r.signal_type
            ]
            if confirmed:
                self.confirmed_column_rows[src.source_id] = confirmed

    def _harvest_ts_selections(self) -> None:
        """Read selected radio buttons and populate selected_timestamp_formats."""
        for (source_id, col_name), group in self._radio_groups.items():
            checked = group.checkedButton()
            if checked is not None:
                fmt = checked.property("format_string")
                if fmt:
                    self.selected_timestamp_formats.setdefault(source_id, {})
                    self.selected_timestamp_formats[source_id][col_name] = fmt

    # ─────────────────────────────────────────────────────────────────────────
    # Section builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_event_section(self) -> QGroupBox:
        box = QGroupBox("Event Summary")
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        grid.setColumnMinimumWidth(0, 136)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        def add_row(row: int, col: int, key: str, value: str) -> None:
            key_lbl = _make_bold_label(key)
            key_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl = QLabel(value)
            val_lbl.setWordWrap(True)
            grid.addWidget(key_lbl, row, col)
            grid.addWidget(val_lbl, row, col + 1)

        s = self._summary

        # Reference start
        ref_str = s.reference_start.isoformat(sep=" ") if s.reference_start else "—"
        add_row(0, 0, "Event ID:", s.event_id)
        add_row(0, 2, "Reference Start:", ref_str)

        # Per-source summary rows
        for idx, src in enumerate(s.sources, start=1):
            src_text = self._fmt_source_row(src)
            add_row(idx, 0, f"{src.source_id} ({src.provider_type}):", src_text)

        return box

    def _fmt_source_row(self, src: SourceReviewSummary) -> str:
        """One-line source summary: samples, rates, offset."""
        rates = _fmt_rates(src.sampling_rates)
        offset = _fmt_offset(src.display_offset_seconds)
        n_a = src.analog_channel_count
        n_d = src.digital_channel_count
        return (
            f"{src.sample_count:,} samples | {n_a}A + {n_d}D | {rates} | "
            f"offset: {offset}"
        )

    def _build_timestamp_section(self) -> QGroupBox | None:
        """Build timestamp section; returns None if there's nothing to flag."""
        if not self._summary.has_timestamp_warnings():
            # Check if any source is non-COMTRADE with low-confidence timestamps
            any_csv_ts = any(
                src.timestamp_summary is not None
                and src.timestamp_summary.confidence < 1.0
                for src in self._summary.sources
            )
            if not any_csv_ts:
                return None

        box = QGroupBox("Timestamp Interpretation")
        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(8, 6, 8, 6)
        vbox.setSpacing(2)

        for src in self._summary.sources:
            ts = src.timestamp_summary
            if ts is None:
                continue

            # Source sub-header
            vbox.addWidget(_make_bold_label(f"Source: {src.source_id}"))

            fmt_text = ts.confirmed_format or ts.raw_format or "unknown"
            is_warn = ts.confidence < 1.0
            vbox.addWidget(_make_kv_row("  Format:", fmt_text, warn=is_warn))
            vbox.addWidget(_make_kv_row("  Confidence:", f"{ts.confidence:.0%}"))
            vbox.addWidget(_make_kv_row("  Inferred from:", ts.inferred_from))

            for w in ts.warnings:
                vbox.addWidget(_make_warning_label(w))

        return box

    def _build_ts_interpretation_section(self) -> QGroupBox | None:
        """Phase D4.3: ranked timestamp interpretation panel.

        Only shown when ts_matrices were provided and at least one column has
        ambiguous or low-confidence timestamp interpretations.

        Rules:
          confidence >= 0.85 and not ambiguous  → auto-apply (not shown)
          confidence >= 0.60 or ambiguous        → show with pre-selected recommendation
          confidence < 0.60                      → show; operator must choose explicitly
        """
        if not self._ts_matrices:
            return None

        # Collect entries that require operator interaction
        panels: list[QWidget] = []

        for source_id, col_matrices in self._ts_matrices.items():
            for col_name, matrix in col_matrices.items():
                if matrix.recommended is None:
                    continue

                rec = matrix.recommended
                auto_apply = (
                    rec.confidence >= _AUTO_APPLY_THRESHOLD
                    and not matrix.is_ambiguous
                )
                if auto_apply:
                    # Auto-applied — pre-populate without showing
                    self.selected_timestamp_formats.setdefault(source_id, {})
                    self.selected_timestamp_formats[source_id][col_name] = rec.format_string
                    continue

                panel = self._build_single_ts_interp_panel(source_id, col_name, matrix)
                panels.append(panel)

        if not panels:
            return None

        box = QGroupBox("Timestamp Format Selection — Operator Confirmation Required")
        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(8, 6, 8, 6)
        vbox.setSpacing(3)

        intro = QLabel(
            "The following timestamp column(s) have ambiguous date formats. "
            "Please confirm the correct interpretation before proceeding."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #856404; padding: 1px 0;")
        vbox.addWidget(intro)

        for panel in panels:
            # Separator line between panels
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            vbox.addWidget(line)
            vbox.addWidget(panel)

        return box

    def _build_single_ts_interp_panel(
        self,
        source_id: str,
        col_name: str,
        matrix,   # TimestampInterpretationMatrix
    ) -> QWidget:
        """Build the ranked-interpretation panel for one timestamp column."""
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(2)

        rec = matrix.recommended

        # Header
        need_choice = rec is not None and rec.confidence < _CONFIRM_THRESHOLD
        header_text = (
            f"Column: <b>{col_name}</b>  |  Source: {source_id}"
        )
        if matrix.is_ambiguous:
            header_text += "  ⚠ Ambiguous date format"
        elif need_choice:
            header_text += "  ⚠ Low confidence — please confirm"

        header = QLabel(header_text)
        header.setTextFormat(Qt.TextFormat.RichText)
        header.setWordWrap(True)
        vbox.addWidget(header)

        # Sample value preview
        if matrix.sample_values:
            sample_text = ", ".join(matrix.sample_values[:3])
            vbox.addWidget(QLabel(f"  Sample values: <i>{sample_text}</i>"))

        # Radio group
        group = QButtonGroup(container)
        self._radio_groups[(source_id, col_name)] = group

        # Show top-N interpretations (max 5)
        top_n = matrix.interpretations[:5]
        for i, interp in enumerate(top_n):
            radio = self._make_interp_radio(interp, is_recommended=(i == 0))
            radio.setProperty("format_string", interp.format_string)
            group.addButton(radio)
            vbox.addWidget(radio)
            # Pre-select recommended when confidence is sufficient
            if i == 0 and rec is not None and rec.confidence >= _CONFIRM_THRESHOLD:
                radio.setChecked(True)

        return container

    def _make_interp_radio(
        self,
        interp,   # TimestampInterpretation
        is_recommended: bool = False,
    ) -> QRadioButton:
        """Build one radio button representing a timestamp interpretation."""
        conf_pct = f"{interp.confidence:.0%}"
        label = interp.format_label

        if is_recommended:
            prefix = "[Recommended] "
            style = "font-weight: bold;"
        else:
            prefix = "[Alternative]  "
            style = ""

        # Show sample parsed value if available
        sample_str = ""
        if interp.parsed_samples:
            sample_str = f"  →  {interp.parsed_samples[0].strftime('%Y-%m-%d %H:%M:%S')}"

        # Reason codes (truncated for display)
        reasons = ", ".join(interp.reason_codes[:3]) if interp.reason_codes else ""
        reason_str = f"  ({reasons})" if reasons else ""

        text = f"{prefix}{label}{sample_str}   Confidence: {conf_pct}{reason_str}"
        radio = QRadioButton(text)
        if style:
            radio.setStyleSheet(style)
        return radio

    def _build_column_section(self) -> QGroupBox:
        box = QGroupBox("Column Classification")
        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(8, 6, 8, 6)
        vbox.setSpacing(4)

        # Collect all rows across sources; skip COMTRADE sources with zero rows
        has_any_rows = any(src.column_rows for src in self._summary.sources)
        if not has_any_rows:
            vbox.addWidget(QLabel(
                "No column classifications to review "
                "(all sources are provider-typed COMTRADE records)."
            ))
            return box

        table = QTableWidget()
        table.setColumnCount(_N_COLS)
        table.setHorizontalHeaderLabels(_HEADER_LABELS)
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(42)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setFixedHeight(_TABLE_HEADER_HEIGHT)
        for col in range(_N_COLS):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_STATUS, QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(False)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(_TABLE_ROW_HEIGHT)
        table.verticalHeader().setMinimumSectionSize(18)

        # Populate rows — sources with no column_rows show a summary line
        all_rows: list[tuple[str, ColumnReviewRow | None, str | None]] = []
        for src in self._summary.sources:
            if src.column_rows:
                for row in src.column_rows:
                    all_rows.append((src.source_id, row, None))
            else:
                # Placeholder summary row for COMTRADE with provider-inferred channels
                all_rows.append((
                    src.source_id,
                    None,
                    f"{src.analog_channel_count} analog channels — provider-inferred, no review needed",
                ))

        table.setRowCount(len(all_rows))
        for r_idx, (source_id, row, summary_text) in enumerate(all_rows):
            table.setRowHeight(r_idx, _TABLE_ROW_HEIGHT)
            if row is not None:
                self._populate_table_row(table, r_idx, source_id, row)
            else:
                self._populate_summary_row(table, r_idx, source_id, summary_text or "")

        table.resizeColumnsToContents()
        table_height = (
            _TABLE_HEADER_HEIGHT
            + (len(all_rows) * _TABLE_ROW_HEIGHT)
            + table.frameWidth() * 2
            + 6
        )
        table.setMinimumHeight(max(_TABLE_MIN_HEIGHT, min(table_height, _TABLE_MAX_HEIGHT)))
        table.setMaximumHeight(_TABLE_MAX_HEIGHT)
        vbox.addWidget(table)

        # Legend
        legend = self._build_legend()
        vbox.addWidget(legend)

        return box

    def _populate_table_row(
        self,
        table: QTableWidget,
        r_idx: int,
        source_id: str,
        row: ColumnReviewRow,
    ) -> None:
        colour = _row_colour(row)
        status = _status_text(row)

        cells = [
            source_id,
            row.column_name,
            row.signal_type or "—",
            row.unit or "—",
            row.display_group,
            f"{row.confidence:.0%}",
            status,
        ]
        for c_idx, text in enumerate(cells):
            item = QTableWidgetItem(str(text))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if colour is not None:
                item.setBackground(colour)
            table.setItem(r_idx, c_idx, item)

    def _populate_summary_row(
        self,
        table: QTableWidget,
        r_idx: int,
        source_id: str,
        summary_text: str,
    ) -> None:
        """Placeholder row for sources with no explicit column review."""
        table.setItem(r_idx, _COL_SOURCE, QTableWidgetItem(source_id))
        summary_item = QTableWidgetItem(summary_text)
        summary_item.setFlags(summary_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        summary_item.setForeground(QColor("#6c757d"))
        table.setItem(r_idx, _COL_NAME, summary_item)
        table.setSpan(r_idx, _COL_NAME, 1, _N_COLS - 1)

    def _build_legend(self) -> QWidget:
        legend = QWidget()
        h = QHBoxLayout(legend)
        h.setContentsMargins(0, 1, 0, 0)
        h.setSpacing(10)
        h.addWidget(QLabel("Legend:"))
        for colour, label in (
            (_COLOUR_CONFIRMED, "✓ Confirmed"),
            (_COLOUR_WARN,      "⚠ Review suggested"),
            (_COLOUR_UNKNOWN,   "⚠ Review required"),
        ):
            swatch = QLabel(f"  {label}  ")
            swatch.setAutoFillBackground(True)
            palette = swatch.palette()
            palette.setColor(swatch.backgroundRole(), colour)
            swatch.setPalette(palette)
            h.addWidget(swatch)
        h.addStretch()
        return legend
