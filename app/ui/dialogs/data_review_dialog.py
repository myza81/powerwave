"""app/ui/dialogs/data_review_dialog.py

Data Mapping Review Dialog — Phase D4.2.

Shows operator-facing review of:
  Section 1 — Event summary: sources, reference start, display offsets
  Section 2 — Timestamp interpretation: format, confidence, ambiguity warnings
  Section 3 — Column classification table: type, unit, group, confidence, status

Low-confidence and unconfirmed columns are visually highlighted.
The operator can accept (Proceed) or reject (Cancel) before visualization starts.

Usage::

    from app.data.review_summary import build_event_review_summary
    from app.ui.dialogs.data_review_dialog import DataReviewDialog

    summary = build_event_review_summary(session, manifest_data)
    dlg = DataReviewDialog(summary, parent=main_window)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        # proceed to visualization
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
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
    return "✓ Confirmed"


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
    h.setContentsMargins(0, 1, 0, 1)
    h.setSpacing(8)
    key_lbl = _make_bold_label(key)
    key_lbl.setFixedWidth(160)
    key_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    val_lbl = QLabel(value)
    val_lbl.setWordWrap(True)
    if warn:
        val_lbl.setStyleSheet("color: #856404; background: #fff3cd; padding: 2px 4px;")
    h.addWidget(key_lbl)
    h.addWidget(val_lbl, 1)
    return row


def _make_warning_label(text: str) -> QLabel:
    lbl = QLabel(f"⚠  {text}")
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        "color: #856404; background: #fff3cd; "
        "padding: 4px 6px; border: 1px solid #ffc107; border-radius: 3px;"
    )
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────────────


class DataReviewDialog(QDialog):
    """Operator-facing data review dialog shown before visualization.

    Presents event summary, timestamp interpretation, and column classification
    review in three clearly delineated sections. Operator can Proceed or Cancel.

    Args:
        summary: EventReviewSummary from build_event_review_summary().
        parent:  Optional parent QWidget.
    """

    def __init__(self, summary: EventReviewSummary, parent=None) -> None:
        super().__init__(parent)
        self._summary = summary
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle(f"Data Review — {self._summary.event_id}")
        self.setMinimumWidth(820)
        self.resize(920, 680)

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Section 1: Event Summary ──────────────────────────────────────────
        root.addWidget(self._build_event_section())

        # ── Section 2: Timestamp Interpretation ──────────────────────────────
        ts_widget = self._build_timestamp_section()
        if ts_widget is not None:
            root.addWidget(ts_widget)

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
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # Make Proceed the default
        proceed_btn.setDefault(True)
        root.addWidget(buttons)

    # ─────────────────────────────────────────────────────────────────────────
    # Section builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_event_section(self) -> QGroupBox:
        box = QGroupBox("Event Summary")
        vbox = QVBoxLayout(box)
        vbox.setSpacing(2)

        s = self._summary
        vbox.addWidget(_make_kv_row("Event ID:", s.event_id))

        # Reference start
        ref_str = s.reference_start.isoformat(sep=" ") if s.reference_start else "—"
        vbox.addWidget(_make_kv_row("Reference Start:", ref_str))

        # Per-source summary rows
        for src in s.sources:
            src_text = self._fmt_source_row(src)
            vbox.addWidget(_make_kv_row(f"  {src.source_id} ({src.provider_type}):", src_text))

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
        vbox.setSpacing(4)

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

    def _build_column_section(self) -> QGroupBox:
        box = QGroupBox("Column Classification")
        vbox = QVBoxLayout(box)

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
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(False)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table.verticalHeader().setVisible(False)

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
            if row is not None:
                self._populate_table_row(table, r_idx, source_id, row)
            else:
                self._populate_summary_row(table, r_idx, source_id, summary_text or "")

        table.resizeColumnsToContents()

        # Wrap in scroll area so the dialog doesn't expand unboundedly
        scroll = QScrollArea()
        scroll.setWidget(table)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        vbox.addWidget(scroll)

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
        h.setContentsMargins(0, 2, 0, 0)
        h.setSpacing(16)
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
