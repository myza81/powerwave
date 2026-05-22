"""Cross-source correlation report panel.

Shows pairwise waveform correlation results for a multi-source session,
helping the analyst determine which sources captured the same event and
what time offset aligns them.

Layout:
  Summary row  — "N of M pairs likely same event"
  Table        — one row per source pair:
                 Source A | Source B | Correlation | Lag (ms) | Confidence | Offset (s)

Colour coding:
  GREEN  — correlation ≥ 0.80 (same event, high confidence)
  AMBER  — correlation 0.50–0.79 (probably same event)
  RED    — correlation < 0.50 (likely different events or no overlap)

Call ``load_results(results)`` to populate.
Call ``clear_results()`` to reset to idle.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.analytics.correlation.cross_correlator import CrossCorrelationResult


def _corr_color(coeff: float) -> str:
    if abs(coeff) >= 0.80:
        return "#55CC55"
    if abs(coeff) >= 0.50:
        return "#FFAA33"
    return "#FF5555"


def _cell(text: str, *, align: int = Qt.AlignmentFlag.AlignCenter) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setTextAlignment(align)
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return it


class CorrelationReportPanel(QWidget):
    """Pairwise cross-correlation results table.

    Public API:
        load_results(results: list[CrossCorrelationResult]) — populate
        clear_results()                                     — reset
    """

    _COL_HEADERS = ["Source A", "Source B", "Correlation", "Lag (ms)",
                    "Confidence", "Suggested Offset (s)"]
    _COL_SRC_A  = 0
    _COL_SRC_B  = 1
    _COL_CORR   = 2
    _COL_LAG    = 3
    _COL_CONF   = 4
    _COL_OFFSET = 5

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CorrelationReportPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # ── Summary row ───────────────────────────────────────────────────────
        summary_frame = QFrame()
        summary_frame.setFrameShape(QFrame.Shape.StyledPanel)
        summary_frame.setStyleSheet("QFrame { background: #252526; border-radius: 4px; }")
        summary_layout = QHBoxLayout(summary_frame)
        summary_layout.setContentsMargins(8, 4, 8, 4)

        self._lbl_summary = QLabel("No multi-source session loaded")
        self._lbl_summary.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        summary_layout.addWidget(self._lbl_summary)
        summary_layout.addStretch()

        self._lbl_hint = QLabel(
            "Apply suggested offsets via the Session Panel offset controls"
        )
        self._lbl_hint.setStyleSheet("color: #555555; font-size: 10px;")
        summary_layout.addWidget(self._lbl_hint)
        root.addWidget(summary_frame)

        # ── Correlation table ─────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(len(self._COL_HEADERS))
        self._table.setHorizontalHeaderLabels(self._COL_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSortingEnabled(True)
        self._table.setShowGrid(True)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                alternate-background-color: #252526;
                color: #CCCCCC;
                gridline-color: #333333;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #2D2D2D;
                color: #AAAAAA;
                border: 1px solid #3C3C3C;
                padding: 2px 4px;
                font-size: 11px;
            }
            QTableWidget::item:selected {
                background-color: #094771;
            }
        """)

        hh = self._table.horizontalHeader()
        hh.setDefaultSectionSize(110)
        hh.setStretchLastSection(True)
        root.addWidget(self._table)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def load_results(self, results: list[CrossCorrelationResult]) -> None:
        """Populate the table with pairwise correlation results."""
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(results))

        bold = QFont()
        bold.setBold(True)

        same_event_count = sum(1 for r in results if r.same_event_likely)

        for row, res in enumerate(results):
            color = _corr_color(res.correlation_coeff)

            src_a = _cell(res.source_a_id,
                          align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, self._COL_SRC_A, src_a)

            src_b = _cell(res.source_b_id,
                          align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, self._COL_SRC_B, src_b)

            corr_item = _cell(f"{res.correlation_coeff:+.3f}")
            corr_item.setForeground(QColor(color))
            corr_item.setFont(bold)
            corr_item.setData(Qt.ItemDataRole.UserRole, res.correlation_coeff)
            tooltip = (
                f"Channels: {res.channel_a} ↔ {res.channel_b}\n"
                f"Same event: {'Yes' if res.same_event_likely else 'No'}"
            )
            corr_item.setToolTip(tooltip)
            self._table.setItem(row, self._COL_CORR, corr_item)

            lag_item = _cell(f"{res.lag_ms:+.2f}")
            lag_item.setData(Qt.ItemDataRole.UserRole, res.lag_ms)
            if abs(res.lag_ms) > 10.0:
                lag_item.setForeground(QColor("#FFAA33"))
            self._table.setItem(row, self._COL_LAG, lag_item)

            conf_item = _cell(res.confidence_label)
            conf_item.setForeground(QColor(res.confidence_color))
            conf_item.setData(Qt.ItemDataRole.UserRole, res.confidence)
            self._table.setItem(row, self._COL_CONF, conf_item)

            offset_item = _cell(f"{res.suggested_offset_s:+.5f}")
            offset_item.setData(Qt.ItemDataRole.UserRole, res.suggested_offset_s)
            offset_item.setForeground(QColor(color))
            offset_item.setToolTip(
                f"Apply {res.suggested_offset_s:+.5f} s to '{res.source_b_id}' "
                f"to align with '{res.source_a_id}'"
            )
            self._table.setItem(row, self._COL_OFFSET, offset_item)

        self._table.setSortingEnabled(True)
        self._table.sortByColumn(self._COL_CORR, Qt.SortOrder.DescendingOrder)
        self._table.resizeColumnToContents(self._COL_SRC_A)
        self._table.resizeColumnToContents(self._COL_SRC_B)

        n_pairs = len(results)
        if same_event_count == n_pairs and n_pairs > 0:
            summary = f"All {n_pairs} pair(s) likely captured the same event"
            style = "color: #55CC55; font-size: 11px;"
        elif same_event_count > 0:
            summary = (
                f"{same_event_count} of {n_pairs} pair(s) likely same event; "
                f"{n_pairs - same_event_count} uncertain"
            )
            style = "color: #FFAA33; font-size: 11px;"
        else:
            summary = f"{n_pairs} pair(s) — no clear same-event match"
            style = "color: #FF5555; font-size: 11px;"

        self._lbl_summary.setText(summary)
        self._lbl_summary.setStyleSheet(style)

    def clear_results(self) -> None:
        """Reset to idle state."""
        self._table.setRowCount(0)
        self._lbl_summary.setText("No multi-source session loaded")
        self._lbl_summary.setStyleSheet("color: #AAAAAA; font-size: 11px;")
