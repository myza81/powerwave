"""Data quality report panel — per-channel health summary dock.

Shows the quality fingerprint computed on file load as a compact table.
Each row is one channel; columns show grade, NaN%, gap count, clipping,
SNR, and DC offset. The header row shows the overall record grade.

Call ``load_quality(result)`` after computing the fingerprint.
Call ``clear_quality()`` when a record is unloaded.
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

from app.analytics.quality.quality_fingerprint import QualityGrade, RecordQuality


_GRADE_COLOR = {
    QualityGrade.OK: "#55CC55",
    QualityGrade.WARN: "#FFAA33",
    QualityGrade.ERROR: "#FF5555",
}
_GRADE_LABEL = {
    QualityGrade.OK: "OK",
    QualityGrade.WARN: "WARN",
    QualityGrade.ERROR: "ERROR",
}


def _cell(text: str, *, align: int = Qt.AlignmentFlag.AlignCenter) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setTextAlignment(align)
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return it


class QualityReportPanel(QWidget):
    """Per-channel data quality table panel.

    Public API:
        load_quality(result: RecordQuality) — populate table
        clear_quality()                     — reset to idle
    """

    _COL_HEADERS = ["Channel", "Grade", "NaN %", "Gaps", "Clipped", "SNR (dB)", "DC Offset"]
    _COL_CHANNEL = 0
    _COL_GRADE = 1
    _COL_NAN = 2
    _COL_GAPS = 3
    _COL_CLIP = 4
    _COL_SNR = 5
    _COL_DC = 6

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("QualityReportPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # Summary header
        summary_frame = QFrame()
        summary_frame.setFrameShape(QFrame.Shape.StyledPanel)
        summary_frame.setStyleSheet("QFrame { background: #252526; border-radius: 4px; }")
        summary_layout = QHBoxLayout(summary_frame)
        summary_layout.setContentsMargins(8, 4, 8, 4)

        self._lbl_overall = QLabel("No record loaded")
        self._lbl_overall.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        summary_layout.addWidget(self._lbl_overall)
        summary_layout.addStretch()

        self._lbl_hint = QLabel("Hover a row for details")
        self._lbl_hint.setStyleSheet("color: #666666; font-size: 10px;")
        summary_layout.addWidget(self._lbl_hint)
        root.addWidget(summary_frame)

        # Quality table
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
        hh.setDefaultSectionSize(80)
        hh.setStretchLastSection(False)
        root.addWidget(self._table)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def load_quality(self, result: RecordQuality) -> None:
        """Populate the table with quality metrics from a fingerprint result."""
        self._table.setSortingEnabled(False)
        channels = list(result.channels.values())
        self._table.setRowCount(len(channels))

        for row, ch in enumerate(channels):
            # Channel name
            name_item = _cell(ch.name, align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, self._COL_CHANNEL, name_item)

            # Grade
            grade_item = _cell(_GRADE_LABEL[ch.grade])
            grade_item.setForeground(QColor(_GRADE_COLOR[ch.grade]))
            bold = QFont()
            bold.setBold(True)
            grade_item.setFont(bold)
            grade_item.setData(Qt.ItemDataRole.UserRole, list(QualityGrade).index(ch.grade))
            self._table.setItem(row, self._COL_GRADE, grade_item)

            # NaN %
            nan_item = _cell(f"{ch.nan_pct:.2f}")
            nan_item.setData(Qt.ItemDataRole.UserRole, ch.nan_pct)
            if ch.nan_pct > 5.0:
                nan_item.setForeground(QColor("#FF5555"))
            elif ch.nan_pct > 0.1:
                nan_item.setForeground(QColor("#FFAA33"))
            self._table.setItem(row, self._COL_NAN, nan_item)

            # Gaps
            gap_item = _cell(str(ch.gap_count))
            gap_item.setData(Qt.ItemDataRole.UserRole, ch.gap_count)
            if ch.gap_count > 10:
                gap_item.setForeground(QColor("#FF5555"))
            elif ch.gap_count > 0:
                gap_item.setForeground(QColor("#FFAA33"))
            self._table.setItem(row, self._COL_GAPS, gap_item)

            # Clipping
            clip_item = _cell(str(ch.clip_count))
            clip_item.setData(Qt.ItemDataRole.UserRole, ch.clip_count)
            if ch.clip_count > 0:
                clip_item.setForeground(QColor("#FFAA33"))
            self._table.setItem(row, self._COL_CLIP, clip_item)

            # SNR
            snr_text = f"{ch.snr_db:.1f}" if ch.snr_db is not None else "—"
            snr_item = _cell(snr_text)
            snr_val = ch.snr_db if ch.snr_db is not None else 999.0
            snr_item.setData(Qt.ItemDataRole.UserRole, snr_val)
            if ch.snr_db is not None and ch.snr_db < 10.0:
                snr_item.setForeground(QColor("#FFAA33"))
            self._table.setItem(row, self._COL_SNR, snr_item)

            # DC offset
            dc_item = _cell(f"{ch.dc_offset_ratio:.3f}")
            dc_item.setData(Qt.ItemDataRole.UserRole, ch.dc_offset_ratio)
            if ch.dc_offset_ratio > 0.3:
                dc_item.setForeground(QColor("#FFAA33"))
            self._table.setItem(row, self._COL_DC, dc_item)

            # Row tooltip from quality issues
            if ch.issues:
                tooltip = "\n".join(f"• {i}" for i in ch.issues)
                for col in range(len(self._COL_HEADERS)):
                    item = self._table.item(row, col)
                    if item:
                        item.setToolTip(tooltip)

        self._table.setSortingEnabled(True)
        self._table.sortByColumn(self._COL_GRADE, Qt.SortOrder.DescendingOrder)
        self._table.resizeColumnToContents(self._COL_CHANNEL)

        overall_color = _GRADE_COLOR[result.overall_grade]
        overall_label = _GRADE_LABEL[result.overall_grade]
        warn_count = sum(1 for ch in channels if ch.grade == QualityGrade.WARN)
        err_count = sum(1 for ch in channels if ch.grade == QualityGrade.ERROR)

        if result.overall_grade == QualityGrade.OK:
            summary = f"All {len(channels)} channel(s) OK"
        else:
            parts = []
            if err_count:
                parts.append(f"{err_count} error")
            if warn_count:
                parts.append(f"{warn_count} warning")
            summary = f"{len(channels)} channel(s) — " + ", ".join(parts)

        self._lbl_overall.setText(
            f"Overall: <span style='color:{overall_color};font-weight:bold'>{overall_label}</span>"
            f" — {summary}"
        )
        self._lbl_overall.setTextFormat(Qt.TextFormat.RichText)

    def clear_quality(self) -> None:
        """Reset to idle state."""
        self._table.setRowCount(0)
        self._lbl_overall.setText("No record loaded")
        self._lbl_overall.setTextFormat(Qt.TextFormat.PlainText)
        self._lbl_overall.setStyleSheet("color: #AAAAAA; font-size: 11px;")
