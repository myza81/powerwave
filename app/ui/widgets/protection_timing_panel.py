"""Protection relay timing panel — extracted timing milestones table.

Shows a summary line (pickup delay, trip delay, clearing time, reclose
interval) and a sortable table listing every detected timing event with
its absolute time, millisecond offset from fault inception, and source
channel.

Call ``load_timing(result)`` after extraction completes.
Call ``clear_timing()`` when a record is unloaded or no fault is present.
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

from app.analytics.protection.timing_extractor import (
    ProtectionRole,
    ProtectionTimingResult,
)

_ROLE_COLOR = {
    ProtectionRole.PICKUP:  "#55AAFF",
    ProtectionRole.TRIP:    "#FF5555",
    ProtectionRole.CB:      "#FFAA33",
    ProtectionRole.RECLOSE: "#55CC55",
    ProtectionRole.UNKNOWN: "#888888",
}


def _ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f} ms"


def _cell(text: str, *, align: int = Qt.AlignmentFlag.AlignCenter) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setTextAlignment(align)
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return it


class ProtectionTimingPanel(QWidget):
    """Timing milestones table dock.

    Public API:
        load_timing(result: ProtectionTimingResult) — populate panel
        clear_timing()                              — reset to idle
    """

    _COL_HEADERS = ["Event", "Time (s)", "Rel. (ms)", "Channel"]
    _COL_EVENT   = 0
    _COL_TIME    = 1
    _COL_REL     = 2
    _COL_CH      = 3

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ProtectionTimingPanel")
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
        summary_layout.setSpacing(20)

        self._chips: dict[str, QLabel] = {}
        for key, label_text in [
            ("pickup",   "Pickup"),
            ("trip",     "Trip"),
            ("clearing", "Clearing"),
            ("reclose",  "Reclose"),
        ]:
            lbl_key = QLabel(f"{label_text}:")
            lbl_key.setStyleSheet("color: #666666; font-size: 10px;")
            lbl_val = QLabel("—")
            lbl_val.setStyleSheet(
                f"color: {_ROLE_COLOR.get(ProtectionRole(key) if key in ('pickup','trip','reclose') else ProtectionRole.CB, '#FFAA33')};"
                "font-size: 11px; font-family: Menlo, Consolas, 'Courier New'; font-weight: bold;"
            )
            pair = QWidget()
            pair.setStyleSheet("background: transparent;")
            pair_lay = QHBoxLayout(pair)
            pair_lay.setContentsMargins(0, 0, 0, 0)
            pair_lay.setSpacing(4)
            pair_lay.addWidget(lbl_key)
            pair_lay.addWidget(lbl_val)
            self._chips[key] = lbl_val
            summary_layout.addWidget(pair)

        summary_layout.addStretch()
        root.addWidget(summary_frame)

        # ── Events table ──────────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(len(self._COL_HEADERS))
        self._table.setHorizontalHeaderLabels(self._COL_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSortingEnabled(False)
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
        hh.setDefaultSectionSize(90)
        hh.setStretchLastSection(True)
        root.addWidget(self._table)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def load_timing(self, result: ProtectionTimingResult) -> None:
        """Populate the panel with a timing extraction result."""
        # Update summary chips
        self._chips["pickup"].setText(_ms(result.pickup_delay_ms))
        self._chips["trip"].setText(_ms(result.trip_delay_ms))
        self._chips["clearing"].setText(_ms(result.clearing_time_ms))
        self._chips["reclose"].setText(_ms(result.reclose_interval_ms))

        # Populate table
        evs = result.events
        self._table.setRowCount(len(evs))

        bold = QFont()
        bold.setBold(True)

        for row, ev in enumerate(evs):
            color = _ROLE_COLOR[ev.role]

            event_item = _cell(ev.label, align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            event_item.setForeground(QColor(color))
            event_item.setFont(bold)
            self._table.setItem(row, self._COL_EVENT, event_item)

            time_item = _cell(f"{ev.t:.5f}")
            time_item.setData(Qt.ItemDataRole.UserRole, ev.t)
            self._table.setItem(row, self._COL_TIME, time_item)

            rel_item = _cell(f"{ev.t_relative_ms:.2f}")
            rel_item.setData(Qt.ItemDataRole.UserRole, ev.t_relative_ms)
            if ev.t_relative_ms > 0:
                rel_item.setForeground(QColor(color))
            self._table.setItem(row, self._COL_REL, rel_item)

            ch_item = _cell(ev.channel, align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            ch_item.setForeground(QColor("#888888"))
            self._table.setItem(row, self._COL_CH, ch_item)

        self._table.resizeColumnToContents(self._COL_EVENT)
        self._table.resizeColumnToContents(self._COL_TIME)

    def clear_timing(self) -> None:
        """Reset to idle state."""
        for lbl in self._chips.values():
            lbl.setText("—")
        self._table.setRowCount(0)
