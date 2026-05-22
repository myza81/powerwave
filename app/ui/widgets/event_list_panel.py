"""Event list panel — shows automatically detected disturbance events.

Presents a sortable QTableWidget listing each DetectedEvent with its type,
channel, time, duration, severity, and description. Clicking a row emits
event_selected(t_start) so the main window can jump the canvas to that event.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
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

from app.analytics.events.event_models import DetectedEvent, EventSeverity


_SEVERITY_COLOR = {
    EventSeverity.HIGH: "#FF5555",
    EventSeverity.MEDIUM: "#FFAA33",
    EventSeverity.LOW: "#AAFFAA",
}


def _cell(text: str, *, align: int = Qt.AlignmentFlag.AlignCenter) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setTextAlignment(align)
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return it


class EventListPanel(QWidget):
    """Sortable event list dock widget.

    Signals:
        event_selected(float): emitted when user clicks a row.
                               Payload is t_start of the event in seconds.
    """

    event_selected = pyqtSignal(float)

    _COL_HEADERS = ["Type", "Channel", "t start (s)", "Duration (ms)", "Severity", "Description"]
    _COL_TYPE = 0
    _COL_CHANNEL = 1
    _COL_TSTART = 2
    _COL_DUR = 3
    _COL_SEV = 4
    _COL_DESC = 5

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("EventListPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._events: list[DetectedEvent] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # Summary row
        summary_frame = QFrame()
        summary_frame.setFrameShape(QFrame.Shape.StyledPanel)
        summary_frame.setStyleSheet("QFrame { background: #252526; border-radius: 4px; }")
        summary_layout = QHBoxLayout(summary_frame)
        summary_layout.setContentsMargins(8, 4, 8, 4)

        self._lbl_summary = QLabel("No events detected")
        self._lbl_summary.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        summary_layout.addWidget(self._lbl_summary)
        summary_layout.addStretch()

        self._lbl_hint = QLabel("Click a row to jump to event")
        self._lbl_hint.setStyleSheet("color: #666666; font-size: 10px;")
        summary_layout.addWidget(self._lbl_hint)
        root.addWidget(summary_frame)

        # Event table
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
        hh.setDefaultSectionSize(90)
        hh.setStretchLastSection(True)

        self._table.cellClicked.connect(self._on_row_clicked)
        root.addWidget(self._table)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def load_events(self, events: list[DetectedEvent]) -> None:
        """Populate the table with a new list of detected events."""
        self._events = list(events)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(events))

        type_counts: dict[str, int] = {}

        for row, ev in enumerate(events):
            type_name = ev.event_type.description
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

            type_item = _cell(ev.event_type.label)
            type_item.setForeground(QColor(ev.event_type.color))
            bold = QFont()
            bold.setBold(True)
            type_item.setFont(bold)
            self._table.setItem(row, self._COL_TYPE, type_item)

            self._table.setItem(
                row, self._COL_CHANNEL,
                _cell(ev.channel, align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            )

            t_item = _cell(f"{ev.t_start:.4f}")
            # Store raw float for sorting
            t_item.setData(Qt.ItemDataRole.UserRole, ev.t_start)
            self._table.setItem(row, self._COL_TSTART, t_item)

            dur_item = _cell(f"{ev.duration_ms:.1f}")
            dur_item.setData(Qt.ItemDataRole.UserRole, ev.duration_ms)
            self._table.setItem(row, self._COL_DUR, dur_item)

            sev_item = _cell(ev.severity.value)
            sev_item.setForeground(QColor(_SEVERITY_COLOR.get(ev.severity, "#CCCCCC")))
            self._table.setItem(row, self._COL_SEV, sev_item)

            self._table.setItem(
                row, self._COL_DESC,
                _cell(ev.description, align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            )

        self._table.setSortingEnabled(True)
        self._table.sortByColumn(self._COL_TSTART, Qt.SortOrder.AscendingOrder)
        self._table.resizeColumnToContents(self._COL_CHANNEL)

        if events:
            parts = [f"{n} {t}" for t, n in type_counts.items()]
            self._lbl_summary.setText(f"{len(events)} event(s): " + ", ".join(parts))
            self._lbl_summary.setStyleSheet("color: #FFAA44; font-size: 11px;")
        else:
            self._lbl_summary.setText("No events detected")
            self._lbl_summary.setStyleSheet("color: #66AA66; font-size: 11px;")

    def clear(self) -> None:
        self._events = []
        self._table.setRowCount(0)
        self._lbl_summary.setText("No events detected")
        self._lbl_summary.setStyleSheet("color: #AAAAAA; font-size: 11px;")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _on_row_clicked(self, row: int, _col: int) -> None:
        t_item = self._table.item(row, self._COL_TSTART)
        if t_item is None:
            return
        t = t_item.data(Qt.ItemDataRole.UserRole)
        if t is not None:
            self.event_selected.emit(float(t))
