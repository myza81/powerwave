"""
src/ui/measurement_panel.py

MeasurementPanel — right-dock panel showing cursor time readouts and
per-channel values at cursor A and cursor B positions.

Layout (top → bottom):
  SECTION 1 — Time readouts (fixed height)
    Cursor A time | Cursor B time | ΔT (B−A)
  SECTION 2 — Channel table (scrollable QTableWidget)
    Columns: Channel | Value at A | Value at B | Δ (B−A) | Unit
  SECTION 3 — RMS summary placeholder (fixed height)

Usage::

    panel = MeasurementPanel()
    # Called from MainWindow whenever a cursor moves:
    panel.update(record, t_a_display, t_b_display)

Architecture: Presentation layer (ui/) — imports engine/ and models/ (LAW 1).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
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

from engine.measurements import display_to_raw_s, get_value_at_time
from models.disturbance_record import DisturbanceRecord

# ── Module constants ───────────────────────────────────────────────────────────

TIME_SECTION_HEIGHT: int     = 90    # px — fixed top section
RMS_SECTION_HEIGHT: int      = 40    # px — fixed bottom section
COL_CHANNEL: int             = 0
COL_VAL_A: int               = 1
COL_VAL_B: int               = 2
COL_DELTA: int               = 3
COL_UNIT: int                = 4
COL_WIDTHS: tuple[int, ...]  = (120, 80, 80, 80, 50)
COL_HEADERS: tuple[str, ...] = ('Channel', 'Value A', 'Value B', 'Δ (B−A)', 'Unit')

DELTA_HIGHLIGHT: str         = '#2A2A2A'   # row bg when Δ is non-zero
BASE_ROW_BG: str             = '#1E1E1E'
HEADER_STYLE: str            = (
    "QHeaderView::section { background-color: #3A3A3A; color: #CCCCCC;"
    " font-size: 8pt; border: none; padding: 2px; }"
)
TABLE_STYLE: str             = (
    "QTableWidget { background-color: #1E1E1E; color: #FFFFFF;"
    " font-size: 9pt; gridline-color: #3A3A3A; border: none; }"
    " QTableWidget::item { padding: 2px; }"
)
LABEL_STYLE: str             = "color: #CCCCCC; font-size: 9pt;"
VALUE_STYLE: str             = "color: #FFFFFF; font-size: 9pt; font-weight: bold;"
SECTION_TITLE_STYLE: str     = (
    "color: #AAAAAA; font-size: 8pt; font-weight: bold; padding-bottom: 2px;"
)
RMS_STYLE: str               = "color: #888888; font-size: 9pt;"

DELTA_THRESHOLD: float       = 1e-6   # treat |Δ| < this as zero (float noise)


class MeasurementPanel(QWidget):
    """Right-dock measurement panel for cursor A / cursor B readouts.

    Displays time positions, per-channel instantaneous values, and a
    placeholder row for the upcoming RMS-at-cursor feature (Milestone 2B).

    Call ``update(record, t_a, t_b)`` whenever either cursor moves.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialise empty panel with three sections."""
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        # Force dark background so white/light-coloured text labels are visible
        # inside QDockWidget, which otherwise inherits the system light palette.
        self.setStyleSheet("QWidget { background-color: #1E1E1E; color: #CCCCCC; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Section 1 — time readouts
        root.addWidget(self._build_time_section())
        root.addWidget(self._make_divider())

        # Section 2 — channel table
        self._table = self._build_table()
        root.addWidget(self._table, stretch=1)
        root.addWidget(self._make_divider())

        # Section 3 — RMS placeholder
        root.addWidget(self._build_rms_section())

    # ── Public API ─────────────────────────────────────────────────────────────

    def refresh(
        self,
        record: DisturbanceRecord,
        t_a: float,
        t_b: float,
    ) -> None:
        """Refresh all displayed values for the current cursor positions.

        Args:
            record: The currently loaded DisturbanceRecord.
            t_a:    Cursor A position in display units (ms / s / min).
            t_b:    Cursor B position in display units (ms / s / min).
        """
        label: str = getattr(record, '_time_axis_label', 'Time (ms)')

        # ── Section 1: time readouts ──────────────────────────────────────────
        if 'ms' in label:
            self._val_a.setText(f'{t_a:.3f} ms')
            self._val_b.setText(f'{t_b:.3f} ms')
            self._val_dt.setText(f'{(t_b - t_a):.3f} ms')
        elif 'min' in label:
            self._val_a.setText(f'{t_a:.4f} min')
            self._val_b.setText(f'{t_b:.4f} min')
            self._val_dt.setText(f'{(t_b - t_a):.4f} min')
        else:
            self._val_a.setText(f'{t_a:.4f} s')
            self._val_b.setText(f'{t_b:.4f} s')
            self._val_dt.setText(f'{(t_b - t_a):.4f} s')

        # ── Section 2: channel table ──────────────────────────────────────────
        raw_a = display_to_raw_s(record, t_a)
        raw_b = display_to_raw_s(record, t_b)

        visible_channels = [
            ch for ch in record.analogue_channels if ch.visible
        ]

        self._table.setRowCount(len(visible_channels))

        for row_idx, ch in enumerate(visible_channels):
            val_a = get_value_at_time(record, ch, raw_a)
            val_b = get_value_at_time(record, ch, raw_b)
            delta = val_b - val_a if not (np.isnan(val_a) or np.isnan(val_b)) else float('nan')

            self._set_cell(row_idx, COL_CHANNEL, ch.name, align_left=True)
            self._set_cell(row_idx, COL_VAL_A, self._fmt(val_a))
            self._set_cell(row_idx, COL_VAL_B, self._fmt(val_b))
            self._set_cell(row_idx, COL_DELTA, self._fmt(delta))
            self._set_cell(row_idx, COL_UNIT, ch.unit or '', align_left=True)

            for col in range(len(COL_WIDTHS)):
                item = self._table.item(row_idx, col)
                if item is not None:
                    item.setBackground(Qt.GlobalColor.transparent)
                    item.setData(
                        Qt.ItemDataRole.BackgroundRole,
                        None,
                    )
            self._table.setRowHeight(row_idx, 20)

            if abs(delta) > DELTA_THRESHOLD and not np.isnan(delta):
                for col in range(len(COL_WIDTHS)):
                    item = self._table.item(row_idx, col)
                    if item is not None:
                        item.setBackground(QColor(DELTA_HIGHLIGHT))

    # ── Private — section builders ─────────────────────────────────────────────

    def _build_time_section(self) -> QWidget:
        """Build the fixed-height time readout section.

        Returns:
            QWidget containing three labelled time value pairs.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel("Cursor Times")
        title.setStyleSheet(SECTION_TITLE_STYLE)
        layout.addWidget(title)

        self._val_a  = self._add_time_row(layout, "Cursor A:")
        self._val_b  = self._add_time_row(layout, "Cursor B:")
        self._val_dt = self._add_time_row(layout, "ΔT (B−A):")

        return widget

    def _add_time_row(self, layout: QVBoxLayout, label_text: str) -> QLabel:
        """Add a label + value pair row to a section layout.

        Args:
            layout:     Parent QVBoxLayout to add the row into.
            label_text: Descriptive label (left side).

        Returns:
            The QLabel that holds the live value (right side).
        """
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        lbl = QLabel(label_text)
        lbl.setStyleSheet(LABEL_STYLE)
        lbl.setFixedWidth(72)
        h.addWidget(lbl)

        val = QLabel("---")
        val.setStyleSheet(VALUE_STYLE)
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(val, stretch=1)

        layout.addWidget(row)
        return val

    def _build_table(self) -> QTableWidget:
        """Build the scrollable channel value table.

        Returns:
            Configured QTableWidget.
        """
        table = QTableWidget(0, len(COL_WIDTHS))
        table.setHorizontalHeaderLabels(list(COL_HEADERS))
        table.horizontalHeader().setStyleSheet(HEADER_STYLE)
        table.setStyleSheet(TABLE_STYLE)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        for col, width in enumerate(COL_WIDTHS):
            table.setColumnWidth(col, width)

        return table

    def _build_rms_section(self) -> QWidget:
        """Build the fixed-height RMS placeholder section.

        Returns:
            QWidget with a single placeholder label.
        """
        widget = QWidget()
        widget.setFixedHeight(RMS_SECTION_HEIGHT)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel("RMS at A (half-cycle):")
        title.setStyleSheet(SECTION_TITLE_STYLE)
        layout.addWidget(title)

        placeholder = QLabel("---")
        placeholder.setStyleSheet(RMS_STYLE)
        layout.addWidget(placeholder)

        return widget

    # ── Private — helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_divider() -> QFrame:
        """Return a thin horizontal divider line.

        Returns:
            QFrame styled as a horizontal rule.
        """
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #3A3A3A;")
        line.setFixedHeight(1)
        return line

    @staticmethod
    def _fmt(value: float) -> str:
        """Format a float for display in the channel table.

        Args:
            value: The value to format.

        Returns:
            Formatted string, or '---' for NaN.
        """
        if np.isnan(value):
            return '---'
        if abs(value) >= 1000:
            return f'{value:.1f}'
        if abs(value) >= 10:
            return f'{value:.2f}'
        return f'{value:.3f}'

    def _set_cell(
        self,
        row: int,
        col: int,
        text: str,
        align_left: bool = False,
    ) -> None:
        """Set a table cell value with right-align by default.

        Args:
            row:        Row index.
            col:        Column index.
            text:       Display string.
            align_left: If True, left-align; otherwise right-align.
        """
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        if align_left:
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
        else:
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
        self._table.setItem(row, col, item)
