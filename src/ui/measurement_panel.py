"""
src/ui/measurement_panel.py

MeasurementPanel — right-dock panel showing cursor time readouts and
per-channel values at cursor C1 and cursor C2 positions.

Layout (top → bottom):
  SECTION 1 — Cursor Times (fixed height)
    Cursor C1 time | Cursor C2 time | ΔT (C2−C1)
  SECTION 2 — Channel table (scrollable QTableWidget)
    Columns: Channel | Val C1 | Val C2 | Δ | Unit

Usage::

    panel = MeasurementPanel()
    # Connected via signal from UnifiedCanvasWidget.readout_updated:
    panel.update_readout(t_c1, t_c2, rows)
    # Called from MainWindow after Preferences dialog closes:
    panel.apply_settings()

Architecture: Presentation layer (ui/) — reads core.AppSettings only (LAW 1).
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

from core.app_settings import AppSettings

# ── Module constants ───────────────────────────────────────────────────────────

COL_CHANNEL: int             = 0
COL_VAL_A: int               = 1
COL_VAL_B: int               = 2
COL_DELTA: int               = 3
COL_UNIT: int                = 4
COL_WIDTHS: tuple[int, ...]  = (100, 65, 65, 65, 42)
COL_HEADERS: tuple[str, ...] = ('Channel', 'Val C1', 'Val C2', 'Δ', 'Unit')

DELTA_HIGHLIGHT: str         = '#2A2A2A'   # row bg when Δ is non-zero
FONT_PT_DEFAULT: int         = 9           # fallback if AppSettings not yet loaded

DELTA_THRESHOLD: float       = 1e-6        # treat |Δ| < this as zero (float noise)


# ── Style generators (parameterised by font size) ──────────────────────────────

def _hdr_style(pt: int) -> str:
    return (
        "QHeaderView::section { background-color: #3A3A3A; color: #CCCCCC;"
        f" font-size: {pt}pt; border: none; padding: 1px; }}"
    )


def _tbl_style(pt: int) -> str:
    return (
        "QTableWidget { background-color: #1E1E1E; color: #FFFFFF;"
        f" font-size: {pt}pt; gridline-color: #3A3A3A; border: none; }}"
        " QTableWidget::item { padding: 1px; }"
    )


def _lbl_style(pt: int) -> str:
    return f"color: #CCCCCC; font-size: {pt}pt;"


def _val_style(pt: int) -> str:
    return f"color: #FFFFFF; font-size: {pt}pt; font-weight: bold;"


def _ttl_style(pt: int) -> str:
    return f"color: #AAAAAA; font-size: {pt}pt; font-weight: bold; padding-bottom: 1px;"


# ── Panel ──────────────────────────────────────────────────────────────────────

class MeasurementPanel(QWidget):
    """Right-dock measurement panel for cursor C1 / cursor C2 readouts.

    Call ``update_readout(t_c1, t_c2, rows)`` whenever either cursor moves.
    Call ``apply_settings()`` after the Preferences dialog closes to pick up
    a changed ``display.panel_font_size``.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialise the panel with a time section and channel table."""
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        self.setStyleSheet("QWidget { background-color: #1E1E1E; color: #CCCCCC; }")

        # Kept for dynamic re-styling in apply_settings()
        self._title_lbls: list[QLabel] = []
        self._key_lbls:   list[QLabel] = []
        self._val_lbls:   list[QLabel] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(self._build_time_section())
        layout.addWidget(self._make_divider())
        self._table = self._build_table()
        layout.addWidget(self._table, stretch=1)

        self.apply_settings()

    # ── Public API ─────────────────────────────────────────────────────────────

    def apply_settings(self) -> None:
        """Re-read AppSettings and update all font-dependent styles.

        Called once at construction and again whenever the user saves new
        preferences via the Preferences dialog.
        """
        pt = int(AppSettings.get('display.panel_font_size', FONT_PT_DEFAULT))
        for lbl in self._title_lbls:
            lbl.setStyleSheet(_ttl_style(pt))
        for lbl in self._key_lbls:
            lbl.setStyleSheet(_lbl_style(pt))
        for lbl in self._val_lbls:
            lbl.setStyleSheet(_val_style(pt))
        self._table.horizontalHeader().setStyleSheet(_hdr_style(pt))
        self._table.setStyleSheet(_tbl_style(pt))

    def update_readout(
        self,
        t_c1: float,
        t_c2: float,
        rows: list,
    ) -> None:
        """Populate cursor times and channel value table.

        Args:
            t_c1:  Cursor C1 position in seconds, or NaN when C1 is inactive.
            t_c2:  Cursor C2 position in seconds, or NaN when C2 is inactive.
            rows:  List of (display_name, unit, val_c1, val_c2) tuples —
                   one entry per selected analogue channel across all files.
        """
        def _fmt_t(t: float) -> str:
            return f'{t:.3f} s' if not np.isnan(t) else '---'

        self._val_c1.setText(_fmt_t(t_c1))
        self._val_c2.setText(_fmt_t(t_c2))
        if not np.isnan(t_c1) and not np.isnan(t_c2):
            self._val_dt.setText(f'{abs(t_c2 - t_c1):.3f} s')
        else:
            self._val_dt.setText('---')

        self._table.setRowCount(len(rows))
        for r_idx, (name, unit, val_c1, val_c2) in enumerate(rows):
            delta = (
                val_c2 - val_c1
                if not (np.isnan(val_c1) or np.isnan(val_c2))
                else float('nan')
            )
            self._set_cell(r_idx, COL_CHANNEL, name,          align_left=True)
            self._set_cell(r_idx, COL_VAL_A,   self._fmt(val_c1))
            self._set_cell(r_idx, COL_VAL_B,   self._fmt(val_c2))
            self._set_cell(r_idx, COL_DELTA,   self._fmt(delta))
            self._set_cell(r_idx, COL_UNIT,    unit or '',     align_left=True)
            self._table.setRowHeight(r_idx, 16)
            if not np.isnan(delta) and abs(delta) > DELTA_THRESHOLD:
                for col in range(len(COL_WIDTHS)):
                    item = self._table.item(r_idx, col)
                    if item:
                        item.setBackground(QColor(DELTA_HIGHLIGHT))

    # ── Private builders ───────────────────────────────────────────────────────

    def _build_time_section(self) -> QWidget:
        """Build the cursor time readout section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel('Cursor Times')
        self._title_lbls.append(title)
        layout.addWidget(title)

        self._val_c1 = self._add_time_row(layout, 'Cursor C1:')
        self._val_c2 = self._add_time_row(layout, 'Cursor C2:')
        self._val_dt = self._add_time_row(layout, 'ΔT (C2−C1):')
        return widget

    def _add_time_row(self, layout: QVBoxLayout, label_text: str) -> QLabel:
        """Add a label + value pair row and register both for re-styling.

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
        lbl.setFixedWidth(60)
        self._key_lbls.append(lbl)
        h.addWidget(lbl)

        val = QLabel('---')
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._val_lbls.append(val)
        h.addWidget(val, stretch=1)

        layout.addWidget(row)
        return val

    def _build_table(self) -> QTableWidget:
        """Build the scrollable channel value table."""
        table = QTableWidget(0, len(COL_WIDTHS))
        table.setHorizontalHeaderLabels(list(COL_HEADERS))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for col, width in enumerate(COL_WIDTHS):
            table.setColumnWidth(col, width)
        return table

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _make_divider() -> QFrame:
        """Return a thin horizontal divider line."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #3A3A3A;")
        line.setFixedHeight(1)
        return line

    @staticmethod
    def _fmt(value: float) -> str:
        """Format a float for display in the channel table."""
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
        """Write text into the channel table at (row, col)."""
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
