"""Two-cursor measurement panel widget.

Displays a compact table of measurements computed between cursor A (yellow)
and cursor B (cyan) by MeasurementEngine. Updates in real-time as either
cursor moves.
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

from app.visualization.interaction.measurement_engine import MeasurementResult


def _item(text: str, *, align: int = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setTextAlignment(align)
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return it


def _fmt(value: float | None, decimals: int = 3, unit: str = "") -> str:
    if value is None:
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{value:,.3f}{suffix}"


class MeasurementPanel(QWidget):
    """Compact widget showing two-cursor measurement statistics.

    Layout:
        ┌─── Time Summary ──────────────────────────────┐
        │  Δt: 20.000 ms   Δf: 50.000 Hz   0.999 cyc   │
        ├─── Channel Table ─────────────────────────────┤
        │ Channel │ ΔY      │ RMS     │ Mean    │ Peak  │
        │ Va      │ 1.23 kV │ 63.5 kV │ 0.1 kV  │ 91 kV │
        │ Ia      │ 12.3 A  │ 450 A   │ 2.1 A   │ 640 A │
        └───────────────────────────────────────────────┘
    """

    _COL_HEADERS = ["Channel", "Unit", "ΔY", "RMS", "Mean", "Peak", "P-P"]
    _COL_CHANNEL = 0
    _COL_UNIT = 1
    _COL_DY = 2
    _COL_RMS = 3
    _COL_MEAN = 4
    _COL_PEAK = 5
    _COL_PP = 6

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MeasurementPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # ── Time summary row ──────────────────────────────────────────────────
        summary_frame = QFrame()
        summary_frame.setFrameShape(QFrame.Shape.StyledPanel)
        summary_frame.setStyleSheet("QFrame { background: #252526; border-radius: 4px; }")
        summary_layout = QHBoxLayout(summary_frame)
        summary_layout.setContentsMargins(8, 4, 8, 4)
        summary_layout.setSpacing(16)

        bold = QFont()
        bold.setBold(True)

        def _summary_label(prefix: str, color: str = "#CCCCCC") -> tuple[QLabel, QLabel]:
            lbl_key = QLabel(prefix)
            lbl_key.setStyleSheet(f"color: #888888; font-size: 11px;")
            lbl_val = QLabel("—")
            lbl_val.setStyleSheet(f"color: {color}; font-size: 11px;")
            lbl_val.setFont(bold)
            return lbl_key, lbl_val

        def _add_pair(layout: QHBoxLayout, key_lbl: QLabel, val_lbl: QLabel) -> None:
            layout.addWidget(key_lbl)
            layout.addWidget(val_lbl)

        self._lbl_dt_key, self._lbl_dt = _summary_label("Δt:", "#FFD700")
        self._lbl_freq_key, self._lbl_freq = _summary_label("Frequency:", "#88DDFF")
        self._lbl_cyc_key, self._lbl_cyc = _summary_label("Cycles:", "#AAFFAA")
        self._lbl_energy_key, self._lbl_energy = _summary_label("Energy:", "#FFAA55")

        for k, v in [
            (self._lbl_dt_key, self._lbl_dt),
            (self._lbl_freq_key, self._lbl_freq),
            (self._lbl_cyc_key, self._lbl_cyc),
            (self._lbl_energy_key, self._lbl_energy),
        ]:
            _add_pair(summary_layout, k, v)

        summary_layout.addStretch()
        root.addWidget(summary_frame)

        # ── Per-channel table ─────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(len(self._COL_HEADERS))
        self._table.setHorizontalHeaderLabels(self._COL_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setShowGrid(True)
        self._table.setMinimumHeight(80)
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
        """)

        hh = self._table.horizontalHeader()
        hh.setDefaultSectionSize(80)
        hh.setStretchLastSection(True)

        root.addWidget(self._table)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update_measurements(self, result: MeasurementResult | None) -> None:
        """Refresh the panel from a MeasurementResult (or clear if None)."""
        if result is None or not result.valid:
            self._clear()
            return

        # Time summary
        if result.delta_t_ms < 1000.0:
            dt_text = f"{result.delta_t_ms:,.3f} ms"
        else:
            dt_text = f"{result.delta_t_s:,.3f} s"
        self._lbl_dt.setText(dt_text)

        freq_text = _fmt(result.frequency_hz, 3, "Hz")
        self._lbl_freq.setText(freq_text)

        cyc_text = _fmt(result.delta_t_cycles, 3, "cyc") if result.delta_t_cycles is not None else "—"
        self._lbl_cyc.setText(cyc_text)

        if result.energy_ws is not None:
            self._lbl_energy_key.setVisible(True)
            self._lbl_energy.setVisible(True)
            energy_ws = result.energy_ws
            if abs(energy_ws) >= 1e6:
                e_text = f"{energy_ws / 1e6:,.3f} MJ"
            elif abs(energy_ws) >= 1e3:
                e_text = f"{energy_ws / 1e3:,.3f} kJ"
            else:
                e_text = f"{energy_ws:,.3f} J"
            self._lbl_energy.setText(e_text)
        else:
            self._lbl_energy_key.setVisible(False)
            self._lbl_energy.setVisible(False)

        # Channel table -- every submitted curve gets a row, available or
        # not (Sprint 1A: never silently omit a visible curve's row).
        channels = result.channels
        self._table.setRowCount(len(channels))

        for row, ch in enumerate(channels):
            unit = ch.unit or ""
            name_item = _item(ch.name, align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            if not ch.available:
                name_item.setToolTip(ch.unavailable_reason or "Unavailable")
            self._table.setItem(row, self._COL_CHANNEL, name_item)
            self._table.setItem(row, self._COL_UNIT, _item(unit))

            if ch.available:
                self._table.setItem(row, self._COL_DY, _item(_fmt(ch.delta_y)))
                self._table.setItem(row, self._COL_RMS, _item(_fmt(ch.rms)))
                self._table.setItem(row, self._COL_MEAN, _item(_fmt(ch.mean)))
                self._table.setItem(row, self._COL_PEAK, _item(_fmt(ch.peak)))
                self._table.setItem(row, self._COL_PP, _item(_fmt(ch.peak_to_peak)))
            else:
                # Explicit, unmissable marker -- distinct from "—" (a
                # computable-but-empty statistic on an otherwise available
                # curve, e.g. coincident cursors) so an engineer can never
                # mistake a genuinely missing measurement for a zero/blank one.
                for col in (self._COL_DY, self._COL_RMS, self._COL_MEAN, self._COL_PEAK, self._COL_PP):
                    unavailable_item = _item("Unavailable")
                    unavailable_item.setForeground(QColor("#CC6666"))
                    font = unavailable_item.font()
                    font.setItalic(True)
                    unavailable_item.setFont(font)
                    unavailable_item.setToolTip(ch.unavailable_reason or "Unavailable")
                    self._table.setItem(row, col, unavailable_item)

        self._table.resizeColumnToContents(self._COL_CHANNEL)

    def _clear(self) -> None:
        self._lbl_dt.setText("—")
        self._lbl_freq.setText("—")
        self._lbl_cyc.setText("—")
        self._lbl_energy_key.setVisible(False)
        self._lbl_energy.setVisible(False)
        self._table.setRowCount(0)

