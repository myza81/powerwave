"""CalculatedSignalRowWidget — one row in SessionPanel's Calculated Signals
section (Phase 3B).

Pure view: displays a CalculatedSignalEntry's current lifecycle state
(visible/hidden, OK/STALE/ERROR/never-calculated) and emits typed signals
for visibility, recalculation, and deletion. No calculation, resolution, or
session mutation happens here -- the caller (SessionPanel -> main window)
owns the EventAnalysisSession and CalculatedSignalResolutionService.

This widget never fabricates a fake SessionSource/SourceRowWidget identity:
it is keyed purely by calc_id.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QWidget

from app.calculated_signals.models import CalculatedSignalResult, CalculationStatus


def _status_text(result: CalculatedSignalResult | None) -> str:
    if result is None:
        return "not calculated"
    if result.status == CalculationStatus.OK:
        return "OK"
    if result.status == CalculationStatus.STALE:
        return "stale"
    if result.status == CalculationStatus.ERROR:
        return "error"
    return result.status.value


def _status_style(result: CalculatedSignalResult | None) -> str:
    if result is None:
        return "color: #888888;"
    if result.status == CalculationStatus.OK:
        return "color: #22aa44;"
    if result.status == CalculationStatus.STALE:
        return "color: #cc8800; font-weight: bold;"
    if result.status == CalculationStatus.ERROR:
        return "color: #cc3333; font-weight: bold;"
    return ""


class CalculatedSignalRowWidget(QWidget):
    """One row: [x] ƒ Name [unit]  status   [Recalculate] [Delete]"""

    visibility_changed = pyqtSignal(str, bool)      # calc_id, visible
    recalculate_requested = pyqtSignal(str)         # calc_id
    delete_requested = pyqtSignal(str)              # calc_id

    def __init__(self, calc_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._calc_id = calc_id
        self._updating = False

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(6)

        self._visible_cb = QCheckBox()
        self._visible_cb.setToolTip("Show/hide this calculated signal's curve")
        self._visible_cb.setChecked(True)
        self._visible_cb.stateChanged.connect(self._on_visible_changed)
        row.addWidget(self._visible_cb)

        self._name_lbl = QLabel()
        self._name_lbl.setToolTip("Calculated signal (derived, not a loaded source)")
        row.addWidget(self._name_lbl, stretch=1)

        self._status_lbl = QLabel()
        self._status_lbl.setMinimumWidth(60)
        row.addWidget(self._status_lbl)

        self._recalc_btn = QPushButton("Recalculate")
        self._recalc_btn.setToolTip("Recalculate this signal from current session state")
        self._recalc_btn.clicked.connect(
            lambda: self.recalculate_requested.emit(self._calc_id)
        )
        row.addWidget(self._recalc_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setToolTip("Remove this calculated signal from the session")
        self._delete_btn.clicked.connect(
            lambda: self.delete_requested.emit(self._calc_id)
        )
        row.addWidget(self._delete_btn)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def calc_id(self) -> str:
        return self._calc_id

    def refresh(self, name: str, result: CalculatedSignalResult | None, is_visible: bool) -> None:
        """Update the row's display from current session state -- never
        recomputes anything."""
        label = f"ƒ {name}"
        if result is not None and result.unit:
            label += f" [{result.unit}]"
        self._name_lbl.setText(label)
        self._status_lbl.setText(_status_text(result))
        self._status_lbl.setStyleSheet(_status_style(result))

        self._updating = True
        try:
            self._visible_cb.setChecked(is_visible)
        finally:
            self._updating = False

    def _on_visible_changed(self, _state) -> None:
        if self._updating:
            return
        self.visibility_changed.emit(self._calc_id, self._visible_cb.isChecked())
