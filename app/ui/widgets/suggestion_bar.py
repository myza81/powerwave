"""Contextual analytics suggestion bar widget.

A 44px-tall horizontal scrollable strip of suggestion chips.
Each chip shows:  [icon] [text]  [action_label]  [×]

Call ``add_suggestion(s)`` to add a chip.
Call ``clear_suggestions()`` to reset.

The bar emits:
    ``action_requested(action_id: str)`` — user clicked the action button
    ``dismissed(suggestion_id: str)``    — user dismissed a chip
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from app.analytics.suggestions.suggestion_engine import Suggestion

_CHIP_STYLE = """
QFrame#SuggestionChip {{
    background: {bg};
    border: 1px solid {border};
    border-radius: 14px;
    margin: 0px 3px;
}}
"""

_ACT_BTN_STYLE = """
QPushButton {
    background: #0E639C;
    color: #FFFFFF;
    border: none;
    border-radius: 9px;
    padding: 1px 8px;
    font-size: 10px;
    font-weight: bold;
}
QPushButton:hover { background: #1177BB; }
QPushButton:pressed { background: #0A4E7A; }
"""

_DISMISS_BTN_STYLE = """
QPushButton {
    background: transparent;
    color: #777777;
    border: none;
    font-size: 12px;
    padding: 0px 2px;
    min-width: 16px;
    max-width: 16px;
}
QPushButton:hover { color: #CCCCCC; }
"""

_PRIORITY_COLORS = {
    # priority ≤ 5   → amber accent (high-priority action)
    "high":   ("#2C2200", "#996600"),
    # priority 6–19  → blue-grey (informational)
    "normal": ("#1A2433", "#2B5880"),
    # priority ≥ 20  → dark grey (low-priority / exploratory)
    "low":    ("#242424", "#404040"),
}


def _chip_colors(priority: int) -> tuple[str, str]:
    if priority <= 5:
        return _PRIORITY_COLORS["high"]
    if priority <= 19:
        return _PRIORITY_COLORS["normal"]
    return _PRIORITY_COLORS["low"]


class _SuggestionChip(QFrame):
    """Single suggestion chip widget."""

    action_clicked = pyqtSignal(str)   # action_id
    dismiss_clicked = pyqtSignal(str)  # suggestion_id

    def __init__(self, suggestion: Suggestion, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._suggestion = suggestion
        self.setObjectName("SuggestionChip")
        bg, border = _chip_colors(suggestion.priority)
        self.setStyleSheet(_CHIP_STYLE.format(bg=bg, border=border))
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 4, 3)
        layout.setSpacing(5)

        icon_lbl = QLabel(self._suggestion.icon)
        icon_lbl.setStyleSheet("font-size: 13px; background: transparent;")
        layout.addWidget(icon_lbl)

        text_lbl = QLabel(self._suggestion.text)
        text_lbl.setStyleSheet("color: #CCCCCC; font-size: 10px; background: transparent;")
        text_lbl.setToolTip(self._suggestion.text)
        layout.addWidget(text_lbl)

        act_btn = QPushButton(self._suggestion.action_label)
        act_btn.setStyleSheet(_ACT_BTN_STYLE)
        act_btn.setFixedHeight(20)
        act_btn.setToolTip("Apply this suggestion")
        act_btn.clicked.connect(self._on_action)
        layout.addWidget(act_btn)

        dismiss_btn = QPushButton("×")
        dismiss_btn.setStyleSheet(_DISMISS_BTN_STYLE)
        dismiss_btn.setFixedHeight(20)
        dismiss_btn.setToolTip("Dismiss")
        dismiss_btn.clicked.connect(self._on_dismiss)
        layout.addWidget(dismiss_btn)

    def _on_action(self) -> None:
        self.action_clicked.emit(self._suggestion.action_id)
        self._on_dismiss()

    def _on_dismiss(self) -> None:
        self.dismiss_clicked.emit(self._suggestion.id)
        self.setVisible(False)
        self.deleteLater()


class SuggestionBar(QWidget):
    """Horizontal scrollable strip of contextual suggestion chips.

    Signals:
        action_requested(action_id: str)   — user clicked an action button
        dismissed(suggestion_id: str)      — user dismissed a chip
        all_dismissed()                    — all chips gone (hide the bar)
    """

    action_requested = pyqtSignal(str)
    dismissed = pyqtSignal(str)
    all_dismissed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SuggestionBar")
        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("""
            QWidget#SuggestionBar {
                background: #1E1E1E;
                border-bottom: 1px solid #333333;
            }
        """)
        self._chips: dict[str, _SuggestionChip] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(0)

        # Label on the far-left
        lbl = QLabel("Suggestions:")
        lbl.setStyleSheet("color: #666666; font-size: 10px; padding-right: 4px;")
        lbl.setFixedWidth(75)
        outer.addWidget(lbl)

        # Scroll area containing the chip strip
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:horizontal {
                height: 6px; background: #2A2A2A;
            }
            QScrollBar::handle:horizontal {
                background: #555555; border-radius: 3px; min-width: 20px;
            }
        """)

        self._strip = QWidget()
        self._strip.setStyleSheet("background: transparent;")
        self._strip_layout = QHBoxLayout(self._strip)
        self._strip_layout.setContentsMargins(0, 0, 0, 0)
        self._strip_layout.setSpacing(4)
        self._strip_layout.addStretch()

        scroll.setWidget(self._strip)
        outer.addWidget(scroll)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def add_suggestion(self, suggestion: Suggestion) -> None:
        """Add a chip for this suggestion (no-op if already present)."""
        if suggestion.id in self._chips:
            return
        chip = _SuggestionChip(suggestion, self._strip)
        chip.action_clicked.connect(self.action_requested)
        chip.dismiss_clicked.connect(self._on_chip_dismissed)
        # Insert before the trailing stretch
        count = self._strip_layout.count()
        self._strip_layout.insertWidget(count - 1, chip)
        self._chips[suggestion.id] = chip

    def clear_suggestions(self) -> None:
        """Remove all chips and reset."""
        for chip in list(self._chips.values()):
            chip.setVisible(False)
            chip.deleteLater()
        self._chips.clear()

    def load_suggestions(self, suggestions: list[Suggestion]) -> None:
        """Replace all chips with a fresh suggestion list."""
        self.clear_suggestions()
        for s in suggestions:
            self.add_suggestion(s)

    def _on_chip_dismissed(self, suggestion_id: str) -> None:
        self._chips.pop(suggestion_id, None)
        self.dismissed.emit(suggestion_id)
        # Count still-visible chips
        visible = [c for c in self._chips.values() if c.isVisible()]
        if not visible:
            self.all_dismissed.emit()
