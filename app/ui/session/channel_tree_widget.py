"""ChannelTreeWidget — grouped channel visibility/panel control for one source.

Handles 100+ channels without full widget rebuilds on single-channel changes.
All business logic stays in EventAnalysisSession; this widget is pure view.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from app.sessions.session_models import PanelConfig, SessionChannel

_ROLE_CH_NAME = Qt.ItemDataRole.UserRole


class ChannelTreeWidget(QTreeWidget):
    """Analog/Digital channel tree for one session source.

    Columns: [✓ display name] [● colour] [Panel ▼]

    populate() builds the tree once. Subsequent updates target individual
    items so the full widget is never discarded for a single-channel change.
    """

    channel_visibility_changed = pyqtSignal(str, str, bool)    # source_id, ch_name, visible
    channel_colour_change_requested = pyqtSignal(str, str)      # source_id, ch_name
    channel_panel_changed = pyqtSignal(str, str, str)           # source_id, ch_name, panel_id

    def __init__(self, source_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_id = source_id
        self._channel_items: dict[str, QTreeWidgetItem] = {}
        self._panel_combos: dict[str, QComboBox] = {}
        self._updating = False

        self.setColumnCount(3)
        self.setHeaderLabels(["Channel", "", "Panel"])
        self.setColumnWidth(0, 160)
        self.setColumnWidth(1, 24)
        self.setColumnWidth(2, 110)
        self.setIndentation(14)
        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(True)
        self.itemChanged.connect(self._on_item_changed)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def populate(
        self,
        analog_channels: list[SessionChannel],
        digital_channels: list[SessionChannel],
        panels: list[PanelConfig],
    ) -> None:
        """Build tree from scratch. Call once when a source is added."""
        self._updating = True
        try:
            self.clear()
            self._channel_items.clear()
            self._panel_combos.clear()

            if analog_channels:
                analog_root = QTreeWidgetItem(self)
                analog_root.setText(0, f"Analog ({len(analog_channels)})")
                analog_root.setExpanded(True)
                for ch in analog_channels:
                    self._add_channel_item(analog_root, ch, panels)

            if digital_channels:
                digital_root = QTreeWidgetItem(self)
                digital_root.setText(0, f"Digital ({len(digital_channels)})")
                digital_root.setExpanded(True)
                for ch in digital_channels:
                    self._add_channel_item(digital_root, ch, panels)
        finally:
            self._updating = False

    def update_channel_visibility(self, channel_name: str, visible: bool) -> None:
        """Update a single checkbox without triggering signal re-emission."""
        item = self._channel_items.get(channel_name)
        if item is None:
            return
        self._updating = True
        try:
            item.setCheckState(
                0,
                Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked,
            )
        finally:
            self._updating = False

    def update_panel_choices(self, panels: list[PanelConfig]) -> None:
        """Refresh panel options in all combos without rebuilding items."""
        self._updating = True
        try:
            for channel_name, combo in self._panel_combos.items():
                current_panel_id = combo.currentData()
                combo.clear()
                for panel in panels:
                    combo.addItem(panel.title, userData=panel.panel_id)
                for i in range(combo.count()):
                    if combo.itemData(i) == current_panel_id:
                        combo.setCurrentIndex(i)
                        break
        finally:
            self._updating = False

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _add_channel_item(
        self,
        parent: QTreeWidgetItem,
        ch: SessionChannel,
        panels: list[PanelConfig],
    ) -> None:
        item = QTreeWidgetItem(parent)
        item.setText(0, ch.display_name)
        item.setData(0, _ROLE_CH_NAME, ch.channel_name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            0,
            Qt.CheckState.Checked if ch.is_visible else Qt.CheckState.Unchecked,
        )
        self._channel_items[ch.channel_name] = item

        # Colour swatch placeholder (Phase 9E will wire colour picker)
        swatch = _ColourSwatch(ch.color_hex or "#888888")
        swatch.clicked.connect(
            lambda name=ch.channel_name: self.channel_colour_change_requested.emit(
                self._source_id, name
            )
        )
        self.setItemWidget(item, 1, swatch)

        # Panel assignment combo
        combo = QComboBox()
        combo.setMaximumHeight(22)
        for panel in panels:
            combo.addItem(panel.title, userData=panel.panel_id)
        for i in range(combo.count()):
            if combo.itemData(i) == ch.panel_id:
                combo.setCurrentIndex(i)
                break
        combo.currentIndexChanged.connect(
            lambda _idx, name=ch.channel_name, cb=combo: self._on_panel_combo_changed(
                name, cb
            )
        )
        self.setItemWidget(item, 2, combo)
        self._panel_combos[ch.channel_name] = combo

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 0:
            return
        channel_name = item.data(0, _ROLE_CH_NAME)
        if channel_name is None:
            return
        visible = item.checkState(0) == Qt.CheckState.Checked
        self.channel_visibility_changed.emit(self._source_id, channel_name, visible)

    def _on_panel_combo_changed(self, channel_name: str, combo: QComboBox) -> None:
        if self._updating:
            return
        panel_id = combo.currentData()
        if panel_id is not None:
            self.channel_panel_changed.emit(self._source_id, channel_name, panel_id)


class _ColourSwatch(QWidget):
    """Tiny clickable colour square used as a channel colour placeholder."""

    from PyQt6.QtCore import pyqtSignal as _sig
    clicked = _sig()

    def __init__(self, colour_hex: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._colour_hex = colour_hex
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        self._label = QLabel()
        self._label.setFixedSize(14, 14)
        self._label.setToolTip("Click to change colour (Phase 9E)")
        self._apply_colour(colour_hex)
        layout.addWidget(self._label)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_colour(self, colour_hex: str) -> None:
        self._colour_hex = colour_hex
        self._apply_colour(colour_hex)

    def _apply_colour(self, colour_hex: str) -> None:
        self._label.setStyleSheet(
            f"background-color: {colour_hex}; border: 1px solid #666; border-radius: 2px;"
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(event)
