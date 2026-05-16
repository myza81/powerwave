"""Dockable signal visibility browser for runtime plot management."""
from __future__ import annotations

import dataclasses

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDockWidget, QTreeWidget, QTreeWidgetItem


@dataclasses.dataclass(frozen=True, slots=True)
class SignalBrowserEntry:
    """One checkable signal entry in the browser."""

    key: str
    source: str
    group: str
    name: str
    visible: bool
    kind: str = "analog"


class SignalBrowserDock(QDockWidget):
    """Small dock widget exposing provider-neutral signal visibility toggles."""

    visibility_changed = pyqtSignal(str, bool)

    def __init__(self, parent=None) -> None:
        super().__init__("Signal Browser", parent)
        self._tree = QTreeWidget(self)
        self._tree.setHeaderLabels(["Signal"])
        self._tree.itemChanged.connect(self._on_item_changed)
        self.setWidget(self._tree)
        self.setObjectName("signalBrowserDock")
        self._updating = False

    def set_entries(self, entries: list[SignalBrowserEntry]) -> None:
        """Replace the browser contents without emitting toggle signals."""
        self._updating = True
        try:
            self._tree.clear()
            source_items: dict[str, QTreeWidgetItem] = {}
            group_items: dict[tuple[str, str], QTreeWidgetItem] = {}

            for entry in entries:
                source_item = source_items.get(entry.source)
                if source_item is None:
                    source_item = QTreeWidgetItem([entry.source])
                    source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                    self._tree.addTopLevelItem(source_item)
                    source_items[entry.source] = source_item

                group_key = (entry.source, entry.group)
                group_item = group_items.get(group_key)
                if group_item is None:
                    group_item = QTreeWidgetItem([entry.group])
                    group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                    source_item.addChild(group_item)
                    group_items[group_key] = group_item

                label = entry.name if entry.kind == "analog" else f"{entry.name} [{entry.kind}]"
                item = QTreeWidgetItem([label])
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsEnabled
                )
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked if entry.visible else Qt.CheckState.Unchecked,
                )
                item.setData(0, Qt.ItemDataRole.UserRole, entry.key)
                group_item.addChild(item)

            self._tree.expandAll()
        finally:
            self._updating = False

    def entry_count(self) -> int:
        """Return the number of checkable signal entries."""
        count = 0
        for source_index in range(self._tree.topLevelItemCount()):
            source_item = self._tree.topLevelItem(source_index)
            for group_index in range(source_item.childCount()):
                group_item = source_item.child(group_index)
                count += group_item.childCount()
        return count

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 0:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key:
            return
        self.visibility_changed.emit(
            str(key),
            item.checkState(0) == Qt.CheckState.Checked,
        )
