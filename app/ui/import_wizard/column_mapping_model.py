"""Qt table model for Import Wizard column mapping review."""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import QComboBox, QStyledItemDelegate, QStyleOptionViewItem

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.models import ColumnMappingCandidate

_TYPE_VALUES = [pt.value for pt in ParameterType if pt != ParameterType.TIMESTAMP]


class ParameterTypeDelegate(QStyledItemDelegate):
    """Combobox delegate for the Type column in column mapping."""

    def createEditor(self, parent, option: QStyleOptionViewItem, index: QModelIndex):  # noqa: N802
        cb = QComboBox(parent)
        cb.addItems(_TYPE_VALUES)
        return cb

    def setEditorData(self, editor: QComboBox, index: QModelIndex) -> None:  # noqa: N802
        value = index.data(Qt.ItemDataRole.EditRole) or ""
        idx = editor.findText(value)
        if idx >= 0:
            editor.setCurrentIndex(idx)

    def setModelData(self, editor: QComboBox, model, index: QModelIndex) -> None:  # noqa: N802
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class ColumnMappingTableModel(QAbstractTableModel):
    """Editable model for include/name/type/unit mapping decisions."""

    HEADERS = ["Include", "Source", "Output Name", "Type", "Unit", "Confidence"]

    def __init__(self, mappings: list[ColumnMappingCandidate] | None = None, parent=None) -> None:
        super().__init__(parent)
        self._mappings = list(mappings or [])

    def set_mappings(self, mappings: list[ColumnMappingCandidate]) -> None:
        self.beginResetModel()
        self._mappings = list(mappings)
        self.endResetModel()

    @property
    def mappings(self) -> list[ColumnMappingCandidate]:
        return self._mappings

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._mappings)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        mapping = self._mappings[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.ToolTipRole:
            markers = []
            if mapping.excluded:
                markers.append("Column is excluded from normalized output.")
            if mapping.user_name_override is not None:
                markers.append("Output name is a user override.")
            if mapping.user_type_override is not None:
                markers.append("Parameter type is a user override.")
            if mapping.user_unit_override is not None:
                markers.append("Engineering unit is a user override.")
            return "\n".join(markers) if markers else None
        if role == Qt.ItemDataRole.CheckStateRole and col == 0:
            return (
                Qt.CheckState.Unchecked
                if mapping.excluded
                else Qt.CheckState.Checked
            )
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None
        if col == 0:
            return ""
        if col == 1:
            return mapping.source_name
        if col == 2:
            value = mapping.effective_name
            if role == Qt.ItemDataRole.DisplayRole and mapping.user_name_override is not None:
                return f"{value} (User Override)"
            return value
        if col == 3:
            value = mapping.effective_type.value
            if role == Qt.ItemDataRole.DisplayRole and mapping.user_type_override is not None:
                return f"{value} (User Override)"
            return value
        if col == 4:
            value = mapping.effective_unit or ""
            if role == Qt.ItemDataRole.DisplayRole and mapping.user_unit_override is not None:
                return f"{value} (User Override)"
            return value
        if col == 5:
            suffix = " | User Override" if mapping.has_user_override or mapping.excluded else ""
            return f"{mapping.confidence:.0%}{suffix}"
        return None

    def setData(  # noqa: N802
        self,
        index: QModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():
            return False
        mapping = self._mappings[index.row()]
        col = index.column()
        if col == 0 and role == Qt.ItemDataRole.CheckStateRole:
            check_value = value.value if hasattr(value, "value") else value
            mapping.excluded = check_value != Qt.CheckState.Checked.value
            self.dataChanged.emit(index, index, [role])
            return True
        if role != Qt.ItemDataRole.EditRole:
            return False
        text = str(value).strip()
        if col == 2:
            mapping.user_name_override = text or None
        elif col == 3:
            try:
                mapping.user_type_override = ParameterType(text)
            except ValueError:
                try:
                    mapping.user_type_override = ParameterType[text.upper()]
                except KeyError:
                    return False
        elif col == 4:
            mapping.user_unit_override = text or None
        else:
            return False
        self.dataChanged.emit(index, index, [role, Qt.ItemDataRole.DisplayRole])
        return True

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        if index.column() in (2, 3, 4):
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return str(section + 1)
