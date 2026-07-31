"""Qt table model for Import Wizard column mapping review."""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import QComboBox, QStyledItemDelegate, QStyleOptionViewItem

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.models import ColumnMappingCandidate

_TYPE_CHOICES: list[tuple[str, ParameterType]] = [
    ("Voltage", ParameterType.VOLTAGE),
    ("Current", ParameterType.CURRENT),
    ("Active Power (MW)", ParameterType.MW),
    ("Reactive Power (Mvar)", ParameterType.MVAR),
    ("Frequency (Hz)", ParameterType.FREQUENCY),
    ("ROCOF (Hz/s)", ParameterType.ROCOF),
    ("Digital / Status", ParameterType.DIGITAL),
    ("Unknown / Analog", ParameterType.UNKNOWN),
]

_UNIT_CHOICES: tuple[str, ...] = (
    "",
    "V",
    "kV",
    "pu",
    "A",
    "kA",
    "MW",
    "Mvar",
    "Hz",
    "Hz/s",
    "s",
    "ms",
)

_DEFAULT_UNIT_BY_TYPE: dict[ParameterType, str] = {
    ParameterType.VOLTAGE: "V",
    ParameterType.CURRENT: "A",
    ParameterType.MW: "MW",
    ParameterType.MVAR: "Mvar",
    ParameterType.FREQUENCY: "Hz",
    ParameterType.ROCOF: "Hz/s",
}


def _type_label(parameter_type: ParameterType) -> str:
    for label, candidate in _TYPE_CHOICES:
        if candidate == parameter_type:
            return label
    return parameter_type.value


def _default_unit_for_type(parameter_type: ParameterType) -> str:
    if parameter_type == ParameterType.DIGITAL:
        return ""
    return _DEFAULT_UNIT_BY_TYPE.get(parameter_type, "")


class ParameterTypeDelegate(QStyledItemDelegate):
    """Combobox delegate for the Type column in column mapping."""

    def createEditor(self, parent, option: QStyleOptionViewItem, index: QModelIndex):  # noqa: N802
        cb = QComboBox(parent)
        cb.setAutoFillBackground(True)
        cb.setMinimumWidth(max(option.rect.width(), 220))
        cb.setMaxVisibleItems(len(_TYPE_CHOICES))
        cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        cb.view().setMinimumWidth(260)
        for label, ptype in _TYPE_CHOICES:
            cb.addItem(label, ptype.value)
        cb.activated.connect(lambda _idx, editor=cb: self.commitData.emit(editor))
        return cb

    def setEditorData(self, editor: QComboBox, index: QModelIndex) -> None:  # noqa: N802
        value = index.data(Qt.ItemDataRole.EditRole) or ""
        idx = editor.findData(value)
        if idx >= 0:
            editor.setCurrentIndex(idx)

    def setModelData(self, editor: QComboBox, model, index: QModelIndex) -> None:  # noqa: N802
        model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor: QComboBox, option: QStyleOptionViewItem, index: QModelIndex) -> None:  # noqa: N802
        editor.setGeometry(option.rect)


class UnitDelegate(QStyledItemDelegate):
    """Editable combobox delegate for engineering units."""

    def createEditor(self, parent, option: QStyleOptionViewItem, index: QModelIndex):  # noqa: N802
        cb = QComboBox(parent)
        cb.setAutoFillBackground(True)
        cb.setEditable(True)
        cb.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        cb.setMinimumWidth(max(option.rect.width(), 140))
        cb.setMaxVisibleItems(len(_UNIT_CHOICES))
        cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        cb.view().setMinimumWidth(170)
        cb.addItems(_UNIT_CHOICES)
        cb.activated.connect(lambda _idx, editor=cb: self.commitData.emit(editor))
        return cb

    def setEditorData(self, editor: QComboBox, index: QModelIndex) -> None:  # noqa: N802
        value = str(index.data(Qt.ItemDataRole.EditRole) or "")
        if value:
            idx = editor.findText(value)
            if idx >= 0:
                editor.setCurrentIndex(idx)
            else:
                editor.setEditText(value)
            return

        row = index.row()
        model = index.model()
        type_index = model.index(row, 3)
        type_value = type_index.data(Qt.ItemDataRole.EditRole) or ""
        try:
            default_unit = _DEFAULT_UNIT_BY_TYPE.get(ParameterType(str(type_value)), "")
        except ValueError:
            default_unit = ""
        if default_unit:
            idx = editor.findText(default_unit)
            if idx >= 0:
                editor.setCurrentIndex(idx)

    def setModelData(self, editor: QComboBox, model, index: QModelIndex) -> None:  # noqa: N802
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor: QComboBox, option: QStyleOptionViewItem, index: QModelIndex) -> None:  # noqa: N802
        editor.setGeometry(option.rect)


class ColumnMappingTableModel(QAbstractTableModel):
    """Editable model for include/name/type/unit mapping decisions."""

    HEADERS = ["Include", "Source", "Output Name", "Type", "Unit", "Confidence"]

    def __init__(self, mappings: list[ColumnMappingCandidate] | None = None, parent=None) -> None:
        super().__init__(parent)
        self._mappings = list(mappings or [])
        self._visible_rows = self._build_visible_rows()

    def set_mappings(self, mappings: list[ColumnMappingCandidate]) -> None:
        self.beginResetModel()
        self._mappings = list(mappings)
        self._visible_rows = self._build_visible_rows()
        self.endResetModel()

    @property
    def mappings(self) -> list[ColumnMappingCandidate]:
        return self._mappings

    @property
    def visible_mappings(self) -> list[ColumnMappingCandidate]:
        return [self._mappings[row] for row in self._visible_rows]

    def hidden_time_axis_count(self) -> int:
        return len(self._mappings) - len(self._visible_rows)

    def visible_row_for_source(self, source_name: str) -> int:
        for visible_row, source_row in enumerate(self._visible_rows):
            if self._mappings[source_row].source_name == source_name:
                return visible_row
        return -1

    def _build_visible_rows(self) -> list[int]:
        return [
            row
            for row, mapping in enumerate(self._mappings)
            if mapping.effective_type != ParameterType.TIMESTAMP
        ]

    def _mapping_for_index(self, index: QModelIndex) -> ColumnMappingCandidate:
        return self._mappings[self._visible_rows[index.row()]]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._visible_rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        mapping = self._mapping_for_index(index)
        col = index.column()
        if role == Qt.ItemDataRole.ToolTipRole:
            markers = []
            if col == 1:
                markers.append(f"Source: {mapping.source_name}")
            elif col == 2:
                markers.append(f"Output name: {mapping.effective_name}")
            elif col == 3:
                markers.append(f"Type: {_type_label(mapping.effective_type)}")
            elif col == 4:
                markers.append(f"Unit: {mapping.effective_unit or '(none)'}")
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
            if role == Qt.ItemDataRole.EditRole:
                return value
            label = _type_label(mapping.effective_type)
            if role == Qt.ItemDataRole.DisplayRole and mapping.user_type_override is not None:
                return f"{label} (User Override)"
            return label
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
        mapping = self._mapping_for_index(index)
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
                new_type = ParameterType(text)
            except ValueError:
                try:
                    new_type = ParameterType[text.upper()]
                except KeyError:
                    return False
            mapping.user_type_override = new_type
            mapping.user_unit_override = _default_unit_for_type(new_type)
            unit_index = self.index(index.row(), 4)
            self.dataChanged.emit(unit_index, unit_index, [role, Qt.ItemDataRole.DisplayRole])
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
