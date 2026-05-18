"""Qt table model for Import Wizard raw previews."""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt

from app.import_wizard.models import RawPreviewModel


class PreviewTableModel(QAbstractTableModel):
    """Lightweight adapter over RawPreviewModel.preview_rows."""

    def __init__(self, preview: RawPreviewModel | None = None, parent=None) -> None:
        super().__init__(parent)
        self._preview = preview or RawPreviewModel(column_names=[], preview_rows=[])

    def set_preview(self, preview: RawPreviewModel | None) -> None:
        self.beginResetModel()
        self._preview = preview or RawPreviewModel(column_names=[], preview_rows=[])
        self.endResetModel()

    @property
    def preview(self) -> RawPreviewModel:
        return self._preview

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._preview.preview_rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._preview.column_names)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role not in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.EditRole,
        ):
            return None
        row = self._preview.preview_rows[index.row()]
        if index.column() >= len(row):
            return ""
        value = row[index.column()]
        return "" if value is None else str(value)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._preview.column_names):
                return self._preview.column_names[section]
            return ""
        return str(section + 1)
