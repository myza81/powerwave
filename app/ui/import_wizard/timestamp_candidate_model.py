"""Qt table model for timestamp candidate review."""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt

from app.import_wizard.models import TimestampCandidate


class TimestampCandidateTableModel(QAbstractTableModel):
    """Displays timestamp candidates and tracks one selected candidate."""

    HEADERS = ["Selected", "Column", "Confidence", "Format", "Invalid Samples"]

    def __init__(self, candidates: list[TimestampCandidate] | None = None, parent=None) -> None:
        super().__init__(parent)
        self._candidates: list[TimestampCandidate] = []
        self.set_candidates(candidates or [])

    def set_candidates(self, candidates: list[TimestampCandidate]) -> None:
        self.beginResetModel()
        self._candidates = list(candidates)
        if self._candidates and not any(c.user_selected for c in self._candidates):
            best = max(self._candidates, key=lambda c: c.confidence)
            best.user_selected = True
        self.endResetModel()

    @property
    def candidates(self) -> list[TimestampCandidate]:
        return self._candidates

    def selected_candidate(self) -> TimestampCandidate | None:
        for candidate in self._candidates:
            if candidate.user_selected:
                return candidate
        return self._candidates[0] if self._candidates else None

    def select_row(self, row: int) -> None:
        if not (0 <= row < len(self._candidates)):
            return
        for idx, candidate in enumerate(self._candidates):
            candidate.user_selected = idx == row
        top_left = self.index(0, 0)
        bottom_right = self.index(max(0, len(self._candidates) - 1), 0)
        self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.CheckStateRole])

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._candidates)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        candidate = self._candidates[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.CheckStateRole and col == 0:
            return (
                Qt.CheckState.Checked
                if candidate.user_selected
                else Qt.CheckState.Unchecked
            )
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None
        if col == 0:
            return ""
        if col == 1:
            return candidate.column_name
        if col == 2:
            return f"{candidate.confidence:.0%}"
        if col == 3:
            return candidate.detected_format or "auto"
        if col == 4:
            return str(candidate.invalid_sample_count)
        return None

    def setData(  # noqa: N802
        self,
        index: QModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid() or index.column() != 0:
            return False
        if role in (Qt.ItemDataRole.CheckStateRole, Qt.ItemDataRole.EditRole):
            self.select_row(index.row())
            return True
        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
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
