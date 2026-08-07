"""Analog-only channel selector for Calculated Signals input binding
(Phase 3A).

Lists eligible analog channels across all active sources in the session,
grouped by source, so identically-named channels from different sources
(e.g. "Va" on both Source A and Source B) remain visually and structurally
unambiguous. Eligibility is derived entirely from the existing session
contract -- EventAnalysisSession.list_analog_channels() (structurally
excludes digital channels) plus
EventAnalysisSession.check_calculated_input_eligibility() (excludes analog
channels whose data column is missing or non-numeric) -- never from reading
DataFrame columns directly. No digital channel can ever appear here, and no
eligibility rule is re-implemented: this widget only calls the session.
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.calculated_signals.models import ChannelRef
from app.sessions.event_session import EventAnalysisSession

_ROLE_SOURCE_ID = Qt.ItemDataRole.UserRole
_ROLE_CHANNEL_NAME = Qt.ItemDataRole.UserRole + 1


@dataclass(frozen=True, slots=True)
class SelectedAnalogChannel:
    """One analog channel chosen from the selector, with display metadata
    for the calling dialog's bindings table.

    ref (source_id + channel_name) is the only identity that matters to the
    backend; source_display_name/channel_display_name/unit are
    presentation-only and are never used to construct a ChannelRef.
    """

    ref: ChannelRef
    source_display_name: str
    channel_display_name: str
    unit: str | None


class AnalogInputSelectorDialog(QDialog):
    """Modal picker for exactly one eligible analog channel, from any
    active source in *session*, grouped by source."""

    def __init__(
        self, session: EventAnalysisSession, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Analog Input")
        self.setMinimumSize(420, 380)

        self._session = session
        self._selection: SelectedAnalogChannel | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose one analog channel as a calculation input:"))

        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Channel", "Unit"])
        self._tree.setRootIsDecorated(True)
        self._tree.itemDoubleClicked.connect(lambda *_: self._accept_if_valid())
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._tree)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._accept_if_valid)
        self._buttons.rejected.connect(self.reject)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        layout.addWidget(self._buttons)

        self._populate()

    # -------------------------------------------------------------------------
    # Population
    # -------------------------------------------------------------------------

    def _populate(self) -> None:
        self._tree.clear()

        channels_by_source: dict[str, list] = {}
        for ch in self._session.list_analog_channels(active_only=True):
            channels_by_source.setdefault(ch.source_id, []).append(ch)

        for source in self._session.list_sources():
            if not source.is_active:
                continue
            channels = channels_by_source.get(source.source_id, [])
            if not channels:
                continue

            unit_by_name = {ch.name: ch.unit for ch in source.record.analog_channels}

            source_item = QTreeWidgetItem(self._tree)
            source_item.setText(0, source.display_name)
            source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            source_item.setExpanded(True)

            for ch in channels:
                ref = ChannelRef(source.source_id, ch.channel_name)
                eligibility = self._session.check_calculated_input_eligibility(ref)
                if not eligibility.is_valid:
                    # Declared analog but data column missing/non-numeric --
                    # excluded entirely (V1 policy, see Phase 3A step 9).
                    continue

                unit = unit_by_name.get(ch.channel_name)
                item = QTreeWidgetItem(source_item)
                item.setText(0, ch.display_name)
                item.setText(1, unit if unit else "Unknown")
                item.setData(0, _ROLE_SOURCE_ID, source.source_id)
                item.setData(0, _ROLE_CHANNEL_NAME, ch.channel_name)

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        is_leaf = bool(items) and items[0].data(0, _ROLE_SOURCE_ID) is not None
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(is_leaf)

    def _accept_if_valid(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        item = items[0]
        source_id = item.data(0, _ROLE_SOURCE_ID)
        channel_name = item.data(0, _ROLE_CHANNEL_NAME)
        if source_id is None or channel_name is None:
            return  # a source (group) header, not a selectable leaf

        source = self._session.get_source(source_id)
        if source is None:
            return
        unit = next(
            (ch.unit for ch in source.record.analog_channels if ch.name == channel_name),
            None,
        )
        self._selection = SelectedAnalogChannel(
            ref=ChannelRef(source_id, channel_name),
            source_display_name=source.display_name,
            channel_display_name=item.text(0),
            unit=unit,
        )
        self.accept()

    def selected_channel(self) -> SelectedAnalogChannel | None:
        """Return the chosen channel, or None if the dialog was cancelled."""
        return self._selection
