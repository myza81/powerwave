"""
src/ui/waveform_panel.py

LabelPanel — fixed-width left channel label strip.

Displays one row per channel, vertically aligned to match the
ChannelCanvas rows exactly.  Row heights come from channel_ordering.py
(single source of truth shared with ChannelCanvas):
  Bay header  : ROW_HEIGHT_BAY_HEADER px — dark grey, bay name bold 9pt
  Analogue row: ROW_HEIGHT_ANALOGUE px   — solid channel colour, name + unit
  Digital row : ROW_HEIGHT_DIGITAL px    — dark background, channel name 6pt

The panel and ChannelCanvas sit side-by-side inside a shared QScrollArea
(in main.py) so vertical scrolling keeps them in sync automatically.

Architecture: Presentation layer (ui/) — imports core/ and models/ only.
              Never import from engine/ or parsers/ (LAW 1).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

from models.channel import AnalogueChannel, DigitalChannel
from models.disturbance_record import DisturbanceRecord
from utils.channel_ordering import (
    ROW_HEIGHT_ANALOGUE,
    ROW_HEIGHT_BAY_HEADER,
    ROW_HEIGHT_DIGITAL,
    get_ordered_rows,
)

# ── Module constants ──────────────────────────────────────────────────────────

PANEL_WIDTH: int          = 160    # px — fixed label strip width
ROW_HEIGHT_TIME_AXIS: int = 30     # px — bottom spacer matches canvas time row


class LabelPanel(QWidget):
    """Fixed-width channel label strip aligned row-for-row with ChannelCanvas.

    Call ``load_record(record)`` whenever a new DisturbanceRecord is loaded.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialise an empty LabelPanel."""
        super().__init__(parent)
        self.setFixedWidth(PANEL_WIDTH)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_record(self, record: DisturbanceRecord) -> None:
        """Rebuild the label strip for ``record``.

        Rows are built in the canonical order from get_ordered_rows():
        analogue channels before digital within each bay, bay headers
        only for named bays.  Mirrors ChannelCanvas row-for-row.

        Args:
            record: The newly loaded DisturbanceRecord.
        """
        self._clear()

        rows = get_ordered_rows(record)
        total_height = ROW_HEIGHT_TIME_AXIS  # bottom spacer matches canvas time axis

        for row in rows:
            if row['type'] == 'bay_header':
                self._layout.addWidget(self._make_bay_header(row['bay_name']))
                total_height += ROW_HEIGHT_BAY_HEADER
            elif row['type'] == 'analogue':
                self._layout.addWidget(self._make_analogue_row(row['channel']))
                total_height += ROW_HEIGHT_ANALOGUE
            elif row['type'] == 'digital':
                self._layout.addWidget(self._make_digital_row(row['channel']))
                total_height += ROW_HEIGHT_DIGITAL

        # Blank spacer matching canvas bottom time-axis row
        spacer = QWidget()
        spacer.setFixedHeight(ROW_HEIGHT_TIME_AXIS)
        spacer.setStyleSheet("background-color: #1E1E1E;")
        self._layout.addWidget(spacer)

        # Pin height to exactly match ChannelCanvas so rows stay pixel-aligned
        # at all window sizes (QVBoxLayout packs top; QGraphicsView scene
        # would otherwise centre vertically, causing a visible offset).
        self.setFixedHeight(total_height)
        print(f'[LabelPanel]   total_height={total_height}  items={self._layout.count()}')

    # ── Private ───────────────────────────────────────────────────────────────

    def _clear(self) -> None:
        """Remove all rows from the layout."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _make_bay_header(self, bay_name: str) -> QLabel:
        """Create a ROW_HEIGHT_BAY_HEADER px bay group header row.

        Args:
            bay_name: The bay label to display (always non-empty here).

        Returns:
            A fixed-height QLabel for the bay header.
        """
        label = QLabel(bay_name)
        label.setFixedHeight(ROW_HEIGHT_BAY_HEADER)
        label.setStyleSheet(
            "background: #3A3A3A; color: white; font-weight: bold;"
            " font-size: 9pt; padding-left: 4px;"
        )
        return label

    def _make_analogue_row(self, ch: AnalogueChannel) -> QWidget:
        """Create a ROW_HEIGHT_ANALOGUE px analogue channel label row.

        Background is the channel's waveform colour.  Channel name appears
        top-left in bold 9pt; unit appears bottom-left in 8pt.

        Args:
            ch: The analogue channel to label.

        Returns:
            A fixed-height QWidget for the analogue row.
        """
        row = QWidget()
        row.setFixedHeight(ROW_HEIGHT_ANALOGUE)
        row.setStyleSheet(f"background-color: {ch.colour};")

        layout = QVBoxLayout(row)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(0)

        name_label = QLabel(ch.name)
        name_label.setStyleSheet(
            "color: white; font-weight: bold; font-size: 9pt;"
            " background: transparent;"
        )
        name_label.setWordWrap(False)
        layout.addWidget(name_label)

        layout.addStretch(1)

        unit_str = f"--- {ch.unit}" if ch.unit else "---"
        unit_label = QLabel(unit_str)
        unit_label.setStyleSheet(
            "color: #CCCCCC; font-size: 8pt; background: transparent;"
        )
        layout.addWidget(unit_label)

        return row

    def _make_digital_row(self, ch: DigitalChannel) -> QLabel:
        """Create a ROW_HEIGHT_DIGITAL px digital channel label row.

        Args:
            ch: The digital channel to label.

        Returns:
            A fixed-height QLabel for the digital row.
        """
        label = QLabel(ch.name)
        label.setFixedHeight(ROW_HEIGHT_DIGITAL)
        label.setStyleSheet(
            "background: #2A2A2A; color: white; font-size: 6pt;"
            " padding-left: 4px;"
        )
        label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        return label
