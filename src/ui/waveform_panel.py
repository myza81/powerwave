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

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

from engine.measurements import display_to_raw_s, get_value_at_time
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

Y_ZOOM_IN_FACTOR: float   = 0.8    # wheel up  → reduce Y range by 20%
Y_ZOOM_OUT_FACTOR: float  = 1.25   # wheel down → expand Y range by 25%


class _AnalogueRow(QWidget):
    """Single analogue channel label row with Y-scale wheel and double-click reset.

    Emits ``y_scale_requested(channel_id, scale_factor)`` on wheel scroll and
    ``y_reset_requested(channel_id)`` on double-click.  Both events are consumed
    so they do not propagate to the parent QScrollArea.
    """

    y_scale_requested:  pyqtSignal = pyqtSignal(int, float)
    y_reset_requested:  pyqtSignal = pyqtSignal(int)
    y_autofit_requested: pyqtSignal = pyqtSignal(int)

    def __init__(self, channel_id: int, parent: Optional[QWidget] = None) -> None:
        """Initialise the row widget for ``channel_id``.

        Args:
            channel_id: The analogue channel this row represents.
            parent:     Optional parent widget.
        """
        super().__init__(parent)
        self._channel_id = channel_id

    def wheelEvent(self, event) -> None:
        """Translate wheel scroll into a Y-scale signal; consume the event.

        Args:
            event: QWheelEvent from Qt.
        """
        delta = event.angleDelta().y()
        factor = Y_ZOOM_IN_FACTOR if delta > 0 else Y_ZOOM_OUT_FACTOR
        self.y_scale_requested.emit(self._channel_id, factor)
        event.accept()

    def mousePressEvent(self, event) -> None:
        """Single click auto-fits Y axis to the visible data range.

        Args:
            event: QMouseEvent from Qt.
        """
        self.y_autofit_requested.emit(self._channel_id)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click resets Y axis to full auto-range.

        Args:
            event: QMouseEvent from Qt.
        """
        self.y_reset_requested.emit(self._channel_id)
        event.accept()


class LabelPanel(QWidget):
    """Fixed-width channel label strip aligned row-for-row with ChannelCanvas.

    Call ``load_record(record)`` whenever a new DisturbanceRecord is loaded.

    Signals:
        y_scale_requested: Emitted when user scrolls wheel over a channel row.
            Args: (channel_id: int, scale_factor: float)
        y_reset_requested: Emitted when user double-clicks a channel row.
            Args: (channel_id: int)
    """

    y_scale_requested:   pyqtSignal = pyqtSignal(int, float)
    y_reset_requested:   pyqtSignal = pyqtSignal(int)
    y_autofit_requested: pyqtSignal = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialise an empty LabelPanel."""
        super().__init__(parent)
        self.setFixedWidth(PANEL_WIDTH)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Maps channel_id → value QLabel for live cursor-A updates
        self._value_labels: dict[int, QLabel] = {}

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

    # ── Public API ────────────────────────────────────────────────────────────

    def update_values(self, record: DisturbanceRecord, t_display: float) -> None:
        """Update every analogue row's value label to the value at cursor A.

        Args:
            record:    The currently loaded DisturbanceRecord.
            t_display: Cursor A time in display units (ms / s / min).
        """
        raw_s = display_to_raw_s(record, t_display)
        for ch in record.analogue_channels:
            lbl = self._value_labels.get(ch.channel_id)
            if lbl is None:
                continue
            val = get_value_at_time(record, ch, raw_s)
            unit = ch.unit or ''
            if val != val:   # NaN check
                lbl.setText(f'--- {unit}')
            elif abs(val) >= 1000:
                lbl.setText(f'{val:.1f} {unit}')
            elif abs(val) >= 10:
                lbl.setText(f'{val:.2f} {unit}')
            else:
                lbl.setText(f'{val:.3f} {unit}')

    # ── Private ───────────────────────────────────────────────────────────────

    def _clear(self) -> None:
        """Remove all rows from the layout."""
        self._value_labels.clear()
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

    def _make_analogue_row(self, ch: AnalogueChannel) -> _AnalogueRow:
        """Create a ROW_HEIGHT_ANALOGUE px analogue channel label row.

        Background is the channel's waveform colour.  Channel name appears
        top-left in bold 9pt; unit appears bottom-left in 8pt.

        Wheel scroll over this row emits ``y_scale_requested`` on the LabelPanel.
        Double-click emits ``y_reset_requested`` to restore auto Y-range.

        Args:
            ch: The analogue channel to label.

        Returns:
            A fixed-height _AnalogueRow for the analogue row.
        """
        row = _AnalogueRow(ch.channel_id)
        row.y_scale_requested.connect(self.y_scale_requested)
        row.y_reset_requested.connect(self.y_reset_requested)
        row.y_autofit_requested.connect(self.y_autofit_requested)
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

        # Store reference so update_values() can refresh it live
        self._value_labels[ch.channel_id] = unit_label

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
