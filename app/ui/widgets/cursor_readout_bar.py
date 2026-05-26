"""Single-cursor live value readout bar.

A slim horizontal strip that shows the interpolated Y value for every visible
channel at the current cursor (A) position. Updates in real time as the cursor
moves. Styled for the dark theme; each channel value is rendered in the
channel's own waveform color.

Intended to be hosted in a slim bottom/right QDockWidget. The widget adapts
to any number of channels by wrapping into a scrollable area when the total
width exceeds the panel.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _fmt_value(value: float | None, unit: str) -> str:
    """Format a channel value as xx,xxx,xxx.xxx [unit]."""
    if value is None:
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{value:,.3f}{suffix}"


class CursorReadoutBar(QWidget):
    """Slim horizontal strip showing live channel values at the cursor position.

    Call ``update_values(t, values)`` whenever the cursor moves.
    Call ``clear()`` when a record is unloaded.

    Args:
        values:  List of (channel_name, value_or_None, unit, hex_color) tuples
                 in the order they should appear.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CursorReadoutBar")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(36)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            "QFrame { background: #1A1A1A; border-top: 1px solid #333333; }"
        )
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(6, 2, 6, 2)
        frame_layout.setSpacing(0)

        # Fixed time label on the left
        self._lbl_time = QLabel("—")
        self._lbl_time.setFixedWidth(130)
        self._lbl_time.setStyleSheet(
            "color: #888888; font-size: 11px; font-family: monospace;"
        )
        self._lbl_time.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        frame_layout.addWidget(self._lbl_time)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setStyleSheet("color: #3C3C3C;")
        divider.setFixedWidth(1)
        frame_layout.addWidget(divider)

        # Scrollable area for channel chips
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedHeight(32)
        scroll.setStyleSheet("background: transparent;")

        self._chips_widget = QWidget()
        self._chips_widget.setStyleSheet("background: transparent;")
        self._chips_layout = QHBoxLayout(self._chips_widget)
        self._chips_layout.setContentsMargins(6, 0, 6, 0)
        self._chips_layout.setSpacing(12)
        self._chips_layout.addStretch()

        scroll.setWidget(self._chips_widget)
        frame_layout.addWidget(scroll, stretch=1)

        root.addWidget(frame)

        # Keep label references for fast in-place updates
        self._chip_labels: list[tuple[QLabel, QLabel]] = []  # (name_lbl, value_lbl)
        self._last_channel_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update_values(
        self,
        t: float,
        values: list[tuple[str, float | None, str, str]],
    ) -> None:
        """Refresh the readout with interpolated values at time t.

        Args:
            t:      Cursor time in seconds (same base as waveform time array).
            values: [(channel_name, value_or_None, unit, hex_color), …]
        """
        # Time label
        if abs(t) < 1.0:
            self._lbl_time.setText(f"t = {t * 1000:,.3f} ms")
        else:
            self._lbl_time.setText(f"t = {t:,.3f} s")

        # Rebuild chips only when channel count changes; otherwise update in-place
        if len(values) != self._last_channel_count:
            self._rebuild_chips(values)
        else:
            self._update_chips_inplace(values)

    def clear(self) -> None:
        """Reset to idle state (no values displayed)."""
        self._lbl_time.setText("—")
        self._rebuild_chips([])

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _rebuild_chips(
        self, values: list[tuple[str, float | None, str, str]]
    ) -> None:
        """Tear down and recreate all channel chip labels."""
        # Remove old chips
        for name_lbl, val_lbl in self._chip_labels:
            name_lbl.setParent(None)
            val_lbl.setParent(None)
        self._chip_labels.clear()

        # Remove all items except the trailing stretch
        while self._chips_layout.count() > 1:
            item = self._chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Insert new chips before the stretch
        for name, value, unit, color in values:
            name_lbl = QLabel(f"{name}:")
            name_lbl.setStyleSheet(
                f"color: {color}; font-size: 11px; font-weight: bold; background: transparent;"
            )
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)

            val_lbl = QLabel(_fmt_value(value, unit))
            val_lbl.setStyleSheet(
                f"color: {color}; font-size: 11px; background: transparent;"
            )
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setMinimumWidth(60)

            pair_widget = QWidget()
            pair_widget.setStyleSheet("background: transparent;")
            pair_layout = QHBoxLayout(pair_widget)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            pair_layout.setSpacing(3)
            pair_layout.addWidget(name_lbl)
            pair_layout.addWidget(val_lbl)

            insert_pos = self._chips_layout.count() - 1  # before stretch
            self._chips_layout.insertWidget(insert_pos, pair_widget)
            self._chip_labels.append((name_lbl, val_lbl))

        self._last_channel_count = len(values)

    def _update_chips_inplace(
        self, values: list[tuple[str, float | None, str, str]]
    ) -> None:
        """Update only the value labels — avoids widget churn on every cursor move."""
        for i, (_, value, unit, _) in enumerate(values):
            if i < len(self._chip_labels):
                self._chip_labels[i][1].setText(_fmt_value(value, unit))
