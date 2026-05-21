"""ChannelLegendWidget — per-panel legend for the Phase 9E session canvas.

Each row shows:  [colour swatch]  [display name]  [source badge]  [unit]

Interactions
------------
- click swatch       → QColorDialog → colour_changed signal
- double-click name  → QInputDialog → display_name_changed signal
- right-click row    → context menu: Hide / Move to panel… / Reset colour / Reset name

Performance
-----------
- rows are keyed by (source_id, channel_name) and reused across updates
- upsert_row() updates in-place; no layout rebuild for single-row changes
- clear_rows() is O(n) but only called on full panel teardown
"""
from __future__ import annotations

import colorsys

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Colour utility
# ---------------------------------------------------------------------------

_SAT_TABLE = [1.0, 0.70, 0.50]


def generate_source_variant_colour(base_colour: str, source_index: int) -> str:
    """Deterministic colour variant for multi-source channel discrimination.

    source_index=0 → full saturation, no hue shift (original colour)
    source_index=1 → 70% saturation, +15° hue
    source_index=2 → 50% saturation, +30° hue
    source_index=n → continue decay (floor 30%) + progressive hue shift

    Returns the input unchanged on malformed hex strings.
    """
    try:
        r = int(base_colour[1:3], 16) / 255.0
        g = int(base_colour[3:5], 16) / 255.0
        b = int(base_colour[5:7], 16) / 255.0
    except (ValueError, IndexError):
        return base_colour

    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    if source_index < len(_SAT_TABLE):
        sat_scale = _SAT_TABLE[source_index]
    else:
        sat_scale = max(0.30, 0.50 - (source_index - 2) * 0.10)

    hue_shift = (source_index * 15.0) / 360.0
    h_new = (h + hue_shift) % 1.0
    s_new = min(1.0, s * sat_scale)

    r2, g2, b2 = colorsys.hsv_to_rgb(h_new, s_new, v)
    return f"#{int(r2 * 255):02x}{int(g2 * 255):02x}{int(b2 * 255):02x}"


# ---------------------------------------------------------------------------
# Single legend row
# ---------------------------------------------------------------------------

_SWATCH_STYLE = (
    "QPushButton {{ background-color: {color}; "
    "border: 1px solid #555; border-radius: 2px; }}"
)
_NAME_ACTIVE = "color: #ddd; font-size: 11px;"
_NAME_HIDDEN = "color: #555; font-size: 11px;"
_BADGE_STYLE = "font-size: 9px; color: #888; padding-left: 3px;"
_UNIT_STYLE  = "font-size: 9px; font-style: italic; color: #666; padding-left: 2px;"


class _LegendRow(QWidget):
    """One channel row inside a ChannelLegendWidget."""

    def __init__(
        self,
        source_id: str,
        channel_name: str,
        display_name: str,
        source_badge: str,
        unit: str | None,
        colour: str,
        visible: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source_id = source_id
        self.channel_name = channel_name
        self._display_name = display_name
        self._colour = colour
        self._line_style = "solid"
        self._line_width = 1.0

        # Callbacks wired by ChannelLegendWidget — keep as callables, not signals,
        # to avoid making _LegendRow a full QObject with its own signal table.
        self._cb_colour: object = None
        self._cb_name: object = None
        self._cb_hide: object = None
        self._cb_move: object = None
        self._cb_reset_colour: object = None
        self._cb_reset_name: object = None
        self._cb_line_style: object = None
        self._cb_line_width: object = None
        self._cb_y_axis: object = None
        self._y_axis_side = "left"

        self._build_ui(display_name, source_badge, unit, colour, visible)

    # ------------------------------------------------------------------
    def _build_ui(
        self,
        display_name: str,
        source_badge: str,
        unit: str | None,
        colour: str,
        visible: bool,
    ) -> None:
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 1, 4, 1)
        lay.setSpacing(4)

        self._swatch = QPushButton()
        self._swatch.setFixedSize(12, 12)
        self._swatch.setFlat(True)
        self._swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self._swatch.clicked.connect(self._pick_colour)
        lay.addWidget(self._swatch)

        self._name_lbl = QLabel(display_name)
        self._name_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        # Double-click → inline rename via dialog
        self._name_lbl.mouseDoubleClickEvent = lambda _e: self._start_name_edit()  # type: ignore[method-assign]
        lay.addWidget(self._name_lbl)

        self._badge_lbl = QLabel(source_badge)
        self._badge_lbl.setStyleSheet(_BADGE_STYLE)
        lay.addWidget(self._badge_lbl)

        self._unit_lbl = QLabel(unit or "")
        self._unit_lbl.setStyleSheet(_UNIT_STYLE)
        lay.addWidget(self._unit_lbl)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

        self.set_colour(colour)
        self.set_visible_state(visible)

    # ------------------------------------------------------------------
    # Public updaters (all O(1), no layout rebuild)
    # ------------------------------------------------------------------

    def set_colour(self, colour: str) -> None:
        self._colour = colour
        self._swatch.setStyleSheet(_SWATCH_STYLE.format(color=colour))

    def set_effective_label(self, label: str) -> None:
        self._name_lbl.setText(label)

    def set_display_name(self, name: str) -> None:
        self._display_name = name

    def set_source_badge(self, badge: str) -> None:
        self._badge_lbl.setText(badge)

    def set_unit(self, unit: str | None) -> None:
        self._unit_lbl.setText(unit or "")

    def set_visible_state(self, visible: bool) -> None:
        self._name_lbl.setStyleSheet(_NAME_ACTIVE if visible else _NAME_HIDDEN)

    # ------------------------------------------------------------------
    # Interaction handlers
    # ------------------------------------------------------------------

    def _pick_colour(self) -> None:
        initial = QColor(self._colour)
        dlg = QColorDialog(initial, self)
        dlg.setWindowTitle("Choose curve colour")
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dlg.setStyleSheet(
            "* { background-color: #f0f0f0; color: #1a1a1a; }"
            "QPushButton { background-color: #e0e0e0; border: 1px solid #b0b0b0;"
            " padding: 4px 10px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #d0d0d0; }"
            "QLineEdit, QSpinBox { background-color: #ffffff; border: 1px solid #b0b0b0; }"
        )
        if dlg.exec() == QColorDialog.DialogCode.Accepted and self._cb_colour is not None:
            self._cb_colour(self.source_id, self.channel_name, dlg.selectedColor().name())

    def _start_name_edit(self) -> None:
        dlg = QInputDialog(self)
        dlg.setWindowTitle("Rename channel")
        dlg.setLabelText("Display name:")
        dlg.setTextValue(self._display_name)
        dlg.setStyleSheet(
            "* { background-color: #f0f0f0; color: #1a1a1a; }"
            "QPushButton { background-color: #e0e0e0; border: 1px solid #b0b0b0;"
            " padding: 4px 10px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #d0d0d0; }"
            "QLineEdit { background-color: #ffffff; border: 1px solid #b0b0b0; }"
        )
        if dlg.exec() == QInputDialog.DialogCode.Accepted:
            new_name = dlg.textValue().strip()
            if new_name and new_name != self._display_name and self._cb_name is not None:
                self._cb_name(self.source_id, self.channel_name, new_name)

    def set_line_style(self, style: str) -> None:
        self._line_style = style

    def set_line_width(self, width: float) -> None:
        self._line_width = width

    def set_y_axis_side(self, side: str) -> None:
        self._y_axis_side = side

    def _context_menu(self, pos) -> None:
        _MENU_SS = (
            "QMenu { background-color: #f0f0f0; color: #1a1a1a; border: 1px solid #b0b0b0; }"
            "QMenu::item:selected { background-color: #0078d4; color: #ffffff; }"
            "QMenu::separator { background-color: #c0c0c0; height: 1px; margin: 2px 4px; }"
        )
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_SS)

        hide_act = menu.addAction("Hide")
        menu.addSeparator()
        move_act = menu.addAction("Move to panel…")
        menu.addSeparator()

        # Line style submenu
        style_menu = menu.addMenu("Line Style")
        style_menu.setStyleSheet(_MENU_SS)
        for label, key in (("Solid", "solid"), ("Dashed", "dashed"), ("Dotted", "dotted")):
            act = style_menu.addAction(("✓ " if self._line_style == key else "   ") + label)
            act.setData(("style", key))

        # Line width submenu
        width_menu = menu.addMenu("Line Width")
        width_menu.setStyleSheet(_MENU_SS)
        for label, val in (
            ("Thin (0.5)", 0.5), ("Normal (1.0)", 1.0),
            ("Medium (1.5)", 1.5), ("Thick (2.0)", 2.0), ("Extra Thick (3.0)", 3.0),
        ):
            act = width_menu.addAction(("✓ " if self._line_width == val else "   ") + label)
            act.setData(("width", val))

        # Y-axis submenu
        yaxis_menu = menu.addMenu("Y-Axis")
        yaxis_menu.setStyleSheet(_MENU_SS)
        for label, side in (("Left", "left"), ("Right", "right")):
            act = yaxis_menu.addAction(("✓ " if self._y_axis_side == side else "   ") + label)
            act.setData(("yaxis", side))

        menu.addSeparator()
        reset_c = menu.addAction("Reset colour")
        reset_n = menu.addAction("Reset name")

        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen is None:
            return
        data = chosen.data()
        if data is not None and isinstance(data, tuple):
            kind, value = data
            if kind == "style" and self._cb_line_style:
                self._line_style = value
                self._cb_line_style(self.source_id, self.channel_name, value)
            elif kind == "width" and self._cb_line_width:
                self._line_width = value
                self._cb_line_width(self.source_id, self.channel_name, value)
            elif kind == "yaxis" and self._cb_y_axis:
                self._y_axis_side = value
                self._cb_y_axis(self.source_id, self.channel_name, value)
        elif chosen == hide_act and self._cb_hide:
            self._cb_hide(self.source_id, self.channel_name)
        elif chosen == move_act and self._cb_move:
            self._cb_move(self.source_id, self.channel_name)
        elif chosen == reset_c and self._cb_reset_colour:
            self._cb_reset_colour(self.source_id, self.channel_name)
        elif chosen == reset_n and self._cb_reset_name:
            self._cb_reset_name(self.source_id, self.channel_name)


# ---------------------------------------------------------------------------
# Legend widget
# ---------------------------------------------------------------------------


class ChannelLegendWidget(QWidget):
    """Scrollable legend showing all curves assigned to one session panel.

    Signals
    -------
    colour_changed(source_id, channel_name, hex_colour)
    display_name_changed(source_id, channel_name, new_name)
    visibility_changed(source_id, channel_name, visible)
    move_to_panel_requested(source_id, channel_name)
    """

    colour_changed = pyqtSignal(str, str, str)
    display_name_changed = pyqtSignal(str, str, str)
    visibility_changed = pyqtSignal(str, str, bool)
    move_to_panel_requested = pyqtSignal(str, str)
    line_style_changed = pyqtSignal(str, str, str)    # source_id, channel_name, style
    line_width_changed = pyqtSignal(str, str, float)  # source_id, channel_name, width
    y_axis_changed = pyqtSignal(str, str, str)        # source_id, channel_name, side

    def __init__(self, panel_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.panel_id = panel_id
        self._rows: dict[tuple[str, str], _LegendRow] = {}

        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setMaximumHeight(120)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: #1a1a1a; }")

        self._container = QWidget()
        self._container.setStyleSheet("background: #1a1a1a;")
        self._row_layout = QVBoxLayout(self._container)
        self._row_layout.setContentsMargins(2, 2, 2, 2)
        self._row_layout.setSpacing(0)
        self._row_layout.addStretch(1)  # pushes rows to top

        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)

    # ------------------------------------------------------------------
    # Row management (all incremental — O(1) per row)
    # ------------------------------------------------------------------

    def upsert_row(
        self,
        source_id: str,
        channel_name: str,
        display_name: str,
        source_badge: str,
        unit: str | None,
        colour: str,
        visible: bool,
    ) -> None:
        """Create or update a legend row. Never rebuilds unaffected rows."""
        key = (source_id, channel_name)
        if key in self._rows:
            row = self._rows[key]
            row.set_colour(colour)
            row.set_display_name(display_name)
            row.set_source_badge(source_badge)
            row.set_unit(unit)
            row.set_visible_state(visible)
        else:
            row = _LegendRow(
                source_id, channel_name, display_name,
                source_badge, unit, colour, visible,
                parent=self._container,
            )
            self._wire_row(row)
            # Insert before the trailing stretch
            insert_pos = self._row_layout.count() - 1
            self._row_layout.insertWidget(insert_pos, row)
            self._rows[key] = row

        self._rebuild_effective_labels()

    def remove_row(self, source_id: str, channel_name: str) -> None:
        key = (source_id, channel_name)
        row = self._rows.pop(key, None)
        if row is not None:
            self._row_layout.removeWidget(row)
            row.deleteLater()
            self._rebuild_effective_labels()

    def remove_source_rows(self, source_id: str) -> None:
        """Remove all rows belonging to one source."""
        stale = [k for k in self._rows if k[0] == source_id]
        for key in stale:
            row = self._rows.pop(key)
            self._row_layout.removeWidget(row)
            row.deleteLater()
        if stale:
            self._rebuild_effective_labels()

    def clear_rows(self) -> None:
        """Remove every row. Named clear_rows (not clear) to avoid PyQtGraph shadowing."""
        for row in list(self._rows.values()):
            self._row_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

    # ------------------------------------------------------------------
    # Targeted incremental updates (called by controller — no full rebuild)
    # ------------------------------------------------------------------

    def update_row_colour(self, source_id: str, channel_name: str, colour: str) -> None:
        row = self._rows.get((source_id, channel_name))
        if row is not None:
            row.set_colour(colour)

    def update_row_display_name(
        self, source_id: str, channel_name: str, name: str
    ) -> None:
        row = self._rows.get((source_id, channel_name))
        if row is not None:
            row.set_display_name(name)
            self._rebuild_effective_labels()

    def update_row_visible(
        self, source_id: str, channel_name: str, visible: bool
    ) -> None:
        row = self._rows.get((source_id, channel_name))
        if row is not None:
            row.set_visible_state(visible)

    def update_row_y_axis_side(
        self, source_id: str, channel_name: str, side: str
    ) -> None:
        row = self._rows.get((source_id, channel_name))
        if row is not None:
            row.set_y_axis_side(side)

    @property
    def row_count(self) -> int:
        return len(self._rows)

    # ------------------------------------------------------------------
    # Duplicate-name disambiguation
    # ------------------------------------------------------------------

    def _rebuild_effective_labels(self) -> None:
        """Set displayed label text, appending source badge on name collision."""
        name_counts: dict[str, int] = {}
        for row in self._rows.values():
            name_counts[row._display_name] = name_counts.get(row._display_name, 0) + 1

        for row in self._rows.values():
            if name_counts.get(row._display_name, 0) > 1:
                badge = row._badge_lbl.text()
                label = f"{row._display_name} [{badge}]" if badge else row._display_name
            else:
                label = row._display_name
            row.set_effective_label(label)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_row(self, row: _LegendRow) -> None:
        row._cb_colour = self._on_row_colour
        row._cb_name = self._on_row_name
        row._cb_hide = self._on_row_hide
        row._cb_move = self._on_row_move
        row._cb_reset_colour = self._on_row_reset_colour
        row._cb_reset_name = self._on_row_reset_name
        row._cb_line_style = self._on_row_line_style
        row._cb_line_width = self._on_row_line_width
        row._cb_y_axis = self._on_row_y_axis

    def _on_row_colour(self, source_id: str, channel_name: str, colour: str) -> None:
        row = self._rows.get((source_id, channel_name))
        if row is not None:
            row.set_colour(colour)
        self.colour_changed.emit(source_id, channel_name, colour)

    def _on_row_name(self, source_id: str, channel_name: str, name: str) -> None:
        row = self._rows.get((source_id, channel_name))
        if row is not None:
            row.set_display_name(name)
            self._rebuild_effective_labels()
        self.display_name_changed.emit(source_id, channel_name, name)

    def _on_row_hide(self, source_id: str, channel_name: str) -> None:
        self.visibility_changed.emit(source_id, channel_name, False)

    def _on_row_move(self, source_id: str, channel_name: str) -> None:
        self.move_to_panel_requested.emit(source_id, channel_name)

    def _on_row_reset_colour(self, source_id: str, channel_name: str) -> None:
        # Reset → emit colour_changed with None-equivalent ("#aaaaaa" placeholder)
        # The controller/session will clear color_hex; canvas will use auto colour.
        self.colour_changed.emit(source_id, channel_name, "")

    def _on_row_reset_name(self, source_id: str, channel_name: str) -> None:
        # Reset to channel_name (canonical) — emit with empty string as sentinel
        self.display_name_changed.emit(source_id, channel_name, channel_name)

    def _on_row_line_style(self, source_id: str, channel_name: str, style: str) -> None:
        self.line_style_changed.emit(source_id, channel_name, style)

    def _on_row_line_width(self, source_id: str, channel_name: str, width: float) -> None:
        self.line_width_changed.emit(source_id, channel_name, width)

    def _on_row_y_axis(self, source_id: str, channel_name: str, side: str) -> None:
        self.y_axis_changed.emit(source_id, channel_name, side)
