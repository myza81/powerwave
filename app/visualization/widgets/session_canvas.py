"""SessionCanvasWidget — per-panel multi-source waveform view (Phase 9D/9E + S1/S2).

Architecture
------------
Wraps a ``pg.GraphicsLayoutWidget`` (GLW).  All channels assigned to one session
panel share this widget; different sources are distinguished by colour.

S1 additions
------------
- **N independent Y-axes** via ``_RightAxisManager``: each unit group that is
  assigned to the right side gets its own independent ViewBox + AxisItem inside
  the GLW layout.  Voltage (left) and current / frequency / power (right) can
  therefore auto-scale independently — no more 110 kV overlapping 50 Hz.
- **Trigger markers** (``set_trigger_marker`` / ``remove_trigger_marker``):
  a per-source vertical DashDot InfiniteLine at the source's trigger time,
  colour-matched to the source, labelled with the source display name.

S2 additions
------------
- **RMS overlay** (``update_rms_curve`` / ``remove_rms_curve`` / ``clear_rms_curves``):
  per-channel sliding-RMS envelope drawn as a lighter dashed line in the same
  ViewBox as the raw waveform.  In RMS_ONLY mode the raw curve is hidden.

A ChannelLegendWidget sits below the plot and reflects the current set of
curves, their colours, display names, and visibility (Phase 9E).

A _PanelHeader strip sits above the plot and provides a right-click context
menu for panel merge and split operations (Phase 9E).

Curve lifecycle
---------------
- update_curve() creates a PlotDataItem on first call for a key, then reuses
  the same item for all subsequent calls (setData only — no recreation).
- update_curve_pen() fast-updates the pen colour without any data copy.
- remove_source() deletes all items belonging to one source.
- clear_all() resets the canvas to empty without destroying the widget.

SynchronizationManager compatibility
-------------------------------------
- _primary_plot  : pg.PlotItem attribute (accessed directly by SyncManager)
- normalize_viewport(t_start, t_end) : called by SyncManager._set_x_range
- getViewBox()   : fallback extraction path used by SyncManager
"""
from __future__ import annotations

import dataclasses
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QAction, QCursor
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)

from app.ui.session.legend_widget import ChannelLegendWidget
from app.visualization.axis.datetime_axis import DatetimeAxisItem


# ---------------------------------------------------------------------------
# Panel header (thin title strip with right-click context menu)
# ---------------------------------------------------------------------------


_CHANNEL_MIME = "application/x-powerwave-channel"


class _PanelHeader(QWidget):
    """Thin title bar above the plot — collapse toggle, right-click merge/split, drop target."""

    channel_dropped = pyqtSignal(str, str, str)   # source_id, channel_name, from_panel_id
    collapse_toggled = pyqtSignal(bool)           # True = collapsed
    rename_requested = pyqtSignal(str)            # new_title

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(20)
        self.setStyleSheet("background: #252525;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(4)

        self._collapsed = False
        self._collapse_btn = QLabel("▼")
        self._collapse_btn.setFixedWidth(14)
        self._collapse_btn.setStyleSheet(
            "color: #888; font-size: 9px; background: transparent;"
        )
        self._collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapse_btn.mousePressEvent = lambda _e: self._toggle_collapse()
        lay.addWidget(self._collapse_btn)

        self._lbl = QLabel(title)
        self._lbl.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #ccc; "
            "background: transparent;"
        )
        self._lbl.mouseDoubleClickEvent = lambda _e: self._start_rename()
        lay.addWidget(self._lbl)
        lay.addStretch()
        self._drop_hint = QLabel("↓ drop here")
        self._drop_hint.setStyleSheet("color: #555; font-size: 9px; background: transparent;")
        self._drop_hint.setVisible(False)
        lay.addWidget(self._drop_hint)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.setAcceptDrops(True)

    def set_title(self, title: str) -> None:
        self._lbl.setText(title)

    def _start_rename(self) -> None:
        current = self._lbl.text()
        new_title, ok = QInputDialog.getText(
            self, "Rename panel", "Panel name:", text=current
        )
        if ok and new_title.strip() and new_title.strip() != current:
            self._lbl.setText(new_title.strip())
            self.rename_requested.emit(new_title.strip())

    def _toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        self._collapse_btn.setText("▶" if self._collapsed else "▼")
        self.collapse_toggled.emit(self._collapsed)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(_CHANNEL_MIME):
            event.acceptProposedAction()
            self.setStyleSheet("background: #3A5A3A;")
            self._drop_hint.setVisible(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet("background: #252525;")
        self._drop_hint.setVisible(False)

    def dropEvent(self, event) -> None:
        self.setStyleSheet("background: #252525;")
        raw = bytes(event.mimeData().data(_CHANNEL_MIME)).decode("utf-8")
        parts = raw.split("|||")
        if len(parts) == 3:
            source_id, channel_name, from_panel_id = parts
            self.channel_dropped.emit(source_id, channel_name, from_panel_id)
            event.acceptProposedAction()


# ---------------------------------------------------------------------------
# SI-prefix Y-axis
# ---------------------------------------------------------------------------


class _SIAxisItem(pg.AxisItem):
    """Y-axis that renders large tick values with K / M SI prefix."""

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            vs = v * scale
            abs_vs = abs(vs)
            if vs == 0:
                strings.append("0")
            elif abs_vs >= 1_000_000:
                strings.append(f"{vs / 1_000_000:.3g}M")
            elif abs_vs >= 1_000:
                strings.append(f"{vs / 1_000:.3g}K")
            else:
                strings.append(f"{vs:.4g}")
        return strings


# ---------------------------------------------------------------------------
# Right-axis group manager (S1)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _RightAxisEntry:
    norm_unit: str
    viewbox: pg.ViewBox
    axis_item: pg.AxisItem


class _RightAxisManager:
    """Independent right-side ViewBox per unit group.

    The first distinct unit passed to ``get_or_create()`` gets column 1 in the
    parent GLW layout; subsequent units get columns 2, 3, … Each ViewBox is
    added to the plot scene and kept geometry-synced with the primary ViewBox
    via ``sigResized``.
    """

    def __init__(self, primary_plot: pg.PlotItem, glw: pg.GraphicsLayoutWidget) -> None:
        self._primary = primary_plot
        self._glw = glw
        self._entries: dict[str, _RightAxisEntry] = {}
        self._next_col: int = 1
        self._primary.getViewBox().sigResized.connect(self._sync)

    def get_or_create(self, norm_unit: str, label: str, color: str) -> pg.ViewBox:
        """Return the ViewBox for this unit group, creating it if needed."""
        if norm_unit in self._entries:
            return self._entries[norm_unit].viewbox

        vb = pg.ViewBox()
        vb.setXLink(self._primary)
        scene = self._primary.scene()
        if scene is not None:
            scene.addItem(vb)

        ax = _SIAxisItem("right")
        ax.enableAutoSIPrefix(False)
        ax.setLabel(label)
        ax.setPen(pg.mkPen(color))
        ax.setTextPen(pg.mkPen(color))
        ax.linkToView(vb)

        self._glw.addItem(ax, row=0, col=self._next_col)
        self._next_col += 1

        # Align new ViewBox with the primary before the first resize
        primary_rect = self._primary.getViewBox().sceneBoundingRect()
        vb.setGeometry(primary_rect)
        vb.linkedViewChanged(self._primary.getViewBox(), vb.XAxis)

        self._entries[norm_unit] = _RightAxisEntry(norm_unit, vb, ax)
        return vb

    def get(self, norm_unit: str) -> pg.ViewBox | None:
        e = self._entries.get(norm_unit)
        return e.viewbox if e else None

    def clear(self) -> None:
        """Remove all secondary ViewBoxes and axes from the scene / layout."""
        for entry in self._entries.values():
            try:
                scene = entry.viewbox.scene()
                if scene is not None:
                    scene.removeItem(entry.viewbox)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._glw.removeItem(entry.axis_item)
            except Exception:  # noqa: BLE001
                pass
        self._entries.clear()
        self._next_col = 1

    def _sync(self) -> None:
        primary_vb = self._primary.getViewBox()
        rect = primary_vb.sceneBoundingRect()
        for entry in self._entries.values():
            entry.viewbox.setGeometry(rect)
            entry.viewbox.linkedViewChanged(primary_vb, entry.viewbox.XAxis)


# ---------------------------------------------------------------------------
# Curve metadata
# ---------------------------------------------------------------------------


_STYLE_MAP = {
    "solid":  Qt.PenStyle.SolidLine,
    "dashed": Qt.PenStyle.DashLine,
    "dotted": Qt.PenStyle.DotLine,
}


def _migrate_item(item: pg.PlotDataItem, old_vb: pg.ViewBox, new_vb: pg.ViewBox) -> None:
    """Move a PlotDataItem between ViewBoxes without crashing on autoRangeEnabled.

    pg.ViewBox.addItem calls scene.addItem(item) before item.setParentItem(childGroup).
    When item already has data, the intermediate state (parentItem=None, in scene) causes
    itemChange → _updateView → getViewBox to fall back to GraphicsLayoutWidget (the scene
    view), which lacks autoRangeEnabled.  Temporarily clearing opts['clipToView'] avoids
    the crash at that exact moment without affecting the final rendered state.
    """
    was_clip = item.opts.get('clipToView', False)
    item.opts['clipToView'] = False
    old_vb.removeItem(item)
    new_vb.addItem(item)
    item.opts['clipToView'] = was_clip


def _rms_pen_color(hex_color: str) -> str:
    """Blend *hex_color* 40 % toward white — lighter shade for RMS overlay curves."""
    c = hex_color.lstrip("#")
    if len(c) != 6:
        return "#FFFFFF"
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r2 = min(255, r + int((255 - r) * 0.4))
    g2 = min(255, g + int((255 - g) * 0.4))
    b2 = min(255, b + int((255 - b) * 0.4))
    return f"#{r2:02X}{g2:02X}{b2:02X}"


def _phasor_mag_pen_color(hex_color: str) -> str:
    """60 % blend toward white — brighter than RMS (40 %) for magnitude overlays."""
    from app.analytics.phasors.phasor_models import PhasorDisplayMode
    from app.visualization.overlays.overlay_colors import phasor_color
    return phasor_color("", PhasorDisplayMode.MAGNITUDE, base_color=hex_color)


def _phasor_angle_pen_color(hex_color: str) -> str:
    """40 % blend toward cyan — distinguishable from raw and magnitude overlays."""
    from app.analytics.phasors.phasor_models import PhasorDisplayMode
    from app.visualization.overlays.overlay_colors import phasor_color
    return phasor_color("", PhasorDisplayMode.ANGLE, base_color=hex_color)


@dataclasses.dataclass
class _CurveMetadata:
    source_id: str
    channel_name: str
    display_name: str
    source_badge: str
    unit: str | None
    colour: str
    visible: bool
    line_style: str = "solid"
    line_width: float = 1.0
    y_axis_side: str = "left"


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------


class SessionCanvasWidget(QWidget):
    """Single-panel waveform canvas for a multi-source EventAnalysisSession.

    Signals
    -------
    merge_with_requested(panel_id_a, panel_id_b)  — user chose "Merge with →"
    split_by_source_requested(panel_id)           — user chose "Split by source"
    split_by_type_requested(panel_id)             — user chose "Split by type"
    """

    merge_with_requested = pyqtSignal(str, str)   # my_panel_id, target_panel_id
    split_by_source_requested = pyqtSignal(str)   # my_panel_id
    split_by_type_requested = pyqtSignal(str)     # my_panel_id
    channel_drop_requested = pyqtSignal(str, str, str, str)  # source_id, ch_name, from_panel, to_panel
    measurement_cursors_moved = pyqtSignal(float, float)  # t_a, t_b  (S6)
    cursor_a_moved = pyqtSignal(float)            # t_a — for cross-panel sync
    cursor_b_moved = pyqtSignal(float)            # t_b — for cross-panel sync
    measurement_mode_toggle_requested = pyqtSignal(bool)  # from right-click menu
    cursor_sync_toggle_requested = pyqtSignal(bool)       # from right-click menu
    crosshair_moved = pyqtSignal(float)           # hover crosshair x position
    crosshair_values_changed = pyqtSignal(float, list)    # (t, [(name, value, unit, color), ...])
    panel_title_changed = pyqtSignal(str, str)    # panel_id, new_title

    def __init__(
        self,
        panel_id: str,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.panel_id = panel_id
        self._panel_title = title
        self._mergeable_panels: list[tuple[str, str]] = []

        self._curves: dict[tuple[str, str], pg.PlotDataItem] = {}
        self._digital_hi_curves: dict[tuple[str, str], pg.PlotDataItem] = {}  # thick — HIGH
        self._digital_lo_curves: dict[tuple[str, str], pg.PlotDataItem] = {}  # thin  — LOW
        self._digital_raw: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, str, str]] = {}  # key → (time, values, label, color)
        self._zero_lines: dict[str, pg.InfiniteLine] = {}
        self._trigger_markers: dict[str, pg.InfiniteLine] = {}  # S1
        self._rms_curves: dict[tuple[str, str], pg.PlotDataItem] = {}  # S2
        self._rms_raw_hidden: set[tuple[str, str]] = set()             # S2
        self._phasor_curves: dict[tuple[str, str], pg.PlotDataItem] = {}  # S3
        self._harmonic_curves: dict[tuple[str, str], dict[int, pg.PlotDataItem]] = {}  # S4
        self._cursor_a: pg.InfiniteLine | None = None  # S6
        self._cursor_b: pg.InfiniteLine | None = None  # S6
        self._measurement_mode: bool = False           # S6
        self._cursor_sync: bool = True                 # mirrors controller state for menu display
        self._metadata: dict[tuple[str, str], _CurveMetadata] = {}
        self._digital_n_rows: int = 0                  # 0 = not a digital panel

        self._annotations: dict[str, pg.InfiniteLine] = {}
        self._annot_counter: int = 0
        self._last_right_click_x: float = 0.0
        self._hover_cursor: pg.InfiniteLine | None = None   # synchronized hover crosshair
        self._hover_proxy: pg.SignalProxy | None = None     # rate-limited mouse-move relay

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Panel header strip (collapse toggle, right-click merge/split, drag target)
        self._header = _PanelHeader(self._panel_title, parent=self)
        self._header.customContextMenuRequested.connect(self._show_panel_menu)
        self._header.channel_dropped.connect(
            lambda sid, cname, from_pid: self.channel_drop_requested.emit(
                sid, cname, from_pid, self.panel_id
            )
        )
        self._header.collapse_toggled.connect(self._on_collapse_toggled)
        self._header.rename_requested.connect(
            lambda title: self.panel_title_changed.emit(self.panel_id, title)
        )
        layout.addWidget(self._header)

        # Main plot area: GraphicsLayoutWidget hosts PlotItem at col=0;
        # _RightAxisManager adds secondary AxisItems at col=1, 2, …
        self._datetime_axis = DatetimeAxisItem(orientation="bottom")
        self._left_axis = _SIAxisItem("left")
        self._left_axis.enableAutoSIPrefix(False)
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("#1E1E1E")
        self._primary_plot: pg.PlotItem = self._glw.addPlot(
            row=0, col=0, axisItems={"bottom": self._datetime_axis, "left": self._left_axis}
        )
        self._primary_plot.showGrid(x=True, y=True, alpha=0.2)
        self._primary_plot.setLabel("bottom", "Time")

        # Inject cursor items at the top of pyqtgraph's built-in right-click menu
        self._inject_cursor_menu_items()

        # Hover crosshair — thin semi-transparent white vertical line, always present
        self._hover_cursor = pg.InfiniteLine(
            pos=0.0, angle=90, movable=False,
            pen=pg.mkPen("#FFFFFF55", width=1),
        )
        self._hover_cursor.setVisible(False)
        self._primary_plot.addItem(self._hover_cursor)

        # Rate-limited mouse-move listener wired to the GLW scene
        self._hover_proxy = pg.SignalProxy(
            self._glw.scene().sigMouseMoved,
            rateLimit=60, delay=0,
            slot=self._on_scene_mouse_moved,
        )

        # N-axis manager for independent right-side Y-axes (S1)
        self._right_mgr = _RightAxisManager(self._primary_plot, self._glw)

        # Legend below the plot (Phase 9E)
        self.legend = ChannelLegendWidget(self.panel_id, parent=self)

        # Plot row: GLW + optional Y scrollbar for digital panels
        self._plot_row_widget = QWidget()
        plot_row = QHBoxLayout(self._plot_row_widget)
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.setSpacing(0)
        plot_row.addWidget(self._glw, stretch=1)

        self._y_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self._y_scrollbar.setVisible(False)
        self._y_scrollbar.setStyleSheet(
            "QScrollBar:vertical { background: #1E1E1E; width: 12px; }"
            "QScrollBar::handle:vertical { background: #555; min-height: 20px; border-radius: 4px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        plot_row.addWidget(self._y_scrollbar)

        layout.addWidget(self._plot_row_widget, stretch=4)
        layout.addWidget(self.legend, stretch=0)

        self._install_keyboard_shortcuts()

    # ─────────────────────────────────────────────────────────────────────────
    # Keyboard shortcuts (zoom / pan)
    # ─────────────────────────────────────────────────────────────────────────

    def _install_keyboard_shortcuts(self) -> None:
        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
        for key, slot in (
            (Qt.Key.Key_Home,  self._kb_fit),
            (Qt.Key.Key_Left,  self._kb_pan_left),
            (Qt.Key.Key_Right, self._kb_pan_right),
            (Qt.Key.Key_Plus,  self._kb_zoom_in),
            (Qt.Key.Key_Equal, self._kb_zoom_in),   # unshifted + on most keyboards
            (Qt.Key.Key_Minus, self._kb_zoom_out),
        ):
            sc = QShortcut(QKeySequence(key), self, context=ctx)
            sc.activated.connect(slot)

    def _kb_fit(self) -> None:
        self._primary_plot.getViewBox().autoRange()

    def _kb_pan_left(self) -> None:
        vb = self._primary_plot.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        step = (x_max - x_min) * 0.1
        vb.translateBy(x=-step)

    def _kb_pan_right(self) -> None:
        vb = self._primary_plot.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        step = (x_max - x_min) * 0.1
        vb.translateBy(x=step)

    def _kb_zoom_in(self) -> None:
        self._primary_plot.getViewBox().scaleBy((0.7, 1.0))

    def _kb_zoom_out(self) -> None:
        self._primary_plot.getViewBox().scaleBy((1.3, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Panel header context menu
    # ─────────────────────────────────────────────────────────────────────────

    def _inject_cursor_menu_items(self) -> None:
        """Prepend Measurement Cursor items to pyqtgraph's ViewBox context menu."""
        vb_menu = self._primary_plot.getViewBox().menu

        self._cursor_menu_act = QAction("Measurement Cursors", vb_menu)
        self._cursor_menu_act.setCheckable(True)

        self._cursor_sync_menu_act = QAction("Sync Cursors Across Panels", vb_menu)
        self._cursor_sync_menu_act.setCheckable(True)

        sep = QAction(vb_menu)
        sep.setSeparator(True)

        first = vb_menu.actions()[0] if vb_menu.actions() else None
        if first:
            vb_menu.insertAction(first, self._cursor_menu_act)
            vb_menu.insertAction(first, self._cursor_sync_menu_act)
            vb_menu.insertAction(first, sep)
        else:
            vb_menu.addAction(self._cursor_menu_act)
            vb_menu.addAction(self._cursor_sync_menu_act)

        self._cursor_menu_act.triggered.connect(
            lambda checked: self.measurement_mode_toggle_requested.emit(checked)
        )
        self._cursor_sync_menu_act.triggered.connect(
            lambda checked: self.cursor_sync_toggle_requested.emit(checked)
        )

        # Annotation + export actions
        sep2 = QAction(vb_menu)
        sep2.setSeparator(True)
        self._annot_add_act = QAction("Add annotation here…", vb_menu)
        self._annot_clear_act = QAction("Clear annotations", vb_menu)
        self._export_act = QAction("Export view…", vb_menu)
        for act in (sep2, self._annot_add_act, self._annot_clear_act, self._export_act):
            vb_menu.addAction(act)
        self._annot_add_act.triggered.connect(self._add_annotation_at_last_x)
        self._annot_clear_act.triggered.connect(self.clear_annotations)
        self._export_act.triggered.connect(self.export_view)

        vb_menu.aboutToShow.connect(self._sync_cursor_menu_state)

    def _sync_cursor_menu_state(self) -> None:
        """Update checkmarks/enabled state before the ViewBox menu is displayed."""
        self._cursor_menu_act.blockSignals(True)
        self._cursor_menu_act.setChecked(self._measurement_mode)
        self._cursor_menu_act.blockSignals(False)

        self._cursor_sync_menu_act.blockSignals(True)
        self._cursor_sync_menu_act.setChecked(self._cursor_sync)
        self._cursor_sync_menu_act.setEnabled(self._measurement_mode)
        self._cursor_sync_menu_act.blockSignals(False)

        # Capture the current cursor x position in view coordinates
        vb = self._primary_plot.getViewBox()
        global_pos = QCursor.pos()
        widget_pos = self._glw.mapFromGlobal(global_pos)
        scene_pos = self._glw.mapToScene(widget_pos.x(), widget_pos.y())
        view_pt = vb.mapSceneToView(scene_pos)
        self._last_right_click_x = float(view_pt.x())

        self._annot_clear_act.setEnabled(bool(self._annotations))

    def _add_annotation_at_last_x(self) -> None:
        t = self._last_right_click_x
        default = f"t={t:.3f}s"
        label, ok = QInputDialog.getText(self, "Add Annotation", "Label:", text=default)
        if ok and label.strip():
            self._add_annotation(t, label.strip())

    def _add_annotation(self, x: float, label: str) -> None:
        self._annot_counter += 1
        ann_id = f"annot_{self._annot_counter}"
        line = pg.InfiniteLine(
            pos=x,
            angle=90,
            movable=True,
            pen=pg.mkPen("#FFB347", width=1, style=Qt.PenStyle.DashLine),
            label=label,
            labelOpts={"position": 0.5, "color": "#FFB347", "fill": (0, 0, 0, 100)},
        )
        self._primary_plot.addItem(line)
        self._annotations[ann_id] = line

    def clear_annotations(self) -> None:
        """Remove all annotation markers from this panel."""
        for line in self._annotations.values():
            try:
                self._primary_plot.removeItem(line)
            except Exception:  # noqa: BLE001
                pass
        self._annotations.clear()

    def set_mergeable_panels(self, panels: list[tuple[str, str]]) -> None:
        """Update the panel list shown in the "Merge with →" submenu."""
        self._mergeable_panels = panels

    def _show_panel_menu(self, pos) -> None:
        menu = QMenu(self)

        # ── Cursor controls ───────────────────────────────────────────────────
        cursor_act = menu.addAction("Measurement Cursors")
        cursor_act.setCheckable(True)
        cursor_act.setChecked(self._measurement_mode)
        cursor_act.toggled.connect(self.measurement_mode_toggle_requested.emit)

        sync_act = menu.addAction("Sync Cursors Across Panels")
        sync_act.setCheckable(True)
        sync_act.setChecked(self._cursor_sync)
        sync_act.setEnabled(self._measurement_mode)
        sync_act.toggled.connect(self.cursor_sync_toggle_requested.emit)

        menu.addSeparator()
        # ── Panel layout ─────────────────────────────────────────────────────
        merge_menu = menu.addMenu("Merge with →")
        if self._mergeable_panels:
            for pid, ptitle in self._mergeable_panels:
                act = merge_menu.addAction(ptitle)
                act.triggered.connect(
                    lambda checked=False, _pid=pid: self.merge_with_requested.emit(
                        self.panel_id, _pid
                    )
                )
        else:
            no_act = merge_menu.addAction("(no other panels)")
            no_act.setEnabled(False)

        exp_act = menu.addAction("Export view…")
        exp_act.triggered.connect(self.export_view)

        menu.addSeparator()
        clr_ann = menu.addAction(f"Clear annotations ({len(self._annotations)})")
        clr_ann.setEnabled(bool(self._annotations))
        clr_ann.triggered.connect(self.clear_annotations)

        menu.addSeparator()
        split_src = menu.addAction("Split by source")
        split_type = menu.addAction("Split by type")

        split_src.triggered.connect(
            lambda: self.split_by_source_requested.emit(self.panel_id)
        )
        split_type.triggered.connect(
            lambda: self.split_by_type_requested.emit(self.panel_id)
        )

        menu.exec(self._header.mapToGlobal(pos))

    # ─────────────────────────────────────────────────────────────────────────
    # SynchronizationManager interface
    # ─────────────────────────────────────────────────────────────────────────

    def normalize_viewport(self, t_start: float, t_end: float) -> None:
        self._primary_plot.setXRange(t_start, t_end, padding=0)

    def getViewBox(self) -> pg.ViewBox:
        return self._primary_plot.getViewBox()

    # ─────────────────────────────────────────────────────────────────────────
    # ViewBox resolution helpers (S1)
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_viewbox(self, y_axis_side: str, unit: str | None, color: str) -> pg.ViewBox:
        """Return the ViewBox for a curve given its axis side and unit."""
        if y_axis_side == "right":
            norm_unit = (unit or "other").lower().strip() or "other"
            label = unit or "Level"
            return self._right_mgr.get_or_create(norm_unit, label, color)
        return self._primary_plot.getViewBox()

    # ─────────────────────────────────────────────────────────────────────────
    # Curve management
    # ─────────────────────────────────────────────────────────────────────────

    def update_curve(
        self,
        source_id: str,
        channel_name: str,
        time: np.ndarray,
        values: np.ndarray,
        *,
        color: str = "#AAAAAA",
        visible: bool = True,
        display_name: str | None = None,
        source_badge: str = "",
        unit: str | None = None,
        line_style: str = "solid",
        line_width: float = 1.0,
        y_axis_side: str = "left",
    ) -> None:
        key = (source_id, channel_name)
        pen = pg.mkPen(color, width=line_width, style=_STYLE_MAP.get(line_style, Qt.PenStyle.SolidLine))

        target_vb = self._resolve_viewbox(y_axis_side, unit, color)
        existing_meta = self._metadata.get(key)

        if key not in self._curves:
            curve = pg.PlotDataItem(pen=pen, skipFiniteCheck=True)
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            target_vb.addItem(curve)
            self._curves[key] = curve
        else:
            self._curves[key].setPen(pen)
            # Migrate ViewBox if axis side or unit changed
            if existing_meta is not None:
                old_vb = self._resolve_viewbox(
                    existing_meta.y_axis_side, existing_meta.unit, existing_meta.colour
                )
                if old_vb is not target_vb:
                    _migrate_item(self._curves[key], old_vb, target_vb)

        curve = self._curves[key]
        if len(time) > 0 and len(values) > 0:
            curve.setData(x=time, y=values)
        else:
            curve.setData(x=np.array([]), y=np.array([]))
        curve.setVisible(visible)

        eff_display = display_name if display_name is not None else channel_name
        self._metadata[key] = _CurveMetadata(
            source_id=source_id,
            channel_name=channel_name,
            display_name=eff_display,
            source_badge=source_badge,
            unit=unit,
            colour=color,
            visible=visible,
            line_style=line_style,
            line_width=line_width,
            y_axis_side=y_axis_side,
        )
        self.legend.upsert_row(
            source_id, channel_name, eff_display, source_badge, unit, color, visible
        )

    def update_curve_pen(self, source_id: str, channel_name: str, color: str) -> None:
        """Update pen colour only — preserves existing line style and width."""
        key = (source_id, channel_name)
        meta = self._metadata.get(key)
        style = _STYLE_MAP.get(meta.line_style if meta else "solid", Qt.PenStyle.SolidLine)
        width = meta.line_width if meta else 1.0
        curve = self._curves.get(key)
        if curve is not None:
            curve.setPen(pg.mkPen(color, width=width, style=style))
        if meta is not None:
            meta.colour = color
        self.legend.update_row_colour(source_id, channel_name, color)

    def update_curve_style(
        self, source_id: str, channel_name: str, line_style: str, line_width: float
    ) -> None:
        """Update pen style and width only — no data copy, O(1)."""
        key = (source_id, channel_name)
        meta = self._metadata.get(key)
        color = meta.colour if meta else "#AAAAAA"
        pen = pg.mkPen(color, width=line_width, style=_STYLE_MAP.get(line_style, Qt.PenStyle.SolidLine))
        curve = self._curves.get(key)
        if curve is not None:
            curve.setPen(pen)
        if meta is not None:
            meta.line_style = line_style
            meta.line_width = line_width

    def set_curve_y_axis(self, source_id: str, channel_name: str, side: str) -> None:
        """Move a curve between left and right Y-axes."""
        key = (source_id, channel_name)
        meta = self._metadata.get(key)
        curve = self._curves.get(key)
        if meta is None or curve is None or meta.y_axis_side == side:
            return
        old_vb = self._resolve_viewbox(meta.y_axis_side, meta.unit, meta.colour)
        new_vb = self._resolve_viewbox(side, meta.unit, meta.colour)
        if old_vb is not new_vb:
            _migrate_item(curve, old_vb, new_vb)
        meta.y_axis_side = side

    def set_curve_visible(
        self, source_id: str, channel_name: str, visible: bool
    ) -> None:
        key = (source_id, channel_name)
        curve = self._curves.get(key)
        if curve is not None:
            # Don't un-hide a raw curve that RMS_ONLY mode is suppressing
            if visible and key in self._rms_raw_hidden:
                pass
            else:
                curve.setVisible(visible)
        meta = self._metadata.get(key)
        if meta is not None:
            meta.visible = visible
        self.legend.update_row_visible(source_id, channel_name, visible)

    def update_digital_curve(
        self,
        source_id: str,
        channel_name: str,
        time: np.ndarray,
        values: np.ndarray,
        *,
        color: str = "#AAAAAA",
        y_offset: float = 0.0,
        visible: bool = True,
        display_name: str | None = None,
        source_badge: str = "",
        unit: str | None = None,
    ) -> None:
        """Render a binary digital channel as thick (HIGH) / thin (LOW) horizontal lines.

        Unlike update_curve(), no fill is drawn and the signal stays at a fixed Y
        position (y_offset) rather than scaling between 0 and 1.
        """
        from app.visualization.rendering.digital_transforms import (
            build_hi_lo_segments,
            extract_transitions,
        )
        key = (source_id, channel_name)
        _empty = np.empty(0, dtype=np.float64)
        vb = self._primary_plot.getViewBox()

        if len(time) >= 2:
            t_tr, d_tr = extract_transitions(np.asarray(time, dtype=np.float64),
                                             np.asarray(values, dtype=np.float64))
            t_hi, y_hi = build_hi_lo_segments(t_tr, d_tr, y_offset, 1)
            t_lo, y_lo = build_hi_lo_segments(t_tr, d_tr, y_offset, 0)
        else:
            t_hi = y_hi = t_lo = y_lo = _empty

        for curves_dict, t_seg, y_seg, width, alpha in (
            (self._digital_hi_curves, t_hi, y_hi, 3,   "FF"),
            (self._digital_lo_curves, t_lo, y_lo, 1,   "44"),
        ):
            pen = pg.mkPen(color if alpha == "FF" else color + alpha, width=width)
            if key not in curves_dict:
                curve = pg.PlotDataItem(pen=pen, connect="finite")
                vb.addItem(curve)
                curves_dict[key] = curve
            else:
                curves_dict[key].setPen(pen)
            curves_dict[key].setData(
                t_seg if len(t_seg) else _empty,
                y_seg if len(y_seg) else _empty,
            )
            curves_dict[key].setVisible(visible)

        eff_display = display_name if display_name is not None else channel_name
        self.legend.upsert_row(
            source_id, channel_name, eff_display, source_badge, unit, color, visible
        )
        self._digital_raw[key] = (
            np.asarray(time, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            eff_display,
            color,
        )

    def _remove_digital_curve(self, key: tuple[str, str]) -> None:
        """Remove both hi and lo digital curves."""
        self._digital_raw.pop(key, None)
        vb = self._primary_plot.getViewBox()
        for curves_dict in (self._digital_hi_curves, self._digital_lo_curves):
            curve = curves_dict.pop(key, None)
            if curve is not None:
                try:
                    vb.removeItem(curve)
                except Exception:  # noqa: BLE001
                    pass

    def remove_curve(self, source_id: str, channel_name: str) -> None:
        """Remove one curve and all its overlays; clean up the legend row."""
        key = (source_id, channel_name)
        self._remove_digital_curve(key)
        meta = self._metadata.pop(key, None)
        curve = self._curves.pop(key, None)
        if curve is not None:
            vb = self._resolve_viewbox(
                meta.y_axis_side if meta else "left",
                meta.unit if meta else None,
                meta.colour if meta else "#AAAAAA",
            )
            try:
                vb.removeItem(curve)
            except Exception:  # noqa: BLE001
                pass
        self.remove_rms_curve(source_id, channel_name)
        self.remove_phasor_curve(source_id, channel_name)
        self.remove_harmonic_curves(source_id, channel_name)
        self.legend.remove_row(source_id, channel_name)

    def remove_source(self, source_id: str) -> None:
        for key in [k for k in self._digital_hi_curves if k[0] == source_id]:
            self._remove_digital_curve(key)
        stale = [k for k in self._curves if k[0] == source_id]
        for key in stale:
            meta = self._metadata.pop(key, None)
            curve = self._curves.pop(key)
            vb = self._resolve_viewbox(
                meta.y_axis_side if meta else "left",
                meta.unit if meta else None,
                meta.colour if meta else "#AAAAAA",
            )
            try:
                vb.removeItem(curve)
            except Exception:  # noqa: BLE001
                pass
        # Clean up RMS overlay curves for this source
        rms_stale = [k for k in self._rms_curves if k[0] == source_id]
        for key in rms_stale:
            self.remove_rms_curve(key[0], key[1])
        # Clean up phasor overlay curves for this source (S3)
        phasor_stale = [k for k in self._phasor_curves if k[0] == source_id]
        for key in phasor_stale:
            self.remove_phasor_curve(key[0], key[1])
        # Clean up harmonic overlay curves for this source (S4)
        harmonic_stale = [k for k in self._harmonic_curves if k[0] == source_id]
        for key in harmonic_stale:
            self.remove_harmonic_curves(key[0], key[1])
        self.legend.remove_source_rows(source_id)
        self.remove_zero_line(source_id)
        self.remove_trigger_marker(source_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Zero-line markers (source time-offset indicators)
    # ─────────────────────────────────────────────────────────────────────────

    def update_zero_line(
        self,
        source_id: str,
        display_name: str,
        offset_s: float,
        color: str,
    ) -> None:
        self.remove_zero_line(source_id)
        label = f"{display_name}  {offset_s:+.3f} s"
        pen = pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine)
        line = pg.InfiniteLine(
            pos=offset_s,
            angle=90,
            movable=False,
            pen=pen,
            label=label,
            labelOpts={"position": 0.05, "color": color, "fill": (0, 0, 0, 80)},
        )
        self._primary_plot.addItem(line)
        self._zero_lines[source_id] = line

    def remove_zero_line(self, source_id: str) -> None:
        line = self._zero_lines.pop(source_id, None)
        if line is not None:
            self._primary_plot.removeItem(line)

    # ─────────────────────────────────────────────────────────────────────────
    # Trigger markers (S1 — per-source vertical line at trigger time)
    # ─────────────────────────────────────────────────────────────────────────

    def set_trigger_marker(
        self,
        source_id: str,
        t_s: float,
        color: str,
        label: str = "",
    ) -> None:
        """Add or replace the trigger marker for a source."""
        self.remove_trigger_marker(source_id)
        pen = pg.mkPen(color, width=1, style=Qt.PenStyle.DashDotLine)
        lbl = label or "▲"
        line = pg.InfiniteLine(
            pos=t_s,
            angle=90,
            movable=False,
            pen=pen,
            label=lbl,
            labelOpts={"position": 0.95, "color": color, "fill": (0, 0, 0, 80)},
        )
        self._primary_plot.addItem(line)
        self._trigger_markers[source_id] = line

    def remove_trigger_marker(self, source_id: str) -> None:
        """Remove the trigger marker for a source if present."""
        line = self._trigger_markers.pop(source_id, None)
        if line is not None:
            self._primary_plot.removeItem(line)

    # ─────────────────────────────────────────────────────────────────────────
    # RMS overlay curves (S2)
    # ─────────────────────────────────────────────────────────────────────────

    def update_rms_curve(
        self,
        source_id: str,
        channel_name: str,
        rms_time: np.ndarray,
        rms_values: np.ndarray,
        color: str,
        unit: str | None = None,
        *,
        rms_only: bool = False,
    ) -> None:
        """Add or update a sliding-RMS overlay for one channel.

        The envelope is drawn as a lighter dashed line in the same ViewBox as
        the raw waveform.  When *rms_only* is True the raw curve is hidden so
        only the envelope is visible.
        """
        key = (source_id, channel_name)
        rms_color = _rms_pen_color(color)
        pen = pg.mkPen(rms_color, width=2, style=Qt.PenStyle.DashLine)

        meta = self._metadata.get(key)
        vb = self._resolve_viewbox(
            meta.y_axis_side if meta else "left",
            meta.unit if meta else unit,
            color,
        )

        if key not in self._rms_curves:
            rms_curve = pg.PlotDataItem(pen=pen, skipFiniteCheck=True)
            rms_curve.setClipToView(True)
            vb.addItem(rms_curve)
            self._rms_curves[key] = rms_curve
        else:
            self._rms_curves[key].setPen(pen)

        rms_curve = self._rms_curves[key]
        if len(rms_time) > 0 and len(rms_values) > 0:
            rms_curve.setData(x=rms_time, y=rms_values)
        else:
            rms_curve.setData(x=np.array([]), y=np.array([]))
        rms_curve.setVisible(True)

        # RMS_ONLY: suppress the raw curve without touching legend visibility
        raw_curve = self._curves.get(key)
        if raw_curve is not None:
            if rms_only:
                raw_curve.setVisible(False)
                self._rms_raw_hidden.add(key)
            else:
                raw_curve.setVisible(meta.visible if meta else True)
                self._rms_raw_hidden.discard(key)

    def remove_rms_curve(self, source_id: str, channel_name: str) -> None:
        """Remove the RMS overlay for one channel and restore raw curve visibility."""
        key = (source_id, channel_name)
        rms_curve = self._rms_curves.pop(key, None)
        if rms_curve is not None:
            try:
                vb = rms_curve.getViewBox()
                if vb is not None:
                    vb.removeItem(rms_curve)
            except Exception:  # noqa: BLE001
                pass
        self._rms_raw_hidden.discard(key)
        raw_curve = self._curves.get(key)
        if raw_curve is not None:
            meta = self._metadata.get(key)
            raw_curve.setVisible(meta.visible if meta else True)

    def clear_rms_curves(self) -> None:
        """Remove all RMS overlays and restore raw curve visibility."""
        for key, rms_curve in list(self._rms_curves.items()):
            try:
                vb = rms_curve.getViewBox()
                if vb is not None:
                    vb.removeItem(rms_curve)
            except Exception:  # noqa: BLE001
                pass
            raw_curve = self._curves.get(key)
            if raw_curve is not None:
                meta = self._metadata.get(key)
                raw_curve.setVisible(meta.visible if meta else True)
        self._rms_curves.clear()
        self._rms_raw_hidden.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Phasor overlay curves (S3)
    # ─────────────────────────────────────────────────────────────────────────

    def update_phasor_curve(
        self,
        source_id: str,
        channel_name: str,
        phasor_time: np.ndarray,
        phasor_values: np.ndarray,
        color: str,
        mode,  # PhasorDisplayMode — imported locally to avoid top-level dep
    ) -> None:
        """Add or update a phasor magnitude/angle overlay for one channel.

        Magnitude overlays use a dotted, brighter-white pen.
        Angle overlays use a dash-dot, cyan-shifted pen.
        Both are drawn in the same ViewBox as the raw waveform.
        """
        from app.analytics.phasors.phasor_models import PhasorDisplayMode
        key = (source_id, channel_name)
        if mode == PhasorDisplayMode.MAGNITUDE:
            pen_color = _phasor_mag_pen_color(color)
            pen_style = Qt.PenStyle.DotLine
        else:  # ANGLE
            pen_color = _phasor_angle_pen_color(color)
            pen_style = Qt.PenStyle.DashDotLine
        pen = pg.mkPen(pen_color, width=1.5, style=pen_style)

        meta = self._metadata.get(key)
        vb = self._resolve_viewbox(
            meta.y_axis_side if meta else "left",
            meta.unit if meta else None,
            color,
        )

        if key not in self._phasor_curves:
            curve = pg.PlotDataItem(pen=pen, skipFiniteCheck=True)
            curve.setClipToView(True)
            vb.addItem(curve)
            self._phasor_curves[key] = curve
        else:
            self._phasor_curves[key].setPen(pen)

        phasor_curve = self._phasor_curves[key]
        if len(phasor_time) > 0 and len(phasor_values) > 0:
            phasor_curve.setData(x=phasor_time, y=phasor_values)
        else:
            phasor_curve.setData(x=np.array([]), y=np.array([]))
        phasor_curve.setVisible(True)

    def remove_phasor_curve(self, source_id: str, channel_name: str) -> None:
        """Remove the phasor overlay for one channel."""
        key = (source_id, channel_name)
        curve = self._phasor_curves.pop(key, None)
        if curve is not None:
            try:
                vb = curve.getViewBox()
                if vb is not None:
                    vb.removeItem(curve)
            except Exception:  # noqa: BLE001
                pass

    def clear_phasor_curves(self) -> None:
        """Remove all phasor overlay curves."""
        for curve in list(self._phasor_curves.values()):
            try:
                vb = curve.getViewBox()
                if vb is not None:
                    vb.removeItem(curve)
            except Exception:  # noqa: BLE001
                pass
        self._phasor_curves.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Harmonic magnitude overlay curves (S4)
    # ─────────────────────────────────────────────────────────────────────────

    def update_harmonic_curves(
        self,
        source_id: str,
        channel_name: str,
        harmonic_time: np.ndarray,
        magnitudes_by_order: dict[int, np.ndarray],
    ) -> None:
        """Add or update per-order harmonic magnitude overlays for one channel.

        Each harmonic order gets its own solid-line curve drawn via
        ``harmonic_order_pen(order)`` — colours are order-fixed and consistent
        across all sources.  The curves share the ViewBox of the raw waveform.
        """
        from app.visualization.overlays.overlay_colors import harmonic_order_pen
        key = (source_id, channel_name)
        meta = self._metadata.get(key)
        vb = self._resolve_viewbox(
            meta.y_axis_side if meta else "left",
            meta.unit if meta else None,
            meta.colour if meta else "#AAAAAA",
        )

        if key not in self._harmonic_curves:
            self._harmonic_curves[key] = {}
        order_curves = self._harmonic_curves[key]

        for order, mag_arr in magnitudes_by_order.items():
            if order not in order_curves:
                curve = pg.PlotDataItem(
                    pen=harmonic_order_pen(order),
                    skipFiniteCheck=True,
                )
                curve.setClipToView(True)
                vb.addItem(curve)
                order_curves[order] = curve

            h_curve = order_curves[order]
            if len(harmonic_time) > 0 and len(mag_arr) > 0:
                h_curve.setData(x=harmonic_time, y=mag_arr)
            else:
                h_curve.setData(x=np.array([]), y=np.array([]))
            h_curve.setVisible(True)

        # Remove curves for orders no longer in magnitudes_by_order
        for order in list(order_curves):
            if order not in magnitudes_by_order:
                stale = order_curves.pop(order)
                try:
                    stale_vb = stale.getViewBox()
                    if stale_vb is not None:
                        stale_vb.removeItem(stale)
                except Exception:  # noqa: BLE001
                    pass

    def remove_harmonic_curves(self, source_id: str, channel_name: str) -> None:
        """Remove all harmonic order curves for one channel."""
        key = (source_id, channel_name)
        order_curves = self._harmonic_curves.pop(key, {})
        for curve in order_curves.values():
            try:
                vb = curve.getViewBox()
                if vb is not None:
                    vb.removeItem(curve)
            except Exception:  # noqa: BLE001
                pass

    def clear_harmonic_curves(self) -> None:
        """Remove all harmonic overlay curves for all channels."""
        for order_curves in self._harmonic_curves.values():
            for curve in order_curves.values():
                try:
                    vb = curve.getViewBox()
                    if vb is not None:
                        vb.removeItem(curve)
                except Exception:  # noqa: BLE001
                    pass
        self._harmonic_curves.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Two-cursor measurement (S6)
    # ─────────────────────────────────────────────────────────────────────────

    def set_measurement_mode(self, enabled: bool) -> None:
        """Enable or disable two-cursor measurement mode.

        When enabled, cursor A (yellow) and cursor B (cyan) both appear as
        draggable InfiniteLines. Either cursor moving emits
        ``measurement_cursors_moved(t_a, t_b)``.
        """
        if enabled == self._measurement_mode:
            return
        self._measurement_mode = enabled
        if enabled:
            self._add_cursor_a()
            self._add_cursor_b()
        else:
            self._remove_cursor_b()
            self._remove_cursor_a()

    def measurement_mode(self) -> bool:
        return self._measurement_mode

    def cursor_positions(self) -> tuple[float, float] | None:
        """Return (t_a, t_b) or None if either cursor is missing."""
        if self._cursor_a is None or self._cursor_b is None:
            return None
        return float(self._cursor_a.value()), float(self._cursor_b.value())

    def _add_cursor_a(self) -> None:
        if self._cursor_a is not None:
            return
        self._cursor_a = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen("#FFFF00", width=1.5, style=Qt.PenStyle.DashLine),
            label="A",
            labelOpts={"color": "#FFFF00", "position": 0.9},
        )
        self._cursor_a.sigPositionChanged.connect(self._on_cursor_a_moved)
        self._primary_plot.addItem(self._cursor_a)

    def _add_cursor_b(self) -> None:
        if self._cursor_b is not None:
            return
        t_ref = float(self._cursor_a.value()) + 0.02 if self._cursor_a is not None else 0.02
        self._cursor_b = pg.InfiniteLine(
            pos=t_ref,
            angle=90,
            movable=True,
            pen=pg.mkPen("#00CCFF", width=1.5, style=Qt.PenStyle.DashLine),
            label="B",
            labelOpts={"color": "#00CCFF", "position": 0.85},
        )
        self._cursor_b.sigPositionChanged.connect(self._on_cursor_b_moved)
        self._primary_plot.addItem(self._cursor_b)
        self._emit_measurement()

    def _remove_cursor_a(self) -> None:
        if self._cursor_a is not None:
            try:
                self._primary_plot.removeItem(self._cursor_a)
            except Exception:  # noqa: BLE001
                pass
            self._cursor_a = None

    def _remove_cursor_b(self) -> None:
        if self._cursor_b is not None:
            try:
                self._primary_plot.removeItem(self._cursor_b)
            except Exception:  # noqa: BLE001
                pass
            self._cursor_b = None

    def _on_cursor_a_moved(self, _line: pg.InfiniteLine) -> None:
        if self._measurement_mode:
            self.cursor_a_moved.emit(float(self._cursor_a.value()))
            self._emit_measurement()

    def _on_cursor_b_moved(self, _line: pg.InfiniteLine) -> None:
        if self._cursor_b is not None:
            self.cursor_b_moved.emit(float(self._cursor_b.value()))
        self._emit_measurement()

    def set_cursor_a_pos(self, t: float) -> None:
        """Move cursor A silently (no signal — prevents cross-panel sync loops)."""
        if self._cursor_a is not None:
            self._cursor_a.blockSignals(True)
            self._cursor_a.setValue(t)
            self._cursor_a.blockSignals(False)

    def set_cursor_b_pos(self, t: float) -> None:
        """Move cursor B silently (no signal — prevents cross-panel sync loops)."""
        if self._cursor_b is not None:
            self._cursor_b.blockSignals(True)
            self._cursor_b.setValue(t)
            self._cursor_b.blockSignals(False)

    def set_cursor_sync_state(self, enabled: bool) -> None:
        """Update local sync flag so the right-click menu shows the correct checkmark."""
        self._cursor_sync = enabled

    # ─────────────────────────────────────────────────────────────────────────
    # Hover crosshair (synchronized across all session panels)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_scene_mouse_moved(self, args: tuple) -> None:
        """Rate-limited slot: move hover crosshair and emit crosshair_moved."""
        pos: QPointF = args[0]
        vb = self._primary_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            if self._hover_cursor is not None:
                self._hover_cursor.setVisible(False)
            return
        mouse_pt = vb.mapSceneToView(pos)
        t = float(mouse_pt.x())
        if self._hover_cursor is not None:
            self._hover_cursor.setValue(t)
            self._hover_cursor.setVisible(True)
        self.crosshair_moved.emit(t)
        # Compute and emit interpolated channel values for the readout bar
        self._emit_crosshair_values(t)

    def _emit_crosshair_values(self, t: float) -> None:
        """Interpolate visible channel values at t and emit crosshair_values_changed."""
        values: list[tuple[str, str, str, float | None, str, str]] = []
        time_ref: np.ndarray | None = None
        for key, curve in self._curves.items():
            if not curve.isVisible():
                continue
            meta = self._metadata.get(key)
            if meta is None:
                continue
            xdata, ydata = curve.getData()
            if xdata is None or len(xdata) < 2:
                continue
            if time_ref is None:
                time_ref = xdata
            try:
                v = float(np.interp(t, xdata, ydata))
            except Exception:  # noqa: BLE001
                v = None
            label = meta.display_name or meta.channel_name
            values.append((key[0], key[1], label, v, meta.unit or "", meta.colour))

        for key, (t_arr, v_arr, label, color) in self._digital_raw.items():
            hi_curve = self._digital_hi_curves.get(key)
            if hi_curve is None or not hi_curve.isVisible():
                continue
            if len(t_arr) == 0:
                continue
            idx = max(0, int(np.searchsorted(t_arr, t, side="right")) - 1)
            state = int(round(v_arr[idx]))
            values.append((key[0], key[1], label, "HIGH" if state == 1 else "LOW", "", color))

        if values:
            self.crosshair_values_changed.emit(t, values)

    def set_crosshair_pos(self, t: float) -> None:
        """Move the hover crosshair to t without emitting crosshair_moved."""
        if self._hover_cursor is not None:
            self._hover_cursor.setValue(t)
            self._hover_cursor.setVisible(True)
        self._emit_crosshair_values(t)

    def hide_crosshair(self) -> None:
        """Hide the hover crosshair (e.g. when mouse leaves the canvas area)."""
        if self._hover_cursor is not None:
            self._hover_cursor.setVisible(False)

    def _on_collapse_toggled(self, collapsed: bool) -> None:
        self._plot_row_widget.setVisible(not collapsed)
        self.legend.setVisible(not collapsed)
        if collapsed:
            self.setMinimumHeight(self._header.height())
            self.setMaximumHeight(self._header.height())
        else:
            self.setMinimumHeight(80)
            self.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX

    # ─────────────────────────────────────────────────────────────────────────
    # Digital panel Y scrollbar
    # ─────────────────────────────────────────────────────────────────────────

    def setup_digital_scroll(self, n_rows: int) -> None:
        """Show a vertical scrollbar for a digital panel with n_rows channel rows.

        Safe to call on every refresh — only rewires if n_rows changed.
        """
        if n_rows == self._digital_n_rows and self._y_scrollbar.isVisible():
            return  # already configured for this row count, don't reset
        self._digital_n_rows = n_rows

        vb = self._primary_plot.getViewBox()
        y_min, y_max = vb.viewRange()[1]
        visible_span = max(0.5, y_max - y_min)
        scroll_max = max(0, int((n_rows - visible_span) * 10 + 0.5))

        # Disconnect stale connections before rewiring
        try:
            self._y_scrollbar.valueChanged.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            vb.sigYRangeChanged.disconnect(self._on_digital_y_range_changed)
        except (RuntimeError, TypeError):
            pass

        if scroll_max <= 0:
            self._y_scrollbar.setVisible(False)
            return

        self._y_scrollbar.blockSignals(True)
        self._y_scrollbar.setRange(0, scroll_max)
        self._y_scrollbar.setSingleStep(5)
        self._y_scrollbar.setPageStep(max(5, int(visible_span * 10)))
        self._y_scrollbar.setValue(0)
        self._y_scrollbar.blockSignals(False)
        self._y_scrollbar.setVisible(True)

        self._y_scrollbar.valueChanged.connect(self._on_digital_y_scroll)
        vb.sigYRangeChanged.connect(self._on_digital_y_range_changed)

    def _on_digital_y_scroll(self, value: int) -> None:
        """Scrollbar moved → shift the Y viewport."""
        if self._digital_n_rows <= 0:
            return
        vb = self._primary_plot.getViewBox()
        y_min, y_max = vb.viewRange()[1]
        span = max(0.5, y_max - y_min)
        new_y_min = -0.5 + value / 10.0
        vb.blockSignals(True)
        vb.setRange(yRange=(new_y_min, new_y_min + span), padding=0)
        vb.blockSignals(False)

    def _on_digital_y_range_changed(self, vb: pg.ViewBox, y_range: tuple) -> None:
        """Y viewport changed (panel resize/pan) → sync the scrollbar."""
        if self._digital_n_rows <= 0:
            return
        y_min, y_max = y_range
        span = max(0.5, y_max - y_min)
        scroll_max = max(0, int((self._digital_n_rows - span) * 10 + 0.5))
        scroll_val = max(0, int((y_min + 0.5) * 10 + 0.5))
        self._y_scrollbar.blockSignals(True)
        self._y_scrollbar.setRange(0, scroll_max)
        self._y_scrollbar.setPageStep(max(5, int(span * 10)))
        self._y_scrollbar.setValue(min(scroll_val, scroll_max))
        self._y_scrollbar.setVisible(scroll_max > 0)
        self._y_scrollbar.blockSignals(False)

    def _emit_measurement(self) -> None:
        if self._cursor_a is None or self._cursor_b is None:
            return
        self.measurement_cursors_moved.emit(
            float(self._cursor_a.value()),
            float(self._cursor_b.value()),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Panel metadata & housekeeping
    # ─────────────────────────────────────────────────────────────────────────

    def export_view(self) -> None:
        """Grab this panel and save as PNG or PDF via a file dialog."""
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export panel view",
            f"{self._panel_title}.png",
            "PNG image (*.png);;PDF document (*.pdf)",
        )
        if not path:
            return
        pixmap = self.grab()
        if path.lower().endswith(".pdf"):
            try:
                from PyQt6.QtGui import QPainter, QPageLayout, QPageSize
                from PyQt6.QtPrintSupport import QPrinter
                printer = QPrinter(QPrinter.PrinterMode.HighResolution)
                printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
                printer.setOutputFileName(path)
                printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
                printer.setPageOrientation(QPageLayout.Orientation.Landscape)
                painter = QPainter(printer)
                rect = painter.viewport()
                scaled = pixmap.size().scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
                painter.setViewport(rect.x(), rect.y(), scaled.width(), scaled.height())
                painter.setWindow(pixmap.rect())
                painter.drawPixmap(0, 0, pixmap)
                painter.end()
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "Export failed", str(exc))
                return
        else:
            if not pixmap.save(path, "PNG"):
                QMessageBox.warning(self, "Export failed", f"Could not write {path}")
                return

    def set_time_reference(self, reference_time: datetime) -> None:
        """Switch the X-axis to absolute wall-clock labels anchored at reference_time."""
        bottom = self._primary_plot.getAxis("bottom")
        if isinstance(bottom, DatetimeAxisItem):
            self._datetime_axis = bottom
            bottom.set_start_time(reference_time)

    def clear_time_reference(self) -> None:
        """Revert the X-axis to elapsed-seconds labels."""
        bottom = self._primary_plot.getAxis("bottom")
        if isinstance(bottom, DatetimeAxisItem):
            self._datetime_axis = bottom
            bottom.set_start_time(None)

    def set_panel_title(self, title: str) -> None:
        self._panel_title = title
        self._header.set_title(title)

    def set_legend_visible(self, visible: bool) -> None:
        self.legend.setVisible(visible)

    def clear_all(self) -> None:
        # Measurement cursors (S6)
        self._remove_cursor_b()
        self._remove_cursor_a()
        self._measurement_mode = False
        # Digital curves
        vb = self._primary_plot.getViewBox()
        for curves_dict in (self._digital_hi_curves, self._digital_lo_curves):
            for curve in curves_dict.values():
                try:
                    vb.removeItem(curve)
                except Exception:  # noqa: BLE001
                    pass
        self._digital_hi_curves.clear()
        self._digital_lo_curves.clear()
        self._digital_raw.clear()
        # RMS overlays first (they hold ViewBox refs via getViewBox)
        for rms_curve in self._rms_curves.values():
            try:
                vb = rms_curve.getViewBox()
                if vb is not None:
                    vb.removeItem(rms_curve)
            except Exception:  # noqa: BLE001
                pass
        self._rms_curves.clear()
        self._rms_raw_hidden.clear()
        # Phasor overlays (S3)
        for curve in self._phasor_curves.values():
            try:
                vb = curve.getViewBox()
                if vb is not None:
                    vb.removeItem(curve)
            except Exception:  # noqa: BLE001
                pass
        self._phasor_curves.clear()
        # Harmonic overlays (S4)
        for order_curves in self._harmonic_curves.values():
            for curve in order_curves.values():
                try:
                    vb = curve.getViewBox()
                    if vb is not None:
                        vb.removeItem(curve)
                except Exception:  # noqa: BLE001
                    pass
        self._harmonic_curves.clear()
        for key, curve in list(self._curves.items()):
            meta = self._metadata.get(key)
            vb = self._resolve_viewbox(
                meta.y_axis_side if meta else "left",
                meta.unit if meta else None,
                meta.colour if meta else "#AAAAAA",
            )
            try:
                vb.removeItem(curve)
            except Exception:  # noqa: BLE001
                pass
        for line in self._zero_lines.values():
            try:
                self._primary_plot.removeItem(line)
            except Exception:  # noqa: BLE001
                pass
        for line in self._trigger_markers.values():
            try:
                self._primary_plot.removeItem(line)
            except Exception:  # noqa: BLE001
                pass
        self._curves.clear()
        self._zero_lines.clear()
        self._trigger_markers.clear()
        self._metadata.clear()
        self._right_mgr.clear()
        self.legend.clear_rows()
        self.clear_annotations()

    @property
    def curve_count(self) -> int:
        return len(self._curves)

    @property
    def zero_line_count(self) -> int:
        return len(self._zero_lines)

    @property
    def trigger_marker_count(self) -> int:
        return len(self._trigger_markers)
