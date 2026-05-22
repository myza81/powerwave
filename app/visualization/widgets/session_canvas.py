"""SessionCanvasWidget — per-panel multi-source waveform view (Phase 9D/9E + S1).

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
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from app.ui.session.legend_widget import ChannelLegendWidget
from app.visualization.axis.datetime_axis import DatetimeAxisItem


# ---------------------------------------------------------------------------
# Panel header (thin title strip with right-click context menu)
# ---------------------------------------------------------------------------


class _PanelHeader(QWidget):
    """Thin title bar above the plot — right-click for merge/split operations."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(20)
        self.setStyleSheet("background: #252525;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        self._lbl = QLabel(title)
        self._lbl.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #ccc; "
            "background: transparent;"
        )
        lay.addWidget(self._lbl)
        lay.addStretch()
        hint = QLabel("▾")
        hint.setStyleSheet("color: #666; font-size: 11px; background: transparent;")
        lay.addWidget(hint)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def set_title(self, title: str) -> None:
        self._lbl.setText(title)


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

        ax = pg.AxisItem("right")
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
            if entry.viewbox.scene() is not None:
                entry.viewbox.scene().removeItem(entry.viewbox)
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
        self._zero_lines: dict[str, pg.InfiniteLine] = {}
        self._trigger_markers: dict[str, pg.InfiniteLine] = {}  # S1
        self._metadata: dict[tuple[str, str], _CurveMetadata] = {}

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Panel header strip (right-click → merge/split menu)
        self._header = _PanelHeader(self._panel_title, parent=self)
        self._header.customContextMenuRequested.connect(self._show_panel_menu)
        layout.addWidget(self._header)

        # Main plot area: GraphicsLayoutWidget hosts PlotItem at col=0;
        # _RightAxisManager adds secondary AxisItems at col=1, 2, …
        self._datetime_axis = DatetimeAxisItem(orientation="bottom")
        self._glw = pg.GraphicsLayoutWidget(background="#1E1E1E")
        self._primary_plot: pg.PlotItem = self._glw.addPlot(
            row=0, col=0, axisItems={"bottom": self._datetime_axis}
        )
        self._primary_plot.showGrid(x=True, y=True, alpha=0.2)
        self._primary_plot.setLabel("bottom", "Time")

        # N-axis manager for independent right-side Y-axes (S1)
        self._right_mgr = _RightAxisManager(self._primary_plot, self._glw)

        # Legend below the plot (Phase 9E)
        self.legend = ChannelLegendWidget(self.panel_id, parent=self)

        layout.addWidget(self._glw, stretch=4)
        layout.addWidget(self.legend, stretch=0)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel header context menu
    # ─────────────────────────────────────────────────────────────────────────

    def set_mergeable_panels(self, panels: list[tuple[str, str]]) -> None:
        """Update the panel list shown in the "Merge with →" submenu."""
        self._mergeable_panels = panels

    def _show_panel_menu(self, pos) -> None:
        menu = QMenu(self)

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
                    old_vb.removeItem(self._curves[key])
                    target_vb.addItem(self._curves[key])

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
            old_vb.removeItem(curve)
            new_vb.addItem(curve)
        meta.y_axis_side = side

    def set_curve_visible(
        self, source_id: str, channel_name: str, visible: bool
    ) -> None:
        key = (source_id, channel_name)
        curve = self._curves.get(key)
        if curve is not None:
            curve.setVisible(visible)
        meta = self._metadata.get(key)
        if meta is not None:
            meta.visible = visible
        self.legend.update_row_visible(source_id, channel_name, visible)

    def remove_source(self, source_id: str) -> None:
        stale = [k for k in self._curves if k[0] == source_id]
        for key in stale:
            meta = self._metadata.pop(key, None)
            curve = self._curves.pop(key)
            vb = self._resolve_viewbox(
                meta.y_axis_side if meta else "left",
                meta.unit if meta else None,
                meta.colour if meta else "#AAAAAA",
            )
            vb.removeItem(curve)
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
    # Panel metadata & housekeeping
    # ─────────────────────────────────────────────────────────────────────────

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
        for key, curve in list(self._curves.items()):
            meta = self._metadata.get(key)
            vb = self._resolve_viewbox(
                meta.y_axis_side if meta else "left",
                meta.unit if meta else None,
                meta.colour if meta else "#AAAAAA",
            )
            vb.removeItem(curve)
        for line in self._zero_lines.values():
            self._primary_plot.removeItem(line)
        for line in self._trigger_markers.values():
            self._primary_plot.removeItem(line)
        self._curves.clear()
        self._zero_lines.clear()
        self._trigger_markers.clear()
        self._metadata.clear()
        self._right_mgr.clear()
        self.legend.clear_rows()

    @property
    def curve_count(self) -> int:
        return len(self._curves)

    @property
    def zero_line_count(self) -> int:
        return len(self._zero_lines)

    @property
    def trigger_marker_count(self) -> int:
        return len(self._trigger_markers)
