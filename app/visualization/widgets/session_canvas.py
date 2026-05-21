"""SessionCanvasWidget — per-panel multi-source waveform view for Phase 9D/9E.

Architecture
------------
Wraps a single pg.PlotWidget.  All channels assigned to one session panel
share this widget; different sources are distinguished by colour.

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

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal  # pyqtSignal used by SessionCanvasWidget
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from app.ui.session.legend_widget import ChannelLegendWidget


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
# Curve metadata
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _CurveMetadata:
    source_id: str
    channel_name: str
    display_name: str
    source_badge: str
    unit: str | None
    colour: str
    visible: bool


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
        self._mergeable_panels: list[tuple[str, str]] = []   # (panel_id, title)

        self._curves: dict[tuple[str, str], pg.PlotDataItem] = {}
        self._zero_lines: dict[str, pg.InfiniteLine] = {}
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

        self._plot_widget = pg.PlotWidget(background="#1E1E1E")
        self._primary_plot: pg.PlotItem = self._plot_widget.getPlotItem()
        self._primary_plot.showGrid(x=True, y=True, alpha=0.2)
        self._primary_plot.setLabel("bottom", "Time (s)")

        # Legend below the plot (Phase 9E)
        self.legend = ChannelLegendWidget(self.panel_id, parent=self)

        layout.addWidget(self._plot_widget, stretch=4)
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
    ) -> None:
        key = (source_id, channel_name)
        if key not in self._curves:
            curve = pg.PlotDataItem(
                pen=pg.mkPen(color, width=1),
                skipFiniteCheck=True,
            )
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            self._primary_plot.addItem(curve)
            self._curves[key] = curve
        else:
            curve = self._curves[key]

        if len(time) > 0 and len(values) > 0:
            curve.setData(x=time, y=values)
        else:
            curve.setData(x=np.array([]), y=np.array([]))
        curve.setVisible(visible)

        eff_display = display_name if display_name is not None else channel_name
        meta = _CurveMetadata(
            source_id=source_id,
            channel_name=channel_name,
            display_name=eff_display,
            source_badge=source_badge,
            unit=unit,
            colour=color,
            visible=visible,
        )
        self._metadata[key] = meta
        self.legend.upsert_row(
            source_id, channel_name, eff_display, source_badge, unit, color, visible
        )

    def update_curve_pen(self, source_id: str, channel_name: str, color: str) -> None:
        """Update pen colour only — no data copy, O(1)."""
        key = (source_id, channel_name)
        curve = self._curves.get(key)
        if curve is not None:
            curve.setPen(pg.mkPen(color, width=1))
        meta = self._metadata.get(key)
        if meta is not None:
            meta.colour = color
        self.legend.update_row_colour(source_id, channel_name, color)

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
            self._primary_plot.removeItem(self._curves.pop(key))
            self._metadata.pop(key, None)
        self.legend.remove_source_rows(source_id)
        self.remove_zero_line(source_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Zero-line markers
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
    # Panel metadata & housekeeping
    # ─────────────────────────────────────────────────────────────────────────

    def set_panel_title(self, title: str) -> None:
        self._panel_title = title
        self._header.set_title(title)

    def set_legend_visible(self, visible: bool) -> None:
        self.legend.setVisible(visible)

    def clear_all(self) -> None:
        for curve in self._curves.values():
            self._primary_plot.removeItem(curve)
        for line in self._zero_lines.values():
            self._primary_plot.removeItem(line)
        self._curves.clear()
        self._zero_lines.clear()
        self._metadata.clear()
        self.legend.clear_rows()

    @property
    def curve_count(self) -> int:
        return len(self._curves)

    @property
    def zero_line_count(self) -> int:
        return len(self._zero_lines)
