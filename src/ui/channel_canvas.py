"""
src/ui/channel_canvas.py

ChannelCanvas — main waveform / trend rendering canvas.

Renders all visible channels from a DisturbanceRecord on a stacked
multi-plot layout powered by PyQtGraph + OpenGL.

Layout (BEN32-style, top → bottom):
  For each bay:
    [ROW_HEIGHT_BAY_HEADER px invisible spacer — matches LabelPanel bay header]
    [ROW_HEIGHT_ANALOGUE px analogue PlotItem per visible analogue channel]
    [ROW_HEIGHT_DIGITAL px digital PlotItem per visible digital channel]
  [TIME_AXIS_HEIGHT px time-axis PlotItem — single shared bottom axis]

Row ordering is determined by utils.channel_ordering.get_ordered_rows(),
the single source of truth shared with LabelPanel.  This guarantees
pixel-perfect vertical alignment at all scroll positions.

Y-axes are hidden on all plots — channel names live in LabelPanel.
All X axes are linked to ONE reference plot (star topology, O(n) signals).

Analogue channels:
  WAVEFORM mode (sample_rate >= 200 Hz) — continuous PlotDataItem line
  TREND mode    (sample_rate <  200 Hz) — ScatterPlotItem, size=4

Digital channels:
  Two overlaid PlotDataItems:
    low_line  — dim baseline at y=0.10 (always visible)
    step_line — bold grey bar at y=0.70 when high, y=0.10 when low
  Y range locked to [DIG_Y_MIN, DIG_Y_MAX]; stepMode='right' for correct edges.

Decimation:
  ALL decimation is pre-computed on the background thread by
  engine.decimator.prepare_display_data() before record_loaded is emitted.
  channel_canvas.py reads ch._display_t / ch._display_d and calls
  setData() only — zero computation on the UI thread (LAW 2, LAW 3).

Architecture: Presentation layer (ui/) — imports core/, models/, pyqtgraph.
              Never import from engine/ or parsers/ (LAW 1).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from models.channel import AnalogueChannel, DigitalChannel
from models.disturbance_record import DisturbanceRecord
from utils.channel_ordering import (
    ROW_HEIGHT_ANALOGUE,
    ROW_HEIGHT_BAY_HEADER,
    ROW_HEIGHT_DIGITAL,
    get_ordered_rows,
)

# ── Module-level PyQtGraph configuration ─────────────────────────────────────
pg.setConfigOptions(
    useOpenGL=True,
    antialias=False,
    foreground='w',
    background='#1E1E1E',
)

# ── Module constants ───────────────────────────────────────────────────────────

BACKGROUND_COLOUR: str      = '#1E1E1E'
GRID_ALPHA: float           = 0.15
WAVEFORM_PEN_WIDTH: int     = 1
TREND_SCATTER_SIZE: int     = 4

TIME_AXIS_HEIGHT: int       = 30     # px — bottom shared time axis
TIME_AXIS_LABEL_FALLBACK: str = 'Time'

# Digital rendering style (BEN32)
DIG_BASELINE_Y: float       = 0.1
DIG_Y_MIN: float            = -0.1
DIG_Y_MAX: float            = 1.2
DIG_FILL_PEN: str           = '#BBBBBB'
DIG_FILL_BRUSH: str         = '#888888'
DIG_FILL_PEN_WIDTH: int     = 2
DIG_BASELINE_PEN: str       = '#555555'
DIG_BASELINE_PEN_WIDTH: int = 1

TRIGGER_COLOUR: str         = '#FF4444'
TRIGGER_LINE_WIDTH: int     = 2
TRIGGER_LABEL_POS: float    = 0.95

AXIS_LABEL_COLOUR: str      = '#AAAAAA'


class ChannelCanvas(pg.GraphicsLayoutWidget):
    """Stacked multi-channel waveform canvas (BEN32 style).

    No Y-axis labels on individual plots — channel names live in the
    adjacent LabelPanel.  A single shared time axis sits at the bottom.
    All X axes are linked in star topology to the first analogue plot
    (or first plot of any kind if no analogue channels are present).

    All decimation is pre-computed on the background thread by
    ``engine.decimator.prepare_display_data()``.  The UI thread only
    calls ``pg.PlotDataItem.setData()`` — zero computation here.

    Usage::

        canvas = ChannelCanvas()
        canvas.load_record(record)          # record already has _display_t/_display_d
        canvas.update_channel_visibility(3, False)
    """

    def __init__(self, parent=None) -> None:
        """Initialise an empty canvas with dark engineering theme."""
        super().__init__(parent=parent)
        self.setBackground(BACKGROUND_COLOUR)

        # Pin the scene to the top-left of the QGraphicsView.
        # Default alignment is AlignCenter which shifts rows down when the
        # view is taller than the scene, breaking alignment with LabelPanel.
        self.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )

        # Zero internal spacing so row pixel heights are exact
        self.ci.layout.setSpacing(0)
        self.ci.layout.setContentsMargins(0, 0, 0, 0)

        self._plots: dict[int, pg.PlotItem]                            = {}
        self._curves: dict[int, pg.PlotDataItem | pg.ScatterPlotItem] = {}
        self._trigger_lines: list[pg.InfiniteLine]                     = []
        self._ref_plot: Optional[pg.PlotItem]                          = None   # X-link reference

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_record(self, record: DisturbanceRecord) -> None:
        """Rebuild the canvas for ``record``.

        All decimated display arrays must already be attached to channels by
        ``engine.decimator.prepare_display_data()`` on the background thread.
        This method only constructs PlotItems and calls setData() — no
        computation occurs here (LAW 2).

        Args:
            record: The DisturbanceRecord to display (with _display_t/_display_d set).
        """
        self._clear_canvas()

        if not record.time_array.size:
            return

        # Time display array and axis label pre-computed by decimator
        t_display: np.ndarray = getattr(
            record, '_t_display', record.time_array
        )
        time_axis_label: str = getattr(
            record, '_time_axis_label', TIME_AXIS_LABEL_FALLBACK
        )
        is_trend = record.display_mode == 'TREND'

        # Compute canvas height from row list
        rows = get_ordered_rows(record)
        total_height = TIME_AXIS_HEIGHT
        for row_spec in rows:
            if row_spec['type'] == 'bay_header':
                total_height += ROW_HEIGHT_BAY_HEADER
            elif row_spec['type'] == 'analogue':
                total_height += ROW_HEIGHT_ANALOGUE
            elif row_spec['type'] == 'digital':
                total_height += ROW_HEIGHT_DIGITAL
        self.setFixedHeight(total_height)

        # Build rows — UI work only (setData, addPlot, configure)
        grid_row = 0
        for row_spec in rows:
            if row_spec['type'] == 'bay_header':
                grid_row = self._add_bay_spacer(grid_row)
            elif row_spec['type'] == 'analogue':
                grid_row = self._add_analogue_plot(
                    row_spec['channel'], is_trend, grid_row
                )
            elif row_spec['type'] == 'digital':
                grid_row = self._add_digital_plot(
                    row_spec['channel'], grid_row
                )

        self._add_time_axis(t_display, time_axis_label)
        print(
            f'[ChannelCanvas] total_height={total_height}'
            f'  grid_rows={self.ci.layout.rowCount()}'
            f'  channel_plots={len(self._plots)}'
        )

    def update_channel_visibility(self, channel_id: int, visible: bool) -> None:
        """Show or hide the PlotItem for ``channel_id``.

        Args:
            channel_id: The channel to update.
            visible:    True = show, False = hide.
        """
        plot = self._plots.get(channel_id)
        if plot is not None:
            plot.setVisible(visible)

    # ── Private — canvas lifecycle ─────────────────────────────────────────────

    def _clear_canvas(self) -> None:
        """Remove all plots and reset tracking state."""
        self.clear()
        self._plots.clear()
        self._curves.clear()
        self._trigger_lines.clear()
        self._ref_plot = None

    # ── Private — row builders ─────────────────────────────────────────────────

    def _add_bay_spacer(self, grid_row: int) -> int:
        """Add an invisible ROW_HEIGHT_BAY_HEADER px spacer for bay alignment.

        Args:
            grid_row: Current GridLayout row index.

        Returns:
            Next available row index.
        """
        spacer = self.addPlot(row=grid_row, col=0)
        self.ci.layout.setRowMinimumHeight(grid_row, ROW_HEIGHT_BAY_HEADER)
        self.ci.layout.setRowMaximumHeight(grid_row, ROW_HEIGHT_BAY_HEADER)
        self._configure_spacer_plot(spacer)
        # Spacer plots are NOT used as X-link reference — they carry no data
        if self._ref_plot is not None:
            spacer.setXLink(self._ref_plot)
        return grid_row + 1

    def _add_analogue_plot(
        self,
        ch: AnalogueChannel,
        is_trend: bool,
        grid_row: int,
    ) -> int:
        """Add one ROW_HEIGHT_ANALOGUE px PlotItem for an analogue channel.

        Reads pre-decimated arrays from ``ch._display_t`` / ``ch._display_d``.

        Args:
            ch:        The analogue channel to render.
            is_trend:  True when display_mode == 'TREND'.
            grid_row:  Current GridLayout row index.

        Returns:
            Next available row index.
        """
        plot = self.addPlot(row=grid_row, col=0)
        self.ci.layout.setRowMinimumHeight(grid_row, ROW_HEIGHT_ANALOGUE)
        self.ci.layout.setRowMaximumHeight(grid_row, ROW_HEIGHT_ANALOGUE)
        self._configure_analogue_plot(plot)

        # First analogue plot becomes the X-link reference (star topology)
        if self._ref_plot is None:
            self._ref_plot = plot
        else:
            plot.setXLink(self._ref_plot)

        t = getattr(ch, '_display_t', np.array([]))
        d = getattr(ch, '_display_d', np.array([]))

        curve = self._make_analogue_curve(t, d, ch, is_trend)
        plot.addItem(curve)
        self._add_trigger_line(plot, trigger_pos=0.0)

        self._plots[ch.channel_id]  = plot
        self._curves[ch.channel_id] = curve
        return grid_row + 1

    def _add_digital_plot(
        self,
        ch: DigitalChannel,
        grid_row: int,
    ) -> int:
        """Add one ROW_HEIGHT_DIGITAL px PlotItem for a digital channel.

        Reads pre-decimated arrays from ``ch._display_t`` / ``ch._display_d``.

        Args:
            ch:       The digital channel to render.
            grid_row: Current GridLayout row index.

        Returns:
            Next available row index.
        """
        plot = self.addPlot(row=grid_row, col=0)
        self.ci.layout.setRowMaximumHeight(grid_row, ROW_HEIGHT_DIGITAL)
        self.ci.layout.setRowMinimumHeight(grid_row, ROW_HEIGHT_DIGITAL)
        self._configure_digital_plot(plot)

        if self._ref_plot is None:
            self._ref_plot = plot
        else:
            plot.setXLink(self._ref_plot)

        t = getattr(ch, '_display_t', np.array([]))
        d = getattr(ch, '_display_d', np.array([]))

        baseline, filled = self._make_digital_curves(t, d)
        plot.addItem(baseline)
        plot.addItem(filled)
        self._add_trigger_line(plot, trigger_pos=0.0, show_label=False)

        self._plots[ch.channel_id]  = plot
        self._curves[ch.channel_id] = filled
        return grid_row + 1

    # ── Private — curve factories ──────────────────────────────────────────────

    def _make_analogue_curve(
        self,
        t: np.ndarray,
        d: np.ndarray,
        ch: AnalogueChannel,
        is_trend: bool,
    ) -> pg.PlotDataItem | pg.ScatterPlotItem:
        """Create the waveform item for one analogue channel.

        Args:
            t:         Pre-decimated time array (display units).
            d:         Pre-decimated data array.
            ch:        Analogue channel (for colour).
            is_trend:  True for TREND scatter mode.

        Returns:
            PlotDataItem (WAVEFORM) or ScatterPlotItem (TREND).
        """
        if is_trend:
            return pg.ScatterPlotItem(
                x=t,
                y=d,
                pen=None,
                brush=pg.mkBrush(QColor(ch.colour)),
                size=TREND_SCATTER_SIZE,
            )

        return pg.PlotDataItem(
            t,
            d,
            pen=pg.mkPen(ch.colour, width=WAVEFORM_PEN_WIDTH),
            antialias=False,
        )

    def _make_digital_curves(
        self,
        t: np.ndarray,
        d: np.ndarray,
    ) -> tuple[pg.PlotDataItem, pg.PlotDataItem]:
        """Create baseline + filled step items for one digital channel.

        Args:
            t: Pre-decimated time array (display units).
            d: Pre-decimated data array (0.0 / 1.0).

        Returns:
            Tuple (baseline, filled).
        """
        baseline = pg.PlotDataItem(
            t,
            np.full(len(t), DIG_BASELINE_Y, dtype=np.float64),
            pen=pg.mkPen(DIG_BASELINE_PEN, width=DIG_BASELINE_PEN_WIDTH),
            antialias=False,
        )

        filled = pg.PlotDataItem(
            t,
            d,
            stepMode='right',
            pen=pg.mkPen(DIG_FILL_PEN, width=DIG_FILL_PEN_WIDTH),
            fillLevel=0.0,
            brush=pg.mkBrush(DIG_FILL_BRUSH),
            antialias=False,
        )

        return baseline, filled

    # ── Private — time axis ───────────────────────────────────────────────────

    def _add_time_axis(
        self, t_display: np.ndarray, label: str = TIME_AXIS_LABEL_FALLBACK
    ) -> None:
        """Add a single shared time axis PlotItem at the bottom.

        Linked to the reference plot (star topology).

        Args:
            t_display: Full time array in display units.
            label:     Axis label string (e.g. 'Time (ms)').
        """
        row = self.ci.layout.rowCount()

        axis_plot = self.addPlot(row=row, col=0)
        self.ci.layout.setRowMaximumHeight(row, TIME_AXIS_HEIGHT)
        self.ci.layout.setRowMinimumHeight(row, TIME_AXIS_HEIGHT)

        axis_plot.hideAxis('left')
        axis_plot.getAxis('bottom').setStyle(showValues=True)
        axis_plot.getAxis('bottom').setLabel(label, color=AXIS_LABEL_COLOUR)
        axis_plot.getViewBox().setBackgroundColor(BACKGROUND_COLOUR)
        axis_plot.getViewBox().setBorder(None)

        # Invisible flat line so X range is initialised to data extent
        if len(t_display) >= 2:
            axis_plot.plot(
                [t_display[0], t_display[-1]],
                [0, 0],
                pen=pg.mkPen(None),
            )

        if self._ref_plot is not None:
            axis_plot.setXLink(self._ref_plot)

    # ── Private — plot helpers ─────────────────────────────────────────────────

    def _configure_spacer_plot(self, plot: pg.PlotItem) -> None:
        """Configure an invisible bay-header spacer plot.

        Args:
            plot: The PlotItem to configure as a spacer.
        """
        plot.hideAxis('left')
        plot.hideAxis('bottom')
        plot.getViewBox().setBackgroundColor(BACKGROUND_COLOUR)
        plot.getViewBox().setBorder(None)
        plot.setMouseEnabled(x=True, y=False)

    def _configure_analogue_plot(self, plot: pg.PlotItem) -> None:
        """Apply settings for an analogue waveform plot.

        Args:
            plot: The PlotItem to configure.
        """
        plot.hideAxis('left')
        plot.hideAxis('bottom')
        plot.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        plot.getViewBox().setBorder(None)
        plot.getViewBox().setBackgroundColor(BACKGROUND_COLOUR)

    def _configure_digital_plot(self, plot: pg.PlotItem) -> None:
        """Apply settings for a compact digital-channel plot.

        Args:
            plot: The PlotItem to configure.
        """
        plot.hideAxis('left')
        plot.hideAxis('bottom')
        plot.setMouseEnabled(x=True, y=False)
        plot.setYRange(DIG_Y_MIN, DIG_Y_MAX, padding=0)
        plot.getViewBox().setBackgroundColor(BACKGROUND_COLOUR)
        plot.getViewBox().setBorder(None)

    def _add_trigger_line(
        self,
        plot: pg.PlotItem,
        trigger_pos: float,
        show_label: bool = True,
    ) -> None:
        """Add a non-movable trigger InfiniteLine to ``plot``.

        Args:
            plot:         The PlotItem to add the line to.
            trigger_pos:  Trigger position in display units (0.0 after centring).
            show_label:   If False, omit the 'T=0' label (narrow digital rows).
        """
        line = pg.InfiniteLine(
            pos=trigger_pos,
            angle=90,
            movable=False,
            pen=pg.mkPen(
                TRIGGER_COLOUR,
                width=TRIGGER_LINE_WIDTH,
                style=Qt.PenStyle.DotLine,
            ),
            label='T=0' if show_label else '',
            labelOpts={
                'color': TRIGGER_COLOUR,
                'position': TRIGGER_LABEL_POS,
            } if show_label else {},
        )
        plot.addItem(line)
        self._trigger_lines.append(line)
