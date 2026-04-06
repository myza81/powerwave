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
All X axes are linked so pan/zoom stays synchronised.

Analogue channels:
  WAVEFORM mode (sample_rate >= 200 Hz) — continuous PlotDataItem line
  TREND mode    (sample_rate <  200 Hz) — ScatterPlotItem, size=4

Digital channels:
  Two overlaid PlotDataItems:
    low_line  — dim baseline at y=0.10 (always visible)
    step_line — bold grey bar at y=0.70 when high, y=0.10 when low
  Y range locked to [DIG_Y_MIN, DIG_Y_MAX]; stepMode='right' for correct edges.

Decimation:
  WAVEFORM: min/max envelope (_decimate) preserves AC waveform peaks.
  TREND:    uniform stride   (_decimate_trend) preserves smooth trend shape.

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

BACKGROUND_COLOUR: str     = '#1E1E1E'
GRID_ALPHA: float          = 0.15
WAVEFORM_PEN_WIDTH: int    = 1
TREND_SCATTER_SIZE: int    = 4

TIME_AXIS_HEIGHT: int      = 30     # px — bottom shared time axis

# Digital rendering style (BEN32)
DIG_BASELINE_Y: float      = 0.1    # y-level for dim LOW baseline
DIG_Y_MIN: float           = -0.1   # Y axis lower bound
DIG_Y_MAX: float           = 1.2    # Y axis upper bound
DIG_FILL_PEN: str          = '#BBBBBB'
DIG_FILL_BRUSH: str        = '#888888'
DIG_FILL_PEN_WIDTH: int    = 2
DIG_BASELINE_PEN: str      = '#555555'
DIG_BASELINE_PEN_WIDTH: int = 1

# Decimation caps
ANALOGUE_MAX_POINTS: int   = 2000
DIGITAL_MAX_POINTS: int    = 500

TRIGGER_COLOUR: str        = '#FF4444'
TRIGGER_LINE_WIDTH: int    = 2
TRIGGER_LABEL_POS: float   = 0.95

AXIS_LABEL_COLOUR: str     = '#AAAAAA'
TIME_AXIS_LABEL_MS: str    = 'Time (ms)'
TIME_AXIS_LABEL_S: str     = 'Time (s)'
TIME_AXIS_LABEL_MIN: str   = 'Time (min)'
TREND_MINUTES_THRESHOLD: float = 60.0  # s — TREND records longer than this use minutes


class ChannelCanvas(pg.GraphicsLayoutWidget):
    """Stacked multi-channel waveform canvas (BEN32 style).

    No Y-axis labels on individual plots — channel names live in the
    adjacent LabelPanel.  A single shared time axis sits at the bottom.
    All X axes are linked; time is in ms (WAVEFORM) or min/s (TREND).

    Row order matches LabelPanel exactly via get_ordered_rows().

    Usage::

        canvas = ChannelCanvas()
        canvas.load_record(record)
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
        self._first_plot: Optional[pg.PlotItem]                        = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_record(self, record: DisturbanceRecord) -> None:
        """Rebuild the canvas for ``record``.

        Clears all existing plots, then builds rows in the canonical order
        from get_ordered_rows() (analogue first, digital last per bay).

        Time axis unit:
          WAVEFORM → milliseconds
          TREND, duration ≤ 60 s → seconds
          TREND, duration > 60 s → minutes

        Args:
            record: The DisturbanceRecord to display.
        """
        self._clear_canvas()

        if not record.time_array.size:
            return

        trigger_offset_s = (
            record.trigger_time - record.start_time
        ).total_seconds()

        # Choose time unit
        is_trend = record.display_mode == 'TREND'
        if is_trend:
            t_raw: np.ndarray = record.time_array - trigger_offset_s
            duration_s = float(t_raw[-1] - t_raw[0]) if len(t_raw) > 1 else 0.0
            if duration_s > TREND_MINUTES_THRESHOLD:
                t_display = t_raw / 60.0
                time_axis_label = TIME_AXIS_LABEL_MIN
            else:
                t_display = t_raw
                time_axis_label = TIME_AXIS_LABEL_S
        else:
            t_display = (record.time_array - trigger_offset_s) * 1000.0
            time_axis_label = TIME_AXIS_LABEL_MS

        # Compute total canvas height from row list
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

        # Build rows
        grid_row = 0
        for row_spec in rows:
            if row_spec['type'] == 'bay_header':
                grid_row = self._add_bay_spacer(grid_row)
            elif row_spec['type'] == 'analogue':
                grid_row = self._add_analogue_plot(
                    row_spec['channel'], record, t_display, grid_row, is_trend
                )
            elif row_spec['type'] == 'digital':
                grid_row = self._add_digital_plot(
                    row_spec['channel'], t_display, grid_row
                )

        self._add_time_axis(t_display, time_axis_label)
        print(
            f'[ChannelCanvas] total_height={total_height}'
            f'  grid_row_count={self.ci.layout.rowCount()}'
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
        self._first_plot = None

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
        self._link_x(spacer)
        return grid_row + 1

    def _add_analogue_plot(
        self,
        ch: AnalogueChannel,
        record: DisturbanceRecord,
        t_display: np.ndarray,
        grid_row: int,
        is_trend: bool,
    ) -> int:
        """Add one ROW_HEIGHT_ANALOGUE px PlotItem for an analogue channel.

        Args:
            ch:         The analogue channel to render.
            record:     Parent record (display_mode).
            t_display:  Time array in display units (ms / s / min).
            grid_row:   Current GridLayout row index.
            is_trend:   True when display_mode == 'TREND'.

        Returns:
            Next available row index.
        """
        plot = self.addPlot(row=grid_row, col=0)
        self.ci.layout.setRowMinimumHeight(grid_row, ROW_HEIGHT_ANALOGUE)
        self.ci.layout.setRowMaximumHeight(grid_row, ROW_HEIGHT_ANALOGUE)
        self._configure_analogue_plot(plot)
        self._link_x(plot)

        curve = self._make_analogue_curve(t_display, ch, is_trend)
        plot.addItem(curve)
        self._add_trigger_line(plot, trigger_pos=0.0)

        self._plots[ch.channel_id] = plot
        self._curves[ch.channel_id] = curve
        return grid_row + 1

    def _add_digital_plot(
        self,
        ch: DigitalChannel,
        t_display: np.ndarray,
        grid_row: int,
    ) -> int:
        """Add one ROW_HEIGHT_DIGITAL px PlotItem for a digital channel.

        Args:
            ch:        The digital channel to render.
            t_display: Time array in display units.
            grid_row:  Current GridLayout row index.

        Returns:
            Next available row index.
        """
        plot = self.addPlot(row=grid_row, col=0)
        self.ci.layout.setRowMaximumHeight(grid_row, ROW_HEIGHT_DIGITAL)
        self.ci.layout.setRowMinimumHeight(grid_row, ROW_HEIGHT_DIGITAL)
        self._configure_digital_plot(plot)
        self._link_x(plot)

        baseline, filled = self._make_digital_curves(t_display, ch)
        plot.addItem(baseline)
        plot.addItem(filled)
        self._add_trigger_line(plot, trigger_pos=0.0, show_label=False)

        self._plots[ch.channel_id] = plot
        self._curves[ch.channel_id] = filled
        return grid_row + 1

    # ── Private — decimation ───────────────────────────────────────────────────

    @staticmethod
    def _decimate_trend(
        time_array: np.ndarray,
        data_array: np.ndarray,
        max_points: int = 2000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce ``data_array`` to at most ``max_points`` via uniform stride.

        Used for TREND (PMU / slow-rate) data where min/max envelope
        decimation creates false extremes on slowly-varying signals.

        Args:
            time_array: 1-D float64 time values.
            data_array: 1-D float64 data values (same length).
            max_points: Maximum number of output points.

        Returns:
            Tuple (t_out, d_out) with length ≤ max_points.
        """
        n = len(time_array)
        if n <= max_points:
            return time_array, data_array
        step = max(1, n // max_points)
        indices = np.arange(0, n, step)
        return time_array[indices], data_array[indices]

    @staticmethod
    def _decimate(
        time_array: np.ndarray,
        data_array: np.ndarray,
        max_points: int = 2000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce ``data_array`` to at most ``max_points`` via min/max envelope.

        Each bucket contributes two output points (min and max within that
        bucket, in chronological order) so AC waveform peaks are preserved
        at all zoom levels.

        Args:
            time_array: 1-D float64 time values.
            data_array: 1-D float64 data values (same length).
            max_points: Maximum number of output points.

        Returns:
            Tuple (t_out, d_out) with length ≤ max_points.
        """
        n = len(time_array)
        if n <= max_points:
            return time_array, data_array

        bucket_size = n // (max_points // 2)
        n_buckets   = n // bucket_size

        t_out = np.empty(n_buckets * 2)
        d_out = np.empty(n_buckets * 2)

        for i in range(n_buckets):
            sl    = slice(i * bucket_size, (i + 1) * bucket_size)
            bt    = time_array[sl]
            bd    = data_array[sl]
            min_i = int(np.argmin(bd))
            max_i = int(np.argmax(bd))
            if min_i < max_i:
                t_out[i * 2],     d_out[i * 2]     = bt[min_i], bd[min_i]
                t_out[i * 2 + 1], d_out[i * 2 + 1] = bt[max_i], bd[max_i]
            else:
                t_out[i * 2],     d_out[i * 2]     = bt[max_i], bd[max_i]
                t_out[i * 2 + 1], d_out[i * 2 + 1] = bt[min_i], bd[min_i]

        return t_out, d_out

    # ── Private — curve factories ──────────────────────────────────────────────

    def _make_analogue_curve(
        self,
        t_display: np.ndarray,
        ch: AnalogueChannel,
        is_trend: bool,
    ) -> pg.PlotDataItem | pg.ScatterPlotItem:
        """Create the waveform item for one analogue channel.

        Args:
            t_display: Time array in display units.
            ch:        Analogue channel.
            is_trend:  True for TREND scatter mode.

        Returns:
            PlotDataItem (WAVEFORM) or ScatterPlotItem (TREND).
        """
        n = min(len(t_display), len(ch.raw_data))
        t = t_display[:n]
        y = ch.raw_data[:n].astype(np.float64)

        if is_trend:
            t, y = self._decimate_trend(t, y, ANALOGUE_MAX_POINTS)
            return pg.ScatterPlotItem(
                x=t,
                y=y,
                pen=None,
                brush=pg.mkBrush(QColor(ch.colour)),
                size=TREND_SCATTER_SIZE,
            )

        t, y = self._decimate(t, y, ANALOGUE_MAX_POINTS)
        return pg.PlotDataItem(
            t,
            y,
            pen=pg.mkPen(ch.colour, width=WAVEFORM_PEN_WIDTH),
            antialias=False,
        )

    def _make_digital_curves(
        self,
        t_display: np.ndarray,
        ch: DigitalChannel,
    ) -> tuple[pg.PlotDataItem, pg.PlotDataItem]:
        """Create baseline + filled step items for one digital channel.

        ``baseline``: dim flat line at DIG_BASELINE_Y.
        ``filled``:   step curve; fills solid bar when data = 1.

        Args:
            t_display: Time array in display units.
            ch:        The digital channel to render.

        Returns:
            Tuple (baseline, filled).
        """
        n = min(len(t_display), len(ch.data))
        t = t_display[:n]
        y = ch.data[:n].astype(np.float64)
        t_dec, y_dec = self._decimate(t, y, DIGITAL_MAX_POINTS)

        baseline = pg.PlotDataItem(
            t_dec,
            np.full(len(t_dec), DIG_BASELINE_Y, dtype=np.float64),
            pen=pg.mkPen(DIG_BASELINE_PEN, width=DIG_BASELINE_PEN_WIDTH),
            antialias=False,
        )

        filled = pg.PlotDataItem(
            t_dec,
            y_dec,
            stepMode='right',
            pen=pg.mkPen(DIG_FILL_PEN, width=DIG_FILL_PEN_WIDTH),
            fillLevel=0.0,
            brush=pg.mkBrush(DIG_FILL_BRUSH),
            antialias=False,
        )

        return baseline, filled

    # ── Private — time axis ───────────────────────────────────────────────────

    def _add_time_axis(
        self, t_display: np.ndarray, label: str = TIME_AXIS_LABEL_MS
    ) -> None:
        """Add a single shared time axis PlotItem at the bottom.

        Linked to the first plot so pan/zoom stays synchronised.

        Args:
            t_display: Full time array in display units, trigger = 0.
            label:     Axis label string.
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

        # Invisible flat line so X range is initialised to the data extent
        axis_plot.plot(
            [t_display[0], t_display[-1]],
            [0, 0],
            pen=pg.mkPen(None),
        )

        self._link_x(axis_plot)

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

    def _link_x(self, plot: pg.PlotItem) -> None:
        """Link ``plot``'s X axis to the first created plot.

        Args:
            plot: The PlotItem to link (or register as first).
        """
        if self._first_plot is None:
            self._first_plot = plot
        else:
            plot.setXLink(self._first_plot)

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
