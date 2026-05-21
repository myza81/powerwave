"""app/visualization/axis/datetime_axis.py

DatetimeAxisItem — PyQtGraph AxisItem subclass that renders elapsed-seconds
X values as absolute timestamps.

The internal axis representation is always float elapsed seconds (compatible
with all viewport code in FlexiblePlotCanvas and DigitalEventTimeline).
Labels are computed as start_time + timedelta(seconds=value) and formatted
automatically based on the visible tick spacing.

When no start_time is set, falls back to plain elapsed-seconds labels.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum

import pyqtgraph as pg

# Explicit axis display mode constants — use these rather than inferring from start_time.
AXIS_MODE_RELATIVE = "relative_seconds"   # COMTRADE / high-rate: elapsed s
AXIS_MODE_ABSOLUTE = "absolute_datetime"  # CSV / Excel trend data: wall-clock labels


class TimeDisplayMode(str, Enum):
    """Visualization-level time-axis display modes."""

    RELATIVE = AXIS_MODE_RELATIVE
    ABSOLUTE = AXIS_MODE_ABSOLUTE

    @classmethod
    def coerce(cls, value: "TimeDisplayMode | str") -> "TimeDisplayMode":
        if isinstance(value, cls):
            return value
        if value == AXIS_MODE_RELATIVE:
            return cls.RELATIVE
        if value == AXIS_MODE_ABSOLUTE:
            return cls.ABSOLUTE
        return cls(value)


def _choose_format(spacing: float) -> str:
    """Select a strftime format string based on tick spacing in seconds.

    For sub-second spacing, returns the sentinel string ``"ms"``; callers must
    append the millisecond field manually (strftime cannot zero-pad sub-seconds
    portably).
    """
    if spacing < 1.0:
        return "ms"            # special sentinel — caller appends .mmm
    if spacing < 3600.0:
        return "%H:%M:%S"      # seconds or minute ticks
    if spacing < 86400.0:
        return "%m-%d %H:%M"   # hour ticks — show date + HH:MM
    return "%Y-%m-%d"          # day ticks — date only


class DatetimeAxisItem(pg.AxisItem):
    """AxisItem that renders elapsed-seconds values as absolute timestamps.

    Usage::

        axis = DatetimeAxisItem(orientation="bottom")
        plot = some_graphics_widget.addPlot(axisItems={"bottom": axis})
        axis.set_start_time(record.timing_info.start_time)

    When *start_time* is None (or not yet set), tick labels fall back to the
    form ``"3.500 s"``, so the axis is always usable immediately.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("orientation", "bottom")
        super().__init__(*args, **kwargs)
        self._start_time: datetime | None = None

    def set_start_time(self, start_time: datetime | None) -> None:
        """Set the absolute reference time for label computation.

        Pass None to revert to elapsed-seconds labels.
        """
        self._start_time = start_time
        self.picture = None  # invalidate rendered-label cache
        try:
            self.update()
        except RuntimeError:
            # C++ QGraphicsItem was destroyed (PyQt6 ownership transfer after
            # PlotItem.clear()).  State is stored; tickStrings() will use it on
            # the next repaint once the axis is re-parented.
            pass

    def tickStrings(  # noqa: N802  (PyQtGraph API name)
        self,
        values,
        scale,
        spacing,
    ) -> list[str]:
        """Return formatted label strings for the given tick positions.

        Args:
            values:  Tick positions in elapsed-seconds data coordinates.
            scale:   Axis scale factor (passed by PyQtGraph; not used here).
            spacing: Distance between major ticks in data units (seconds).
        """
        if not values:
            return []

        if self._start_time is None:
            return [f"{v:.3f} s" for v in values]

        fmt = _choose_format(float(spacing))
        strs: list[str] = []
        for v in values:
            try:
                dt = self._start_time + timedelta(seconds=float(v))
                if fmt == "ms":
                    strs.append(
                        dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
                    )
                else:
                    strs.append(dt.strftime(fmt))
            except (OverflowError, OSError, ValueError):
                strs.append(f"{v:.3f}")
        return strs
