from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PyQt6.QtGui import QFont

PLOT_BACKGROUND = "#181B1F"
GRID_ALPHA = 0.14
AXIS_LINE = "#4D5560"
AXIS_TEXT = "#D6DCE3"
AXIS_LABEL = "#E7EBF0"
PANEL_TITLE = "#C9D2DC"
CROSSHAIR_LINE = "#B8C1CC"
CROSSHAIR_TEXT = "#F2F5F8"
CROSSHAIR_FILL = (12, 14, 16, 210)
CROSSHAIR_BORDER = "#5E6975"

_AXIS_LABEL_STYLE = {
    "color": AXIS_LABEL,
    "font-size": "9pt",
    "font-weight": "600",
}


@dataclass(frozen=True)
class PlotTheme:
    name: str
    background: str
    grid_alpha: float
    axis_line: str
    axis_text: str
    axis_label: str
    panel_title: str
    crosshair_line: str
    crosshair_text: str
    crosshair_fill: tuple[int, int, int, int]
    crosshair_border: str


_THEMES = {
    "dark": PlotTheme(
        name="dark",
        background="#181B1F",
        grid_alpha=0.14,
        axis_line="#4D5560",
        axis_text="#D6DCE3",
        axis_label="#E7EBF0",
        panel_title="#C9D2DC",
        crosshair_line="#B8C1CC",
        crosshair_text="#F2F5F8",
        crosshair_fill=(12, 14, 16, 210),
        crosshair_border="#5E6975",
    ),
    "light": PlotTheme(
        name="light",
        background="#FAFAFA",
        grid_alpha=0.28,
        axis_line="#5E6670",
        axis_text="#20252B",
        axis_label="#111827",
        panel_title="#0B1220",
        crosshair_line="#303640",
        crosshair_text="#111827",
        crosshair_fill=(255, 255, 255, 225),
        crosshair_border="#9AA3AD",
    ),
}


def get_plot_theme(theme: str | PlotTheme = "dark") -> PlotTheme:
    if isinstance(theme, PlotTheme):
        return theme
    return _THEMES.get(str(theme).lower(), _THEMES["dark"])


def axis_tick_font() -> QFont:
    font = QFont()
    font.setPointSize(9)
    return font


def info_box_font() -> QFont:
    font = QFont("Menlo")
    font.setPointSize(8)
    return font


def apply_axis_style(
    axis: pg.AxisItem,
    *,
    text_color: str | None = None,
    theme: str | PlotTheme = "dark",
) -> None:
    t = get_plot_theme(theme)
    axis.enableAutoSIPrefix(False)
    axis.setPen(pg.mkPen(t.axis_line, width=1))
    axis.setTextPen(pg.mkPen(text_color or t.axis_text))
    axis.setStyle(
        tickFont=axis_tick_font(),
        tickTextOffset=8,
        autoExpandTextSpace=True,
    )


def apply_plot_style(
    plot: pg.PlotItem,
    *,
    y_grid: bool = True,
    theme: str | PlotTheme = "dark",
) -> None:
    t = get_plot_theme(theme)
    if hasattr(plot, "hideButtons"):
        plot.hideButtons()
    if hasattr(plot, "setMenuEnabled"):
        plot.setMenuEnabled(False)
    auto_button = getattr(plot, "autoBtn", None)
    if auto_button is not None:
        auto_button.hide()
    plot.showGrid(x=True, y=y_grid, alpha=t.grid_alpha)
    for axis_name in ("bottom", "left", "right", "top"):
        try:
            apply_axis_style(plot.getAxis(axis_name), theme=t)
        except Exception:  # noqa: BLE001
            pass


def set_axis_label(
    plot: pg.PlotItem,
    axis_name: str,
    text: str,
    *,
    units: str | None = None,
    theme: str | PlotTheme = "dark",
) -> None:
    t = get_plot_theme(theme)
    style = {
        "color": t.axis_label,
        "font-size": _AXIS_LABEL_STYLE["font-size"],
        "font-weight": _AXIS_LABEL_STYLE["font-weight"],
    }
    if units:
        plot.setLabel(axis_name, text, units=units, **style)
    else:
        plot.setLabel(axis_name, text, **style)


def set_axis_item_label(
    axis: pg.AxisItem,
    text: str,
    *,
    units: str | None = None,
    theme: str | PlotTheme = "dark",
) -> None:
    t = get_plot_theme(theme)
    style = {
        "color": t.axis_label,
        "font-size": _AXIS_LABEL_STYLE["font-size"],
        "font-weight": _AXIS_LABEL_STYLE["font-weight"],
    }
    if units:
        axis.setLabel(text, units=units, **style)
    else:
        axis.setLabel(text, **style)


def set_panel_title(plot: pg.PlotItem, title: str, *, theme: str | PlotTheme = "dark") -> None:
    t = get_plot_theme(theme)
    plot.setTitle(title, color=t.panel_title, size="11pt")
    label = getattr(plot, "titleLabel", None)
    if label is not None:
        label.setAttr("bold", True)


def make_crosshair_label(theme: str | PlotTheme = "dark") -> pg.TextItem:
    t = get_plot_theme(theme)
    item = pg.TextItem(
        "",
        color=t.crosshair_text,
        fill=pg.mkBrush(*t.crosshair_fill),
        border=pg.mkPen(t.crosshair_border, width=1),
        anchor=(0, 0),
    )
    item.setFont(info_box_font())
    return item


def position_crosshair_label(label: pg.TextItem, plot: pg.PlotItem) -> None:
    x_range, y_range = plot.getViewBox().viewRange()
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    x_span = max(abs(x1 - x0), 1e-12)
    y_span = max(abs(y1 - y0), 1e-12)
    label.setAnchor((1, 0))
    label.setPos(x1 - x_span * 0.012, y1 - y_span * 0.045)
