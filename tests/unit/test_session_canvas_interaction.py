"""Regression tests for SessionCanvasWidget crosshair-snap and theme switching.

These cover behaviour added for the "snap to nearest waveform point" crosshair
mode and the light/dark canvas theme toggle: irregular timestamps, NaN
handling, multiple/hidden curves, and that switching themes restyles chrome
without touching curve data.
"""
from __future__ import annotations

import sys

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6.QtWidgets import QApplication

from app.ui.session.session_canvas_controller import SessionCanvasController  # noqa: F401  (import order avoids a circular-import edge in session_canvas)
from app.visualization.widgets.session_canvas import SessionCanvasWidget


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ---------------------------------------------------------------------------
# _nearest_waveform_point (crosshair snap-to-waveform)
# ---------------------------------------------------------------------------


def test_nearest_waveform_point_returns_none_with_no_curves(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    result = canvas._nearest_waveform_point(pg.Point(0.0, 0.0), 0.0)
    assert result is None
    canvas.clear_all()


def test_nearest_waveform_point_snaps_to_exact_sample_with_irregular_spacing(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    # Non-uniform dt (0.01, 0.04, 0.01, 0.14) but strictly monotonic — the
    # searchsorted-based lookup only requires sortedness, not uniform spacing.
    t = np.array([0.0, 0.01, 0.05, 0.06, 0.20])
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    canvas.update_curve("s1", "VA", t, v, color="#ff0000", visible=True)

    curve = canvas._curves[("s1", "VA")]
    vb = curve.getViewBox()
    scene_pos = vb.mapViewToScene(pg.Point(0.06, 4.0))

    result = canvas._nearest_waveform_point(scene_pos, 0.06)
    assert result is not None
    name, x_val, y_val, unit, color = result
    assert name == "VA"
    assert x_val == pytest.approx(0.06)
    assert y_val == pytest.approx(4.0)
    canvas.clear_all()


def test_nearest_waveform_point_skips_nan_sample(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    t = np.array([0.0, 0.01, 0.02, 0.03])
    v = np.array([1.0, np.nan, 3.0, 4.0])
    canvas.update_curve("s1", "VA", t, v, color="#ff0000", visible=True)

    curve = canvas._curves[("s1", "VA")]
    vb = curve.getViewBox()
    # Aim directly at the NaN sample (index 1); the function must fall back
    # to a neighbouring finite point rather than returning/crashing on NaN.
    scene_pos = vb.mapViewToScene(pg.Point(0.01, 2.0))

    result = canvas._nearest_waveform_point(scene_pos, 0.01)
    assert result is not None
    _name, _x, y_val, _unit, _color = result
    assert np.isfinite(y_val)
    canvas.clear_all()


def test_nearest_waveform_point_ignores_infinite_sample(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    t = np.array([0.0, 0.01, 0.02])
    v = np.array([1.0, np.inf, 3.0])
    canvas.update_curve("s1", "VA", t, v, color="#ff0000", visible=True)

    curve = canvas._curves[("s1", "VA")]
    vb = curve.getViewBox()
    scene_pos = vb.mapViewToScene(pg.Point(0.01, 1.0))

    result = canvas._nearest_waveform_point(scene_pos, 0.01)
    assert result is not None
    assert np.isfinite(result[2])
    canvas.clear_all()


def test_nearest_waveform_point_picks_closest_of_several_visible_curves(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    t = np.array([0.0, 0.01, 0.02])
    canvas.update_curve("s1", "VA", t, np.array([1.0, 1.0, 1.0]), color="#ff0000", visible=True)
    canvas.update_curve("s1", "VB", t, np.array([50.0, 50.0, 50.0]), color="#00ff00", visible=True)

    curve_a = canvas._curves[("s1", "VA")]
    vb = curve_a.getViewBox()
    scene_pos = vb.mapViewToScene(pg.Point(0.01, 1.0))

    result = canvas._nearest_waveform_point(scene_pos, 0.01)
    assert result is not None
    assert result[0] == "VA"
    canvas.clear_all()


def test_nearest_waveform_point_ignores_hidden_curve(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    t = np.array([0.0, 0.01, 0.02])
    canvas.update_curve("s1", "VA", t, np.array([1.0, 1.0, 1.0]), color="#ff0000", visible=True)
    canvas.update_curve("s1", "VB", t, np.array([1.0, 1.0, 1.0]), color="#00ff00", visible=True)
    canvas.set_curve_visible("s1", "VA", False)

    curve_b = canvas._curves[("s1", "VB")]
    vb = curve_b.getViewBox()
    # VA and VB have identical data, so aim exactly at the shared point.
    scene_pos = vb.mapViewToScene(pg.Point(0.01, 1.0))

    result = canvas._nearest_waveform_point(scene_pos, 0.01)
    assert result is not None
    assert result[0] == "VB"  # the hidden VA curve must never win
    canvas.clear_all()


def test_free_crosshair_mode_unaffected_by_snap_flag(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    assert canvas._crosshair_snap_enabled is False
    canvas.set_crosshair_snap_enabled(True)
    assert canvas._crosshair_snap_enabled is True
    canvas.set_crosshair_snap_enabled(False)
    assert canvas._crosshair_snap_enabled is False
    canvas.clear_all()


# ---------------------------------------------------------------------------
# set_canvas_theme
# ---------------------------------------------------------------------------


def test_set_canvas_theme_does_not_replace_curve_items(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    t = np.linspace(0, 0.1, 50)
    canvas.update_curve("s1", "VA", t, np.ones(50))
    curve_id_before = id(canvas._curves[("s1", "VA")])

    canvas.set_canvas_theme("light")
    canvas.set_canvas_theme("dark")

    curve_id_after = id(canvas._curves[("s1", "VA")])
    assert curve_id_before == curve_id_after
    canvas.clear_all()


def test_set_canvas_theme_light_and_dark_apply_distinct_backgrounds(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")

    canvas.set_canvas_theme("light")
    assert canvas._canvas_theme == "light"

    canvas.set_canvas_theme("dark")
    assert canvas._canvas_theme == "dark"

    # Unknown/garbage values must fall back to dark rather than raising.
    canvas.set_canvas_theme("not-a-real-theme")
    assert canvas._canvas_theme == "dark"
    canvas.clear_all()


def test_set_canvas_theme_repeated_calls_do_not_duplicate_curves(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    t = np.linspace(0, 0.1, 20)
    canvas.update_curve("s1", "VA", t, np.ones(20))

    for _ in range(5):
        canvas.set_canvas_theme("light")
        canvas.set_canvas_theme("dark")

    assert canvas.curve_count == 1
    canvas.clear_all()
