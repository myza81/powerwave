"""Unit tests for Phase 9E — ChannelLegendWidget and generate_source_variant_colour.

Coverage
--------
 1.  generate_source_variant_colour: index 0 returns base colour unchanged
 2.  generate_source_variant_colour: saturation/hue variants are deterministic
 3.  generate_source_variant_colour: index 1 → 70% saturation, +15° hue
 4.  generate_source_variant_colour: index 2 → 50% saturation, +30° hue
 5.  generate_source_variant_colour: malformed hex passes through unchanged
 6.  generate_source_variant_colour: additional sources stay above floor
 7.  ChannelLegendWidget: upsert_row creates a row
 8.  ChannelLegendWidget: upsert_row updates existing row in-place (no duplicate)
 9.  ChannelLegendWidget: remove_row removes the row
10.  ChannelLegendWidget: remove_source_rows removes all rows for a source
11.  ChannelLegendWidget: clear_rows empties all rows
12.  ChannelLegendWidget: duplicate display names get source badge appended
13.  ChannelLegendWidget: unique display names are shown as-is
14.  ChannelLegendWidget: display_name_changed emitted on name edit callback
15.  ChannelLegendWidget: colour_changed emitted on colour callback
16.  ChannelLegendWidget: visibility_changed emitted on hide callback
17.  ChannelLegendWidget: reset-colour emits colour_changed with empty string
18.  ChannelLegendWidget: update_row_colour updates swatch style
19.  ChannelLegendWidget: update_row_display_name triggers label disambiguation
20.  ChannelLegendWidget: row_count property is accurate
"""
from __future__ import annotations

import colorsys
import sys

import pytest

from PyQt6.QtWidgets import QApplication

from app.ui.session.legend_widget import (
    ChannelLegendWidget,
    generate_source_variant_colour,
)


# ---------------------------------------------------------------------------
# QApplication fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hsv(hex_color: str) -> tuple[float, float, float]:
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return colorsys.rgb_to_hsv(r, g, b)


def _make_legend(qapp, panel_id="p1") -> ChannelLegendWidget:
    return ChannelLegendWidget(panel_id)


def _upsert(legend, source_id, channel_name, display_name=None, badge="SRC", unit="kV", colour="#1f77b4", visible=True):
    legend.upsert_row(
        source_id, channel_name,
        display_name or channel_name,
        badge, unit, colour, visible,
    )


# ---------------------------------------------------------------------------
# 1. index 0 returns base colour unchanged
# ---------------------------------------------------------------------------


def test_generate_source_variant_index_0_unchanged():
    base = "#1f77b4"
    result = generate_source_variant_colour(base, 0)
    assert result == base


# ---------------------------------------------------------------------------
# 2. Deterministic — same inputs always produce same output
# ---------------------------------------------------------------------------


def test_generate_source_variant_deterministic():
    base = "#ff7f0e"
    r1 = generate_source_variant_colour(base, 1)
    r2 = generate_source_variant_colour(base, 1)
    assert r1 == r2


# ---------------------------------------------------------------------------
# 3. Index 1 → ~70% saturation, hue shifted +15°
# ---------------------------------------------------------------------------


def test_generate_source_variant_index_1_saturation():
    base = "#ff0000"  # pure red, full saturation in HSV
    result = generate_source_variant_colour(base, 1)

    h_b, s_b, v_b = _hsv(base)
    h_r, s_r, v_r = _hsv(result)

    # Saturation should be ~70% of original (within a tolerance)
    assert abs(s_r - s_b * 0.70) < 0.05

    # Hue shift should be approximately +15° (= +15/360 ≈ 0.0417)
    hue_delta = (h_r - h_b) % 1.0
    assert abs(hue_delta - 15.0 / 360.0) < 0.02


# ---------------------------------------------------------------------------
# 4. Index 2 → ~50% saturation, hue shifted +30°
# ---------------------------------------------------------------------------


def test_generate_source_variant_index_2_saturation():
    base = "#ff0000"
    result = generate_source_variant_colour(base, 2)

    h_b, s_b, v_b = _hsv(base)
    h_r, s_r, v_r = _hsv(result)

    assert abs(s_r - s_b * 0.50) < 0.05

    hue_delta = (h_r - h_b) % 1.0
    assert abs(hue_delta - 30.0 / 360.0) < 0.02


# ---------------------------------------------------------------------------
# 5. Malformed hex passes through unchanged
# ---------------------------------------------------------------------------


def test_generate_source_variant_malformed_hex():
    bad = "not-a-color"
    assert generate_source_variant_colour(bad, 1) == bad


# ---------------------------------------------------------------------------
# 6. Additional sources stay above saturation floor (30%)
# ---------------------------------------------------------------------------


def test_generate_source_variant_saturation_floor():
    base = "#ff0000"  # s=1.0 in HSV
    for i in range(10):
        result = generate_source_variant_colour(base, i)
        _, s, _ = _hsv(result)
        assert s >= 0.25, f"Saturation {s:.3f} dropped below floor at index {i}"


# ---------------------------------------------------------------------------
# 7. upsert_row creates a row
# ---------------------------------------------------------------------------


def test_upsert_row_creates(qapp):
    legend = _make_legend(qapp)
    assert legend.row_count == 0
    _upsert(legend, "s1", "VA")
    assert legend.row_count == 1


# ---------------------------------------------------------------------------
# 8. upsert_row reuses existing row — no duplicate
# ---------------------------------------------------------------------------


def test_upsert_row_no_duplicate(qapp):
    legend = _make_legend(qapp)
    _upsert(legend, "s1", "VA")
    _upsert(legend, "s1", "VA", colour="#ff0000")
    assert legend.row_count == 1


# ---------------------------------------------------------------------------
# 9. remove_row removes the row
# ---------------------------------------------------------------------------


def test_remove_row(qapp):
    legend = _make_legend(qapp)
    _upsert(legend, "s1", "VA")
    legend.remove_row("s1", "VA")
    assert legend.row_count == 0


# ---------------------------------------------------------------------------
# 10. remove_source_rows removes all rows for that source
# ---------------------------------------------------------------------------


def test_remove_source_rows(qapp):
    legend = _make_legend(qapp)
    _upsert(legend, "s1", "VA")
    _upsert(legend, "s1", "VB")
    _upsert(legend, "s2", "VA")
    legend.remove_source_rows("s1")
    assert legend.row_count == 1
    assert ("s2", "VA") in legend._rows


# ---------------------------------------------------------------------------
# 11. clear_rows empties all rows
# ---------------------------------------------------------------------------


def test_clear_rows(qapp):
    legend = _make_legend(qapp)
    _upsert(legend, "s1", "VA")
    _upsert(legend, "s2", "VB")
    legend.clear_rows()
    assert legend.row_count == 0


# ---------------------------------------------------------------------------
# 12. Duplicate display names get source badge appended
# ---------------------------------------------------------------------------


def test_duplicate_display_names_get_badge(qapp):
    legend = _make_legend(qapp)
    legend.upsert_row("s1", "VA", "VA", "COMTRADE", "kV", "#1f77b4", True)
    legend.upsert_row("s2", "VA", "VA", "CSV", "kV", "#ff7f0e", True)

    # Both rows share the same display_name → effective label must include badge
    row1 = legend._rows[("s1", "VA")]
    row2 = legend._rows[("s2", "VA")]
    label1 = row1._name_lbl.text()
    label2 = row2._name_lbl.text()
    assert "COMTRADE" in label1 or "CSV" in label1
    assert "COMTRADE" in label2 or "CSV" in label2


# ---------------------------------------------------------------------------
# 13. Unique display names shown as-is (no badge appended)
# ---------------------------------------------------------------------------


def test_unique_display_names_no_badge(qapp):
    legend = _make_legend(qapp)
    legend.upsert_row("s1", "VA", "Voltage A", "COMTRADE", "kV", "#1f77b4", True)
    legend.upsert_row("s2", "VB", "Voltage B", "CSV", "kV", "#ff7f0e", True)

    row1 = legend._rows[("s1", "VA")]
    assert row1._name_lbl.text() == "Voltage A"


# ---------------------------------------------------------------------------
# 14. display_name_changed emitted on name edit callback
# ---------------------------------------------------------------------------


def test_display_name_changed_signal(qapp):
    legend = _make_legend(qapp)
    _upsert(legend, "s1", "VA")

    received = []
    legend.display_name_changed.connect(lambda s, c, n: received.append((s, c, n)))

    row = legend._rows[("s1", "VA")]
    row._cb_name("s1", "VA", "New Name")

    assert received == [("s1", "VA", "New Name")]


# ---------------------------------------------------------------------------
# 15. colour_changed emitted on colour callback
# ---------------------------------------------------------------------------


def test_colour_changed_signal(qapp):
    legend = _make_legend(qapp)
    _upsert(legend, "s1", "VA")

    received = []
    legend.colour_changed.connect(lambda s, c, col: received.append((s, c, col)))

    row = legend._rows[("s1", "VA")]
    row._cb_colour("s1", "VA", "#ff0000")

    assert received == [("s1", "VA", "#ff0000")]


# ---------------------------------------------------------------------------
# 16. visibility_changed emitted on hide callback
# ---------------------------------------------------------------------------


def test_visibility_changed_signal(qapp):
    legend = _make_legend(qapp)
    _upsert(legend, "s1", "VA")

    received = []
    legend.visibility_changed.connect(lambda s, c, v: received.append((s, c, v)))

    row = legend._rows[("s1", "VA")]
    row._cb_hide("s1", "VA")

    assert received == [("s1", "VA", False)]


# ---------------------------------------------------------------------------
# 17. reset-colour emits colour_changed with empty string
# ---------------------------------------------------------------------------


def test_reset_colour_emits_empty(qapp):
    legend = _make_legend(qapp)
    _upsert(legend, "s1", "VA")

    received = []
    legend.colour_changed.connect(lambda s, c, col: received.append(col))

    row = legend._rows[("s1", "VA")]
    row._cb_reset_colour("s1", "VA")

    assert received == [""]


# ---------------------------------------------------------------------------
# 18. update_row_colour updates swatch style
# ---------------------------------------------------------------------------


def test_update_row_colour(qapp):
    legend = _make_legend(qapp)
    _upsert(legend, "s1", "VA", colour="#1f77b4")
    legend.update_row_colour("s1", "VA", "#ff0000")

    row = legend._rows[("s1", "VA")]
    assert "#ff0000" in row._swatch.styleSheet()


# ---------------------------------------------------------------------------
# 19. update_row_display_name triggers label disambiguation
# ---------------------------------------------------------------------------


def test_update_row_display_name_disambiguation(qapp):
    legend = _make_legend(qapp)
    legend.upsert_row("s1", "VA", "Voltage A", "COMTRADE", "kV", "#1f77b4", True)
    legend.upsert_row("s2", "VA", "Voltage B", "CSV", "kV", "#ff7f0e", True)

    # Rename s2/VA to match s1/VA
    legend.update_row_display_name("s2", "VA", "Voltage A")

    row1 = legend._rows[("s1", "VA")]
    row2 = legend._rows[("s2", "VA")]
    assert "COMTRADE" in row1._name_lbl.text() or "CSV" in row2._name_lbl.text()


# ---------------------------------------------------------------------------
# 20. row_count property is accurate
# ---------------------------------------------------------------------------


def test_row_count(qapp):
    legend = _make_legend(qapp)
    assert legend.row_count == 0
    _upsert(legend, "s1", "VA")
    assert legend.row_count == 1
    _upsert(legend, "s1", "VB")
    assert legend.row_count == 2
    legend.remove_row("s1", "VA")
    assert legend.row_count == 1
    legend.clear_rows()
    assert legend.row_count == 0
