"""Unit tests for FlexiblePlotCanvas Phase 5B engineering scaling.

Tests cover the pure-logic methods without requiring a running Qt event loop.
Following the same MagicMock(spec=FlexiblePlotCanvas) pattern used in
test_d441_stabilization.py — class methods are called directly on the mock
so only the method under test runs (not the Qt widget plumbing).

Covers:
  - _get_display_data: raw fallback when no scaled cache entry
  - _get_display_data: returns scaled data when cache entry exists
  - _build_scaled_arrays: RAW mode leaves scaled cache empty
  - _build_scaled_arrays: no record leaves scaled cache empty
  - _build_scaled_arrays: PRIMARY mode fills scaled cache and effective_units
  - _build_scaled_arrays: unconfigured PER_UNIT does not add to scaled cache
  - _build_scaled_arrays: factor==1.0 skips scaled cache (avoid redundant copy)
  - set_scaling_mode: same mode + no registry is a no-op (no rebuild)
  - set_scaling_mode: RMS caches are cleared on mode change
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.analytics.scaling.scaling_models import (
    EngineeringScalingMode,
    GlobalScalingConfig,
    ScalingResult,
)
from app.analytics.scaling.scaling_registry import ScalingRegistry
from app.analytics.rms.rms_models import RMSDisplayMode
from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def _make_channel(name: str, unit: str = "kV") -> MagicMock:
    ch = MagicMock()
    ch.name = name
    ch.unit = unit
    return ch


def _make_canvas(**kwargs) -> MagicMock:
    """Create a canvas mock with the minimum scaling state attributes."""
    canvas = MagicMock(spec=FlexiblePlotCanvas)
    canvas._scaled_data_cache = {}
    canvas._effective_units = {}
    canvas._scaling_mode = EngineeringScalingMode.RAW
    canvas._scaling_registry = None
    canvas._data_cache = {}
    canvas._record = None
    canvas._rms_cache = None
    canvas._rms_curves = {}
    canvas._rms_time_cache = {}
    canvas._rms_data_cache = {}
    canvas._rms_display_mode = RMSDisplayMode.OFF
    canvas._rms_config = MagicMock()
    canvas._rms_signal_metadata = None
    canvas._rms_force_channels = None
    for k, v in kwargs.items():
        setattr(canvas, k, v)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# TestGetDisplayData
# ─────────────────────────────────────────────────────────────────────────────


class TestGetDisplayData:
    def test_returns_raw_when_no_scaled_entry(self) -> None:
        raw = np.array([1.0, 2.0, 3.0])
        canvas = _make_canvas(_data_cache={"VA": raw}, _scaled_data_cache={})
        result = FlexiblePlotCanvas._get_display_data(canvas, "VA")
        np.testing.assert_array_equal(result, raw)

    def test_returns_scaled_when_cache_entry_exists(self) -> None:
        raw = np.array([1.0, 2.0, 3.0])
        scaled = raw * 100.0
        canvas = _make_canvas(
            _data_cache={"VA": raw},
            _scaled_data_cache={"VA": scaled},
        )
        result = FlexiblePlotCanvas._get_display_data(canvas, "VA")
        np.testing.assert_array_equal(result, scaled)

    def test_returns_none_for_unknown_channel(self) -> None:
        canvas = _make_canvas(_data_cache={}, _scaled_data_cache={})
        assert FlexiblePlotCanvas._get_display_data(canvas, "MISSING") is None

    def test_scaled_takes_priority_over_raw(self) -> None:
        raw = np.array([1.0])
        scaled = np.array([999.0])
        canvas = _make_canvas(
            _data_cache={"VA": raw},
            _scaled_data_cache={"VA": scaled},
        )
        result = FlexiblePlotCanvas._get_display_data(canvas, "VA")
        assert result[0] == 999.0


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildScaledArrays
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildScaledArrays:
    def test_raw_mode_leaves_scaled_cache_empty(self) -> None:
        canvas = _make_canvas(_scaling_mode=EngineeringScalingMode.RAW)
        FlexiblePlotCanvas._build_scaled_arrays(canvas)
        assert canvas._scaled_data_cache == {}
        assert canvas._effective_units == {}

    def test_no_record_leaves_scaled_cache_empty(self) -> None:
        canvas = _make_canvas(
            _scaling_mode=EngineeringScalingMode.PRIMARY,
            _record=None,
        )
        FlexiblePlotCanvas._build_scaled_arrays(canvas)
        assert canvas._scaled_data_cache == {}

    def test_primary_mode_fills_scaled_cache(self) -> None:
        raw = np.array([1.0, 2.0, 3.0])
        ch = _make_channel("VA", "kV")
        record = MagicMock()
        record.analog_channels = [ch]

        registry = ScalingRegistry()
        registry.set_global_config(GlobalScalingConfig(pt_ratio=100.0))

        canvas = _make_canvas(
            _scaling_mode=EngineeringScalingMode.PRIMARY,
            _scaling_registry=registry,
            _data_cache={"VA": raw},
            _record=record,
        )
        FlexiblePlotCanvas._build_scaled_arrays(canvas)

        assert "VA" in canvas._scaled_data_cache
        np.testing.assert_allclose(canvas._scaled_data_cache["VA"], raw * 100.0)

    def test_effective_units_populated(self) -> None:
        raw = np.array([1.0])
        ch = _make_channel("VA", "kV")
        record = MagicMock()
        record.analog_channels = [ch]

        registry = ScalingRegistry()
        registry.set_global_config(GlobalScalingConfig(pt_ratio=100.0))

        canvas = _make_canvas(
            _scaling_mode=EngineeringScalingMode.PRIMARY,
            _scaling_registry=registry,
            _data_cache={"VA": raw},
            _record=record,
        )
        FlexiblePlotCanvas._build_scaled_arrays(canvas)
        assert "VA" in canvas._effective_units

    def test_unconfigured_pu_not_added_to_scaled_cache(self) -> None:
        raw = np.array([1.0, 2.0])
        ch = _make_channel("VA", "kV")
        record = MagicMock()
        record.analog_channels = [ch]

        registry = ScalingRegistry()
        # No voltage_base_kv set → PER_UNIT is unconfigured

        canvas = _make_canvas(
            _scaling_mode=EngineeringScalingMode.PER_UNIT,
            _scaling_registry=registry,
            _data_cache={"VA": raw},
            _record=record,
        )
        FlexiblePlotCanvas._build_scaled_arrays(canvas)
        assert "VA" not in canvas._scaled_data_cache

    def test_factor_1_skips_scaled_cache(self) -> None:
        # PT=1.0 → factor==1.0 → skip copy (raw display is identical)
        raw = np.array([1.0, 2.0])
        ch = _make_channel("VA", "kV")
        record = MagicMock()
        record.analog_channels = [ch]

        registry = ScalingRegistry()
        registry.set_global_config(GlobalScalingConfig(pt_ratio=1.0))

        canvas = _make_canvas(
            _scaling_mode=EngineeringScalingMode.PRIMARY,
            _scaling_registry=registry,
            _data_cache={"VA": raw},
            _record=record,
        )
        FlexiblePlotCanvas._build_scaled_arrays(canvas)
        assert "VA" not in canvas._scaled_data_cache

    def test_channel_absent_from_data_cache_skipped(self) -> None:
        ch = _make_channel("VA", "kV")
        record = MagicMock()
        record.analog_channels = [ch]

        registry = ScalingRegistry()
        registry.set_global_config(GlobalScalingConfig(pt_ratio=100.0))

        canvas = _make_canvas(
            _scaling_mode=EngineeringScalingMode.PRIMARY,
            _scaling_registry=registry,
            _data_cache={},   # "VA" not loaded
            _record=record,
        )
        FlexiblePlotCanvas._build_scaled_arrays(canvas)
        assert "VA" not in canvas._scaled_data_cache

    def test_lazy_creates_registry_when_none(self) -> None:
        raw = np.array([1.0])
        ch = _make_channel("VA", "kV")
        record = MagicMock()
        record.analog_channels = [ch]

        canvas = _make_canvas(
            _scaling_mode=EngineeringScalingMode.PRIMARY,
            _scaling_registry=None,
            _data_cache={"VA": raw},
            _record=record,
        )
        FlexiblePlotCanvas._build_scaled_arrays(canvas)
        assert canvas._scaling_registry is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestSetScalingMode
# ─────────────────────────────────────────────────────────────────────────────


class TestSetScalingMode:
    def test_same_mode_no_registry_is_noop(self) -> None:
        canvas = _make_canvas(_scaling_mode=EngineeringScalingMode.RAW)
        FlexiblePlotCanvas.set_scaling_mode(canvas, EngineeringScalingMode.RAW)
        # _build_scaled_arrays should NOT be called (via mock tracking)
        canvas._rebuild_visible_channel_axes.assert_not_called()

    def test_mode_change_clears_rms_caches(self) -> None:
        rms_cache = MagicMock()
        canvas = _make_canvas(
            _scaling_mode=EngineeringScalingMode.RAW,
            _rms_cache=rms_cache,
            _rms_curves={"VA": MagicMock()},
            _rms_time_cache={"VA": MagicMock()},
            _rms_data_cache={"VA": MagicMock()},
            _record=None,
        )
        FlexiblePlotCanvas.set_scaling_mode(canvas, EngineeringScalingMode.PRIMARY)

        rms_cache.clear.assert_called_once()
        assert canvas._rms_curves == {}
        assert canvas._rms_time_cache == {}
        assert canvas._rms_data_cache == {}

    def test_registry_kwarg_replaces_stored_registry(self) -> None:
        canvas = _make_canvas(_scaling_mode=EngineeringScalingMode.RAW)
        new_registry = ScalingRegistry()
        FlexiblePlotCanvas.set_scaling_mode(
            canvas, EngineeringScalingMode.RAW, registry=new_registry
        )
        assert canvas._scaling_registry is new_registry

    def test_mode_updated_on_canvas(self) -> None:
        canvas = _make_canvas(_scaling_mode=EngineeringScalingMode.RAW, _record=None)
        FlexiblePlotCanvas.set_scaling_mode(canvas, EngineeringScalingMode.PRIMARY)
        assert canvas._scaling_mode == EngineeringScalingMode.PRIMARY
