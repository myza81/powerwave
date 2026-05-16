"""Unit tests for app/analytics/rms/rms_cache.py — RMSCache."""
from __future__ import annotations

import numpy as np
import pytest

from app.analytics.rms.rms_cache import RMSCache


class TestRMSCacheBasic:
    def test_empty_cache_returns_none(self) -> None:
        cache = RMSCache()
        assert cache.get("VA", 100, 5000.0) is None

    def test_put_then_get_returns_same_arrays(self) -> None:
        cache = RMSCache()
        rms_t = np.array([0.1, 0.2, 0.3])
        rms_v = np.array([1.0, 2.0, 3.0])
        cache.put("VA", 100, 5000.0, rms_t, rms_v)
        result = cache.get("VA", 100, 5000.0)
        assert result is not None
        t_out, v_out = result
        np.testing.assert_array_equal(t_out, rms_t)
        np.testing.assert_array_equal(v_out, rms_v)

    def test_different_window_gives_different_entry(self) -> None:
        cache = RMSCache()
        t1, v1 = np.array([0.1]), np.array([1.0])
        t2, v2 = np.array([0.2]), np.array([2.0])
        cache.put("VA", 100, 5000.0, t1, v1)
        cache.put("VA", 200, 5000.0, t2, v2)
        assert len(cache) == 2

    def test_different_sample_rate_gives_different_entry(self) -> None:
        cache = RMSCache()
        t1, v1 = np.array([0.1]), np.array([1.0])
        t2, v2 = np.array([0.2]), np.array([2.0])
        cache.put("VA", 100, 5000.0, t1, v1)
        cache.put("VA", 100, 3200.0, t2, v2)
        assert len(cache) == 2
        # Each key is independent
        assert cache.get("VA", 100, 5000.0) is not None
        assert cache.get("VA", 100, 3200.0) is not None

    def test_different_channel_gives_different_entry(self) -> None:
        cache = RMSCache()
        t, v = np.array([0.0]), np.array([1.0])
        cache.put("VA", 100, 5000.0, t, v)
        cache.put("VB", 100, 5000.0, t, v)
        assert len(cache) == 2

    def test_overwrite_same_key_replaces_value(self) -> None:
        cache = RMSCache()
        cache.put("VA", 100, 5000.0, np.array([0.0]), np.array([1.0]))
        cache.put("VA", 100, 5000.0, np.array([9.0]), np.array([99.0]))
        assert len(cache) == 1
        _, v = cache.get("VA", 100, 5000.0)
        assert v[0] == pytest.approx(99.0)


class TestRMSCacheLen:
    def test_empty_len_zero(self) -> None:
        assert len(RMSCache()) == 0

    def test_len_increments_on_put(self) -> None:
        cache = RMSCache()
        t, v = np.array([0.0]), np.array([1.0])
        cache.put("VA", 100, 5000.0, t, v)
        assert len(cache) == 1
        cache.put("VB", 100, 5000.0, t, v)
        assert len(cache) == 2


class TestRMSCacheInvalidate:
    def test_invalidate_channel_removes_all_its_entries(self) -> None:
        cache = RMSCache()
        t, v = np.array([0.0]), np.array([1.0])
        cache.put("VA", 100, 5000.0, t, v)
        cache.put("VA", 200, 5000.0, t, v)
        cache.put("VB", 100, 5000.0, t, v)
        cache.invalidate_channel("VA")
        assert cache.get("VA", 100, 5000.0) is None
        assert cache.get("VA", 200, 5000.0) is None
        assert cache.get("VB", 100, 5000.0) is not None

    def test_invalidate_nonexistent_channel_is_no_op(self) -> None:
        cache = RMSCache()
        t, v = np.array([0.0]), np.array([1.0])
        cache.put("VA", 100, 5000.0, t, v)
        cache.invalidate_channel("NONEXISTENT")
        assert len(cache) == 1

    def test_clear_removes_all_entries(self) -> None:
        cache = RMSCache()
        t, v = np.array([0.0]), np.array([1.0])
        cache.put("VA", 100, 5000.0, t, v)
        cache.put("VB", 100, 5000.0, t, v)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("VA", 100, 5000.0) is None

    def test_clear_then_put_works(self) -> None:
        cache = RMSCache()
        t, v = np.array([0.0]), np.array([1.0])
        cache.put("VA", 100, 5000.0, t, v)
        cache.clear()
        cache.put("VA", 100, 5000.0, t, v)
        assert len(cache) == 1


class TestRMSCacheLargeArrays:
    def test_large_array_stored_and_retrieved(self) -> None:
        cache = RMSCache()
        n = 100_000
        rms_t = np.linspace(0.0, 10.0, n)
        rms_v = np.random.default_rng(42).standard_normal(n)
        cache.put("VA", 100, 5000.0, rms_t, rms_v)
        result = cache.get("VA", 100, 5000.0)
        assert result is not None
        t_out, v_out = result
        assert len(t_out) == n
        assert len(v_out) == n
        # Arrays stored by reference — no copy overhead
        assert t_out is rms_t
        assert v_out is rms_v
