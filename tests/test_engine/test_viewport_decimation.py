"""
tests/test_engine/test_viewport_decimation.py

Tests for viewport-aware decimation (Milestone 1F).

No Qt dependency — pure NumPy + decimator functions.  These tests validate
the core logic that _update_viewport() relies on:

  1. Viewport slice correctness — mask produces only in-window points.
  2. Full resolution in narrow zoom — no decimation when slice < max_points.
  3. Fault window contains trigger — trigger at t=0 is inside ±200 ms.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.decimator import decimate_minmax


# ── Test 1 — Viewport slice correctness ───────────────────────────────────────

def test_viewport_slice_correctness() -> None:
    """Masking to a time window returns only in-window points.

    Simulates the boolean mask used in _update_viewport() on a 60 000-point
    array covering 0–10 s, sliced to 4–6 s.
    """
    t = np.linspace(0.0, 10.0, 60_000, dtype=np.float64)
    d = np.sin(t)
    t_start, t_end = 4.0, 6.0

    mask = (t >= t_start) & (t <= t_end)
    t_vis = t[mask]
    d_vis = d[mask]

    assert len(t_vis) > 0,            "Sliced window must be non-empty"
    assert t_vis[0] >= t_start,       "First point must be >= t_start"
    assert t_vis[-1] <= t_end,        "Last point must be <= t_end"
    assert len(t_vis) == mask.sum(),  "Slice length must match mask count"
    assert len(d_vis) == len(t_vis),  "Data and time slices must be same length"


# ── Test 2 — Full resolution in narrow zoom ────────────────────────────────────

def test_narrow_zoom_no_decimation() -> None:
    """120 raw points (< 2 000 max) — decimate_minmax returns all 120.

    At 6 000 Hz, one 50 Hz cycle is 0.020 s = 120 samples.
    When the viewport contains only 120 points, no decimation is needed
    and the caller receives full waveform resolution.
    """
    sample_rate = 6_000          # Hz
    window_s    = 0.020          # one 50 Hz cycle
    n           = int(sample_rate * window_s)   # 120 points

    t = np.linspace(0.0, window_s, n, dtype=np.float64)
    d = np.sin(2.0 * np.pi * 50.0 * t)

    t_out, d_out = decimate_minmax(t, d, max_points=2_000)

    assert len(t_out) == n, (
        f"Expected all {n} points (no decimation), got {len(t_out)}"
    )


# ── Test 3 — Fault window contains trigger ────────────────────────────────────

def test_fault_window_contains_trigger() -> None:
    """Trigger is at t=0 on the display axis; ±200 ms window contains it.

    Simulates a 6 000 Hz, 10-second COMTRADE record with the trigger at
    t=2 s into the record.  After trigger-centring and conversion to ms,
    t=0 ms should fall inside the ±200 ms fault window.
    """
    half_ms = 200.0          # ms — FAULT_WINDOW_S × 1000
    t_start = -half_ms
    t_end   =  half_ms

    # Trigger at t=0 must lie strictly inside the window
    assert t_start < 0.0 < t_end, "Trigger (0.0) must be inside ±200 ms window"

    # Build a realistic display-unit time array
    sample_rate = 6_000
    n           = sample_rate * 10                              # 60 000 samples
    t_raw_s     = np.arange(n, dtype=np.float64) / sample_rate  # 0 .. 9.9998 s
    trigger_s   = 2.0                                           # trigger at 2 s
    t_ms        = (t_raw_s - trigger_s) * 1_000.0              # trigger-centred ms

    mask = (t_ms >= t_start) & (t_ms <= t_end)

    assert mask.any(), "Fault window must contain at least one sample"

    t_window = t_ms[mask]
    assert t_window.min() <= 0.0 <= t_window.max(), (
        "Trigger time (0.0 ms) must fall within the windowed samples"
    )
