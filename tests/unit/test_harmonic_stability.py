"""Phase 7.5 stabilization tests for harmonic analytics foundations."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app.analytics.harmonics import (
    HarmonicCache,
    HarmonicConfig,
    HarmonicDisplayMode,
    HarmonicRegistry,
    HarmonicWindowMode,
    classify_harmonic_role,
    compute_harmonic_window_samples,
    compute_thd,
    compute_thd_array,
    extract_harmonics,
)
from app.analytics.harmonics.harmonic_models import (
    HarmonicChannelRole,
    HarmonicResult,
)
from app.data.signal_metadata import SignalMetadata
from app.visualization.managers.visualization_manager import VisualizationManager
from app.visualization.widgets.digital_event_timeline import DigitalEventTimeline
from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas


def _sine(
    amplitude: float,
    freq_hz: float,
    sample_rate: float,
    duration_s: float,
    phase_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(round(sample_rate * duration_s))) / sample_rate
    y = amplitude * np.sin(2.0 * np.pi * freq_hz * t + np.radians(phase_deg))
    return t, y


def _mixture(
    sample_rate: float = 5000.0,
    nominal_hz: float = 50.0,
    duration_s: float = 0.24,
    fundamental: float = 100.0,
    harmonics: dict[int, float] | None = None,
    dc_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    harmonics = harmonics or {}
    t, sig = _sine(fundamental, nominal_hz, sample_rate, duration_s)
    sig = sig + dc_offset
    for order, amplitude in harmonics.items():
        _, hn = _sine(amplitude, order * nominal_hz, sample_rate, duration_s)
        sig = sig + hn
    return t, sig


def _cfg(
    nominal_hz: float = 50.0,
    max_order: int = 13,
    window_mode: HarmonicWindowMode = HarmonicWindowMode.TWO_CYCLE,
) -> HarmonicConfig:
    return HarmonicConfig(
        nominal_hz=nominal_hz,
        max_order=max_order,
        window_mode=window_mode,
        overlap=0.5,
    )


def _result(n: int = 5) -> HarmonicResult:
    return HarmonicResult(
        harmonic_time=np.arange(n, dtype=np.float64),
        magnitudes={1: np.ones(n), 5: np.ones(n) * 0.05},
        sample_rate_hz=5000.0,
        nominal_hz=50.0,
        window_samples=200,
        hop_samples=100,
    )


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_short_signal_returns_empty_result() -> None:
    result = extract_harmonics(np.ones(8), 5000.0, _cfg())

    assert result.n_windows == 0
    assert result.magnitudes == {}


def test_pure_fundamental_has_low_higher_harmonics() -> None:
    _, sig = _mixture(harmonics={})
    result = extract_harmonics(sig, 5000.0, _cfg(max_order=13))

    h1 = float(np.mean(result.magnitudes[1]))
    assert h1 > 0.0
    for order in (3, 5, 7, 11, 13):
        assert float(np.mean(result.magnitudes[order])) < h1 * 0.025


def test_injected_odd_harmonics_are_detected() -> None:
    expected = {3: 6.0, 5: 5.0, 7: 4.0, 11: 3.0, 13: 2.0}
    _, sig = _mixture(harmonics=expected)
    result = extract_harmonics(sig, 5000.0, _cfg(max_order=13))

    for order, amplitude in expected.items():
        measured = float(np.mean(result.magnitudes[order]))
        assert measured == pytest.approx(amplitude / np.sqrt(2.0), rel=0.08)


def test_noisy_waveform_with_known_5th_harmonic_remains_finite() -> None:
    rng = np.random.default_rng(7)
    _, sig = _mixture(harmonics={5: 8.0})
    sig = sig + rng.normal(0.0, 0.8, size=len(sig))

    result = extract_harmonics(sig, 5000.0, _cfg(max_order=7))

    h5 = result.magnitudes[5]
    assert np.all(np.isfinite(h5))
    assert float(np.mean(h5)) == pytest.approx(8.0 / np.sqrt(2.0), rel=0.15)


def test_low_amplitude_and_near_zero_waveforms_are_safe() -> None:
    _, low = _mixture(fundamental=1e-6, harmonics={5: 2e-7})
    low_result = extract_harmonics(low, 5000.0, _cfg(max_order=5))
    zero_result = extract_harmonics(np.zeros_like(low), 5000.0, _cfg(max_order=5))

    for result in (low_result, zero_result):
        assert result.n_windows > 0
        for arr in result.magnitudes.values():
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0.0)


def test_non_integer_cycles_and_dc_offset_do_not_crash_or_explode() -> None:
    _, sig = _mixture(duration_s=0.235, harmonics={5: 4.0}, dc_offset=25.0)
    result = extract_harmonics(sig, 5000.0, _cfg(max_order=5))

    assert result.n_windows > 0
    assert np.all(np.isfinite(result.magnitudes[1]))
    assert float(np.mean(result.magnitudes[1])) < 90.0


def test_50hz_and_60hz_nominal_frequency_are_supported() -> None:
    for nominal_hz, sample_rate in ((50.0, 5000.0), (60.0, 6000.0)):
        _, sig = _mixture(
            sample_rate=sample_rate,
            nominal_hz=nominal_hz,
            duration_s=0.24,
            fundamental=120.0,
            harmonics={5: 6.0},
        )
        result = extract_harmonics(sig, sample_rate, _cfg(nominal_hz, max_order=5))

        assert result.n_windows > 0
        assert result.nominal_hz == nominal_hz
        assert float(np.mean(result.magnitudes[1])) == pytest.approx(
            120.0 / np.sqrt(2.0),
            rel=0.03,
        )


def test_invalid_nominal_frequency_and_above_nyquist_orders_are_safe() -> None:
    cfg = HarmonicConfig(nominal_hz=0.0, max_order=5)
    assert compute_harmonic_window_samples(5000.0, cfg) == 1

    _, sig = _mixture(sample_rate=1000.0, duration_s=0.4, harmonics={})
    result = extract_harmonics(sig, 1000.0, _cfg(max_order=30))

    assert np.all(result.magnitudes[11] == 0.0)
    assert np.all(np.isfinite(result.magnitudes[30]))


def test_thd_near_zero_and_non_finite_inputs_are_safe() -> None:
    assert compute_thd({1: 1e-12, 5: 10.0}) == 0.0

    thd = compute_thd_array(
        {
            1: np.array([100.0, 0.0, np.nan, 100.0]),
            3: np.array([5.0, 5.0, 5.0, np.inf]),
            5: np.array([3.0, 3.0, 3.0, 4.0]),
        }
    )
    assert np.all(np.isfinite(thd))
    assert thd[0] == pytest.approx(np.sqrt(5.0**2 + 3.0**2) / 100.0)
    assert thd[1] == 0.0
    assert thd[2] == 0.0
    assert thd[3] == pytest.approx(4.0 / 100.0)


@pytest.mark.parametrize(
    ("name", "unit", "meta", "expected"),
    [
        ("Va", None, None, HarmonicChannelRole.VOLTAGE_HARMONIC),
        ("Vb", None, None, HarmonicChannelRole.VOLTAGE_HARMONIC),
        ("Vc", None, None, HarmonicChannelRole.VOLTAGE_HARMONIC),
        ("Ia", None, None, HarmonicChannelRole.CURRENT_HARMONIC),
        ("Ib", None, None, HarmonicChannelRole.CURRENT_HARMONIC),
        ("Ic", None, None, HarmonicChannelRole.CURRENT_HARMONIC),
        ("Voltage A", None, None, HarmonicChannelRole.VOLTAGE_HARMONIC),
        ("Current B", None, None, HarmonicChannelRole.CURRENT_HARMONIC),
        ("MW", None, None, HarmonicChannelRole.UNKNOWN),
        ("MVar", None, None, HarmonicChannelRole.UNKNOWN),
        ("Freq", None, None, HarmonicChannelRole.UNKNOWN),
        ("ROCOF", None, None, HarmonicChannelRole.UNKNOWN),
        ("Va RMS", None, None, HarmonicChannelRole.UNKNOWN),
        ("V1 Magnitude", None, None, HarmonicChannelRole.UNKNOWN),
        ("I2 Magnitude", None, None, HarmonicChannelRole.UNKNOWN),
        ("Voltage Phasor", None, None, HarmonicChannelRole.UNKNOWN),
        (
            "calc_Va",
            "kV",
            SignalMetadata(name="calc_Va", measurement_kind="calculated"),
            HarmonicChannelRole.UNKNOWN,
        ),
        (
            "slow_Va",
            "kV",
            SignalMetadata(name="slow_Va", measurement_kind="telemetry"),
            HarmonicChannelRole.UNKNOWN,
        ),
    ],
)
def test_classifier_includes_only_raw_ac_waveform_channels(
    name: str,
    unit: str | None,
    meta: SignalMetadata | None,
    expected: HarmonicChannelRole,
) -> None:
    assert classify_harmonic_role(name, unit=unit, signal_meta=meta).role == expected


def test_registry_eligible_channels_exclude_harmonic_result_names() -> None:
    registry = HarmonicRegistry()
    names = ["Va", "Ia", "V1 Magnitude", "I2 Magnitude", "Freq", "MW"]

    assert registry.harmonic_eligible_channels(names) == ["Va", "Ia"]


def test_cache_differentiates_configuration_and_clears() -> None:
    cache = HarmonicCache()
    r1 = _result()
    r2 = _result()
    r3 = _result()
    r4 = _result()

    cache.put("Va", 200, 100, 50.0, 25, r1)
    cache.put("Va", 100, 50, 50.0, 25, r2)
    cache.put("Va", 200, 100, 60.0, 25, r3)
    cache.put("Va", 200, 100, 50.0, 13, r4)

    assert cache.get("Va", 200, 100, 50.0, 25) is r1
    assert cache.get("Va", 100, 50, 50.0, 25) is r2
    assert cache.get("Va", 200, 100, 60.0, 25) is r3
    assert cache.get("Va", 200, 100, 50.0, 13) is r4
    cache.clear()
    assert cache.entry_count == 0


def test_registry_defaults_to_off_and_clears_roles() -> None:
    registry = HarmonicRegistry()

    assert registry.display_mode is HarmonicDisplayMode.OFF
    registry.classify("Va")
    registry.classify("Va RMS")
    assert registry.cached_roles
    registry.clear_cache()
    assert registry.cached_roles == {}


def test_canvas_harmonic_hooks_are_safe_noops(qapp: QApplication) -> None:
    canvas = FlexiblePlotCanvas()

    canvas.set_harmonic_display_mode(HarmonicDisplayMode.THD, _cfg())
    canvas.set_harmonic_display_mode(HarmonicDisplayMode.SPECTRUM)
    canvas.set_harmonic_display_mode(HarmonicDisplayMode.OFF)

    assert canvas._harmonic_display_mode is HarmonicDisplayMode.OFF
    assert canvas._harmonic_curves == {}


def test_visualization_manager_harmonic_hooks_are_safe_noops(qapp: QApplication) -> None:
    canvas = FlexiblePlotCanvas()
    timeline = DigitalEventTimeline()
    manager = VisualizationManager(canvas, timeline)

    manager.show_harmonic_panels({"Va": _result()})
    manager.set_harmonic_panel_data(mode=HarmonicDisplayMode.THD)
    manager.hide_harmonic_panels()

    assert manager._harmonic_panels_visible is False
    assert "kwargs" in manager._harmonic_panel_data
