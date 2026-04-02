# SKILL: Signal Processing Engines

## Trigger
Load this skill when implementing anything in `src/engine/`:
rms_calculator.py, phasor_calculator.py, symmetrical_components.py,
fft_analyzer.py, frequency_tracker.py, decimator.py

---

## ENGINE 1 — RMS Calculator (engine/rms_calculator.py)

### Formula
```
RMS = sqrt( (1/N) * sum(x[i]^2) )
N = samples per window
```

### Window Sizes (50 Hz system, scale for 60 Hz)
```python
half_cycle_samples = int(sample_rate / (2 * nominal_freq))   # e.g. 60 at 6000Hz/50Hz
full_cycle_samples = int(sample_rate / nominal_freq)          # e.g. 120 at 6000Hz/50Hz
```

### Implementation Pattern (NumPy stride_tricks — no Python loops)
```python
import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def _rolling_rms_numba(data: np.ndarray, window: int) -> np.ndarray:
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            s += data[j] * data[j]
        out[i] = np.sqrt(s / window)
    return out

def full_cycle_rms(data: np.ndarray, sample_rate: float, freq: float) -> np.ndarray:
    window = int(round(sample_rate / freq))
    return _rolling_rms_numba(data.astype(np.float64), window)
```

### Validation Test
```python
t = np.linspace(0, 1, int(6000), endpoint=False)
sine = np.sin(2 * np.pi * 50 * t)  # amplitude 1.0
rms = full_cycle_rms(sine, 6000.0, 50.0)
valid = rms[~np.isnan(rms)]
assert np.allclose(valid, 1.0 / np.sqrt(2), atol=0.001)  # must be 0.7071 ± 0.001
```

---

## ENGINE 2 — Phasor Calculator (engine/phasor_calculator.py)

### Method: Full-Cycle DFT at Fundamental
```python
def extract_phasor(data: np.ndarray, sample_rate: float,
                   freq: float, sample_index: int) -> complex:
    N = int(round(sample_rate / freq))          # samples per cycle
    half = N // 2
    start = max(0, sample_index - half)
    end = start + N
    if end > len(data):
        return complex(np.nan, np.nan)
    window = data[start:end].astype(np.float64)
    k = 1                                        # fundamental = 1st harmonic
    n = np.arange(N)
    real_part = (2.0 / N) * np.sum(window * np.cos(2 * np.pi * k * n / N))
    imag_part = (2.0 / N) * np.sum(window * np.sin(2 * np.pi * k * n / N))
    return complex(real_part, -imag_part)        # negative imag for standard convention
```

### Pre-computation (vectorised over all samples)
```python
def compute_phasor_array(data: np.ndarray, sample_rate: float,
                         freq: float) -> np.ndarray:
    # Returns complex64 array, same length as data
    # NaN-filled at start/end where full window unavailable
```

### Validation Test
```python
t = np.linspace(0, 1, 6000, endpoint=False)
sig = np.sin(2 * np.pi * 50 * t)           # 1.0A, 0° reference
p = extract_phasor(sig, 6000.0, 50.0, 3000)
assert abs(abs(p) - 1.0) < 0.01            # magnitude 1.0 ± 1%
assert abs(np.angle(p, deg=True)) < 1.0    # angle 0° ± 1°
```

---

## ENGINE 3 — Symmetrical Components (engine/symmetrical_components.py)

### Fortescue Transformation
```python
import numpy as np

A = np.exp(1j * 2 * np.pi / 3)       # operator a = 1∠120°
A2 = A * A                            # operator a² = 1∠240°

TRANSFORM = (1/3) * np.array([
    [1,  1,   1  ],   # zero sequence row
    [1,  A,   A2 ],   # positive sequence row
    [1,  A2,  A  ],   # negative sequence row
])

def sequence_components(Va: complex, Vb: complex, Vc: complex
                        ) -> tuple[complex, complex, complex]:
    """Returns (V0, V1, V2) — zero, positive, negative sequence."""
    phasors = np.array([Va, Vb, Vc])
    result = TRANSFORM @ phasors
    return result[0], result[1], result[2]
```

### Fault Identification Reference
| Fault Type          | V1 | V2 | V0 |
|---------------------|----|----|----|
| 3-phase             | High | ~0 | ~0 |
| Phase-phase         | High | High | ~0 |
| 1-phase-earth       | High | Medium | High |
| 2-phase-earth       | High | Medium | Medium |

### Validation Test
```python
Va = complex(1.0, 0)
Vb = complex(-0.5, -np.sqrt(3)/2)
Vc = complex(-0.5,  np.sqrt(3)/2)
V0, V1, V2 = sequence_components(Va, Vb, Vc)
assert abs(V0) < 0.001   # balanced → zero sequence ≈ 0
assert abs(abs(V1) - 1.0) < 0.001  # positive sequence = 1.0
assert abs(V2) < 0.001   # balanced → negative sequence ≈ 0
```

---

## ENGINE 4 — FFT Analyzer (engine/fft_analyzer.py)

### Implementation Pattern
```python
from scipy.fft import fft, fftfreq
import numpy as np

WINDOW_FUNCTIONS = {
    'rectangular': np.ones,
    'hanning':     np.hanning,
    'hamming':     np.hamming,
    'blackman':    np.blackman,
}

def compute_spectrum(data: np.ndarray, sample_rate: float,
                     window_name: str = 'rectangular') -> tuple[np.ndarray, np.ndarray]:
    """Returns (frequencies, magnitudes) up to Nyquist."""
    N = len(data)
    win = WINDOW_FUNCTIONS[window_name](N)
    spectrum = fft(data * win)
    freqs = fftfreq(N, d=1.0/sample_rate)
    mags = (2.0 / N) * np.abs(spectrum[:N//2])   # one-sided, scaled
    return freqs[:N//2], mags

def compute_thd(mags: np.ndarray, freqs: np.ndarray,
                fundamental_freq: float) -> float:
    """THD as percentage of fundamental magnitude."""
    bin_width = freqs[1] - freqs[0]
    fund_idx = int(round(fundamental_freq / bin_width))
    fund_mag = mags[fund_idx]
    harmonic_power = sum(
        mags[fund_idx * h] ** 2
        for h in range(2, 51)
        if fund_idx * h < len(mags)
    )
    return 100.0 * np.sqrt(harmonic_power) / fund_mag
```

### Validation Test
```python
t = np.linspace(0, 0.04, 240, endpoint=False)  # 2 cycles at 6000Hz
sig = np.sin(2*np.pi*50*t)                      # pure 50Hz sine
freqs, mags = compute_spectrum(sig, 6000.0)
fund_idx = np.argmax(mags)
assert abs(freqs[fund_idx] - 50.0) < 1.0        # peak at 50Hz
thd = compute_thd(mags, freqs, 50.0)
assert thd < 0.1                                  # pure sine → THD < 0.1%
```

---

## ENGINE 5 — Decimator (engine/decimator.py)

### Critical Rule
MAX_DISPLAY_POINTS = 4000  # never exceed this per channel per render

### Min/Max Envelope Algorithm
```python
def decimate_for_display(time_array: np.ndarray, data_array: np.ndarray,
                         t_start: float, t_end: float,
                         max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (x_display, y_display) with at most max_points.
    Uses min/max envelope within each display bucket to preserve peaks.
    """
    mask = (time_array >= t_start) & (time_array <= t_end)
    t_vis = time_array[mask]
    d_vis = data_array[mask]
    n_vis = len(t_vis)

    if n_vis <= max_points:
        return t_vis, d_vis          # full resolution — no decimation needed

    bucket_size = n_vis // (max_points // 2)
    n_buckets = n_vis // bucket_size
    t_out = np.empty(n_buckets * 2)
    d_out = np.empty(n_buckets * 2)

    for i in range(n_buckets):
        sl = slice(i * bucket_size, (i + 1) * bucket_size)
        bucket_t = t_vis[sl]
        bucket_d = d_vis[sl]
        min_idx = np.argmin(bucket_d)
        max_idx = np.argmax(bucket_d)
        if min_idx < max_idx:
            t_out[i*2], d_out[i*2] = bucket_t[min_idx], bucket_d[min_idx]
            t_out[i*2+1], d_out[i*2+1] = bucket_t[max_idx], bucket_d[max_idx]
        else:
            t_out[i*2], d_out[i*2] = bucket_t[max_idx], bucket_d[max_idx]
            t_out[i*2+1], d_out[i*2+1] = bucket_t[min_idx], bucket_d[min_idx]

    return t_out, d_out
```

### Validation Test
```python
# Peak must never be lost at any zoom level
t = np.linspace(0, 10.0, 60_000_000)   # 10s at 6MHz
d = np.sin(2 * np.pi * 50 * t)
d[30_000_000] = 5.0                      # inject a spike
t_dec, d_dec = decimate_for_display(t, d, 0.0, 10.0, 4000)
assert np.max(d_dec) >= 4.9             # spike must survive decimation
assert len(d_dec) <= 4000
```

---

## ENGINE 6 — Frequency Tracker (engine/frequency_tracker.py)

```python
def zero_crossing_frequency(voltage: np.ndarray, time_array: np.ndarray,
                             nominal_freq: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (freq_times, freq_values) — instantaneous frequency at each
    positive zero crossing, interpolated to sub-sample precision.
    """
    # Detect positive zero crossings (sign change -→+)
    signs = np.sign(voltage)
    crossings = np.where((signs[:-1] < 0) & (signs[1:] >= 0))[0]

    if len(crossings) < 2:
        return np.array([]), np.array([])

    # Interpolate exact crossing time
    crossing_times = []
    for idx in crossings:
        frac = -voltage[idx] / (voltage[idx+1] - voltage[idx])
        crossing_times.append(time_array[idx] + frac * (time_array[idx+1] - time_array[idx]))

    crossing_times = np.array(crossing_times)
    periods = np.diff(crossing_times)
    freq_values = 1.0 / periods
    freq_times  = (crossing_times[:-1] + crossing_times[1:]) / 2

    return freq_times, freq_values
```
