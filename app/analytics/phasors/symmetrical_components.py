"""Symmetrical (sequence) component computation using the Fortescue transform.

Standard IEC/IEEE Fortescue convention:

    a  = e^(j 2π/3)           rotation operator (120° advance)

    V0 = (Va + Vb  + Vc) / 3  zero-sequence
    V1 = (Va + a·Vb + a²·Vc) / 3  positive-sequence
    V2 = (Va + a²·Vb + a·Vc) / 3  negative-sequence

All inputs and outputs are complex arrays (complex128).  RMS magnitudes
are derived by taking |V1|, |V2|, |V0|; the caller's phasors should
already be RMS if magnitude interpretation is desired.

Engineering validation (unit tests confirm):
    Balanced three-phase → |V1| ≈ Vnom, |V2| ≈ 0, |V0| ≈ 0
    SLG fault (Va depressed, Vb/Vc nominal) → elevated |V0| and |V2|
    Phase reversal (ABC → ACB) → elevated |V2|, |V1| ≈ Vnom, |V0| ≈ 0
"""
from __future__ import annotations

import numpy as np

# Fortescue rotation operator: a = e^(j*2π/3)
_A = np.exp(1j * 2.0 * np.pi / 3.0)
_A2 = _A * _A  # a² = e^(j*4π/3) = e^(-j*2π/3)


def compute_sequence_components(
    va: np.ndarray,
    vb: np.ndarray,
    vc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Fortescue transform to three-phase phasors.

    Args:
        va: Complex phasor array for phase A, shape (N,).
        vb: Complex phasor array for phase B, shape (N,).
        vc: Complex phasor array for phase C, shape (N,).

    Returns:
        (v1, v2, v0): Positive, negative, and zero sequence complex arrays,
                      each shape (N,).

    Note:
        All three input arrays must have the same length.
        The function works identically for voltage (Va/Vb/Vc) and
        current (Ia/Ib/Ic) phasors.
    """
    va_c = np.asarray(va, dtype=np.complex128)
    vb_c = np.asarray(vb, dtype=np.complex128)
    vc_c = np.asarray(vc, dtype=np.complex128)

    v0 = (va_c + vb_c + vc_c) / 3.0
    v1 = (va_c + _A * vb_c + _A2 * vc_c) / 3.0
    v2 = (va_c + _A2 * vb_c + _A * vc_c) / 3.0

    return v1, v2, v0


def sequence_magnitudes(
    v1: np.ndarray,
    v2: np.ndarray,
    v0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RMS magnitudes of the three sequence components.

    Args:
        v1: Positive sequence complex array.
        v2: Negative sequence complex array.
        v0: Zero sequence complex array.

    Returns:
        (mag_v1, mag_v2, mag_v0): Float64 magnitude arrays.
    """
    return np.abs(v1), np.abs(v2), np.abs(v0)


def unbalance_factor(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute negative-sequence voltage unbalance factor (NSVUF).

    NSVUF = |V2| / |V1| × 100 %

    A value near 0 % indicates a balanced system.  IEC 61000-2-2 limits
    are typically 2 % for continuous operation.

    Args:
        v1: Positive sequence complex array.
        v2: Negative sequence complex array.

    Returns:
        Float64 percentage array.  Returns 0.0 where |V1| == 0.
    """
    mag_v1 = np.abs(v1)
    mag_v2 = np.abs(v2)
    with np.errstate(divide="ignore", invalid="ignore"):
        uf = np.where(mag_v1 > 0, mag_v2 / mag_v1 * 100.0, 0.0)
    return uf.astype(np.float64)


def compute_sequence_from_phasor_arrays(
    phasor_a: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    phasor_b: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    phasor_c: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute sequence components from three extract_phasor() result tuples.

    Each phasor tuple is (phasor_time, magnitude_rms, angle_deg, complex_phasor)
    as returned by phasor_extraction.extract_phasor().

    The three phasor arrays must share the same time base and length — i.e.
    they must have been extracted from the same record with the same config.

    Args:
        phasor_a: Extract result for phase A.
        phasor_b: Extract result for phase B.
        phasor_c: Extract result for phase C.

    Returns:
        Dictionary with keys:
          'time'  — shared time array
          'v1'    — positive sequence complex array
          'v2'    — negative sequence complex array
          'v0'    — zero sequence complex array
          'mag_v1', 'mag_v2', 'mag_v0' — RMS magnitude float64 arrays
    """
    time_a, _, _, cpx_a = phasor_a
    _,      _, _, cpx_b = phasor_b
    _,      _, _, cpx_c = phasor_c

    if len(cpx_a) != len(cpx_b) or len(cpx_a) != len(cpx_c):
        raise ValueError(
            "Phasor arrays must have equal length for sequence component computation. "
            f"Got {len(cpx_a)}, {len(cpx_b)}, {len(cpx_c)}."
        )

    # Use peak phasors for Fortescue (magnitudes are still proportional)
    v1, v2, v0 = compute_sequence_components(cpx_a, cpx_b, cpx_c)
    mag_v1, mag_v2, mag_v0 = sequence_magnitudes(v1, v2, v0)

    # Convert peak → RMS for displayed magnitudes
    sqrt2 = np.sqrt(2.0)
    return {
        "time": time_a,
        "v1": v1 / sqrt2,
        "v2": v2 / sqrt2,
        "v0": v0 / sqrt2,
        "mag_v1": mag_v1 / sqrt2,
        "mag_v2": mag_v2 / sqrt2,
        "mag_v0": mag_v0 / sqrt2,
    }
