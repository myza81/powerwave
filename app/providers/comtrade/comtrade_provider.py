from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.models import (
    AnalogChannel,
    DigitalChannel,
    DisturbanceRecord,
    RecordingMetadata,
    SamplingInformation,
    TimingInformation,
)
from app.providers.base.base_provider import BaseProvider
from app.providers.base.exceptions import ProviderLoadError

_SUPPORTED_SUFFIXES = {".cfg", ".comtrade"}
_DEFAULT_REV_YR = "1999"
_DEFAULT_NOMINAL_FREQ = 50.0
_DEFAULT_TIMEMULT = 1.0
_MICROSECONDS_PER_SECOND = 1_000_000.0
_VALID_REV_YRS = {"1991", "1999", "2013"}
# Non-standard revision years written by some IEDs — mapped silently to nearest standard.
_NONSTANDARD_REV_YR_MAP = {"2001": "1999", "2005": "1999"}
_VALID_DAT_FORMATS = {"ASCII", "BINARY", "BINARY32"}


# ─────────────────────────────────────────────────────────────────────────────
# Internal CFG data containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _AnalogDef:
    index: int
    ch_id: str
    ph: str | None
    uu: str
    a: float
    b: float
    skew: float
    primary: float | None
    secondary: float | None
    ps_flag: str | None


@dataclass
class _DigitalDef:
    index: int
    ch_id: str
    ph: str | None
    y: int


@dataclass
class _CfgData:
    station_name: str
    rec_dev_id: str
    rev_yr: str
    n_analog: int
    n_digital: int
    analog_defs: list[_AnalogDef]
    digital_defs: list[_DigitalDef]
    nominal_freq: float
    n_rates: int
    sampling_rates: list[float]
    samples_per_rate: list[int]
    total_samples: int
    start_time: datetime
    trigger_time: datetime
    dat_format: str
    timemult: float
    dat_file: Path


# ─────────────────────────────────────────────────────────────────────────────
# CFG helper functions
# ─────────────────────────────────────────────────────────────────────────────


def _get_cfg_line(lines: list[str], idx: int, field_name: str, path: Path) -> str:
    """Return stripped CFG line at *idx*, raising ProviderLoadError if missing."""
    if idx >= len(lines):
        raise ProviderLoadError(
            f"CFG '{path}' is truncated: missing field '{field_name}'"
        )
    return lines[idx].strip()


def _parse_channel_counts(line: str, path: Path) -> tuple[int, int]:
    """Parse CFG line 2.

    Accepts both modern "TT,##A,##D" (e.g. "3,2A,1D") and legacy "nA,nD"
    (e.g. "2,1") forms.
    """
    parts = [p.strip() for p in line.split(",")]
    try:
        if len(parts) >= 3:
            n_analog = int(parts[1].rstrip("AaBb"))
            n_digital = int(parts[2].rstrip("Dd"))
        elif len(parts) == 2:
            n_analog = int(parts[0].rstrip("AaBb"))
            n_digital = int(parts[1].rstrip("Dd"))
        else:
            raise ValueError
        return n_analog, n_digital
    except (ValueError, IndexError) as exc:
        raise ProviderLoadError(
            f"Cannot parse channel counts from '{line}' in '{path}'"
        ) from exc


def _parse_analog_line(line: str, default_index: int) -> _AnalogDef:
    """Parse one analog channel definition line from CFG.

    Field order (COMTRADE 1999, 13 fields):
      0:An#  1:ch_id  2:ph  3:ccbm  4:uu  5:a  6:b  7:skew  8:min  9:max
      10:primary  11:secondary  12:PS
    COMTRADE 1991 has only 10 fields (stops at max).
    """
    parts = line.split(",")

    def _get(i: int, default: str = "") -> str:
        return parts[i].strip() if i < len(parts) else default

    index_str = _get(0).lstrip("AaBb")
    try:
        index = int(index_str) if index_str else default_index
    except ValueError:
        index = default_index

    ch_id = _get(1) or f"A{default_index}"
    ph = _get(2) or None
    uu = _get(4) or ""

    try:
        a = float(_get(5, "1.0"))
    except ValueError:
        warnings.warn(f"Cannot parse scale factor 'a' for channel '{ch_id}', using 1.0")
        a = 1.0

    try:
        b = float(_get(6, "0.0"))
    except ValueError:
        warnings.warn(f"Cannot parse offset 'b' for channel '{ch_id}', using 0.0")
        b = 0.0

    try:
        skew = float(_get(7, "0.0"))
    except ValueError:
        skew = 0.0

    primary: float | None = None
    secondary: float | None = None
    ps_flag: str | None = None

    if len(parts) >= 13:
        try:
            primary = float(_get(10)) if _get(10) else None
        except ValueError:
            primary = None
        try:
            secondary = float(_get(11)) if _get(11) else None
        except ValueError:
            secondary = None
        ps_val = _get(12)
        ps_flag = ps_val if ps_val else None

    if a == 0.0:
        warnings.warn(
            f"Analog channel '{ch_id}' has scale factor a=0.0 — "
            "channel will be rendered as constant value b"
        )

    return _AnalogDef(
        index=index,
        ch_id=ch_id,
        ph=ph,
        uu=uu,
        a=a,
        b=b,
        skew=skew,
        primary=primary,
        secondary=secondary,
        ps_flag=ps_flag,
    )


def _parse_digital_line(line: str, default_index: int) -> _DigitalDef:
    """Parse one digital channel definition line from CFG.

    5-field format (1999): Dn#,ch_id,ph,ccbm,y
    3-field format (some vendors): Dn#,ch_id,y
    """
    parts = line.split(",")

    def _get(i: int, default: str = "") -> str:
        return parts[i].strip() if i < len(parts) else default

    index_str = _get(0).lstrip("DdBb")
    try:
        index = int(index_str) if index_str else default_index
    except ValueError:
        index = default_index

    ch_id = _get(1) or f"D{default_index}"
    ph: str | None = None
    y_str: str

    if len(parts) >= 5:
        ph = _get(2) or None
        y_str = _get(4, "0")
    elif len(parts) == 3:
        y_str = _get(2, "0")
    else:
        y_str = "0"

    try:
        y = int(y_str)
    except ValueError:
        y = 0

    return _DigitalDef(index=index, ch_id=ch_id, ph=ph, y=y)


def _parse_timestamp(s: str, path: Path) -> datetime:
    """Parse a COMTRADE CFG timestamp into a datetime.

    Tolerates the real-world format variety found in relay recordings:

    Separators
        Comma (standard) or space between date and time parts.

    Date order
        DD/MM/YYYY (IEEE C37.111 standard, default).
        MM/DD/YY or MM/DD/YYYY — auto-detected when the middle component
        exceeds 12 (cannot be a valid month, so must be the day).

    Year width
        4-digit years (preferred) or 2-digit years:
        00–69 → 2000–2069, 70–99 → 1970–1999.

    Subseconds
        1–6 fractional digits (padded or truncated to microseconds).
        Missing fractional part → microsecond = 0.
    """
    s = s.strip()
    # Split date and time on comma (standard) or first space
    if "," in s:
        date_str, time_str = s.split(",", 1)
    elif " " in s:
        date_str, time_str = s.split(" ", 1)
    else:
        raise ProviderLoadError(
            f"Cannot parse timestamp '{s}' in '{path}': "
            "expected date,time or date time"
        )

    date_str = date_str.strip()
    time_str = time_str.strip()

    d_parts = date_str.split("/")
    if len(d_parts) != 3:
        raise ProviderLoadError(
            f"Cannot parse timestamp date '{date_str}' in '{path}'"
        )
    try:
        p1, p2, p3 = int(d_parts[0]), int(d_parts[1]), int(d_parts[2])
    except ValueError as exc:
        raise ProviderLoadError(
            f"Cannot parse timestamp date '{date_str}' in '{path}'"
        ) from exc

    # If the middle component > 12 it cannot be a month → MM/DD/Y[Y] format
    if p2 > 12:
        month, day, year = p1, p2, p3
    else:
        day, month, year = p1, p2, p3

    # 2-digit year expansion: 00-69 → 2000s, 70-99 → 1900s
    if year < 100:
        year += 2000 if year < 70 else 1900

    t_parts = time_str.split(":")
    try:
        hour = int(t_parts[0])
        minute = int(t_parts[1])
        sec_str = t_parts[2] if len(t_parts) > 2 else "0"
        sec_parts = sec_str.split(".")
        second = int(sec_parts[0])
        microsecond = int(sec_parts[1].ljust(6, "0")[:6]) if len(sec_parts) > 1 else 0
    except (IndexError, ValueError) as exc:
        raise ProviderLoadError(
            f"Cannot parse timestamp time '{time_str}' in '{path}'"
        ) from exc

    try:
        return datetime(year, month, day, hour, minute, second, microsecond)
    except ValueError as exc:
        raise ProviderLoadError(
            f"Cannot parse timestamp '{s}' in '{path}': {exc}"
        ) from exc


def _find_dat_file(cfg_path: Path) -> Path:
    """Locate the DAT file associated with a CFG file (same directory, same stem)."""
    for suffix in (".dat", ".DAT"):
        candidate = cfg_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise ProviderLoadError(
        f"DAT file not found for '{cfg_path}': "
        f"tried '{cfg_path.with_suffix('.dat')}' and '{cfg_path.with_suffix('.DAT')}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CFG parser
# ─────────────────────────────────────────────────────────────────────────────


def _parse_cfg(cfg_path: Path) -> _CfgData:
    """Read and parse a COMTRADE CFG file into a _CfgData container."""
    try:
        text = cfg_path.read_text(encoding="latin-1")
    except OSError as exc:
        raise ProviderLoadError(f"Cannot read CFG file '{cfg_path}'") from exc

    lines = text.splitlines()
    idx = 0

    # Line 1 — station_name,rec_dev_id[,rev_yr]
    raw1 = _get_cfg_line(lines, idx, "station/device/rev_yr", cfg_path)
    idx += 1
    parts1 = [p.strip() for p in raw1.split(",")]
    station_name = parts1[0] if len(parts1) > 0 else ""
    rec_dev_id = parts1[1] if len(parts1) > 1 else ""
    rev_yr_raw = parts1[2] if len(parts1) > 2 else ""
    if rev_yr_raw in _VALID_REV_YRS:
        rev_yr = rev_yr_raw
    elif rev_yr_raw in _NONSTANDARD_REV_YR_MAP:
        rev_yr = _NONSTANDARD_REV_YR_MAP[rev_yr_raw]
    else:
        if rev_yr_raw:
            warnings.warn(
                f"Unrecognized rev_yr '{rev_yr_raw}' in '{cfg_path}', defaulting to '1999'"
            )
        rev_yr = _DEFAULT_REV_YR

    # Line 2 — channel counts
    line2 = _get_cfg_line(lines, idx, "channel counts", cfg_path)
    idx += 1
    n_analog, n_digital = _parse_channel_counts(line2, cfg_path)

    # Analog channel definitions
    analog_defs: list[_AnalogDef] = []
    for i in range(n_analog):
        line = _get_cfg_line(lines, idx, f"analog channel {i + 1}", cfg_path)
        idx += 1
        analog_defs.append(_parse_analog_line(line, i + 1))

    # Digital channel definitions
    digital_defs: list[_DigitalDef] = []
    for i in range(n_digital):
        line = _get_cfg_line(lines, idx, f"digital channel {i + 1}", cfg_path)
        idx += 1
        digital_defs.append(_parse_digital_line(line, i + 1))

    # lf — nominal frequency
    lf_line = _get_cfg_line(lines, idx, "nominal frequency (lf)", cfg_path)
    idx += 1
    try:
        nominal_freq = float(lf_line)
    except ValueError:
        warnings.warn(
            f"Cannot parse nominal frequency '{lf_line}' in '{cfg_path}', defaulting to 50.0"
        )
        nominal_freq = _DEFAULT_NOMINAL_FREQ

    # nrates
    nrates_line = _get_cfg_line(lines, idx, "nrates", cfg_path)
    idx += 1
    try:
        n_rates = int(nrates_line)
    except ValueError as exc:
        raise ProviderLoadError(
            f"Cannot parse nrates '{nrates_line}' in '{cfg_path}'"
        ) from exc

    # Sampling rate sections.
    # nrates == 0 means variable rate, but the CFG still contains exactly one
    # sentinel line "0,<total_samples>" — consume it unconditionally.
    sampling_rates: list[float] = []
    endsamps: list[int] = []
    n_sections = max(n_rates, 1)
    for i in range(n_sections):
        rate_line = _get_cfg_line(lines, idx, f"rate section {i + 1}", cfg_path)
        idx += 1
        rate_parts = [p.strip() for p in rate_line.split(",")]
        try:
            samp = float(rate_parts[0])
            endsamp = int(rate_parts[1])
        except (IndexError, ValueError) as exc:
            raise ProviderLoadError(
                f"Cannot parse rate section '{rate_line}' in '{cfg_path}'"
            ) from exc
        sampling_rates.append(samp)
        endsamps.append(endsamp)

    # Compute samples_per_rate from cumulative endsamp values
    samples_per_rate: list[int] = []
    prev = 0
    for es in endsamps:
        samples_per_rate.append(es - prev)
        prev = es
    total_samples = endsamps[-1] if endsamps else 0

    # start_time
    st_line = _get_cfg_line(lines, idx, "start_time", cfg_path)
    idx += 1
    start_time = _parse_timestamp(st_line, cfg_path)

    # trigger_time
    trig_line = _get_cfg_line(lines, idx, "trigger_time", cfg_path)
    idx += 1
    trigger_time = _parse_timestamp(trig_line, cfg_path)

    # ft — file type
    ft_raw = _get_cfg_line(lines, idx, "ft (file type)", cfg_path).upper()
    idx += 1
    if ft_raw not in _VALID_DAT_FORMATS:
        raise ProviderLoadError(
            f"Unrecognized DAT format '{ft_raw}' in '{cfg_path}': "
            f"expected one of {sorted(_VALID_DAT_FORMATS)}"
        )
    dat_format = ft_raw

    # TIMEMULT — 1999+ optional
    timemult = _DEFAULT_TIMEMULT
    if rev_yr in {"1999", "2013"} and idx < len(lines):
        tm_line = lines[idx].strip()
        if tm_line:
            try:
                timemult = float(tm_line)
                idx += 1
            except ValueError:
                pass  # leave default, line may be TIMECODE or something else

    # Locate DAT file
    dat_file = _find_dat_file(cfg_path)

    return _CfgData(
        station_name=station_name,
        rec_dev_id=rec_dev_id,
        rev_yr=rev_yr,
        n_analog=n_analog,
        n_digital=n_digital,
        analog_defs=analog_defs,
        digital_defs=digital_defs,
        nominal_freq=nominal_freq,
        n_rates=n_rates,
        sampling_rates=sampling_rates,
        samples_per_rate=samples_per_rate,
        total_samples=total_samples,
        start_time=start_time,
        trigger_time=trigger_time,
        dat_format=dat_format,
        timemult=timemult,
        dat_file=dat_file,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DAT parsers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_ascii_dat(
    dat_path: Path,
    cfg: _CfgData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse an ASCII COMTRADE DAT file.

    In ASCII format each digital channel occupies its own column (values 0 or 1),
    unlike binary format where channels are packed into 16-bit words.

    Returns:
        time_array:     float64 (n_samples,)           — seconds from start
        analog_raw:     float64 (n_samples, n_analog)  — unscaled sample values
        digital_states: int8    (n_samples, n_digital) — individual channel states
    """
    expected_cols = 2 + cfg.n_analog + cfg.n_digital

    try:
        raw = dat_path.read_text(encoding="latin-1")
    except OSError as exc:
        raise ProviderLoadError(f"Cannot read DAT file '{dat_path}'") from exc

    # Strip DOS EOF markers (\x1a / CTRL-Z) present in some Windows-generated files.
    # np.loadtxt treats a lone \x1a as a 1-column row and raises a column-count mismatch.
    content = raw.replace("\x1a", "")

    try:
        data = np.loadtxt(io.StringIO(content), delimiter=",", dtype=np.float64, comments=None)
    except Exception as exc:
        raise ProviderLoadError(
            f"Cannot parse ASCII DAT '{dat_path}': {exc}"
        ) from exc

    if data.size == 0:
        raise ProviderLoadError(f"DAT file '{dat_path}' is empty")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != expected_cols:
        raise ProviderLoadError(
            f"ASCII DAT column count mismatch in '{dat_path}': "
            f"expected {expected_cols} (2 + {cfg.n_analog}A + {cfg.n_digital}D), "
            f"got {data.shape[1]}"
        )

    raw_ts = data[:, 1]
    time_array = (raw_ts * cfg.timemult) / _MICROSECONDS_PER_SECOND

    analog_raw = data[:, 2 : 2 + cfg.n_analog]

    if cfg.n_digital > 0:
        digital_states = data[:, 2 + cfg.n_analog :].astype(np.int8)
    else:
        digital_states = np.empty((data.shape[0], 0), dtype=np.int8)

    return time_array, analog_raw, digital_states


def _parse_binary_dat(
    dat_path: Path,
    cfg: _CfgData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Binary (16-bit integer, little-endian) COMTRADE DAT file.

    Row layout: uint32(n) + uint32(ts) + int16*nA + uint16*nDw (LE throughout).

    Returns:
        time_array:     float64 (n_samples,)
        analog_raw:     float64 (n_samples, n_analog) — raw int16 values as float64
        digital_states: int8    (n_samples, n_digital) — individual channel states
    """
    n_dwords = math.ceil(cfg.n_digital / 16) if cfg.n_digital > 0 else 0
    row_size = 8 + 2 * cfg.n_analog + 2 * n_dwords  # 4(n) + 4(ts) + 2*nA + 2*nDw

    try:
        raw_bytes = dat_path.read_bytes()
    except OSError as exc:
        raise ProviderLoadError(f"Cannot read DAT file '{dat_path}'") from exc

    if not raw_bytes:
        raise ProviderLoadError(f"DAT file '{dat_path}' is empty")

    if len(raw_bytes) % row_size != 0:
        raise ProviderLoadError(
            f"Binary DAT '{dat_path}' file size {len(raw_bytes)} bytes is not a "
            f"multiple of row size {row_size} (nA={cfg.n_analog}, nDw={n_dwords})"
        )

    n_samples = len(raw_bytes) // row_size

    # Build structured dtype for vectorized frombuffer parsing
    dtype_fields: list = [("n", "<u4"), ("ts", "<u4")]
    if cfg.n_analog > 0:
        dtype_fields.append(("ana", "<i2", (cfg.n_analog,)))
    if n_dwords > 0:
        dtype_fields.append(("dw", "<u2", (n_dwords,)))
    dt = np.dtype(dtype_fields)

    rows = np.frombuffer(raw_bytes, dtype=dt, count=n_samples)

    time_array = (rows["ts"].astype(np.float64) * cfg.timemult) / _MICROSECONDS_PER_SECOND

    if cfg.n_analog > 0:
        analog_raw = rows["ana"].astype(np.float64)  # (n_samples, n_analog)
    else:
        analog_raw = np.empty((n_samples, 0), dtype=np.float64)

    if n_dwords > 0:
        digital_words = np.array(rows["dw"])  # (n_samples, n_dwords), uint16
    else:
        digital_words = np.empty((n_samples, 0), dtype=np.uint16)

    digital_states = _extract_digital_channels(digital_words, cfg.n_digital)
    return time_array, analog_raw, digital_states


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────


def _apply_analog_scaling(
    analog_raw: np.ndarray,
    analog_defs: list[_AnalogDef],
) -> np.ndarray:
    """Apply COMTRADE a*raw + b scaling vectorized across all samples.

    Returns float64 (n_samples, n_analog) with physical engineering values.
    Binary32 data must NOT pass through this function (already pre-scaled).
    """
    if not analog_defs or analog_raw.shape[1] == 0:
        return analog_raw

    a_vec = np.array([d.a for d in analog_defs], dtype=np.float64)  # (nA,)
    b_vec = np.array([d.b for d in analog_defs], dtype=np.float64)  # (nA,)

    # Broadcast: (n_samples, nA) * (nA,) + (nA,) → (n_samples, nA)
    return analog_raw * a_vec + b_vec


def _extract_digital_channels(
    digital_words: np.ndarray,
    n_digital: int,
) -> np.ndarray:
    """Unpack packed 16-bit digital words into individual channel bit states.

    Vectorized: no per-sample Python loops.

    Returns int8 (n_samples, n_digital) with values 0 or 1.
    Normal state inversion is NOT applied — raw bit states are preserved.
    """
    n_samples = digital_words.shape[0]

    if n_digital == 0 or digital_words.shape[1] == 0:
        return np.empty((n_samples, 0), dtype=np.int8)

    # For channel d (0-indexed): word = d // 16, bit = d % 16
    ch_indices = np.arange(n_digital)
    word_indices = ch_indices // 16   # (n_digital,)
    bit_indices = ch_indices % 16     # (n_digital,)

    # Select the correct word column for each channel: (n_samples, n_digital)
    words_for_channels = digital_words[:, word_indices]

    # Shift right by bit index and mask LSB: (n_samples, n_digital)
    return ((words_for_channels >> bit_indices) & np.uint16(1)).astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame assembly
# ─────────────────────────────────────────────────────────────────────────────


def _build_dataframe(
    time_array: np.ndarray,
    analog_physical: np.ndarray,
    digital_states: np.ndarray,
    analog_defs: list[_AnalogDef],
    digital_defs: list[_DigitalDef],
) -> pd.DataFrame:
    """Assemble waveform_data DataFrame from pre-computed arrays.

    Column order: time, [analog channels in CFG order], [digital channels in CFG order].
    Duplicate channel names are suffixed with _1, _2, ... and warned about.
    """
    col_data: dict[str, np.ndarray] = {"time": time_array}
    seen: set[str] = {"time"}
    suffix_count: dict[str, int] = {}

    def _unique_name(raw_name: str) -> str:
        if raw_name not in seen:
            seen.add(raw_name)
            return raw_name
        count = suffix_count.get(raw_name, 0) + 1
        suffix_count[raw_name] = count
        new_name = f"{raw_name}_{count}"
        warnings.warn(
            f"Duplicate channel name '{raw_name}' in COMTRADE file — "
            f"renamed to '{new_name}'"
        )
        seen.add(new_name)
        return new_name

    for i, adef in enumerate(analog_defs):
        name = _unique_name(adef.ch_id)
        col_data[name] = analog_physical[:, i]

    for i, ddef in enumerate(digital_defs):
        name = _unique_name(ddef.ch_id)
        col_data[name] = digital_states[:, i]

    return pd.DataFrame(col_data)


# ─────────────────────────────────────────────────────────────────────────────
# DisturbanceRecord assembler
# ─────────────────────────────────────────────────────────────────────────────


def _build_record(cfg: _CfgData) -> DisturbanceRecord:
    """Parse the DAT file and assemble a fully normalized DisturbanceRecord."""
    if cfg.dat_format == "ASCII":
        time_arr, analog_raw, digital_states = _parse_ascii_dat(cfg.dat_file, cfg)
    elif cfg.dat_format == "BINARY":
        time_arr, analog_raw, digital_states = _parse_binary_dat(cfg.dat_file, cfg)
    elif cfg.dat_format == "BINARY32":
        raise ProviderLoadError(
            f"BINARY32 (COMTRADE 2013 float32) is not yet supported by this provider. "
            f"DAT file: '{cfg.dat_file}'"
        )
    else:
        raise ProviderLoadError(f"Unknown DAT format '{cfg.dat_format}'")

    actual_samples = len(time_arr)
    if actual_samples == 0:
        raise ProviderLoadError(
            f"DAT file '{cfg.dat_file}' contains no samples after parsing"
        )
    if actual_samples != cfg.total_samples:
        warnings.warn(
            f"DAT sample count mismatch in '{cfg.dat_file}': "
            f"CFG declares {cfg.total_samples} samples, DAT contains {actual_samples}. "
            "Processing available samples."
        )

    analog_physical = _apply_analog_scaling(analog_raw, cfg.analog_defs)

    analog_channels = [
        AnalogChannel(
            name=adef.ch_id,
            unit=adef.uu,
            index=adef.index,
            phase=adef.ph,
            description=_analog_description(adef),
            scale=adef.a,
            offset=adef.b,
            primary_ratio=adef.primary,
            secondary_ratio=adef.secondary,
        )
        for adef in cfg.analog_defs
    ]

    digital_channels = [
        DigitalChannel(
            name=ddef.ch_id,
            index=ddef.index,
            normal_state=ddef.y,
            description=ddef.ph,
        )
        for ddef in cfg.digital_defs
    ]

    waveform_data = _build_dataframe(
        time_arr, analog_physical, digital_states, cfg.analog_defs, cfg.digital_defs
    )

    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name=cfg.station_name,
            recorder_name=cfg.rec_dev_id,
            source_file=str(cfg.dat_file.parent / (cfg.dat_file.stem + ".cfg")),
            provider_type="COMTRADE",
            nominal_frequency=cfg.nominal_freq,
            timezone=None,
        ),
        waveform_data=waveform_data,
        analog_channels=analog_channels,
        digital_channels=digital_channels,
        sampling_info=SamplingInformation(
            sampling_rates=cfg.sampling_rates,
            samples_per_rate=cfg.samples_per_rate,
            nominal_frequency=cfg.nominal_freq,
        ),
        timing_info=TimingInformation(
            start_time=cfg.start_time,
            trigger_time=cfg.trigger_time,
            time_multiplier=cfg.timemult,
            timezone=None,
        ),
        disturbance_info=None,
    )


def _analog_description(adef: _AnalogDef) -> str | None:
    """Compose an AnalogChannel description from PS flag and skew if non-trivial."""
    parts: list[str] = []
    if adef.ps_flag:
        parts.append(f"ps={adef.ps_flag}")
    if adef.skew != 0.0:
        parts.append(f"skew={adef.skew}us")
    return ";".join(parts) if parts else None


# ─────────────────────────────────────────────────────────────────────────────
# Provider
# ─────────────────────────────────────────────────────────────────────────────


class ComtradeProvider(BaseProvider):
    """COMTRADE (IEEE C37.111) ingestion provider.

    Supports:
      .cfg + .dat  — COMTRADE 1991, 1999 (ASCII and Binary 16-bit)
      .comtrade    — looks for companion .dat in same directory

    Does NOT support:
      BINARY32 / COMTRADE 2013 float32 DAT format
    """

    provider_name: str = "comtrade"

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in _SUPPORTED_SUFFIXES

    def load(self, path: Path) -> DisturbanceRecord:
        """Parse the COMTRADE file at *path* and return a normalized DisturbanceRecord."""
        try:
            cfg = _parse_cfg(path)
        except ProviderLoadError:
            raise
        except Exception as exc:
            raise ProviderLoadError(
                f"Unexpected error parsing CFG '{path}': {exc}"
            ) from exc

        try:
            return _build_record(cfg)
        except ProviderLoadError:
            raise
        except Exception as exc:
            raise ProviderLoadError(
                f"Unexpected error building record from '{path}': {exc}"
            ) from exc
