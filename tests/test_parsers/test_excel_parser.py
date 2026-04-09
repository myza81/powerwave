"""
tests/test_parsers/test_excel_parser.py

Unit and integration tests for ExcelParser (src/parsers/excel_parser.py).

Excel workbooks are generated on-the-fly using openpyxl + pandas via pytest
tmp_path — no committed binary fixtures required.

Coverage:
  - Single-sheet file loads automatically (no NeedsSheetSelection)
  - Multi-sheet file raises NeedsSheetSelection with correct sheet names
  - Sheet selection (sheet_name=...) then loads correctly
  - Role detection works identically to CSV after sheet selection
  - source_format is 'EXCEL' not 'CSV'
  - column_map applied correctly through ExcelParser → CsvParser delegation
  - NeedsMappingDialog raised through the delegation chain
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.channel import SignalRole
from models.disturbance_record import DisturbanceRecord, SourceFormat
from parsers.excel_parser import ExcelParser
from parsers.parser_exceptions import NeedsMappingDialog, NeedsSheetSelection


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _make_waveform_df(n: int = 60, rate_hz: float = 1000.0) -> pd.DataFrame:
    """Build a small V/I waveform DataFrame identical in structure to the CSV files."""
    dt = 1.0 / rate_hz
    base = datetime(2024, 3, 15, 10, 30, 0)
    rows = []
    for i in range(n):
        t = base + timedelta(seconds=i * dt)
        angle = 2 * math.pi * 50 * i * dt
        rows.append({
            'Timestamp':  t.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'Va (kV)':   round(230.0 * math.sin(angle), 4),
            'Vb (kV)':   round(230.0 * math.sin(angle - 2 * math.pi / 3), 4),
            'Vc (kV)':   round(230.0 * math.sin(angle + 2 * math.pi / 3), 4),
            'Ia (kA)':   round(1.0 * math.sin(angle - 0.3), 6),
        })
    return pd.DataFrame(rows)


def _make_trend_df(n: int = 100) -> pd.DataFrame:
    """Build a trend (50 Hz) DataFrame with P/Q/Freq channels."""
    base = datetime(2024, 3, 15, 10, 30, 0)
    rows = []
    for i in range(n):
        t = base + timedelta(seconds=i * 0.02)
        rows.append({
            'Timestamp':      t.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'P_MW (MW)':     round(120.5 + 2.0 * math.sin(0.1 * i), 3),
            'Q_MVAR (MVAR)': round(35.2  + 1.0 * math.cos(0.1 * i), 3),
            'Freq (Hz)':     round(50.02 + 0.01 * math.sin(0.05 * i), 4),
        })
    return pd.DataFrame(rows)


def _make_ambiguous_df(n: int = 30) -> pd.DataFrame:
    """Generic column names — should trigger NeedsMappingDialog.

    Values are in the 50–500 range so they fall outside both heuristic
    windows (I_PHASE: 0.1–50, V_PHASE: >1000), guaranteeing ANALOGUE/LOW
    for all channels regardless of magnitude.
    """
    return pd.DataFrame({
        'ch1': [i * 0.001 for i in range(n)],
        'ch2': [55.0 + float(i) for i in range(n)],   # 55 → 84
        'ch3': [60.0 + float(i) * 2 for i in range(n)],  # 60 → 118
    })


@pytest.fixture
def single_sheet_xlsx(tmp_path) -> Path:
    """One-sheet workbook with V/I waveform data."""
    path = tmp_path / 'single_sheet.xlsx'
    df = _make_waveform_df()
    df.to_excel(path, index=False, sheet_name='Waveform')
    return path


@pytest.fixture
def multi_sheet_xlsx(tmp_path) -> Path:
    """Two-sheet workbook: 'Waveform' and 'Trend'."""
    path = tmp_path / 'multi_sheet.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        _make_waveform_df().to_excel(writer, index=False, sheet_name='Waveform')
        _make_trend_df().to_excel(writer,    index=False, sheet_name='Trend')
    return path


@pytest.fixture
def ambiguous_xlsx(tmp_path) -> Path:
    """One-sheet workbook with generic column names."""
    path = tmp_path / 'ambiguous.xlsx'
    _make_ambiguous_df().to_excel(path, index=False, sheet_name='Data')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# TestNeedsSheetSelection
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeedsSheetSelection:
    """Multi-sheet workbooks must raise NeedsSheetSelection when no sheet named."""

    def test_raises_on_multi_sheet(self, multi_sheet_xlsx):
        with pytest.raises(NeedsSheetSelection) as exc_info:
            ExcelParser().load(multi_sheet_xlsx)
        exc = exc_info.value
        assert isinstance(exc.sheet_names, list)

    def test_exception_carries_both_sheet_names(self, multi_sheet_xlsx):
        with pytest.raises(NeedsSheetSelection) as exc_info:
            ExcelParser().load(multi_sheet_xlsx)
        names = exc_info.value.sheet_names
        assert 'Waveform' in names
        assert 'Trend' in names

    def test_exactly_two_sheets(self, multi_sheet_xlsx):
        with pytest.raises(NeedsSheetSelection) as exc_info:
            ExcelParser().load(multi_sheet_xlsx)
        assert len(exc_info.value.sheet_names) == 2

    def test_single_sheet_does_not_raise(self, single_sheet_xlsx):
        """Single-sheet file must load without any exception."""
        record = ExcelParser().load(single_sheet_xlsx)
        assert isinstance(record, DisturbanceRecord)


# ═══════════════════════════════════════════════════════════════════════════════
# TestSingleSheetLoad
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingleSheetLoad:
    """Single-sheet workbook loads automatically with correct metadata."""

    def test_returns_disturbance_record(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        assert isinstance(record, DisturbanceRecord)

    def test_source_format_is_excel(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        assert record.source_format == SourceFormat.EXCEL

    def test_source_format_not_csv(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        assert record.source_format != SourceFormat.CSV

    def test_channel_count(self, single_sheet_xlsx):
        # Va Vb Vc Ia = 4 analogue channels
        record = ExcelParser().load(single_sheet_xlsx)
        assert record.n_analogue == 4

    def test_no_digital_channels(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        assert record.n_digital == 0

    def test_sample_rate_approx_1000hz(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        assert abs(record.sample_rate - 1000.0) < 5.0

    def test_display_mode_waveform(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        assert record.display_mode == 'WAVEFORM'

    def test_time_array_length(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        assert len(record.time_array) == 60

    def test_va_role_v_phase(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        va = next(c for c in record.analogue_channels if 'Va' in c.name)
        assert va.signal_role == SignalRole.V_PHASE

    def test_ia_role_i_phase(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        ia = next(c for c in record.analogue_channels if 'Ia' in c.name)
        assert ia.signal_role == SignalRole.I_PHASE

    def test_va_unit_kv(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        va = next(c for c in record.analogue_channels if 'Va' in c.name)
        assert va.unit == 'kV'

    def test_raw_data_dtype_float32(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        for ch in record.analogue_channels:
            assert ch.raw_data.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# TestSheetSelection
# ═══════════════════════════════════════════════════════════════════════════════

class TestSheetSelection:
    """Specifying sheet_name after NeedsSheetSelection loads correctly."""

    def test_waveform_sheet_loads(self, multi_sheet_xlsx):
        record = ExcelParser().load(multi_sheet_xlsx, sheet_name='Waveform')
        assert isinstance(record, DisturbanceRecord)

    def test_trend_sheet_loads(self, multi_sheet_xlsx):
        record = ExcelParser().load(multi_sheet_xlsx, sheet_name='Trend')
        assert isinstance(record, DisturbanceRecord)

    def test_waveform_sheet_sample_rate(self, multi_sheet_xlsx):
        record = ExcelParser().load(multi_sheet_xlsx, sheet_name='Waveform')
        assert abs(record.sample_rate - 1000.0) < 5.0

    def test_trend_sheet_sample_rate(self, multi_sheet_xlsx):
        record = ExcelParser().load(multi_sheet_xlsx, sheet_name='Trend')
        assert abs(record.sample_rate - 50.0) < 1.0

    def test_waveform_sheet_display_mode(self, multi_sheet_xlsx):
        record = ExcelParser().load(multi_sheet_xlsx, sheet_name='Waveform')
        assert record.display_mode == 'WAVEFORM'

    def test_trend_sheet_display_mode(self, multi_sheet_xlsx):
        record = ExcelParser().load(multi_sheet_xlsx, sheet_name='Trend')
        assert record.display_mode == 'TREND'

    def test_waveform_sheet_channel_count(self, multi_sheet_xlsx):
        record = ExcelParser().load(multi_sheet_xlsx, sheet_name='Waveform')
        assert record.n_analogue == 4   # Va Vb Vc Ia

    def test_trend_sheet_channel_count(self, multi_sheet_xlsx):
        record = ExcelParser().load(multi_sheet_xlsx, sheet_name='Trend')
        assert record.n_analogue == 3   # P_MW Q_MVAR Freq

    def test_trend_p_mw_role(self, multi_sheet_xlsx):
        record = ExcelParser().load(multi_sheet_xlsx, sheet_name='Trend')
        p = next(c for c in record.analogue_channels if 'P_MW' in c.name)
        assert p.signal_role == SignalRole.P_MW

    def test_trend_freq_role(self, multi_sheet_xlsx):
        record = ExcelParser().load(multi_sheet_xlsx, sheet_name='Trend')
        f = next(c for c in record.analogue_channels if 'Freq' in c.name)
        assert f.signal_role == SignalRole.FREQ

    def test_source_format_excel_after_selection(self, multi_sheet_xlsx):
        record = ExcelParser().load(multi_sheet_xlsx, sheet_name='Waveform')
        assert record.source_format == SourceFormat.EXCEL


# ═══════════════════════════════════════════════════════════════════════════════
# TestExcelNeedsMappingDialog
# ═══════════════════════════════════════════════════════════════════════════════

class TestExcelNeedsMappingDialog:
    """Ambiguous headers in Excel must raise NeedsMappingDialog through the chain."""

    def test_ambiguous_raises_needs_mapping(self, ambiguous_xlsx):
        with pytest.raises(NeedsMappingDialog):
            ExcelParser().load(ambiguous_xlsx)

    def test_column_map_suppresses_dialog(self, ambiguous_xlsx):
        column_map = {
            'time_column': 'ch1',
            'channels': {
                'ch2': {'role': SignalRole.V_PHASE, 'phase': 'A', 'unit': 'kV'},
                'ch3': {'role': SignalRole.I_PHASE, 'phase': 'A', 'unit': 'A'},
            }
        }
        record = ExcelParser().load(ambiguous_xlsx, column_map=column_map)
        assert isinstance(record, DisturbanceRecord)

    def test_column_map_roles_applied(self, ambiguous_xlsx):
        column_map = {
            'time_column': 'ch1',
            'channels': {
                'ch2': {'role': SignalRole.V_PHASE, 'phase': 'A', 'unit': 'kV'},
                'ch3': {'role': SignalRole.I_PHASE, 'phase': 'A', 'unit': 'A'},
            }
        }
        record = ExcelParser().load(ambiguous_xlsx, column_map=column_map)
        ch2 = next(c for c in record.analogue_channels if c.name == 'ch2')
        assert ch2.signal_role == SignalRole.V_PHASE
        assert ch2.role_confirmed is True


# ═══════════════════════════════════════════════════════════════════════════════
# TestExcelDisturbanceRecordContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestExcelDisturbanceRecordContract:
    """LAW 5 checks — Excel record must satisfy the same model contract as CSV."""

    def test_file_path_stored(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        assert record.file_path == single_sheet_xlsx

    def test_trigger_sample_within_bounds(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        assert 0 <= record.trigger_sample < len(record.time_array)

    def test_channel_ids_unique(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        ids = [c.channel_id for c in record.analogue_channels]
        assert len(ids) == len(set(ids))

    def test_multiplier_and_offset_identity(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        for ch in record.analogue_channels:
            assert ch.multiplier == pytest.approx(1.0)
            assert ch.offset == pytest.approx(0.0)

    def test_colours_assigned(self, single_sheet_xlsx):
        record = ExcelParser().load(single_sheet_xlsx)
        for ch in record.analogue_channels:
            assert ch.colour.startswith('#')
