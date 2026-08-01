"""Cross-path timestamp consistency: CsvProvider.load() vs the Import Wizard's
headless pipeline must resolve the same ambiguous date the same way.

This is the regression test for the divergence found by the architecture
audit: "3/6/2026" was interpreted as 6 March by the direct CsvProvider path
(bare pandas, month-first default) and as 3 June by the Import Wizard
(day-first-preferred format probing). Both now apply Powerwave's approved
day-first-default policy for ambiguous CSV/Excel dates.

Per task scope, only timestamp consistency is asserted here — channel
naming/units/classification are known to differ between the two pipelines
and are explicitly out of scope for this fix.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

_CSV_PATH = Path("samples/csv/pulu_20260306.csv")


@pytest.mark.skipif(not _CSV_PATH.exists(), reason="sample fixture not present")
def test_csv_provider_and_wizard_agree_on_ambiguous_start_time() -> None:
    from app.providers.csv.csv_provider import CsvProvider
    from app.import_wizard.import_pipeline import run_import_pipeline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct_record = CsvProvider().load(_CSV_PATH)

    result = run_import_pipeline(str(_CSV_PATH), provider_type="csv")

    assert result.success
    assert result.record is not None
    assert direct_record.timing_info.start_time == result.record.timing_info.start_time


@pytest.mark.skipif(not _CSV_PATH.exists(), reason="sample fixture not present")
def test_csv_provider_and_wizard_agree_on_elapsed_time_axis() -> None:
    from app.providers.csv.csv_provider import CsvProvider
    from app.import_wizard.import_pipeline import run_import_pipeline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct_record = CsvProvider().load(_CSV_PATH)

    result = run_import_pipeline(str(_CSV_PATH), provider_type="csv")

    direct_deltas = direct_record.waveform_data["time"].head(5).round(3).tolist()
    wizard_deltas = result.record.waveform_data["time"].head(5).round(3).tolist()
    assert direct_deltas == wizard_deltas


def test_excel_provider_and_wizard_agree_on_ambiguous_start_time(tmp_path: Path) -> None:
    import openpyxl

    from app.providers.excel.excel_provider import ExcelProvider
    from app.import_wizard.import_pipeline import run_import_pipeline

    xlsx_path = tmp_path / "ambiguous.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Time", "Value"])
    ws.append(["3/6/2026 17:25", 1.0])
    ws.append(["3/6/2026 17:26", 2.0])
    ws.append(["3/6/2026 17:27", 3.0])
    wb.save(str(xlsx_path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct_record = ExcelProvider().load(xlsx_path)

    result = run_import_pipeline(str(xlsx_path), provider_type="excel")

    assert result.success
    assert result.record is not None
    assert direct_record.timing_info.start_time == result.record.timing_info.start_time
