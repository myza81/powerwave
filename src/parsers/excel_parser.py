"""
src/parsers/excel_parser.py

Excel parser — loads a single worksheet from an .xlsx / .xls / .xlsm file
into a DisturbanceRecord.

Sheet selection:
  Single-sheet files → loaded automatically.
  Multi-sheet files  → NeedsSheetSelection raised so the UI can prompt the
                        engineer to choose a sheet, then re-call load() with
                        the chosen sheet_name.

After sheet selection the column-detection and record-building logic is
identical to CsvParser — ExcelParser delegates to CsvParser._parse_dataframe().

Architecture: Data layer (parsers/) — imports models/ only.
              Never import from ui/ or engine/ here (LAW 1).

Dependencies:
  pandas   — sheet reading (data I/O; not used in internal data model)
  openpyxl — required by pandas for .xlsx (listed in requirements.txt)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from models.disturbance_record import DisturbanceRecord, SourceFormat
from parsers.csv_parser import CsvParser
from parsers.parser_exceptions import NeedsSheetSelection

# ── Module-level constants ────────────────────────────────────────────────────

# pandas engine selection by file extension
_ENGINE_MAP: dict[str, str] = {
    '.xlsx':  'openpyxl',
    '.xlsm':  'openpyxl',
    '.xlsb':  'pyxlsb',
    '.xls':   'xlrd',
    '.ods':   'odf',
}

# Fallback engine when extension is unrecognised
_DEFAULT_ENGINE: str = 'openpyxl'


# ── ExcelParser ───────────────────────────────────────────────────────────────

class ExcelParser:
    """Parser for Excel workbook files (.xlsx, .xls, .xlsm).

    Produces a single DisturbanceRecord (LAW 5) from one worksheet.
    Multi-sheet workbooks require the caller to specify ``sheet_name``
    (or catch NeedsSheetSelection and re-call with the chosen sheet).

    Column detection, time parsing, unit inference, and signal role
    assignment are performed by CsvParser._parse_dataframe() — no logic
    is duplicated.

    Usage::

        # Single-sheet file — loads automatically
        record = ExcelParser().load(Path('data/fault.xlsx'))

        # Multi-sheet file — first call raises NeedsSheetSelection
        try:
            record = ExcelParser().load(Path('data/multi.xlsx'))
        except NeedsSheetSelection as exc:
            record = ExcelParser().load(
                Path('data/multi.xlsx'),
                sheet_name=exc.sheet_names[0],
            )
    """

    def load(
        self,
        filepath: Path,
        sheet_name: Optional[str] = None,
        column_map: Optional[dict] = None,
    ) -> DisturbanceRecord:
        """Load one worksheet from an Excel file and return a DisturbanceRecord.

        Args:
            filepath:   Path to the Excel workbook.
            sheet_name: Worksheet name to load.  When None and the workbook
                        has exactly one sheet, that sheet is used
                        automatically.  When None and multiple sheets exist,
                        NeedsSheetSelection is raised.
            column_map: Optional channel mapping dict (same schema as
                        CsvParser.load(); see that docstring for details).

        Returns:
            DisturbanceRecord with source_format='EXCEL'.

        Raises:
            NeedsSheetSelection: When sheet_name is None and the workbook
                                 contains more than one sheet.
            ValueError: On unreadable or empty workbooks.
        """
        filepath = Path(filepath)
        engine = _ENGINE_MAP.get(filepath.suffix.lower(), _DEFAULT_ENGINE)

        # ── Inspect available sheets ──────────────────────────────────────────
        with pd.ExcelFile(filepath, engine=engine) as xf:
            sheet_names: list[str] = xf.sheet_names

            if sheet_name is None:
                if len(sheet_names) > 1:
                    raise NeedsSheetSelection(sheet_names)
                sheet_name = sheet_names[0]

            # ── Read the selected sheet ───────────────────────────────────────
            df = pd.read_excel(xf, sheet_name=sheet_name, dtype=str)

        df.columns = [str(c).strip() for c in df.columns]

        # ── Delegate to CsvParser._parse_dataframe ────────────────────────────
        csv_parser = CsvParser()
        record = csv_parser._parse_dataframe(df, filepath, column_map)

        # Override source_format — this file came from Excel, not CSV
        record.source_format = SourceFormat.EXCEL

        return record
