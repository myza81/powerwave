"""
src/parsers/parser_exceptions.py

Exceptions raised by CsvParser and ExcelParser when the UI layer must
step in to provide additional information before parsing can complete.

These exceptions are part of the normal parser contract — they are not
error conditions but requests for user input.  The UI catches them,
shows the appropriate dialog, and re-calls the parser with the resolved
information.

Architecture: Data layer (parsers/) — no UI imports.  LAW 1.
"""

from __future__ import annotations


class NeedsMappingDialog(Exception):
    """Raised when column roles cannot be auto-detected from CSV/Excel headers.

    The UI layer should catch this exception, show the channel mapping
    dialog populated with ``columns``, collect a ``column_map`` dict
    from the user, then re-call ``CsvParser.load()`` or
    ``ExcelParser.load()`` with that ``column_map``.

    Attributes:
        columns: List of column header strings from the file, in the
                 order they appear.  Excludes any column already
                 identified as the time axis.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns: list[str] = columns
        super().__init__(
            f"Channel mapping required: {len(columns)} columns could not be "
            f"auto-detected.  Provide a column_map and re-call load()."
        )


class NeedsSheetSelection(Exception):
    """Raised by ExcelParser when the file has multiple sheets and none is specified.

    The UI layer should catch this exception, show a sheet-selection
    dialog populated with ``sheet_names``, then re-call
    ``ExcelParser.load()`` with the chosen ``sheet_name``.

    Attributes:
        sheet_names: List of worksheet names found in the Excel file.
    """

    def __init__(self, sheet_names: list[str]) -> None:
        self.sheet_names: list[str] = sheet_names
        super().__init__(
            f"Sheet selection required: file contains {len(sheet_names)} sheets — "
            f"{sheet_names}.  Provide sheet_name and re-call load()."
        )
