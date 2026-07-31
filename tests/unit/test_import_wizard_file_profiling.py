"""Tests for Phase 8.55B: file_profiler.py, csv_profiler.py, excel_profiler.py."""
from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

import openpyxl
import pytest

from app.import_wizard.contracts import ValidationSeverity
from app.import_wizard.file_profiler import (
    FileProfileResult,
    populate_session,
    profile_import_file,
)
from app.import_wizard.models import ImportWizardSession, RawPreviewModel
from app.import_wizard.wizard_state import WizardStep


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(rows: list[list[str]], delimiter: str = ",", encoding: str = "utf-8") -> str:
    """Write rows to a temporary CSV file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerows(rows)
    return path


def _sample_csv_rows() -> list[list[str]]:
    return [
        ["Timestamp", "Voltage_A", "Current_A", "MW", "Freq"],
        ["2024-01-01 00:00:00", "230.1", "100.2", "23.0", "50.01"],
        ["2024-01-01 00:00:01", "230.2", "100.3", "23.1", "50.00"],
        ["2024-01-01 00:00:02", "230.0", "100.1", "22.9", "49.99"],
        ["2024-01-01 00:00:03", "230.3", "100.4", "23.2", "50.02"],
    ]


def _write_xlsx(rows: list[list[object]], tmp_path: Path) -> str:
    path = tmp_path / "sample.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Loss of 2000MW"
    for row in rows:
        ws.append(row)
    wb.save(path)
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# preview_sampler
# ─────────────────────────────────────────────────────────────────────────────

class TestReadTextSample:
    def test_returns_lines_and_encoding(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        from app.import_wizard.preview_sampler import read_text_sample
        lines, enc = read_text_sample(str(f))
        assert lines == ["line1", "line2", "line3"]
        assert "utf" in enc

    def test_respects_max_lines(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(str(i) for i in range(500)), encoding="utf-8")
        from app.import_wizard.preview_sampler import read_text_sample
        lines, _ = read_text_sample(str(f), max_lines=10)
        assert len(lines) == 10

    def test_returns_empty_for_missing_file(self, tmp_path):
        from app.import_wizard.preview_sampler import read_text_sample
        lines, enc = read_text_sample(str(tmp_path / "no_such_file.txt"))
        assert lines == []

    def test_strips_crlf(self, tmp_path):
        f = tmp_path / "crlf.txt"
        f.write_bytes(b"col1,col2\r\nval1,val2\r\n")
        from app.import_wizard.preview_sampler import read_text_sample
        lines, _ = read_text_sample(str(f))
        assert lines[0] == "col1,col2"
        assert lines[1] == "val1,val2"

    def test_handles_latin1_file(self, tmp_path):
        f = tmp_path / "latin.csv"
        f.write_bytes(b"caf\xe9,bar\n1,2\n")
        from app.import_wizard.preview_sampler import read_text_sample
        lines, enc = read_text_sample(str(f))
        assert len(lines) == 2


class TestEstimateCsvRowCount:
    def test_returns_zero_for_missing_file(self, tmp_path):
        from app.import_wizard.preview_sampler import estimate_csv_row_count
        assert estimate_csv_row_count(str(tmp_path / "nope.csv")) == 0

    def test_returns_zero_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("", encoding="utf-8")
        from app.import_wizard.preview_sampler import estimate_csv_row_count
        assert estimate_csv_row_count(str(f)) == 0

    def test_estimates_for_simple_file(self, tmp_path):
        rows = [["a", "b"]] + [[str(i), str(i * 2)] for i in range(100)]
        path = _write_csv(rows)
        from app.import_wizard.preview_sampler import estimate_csv_row_count
        count = estimate_csv_row_count(path, header_rows=1)
        os.unlink(path)
        assert count > 50


# ─────────────────────────────────────────────────────────────────────────────
# csv_profiler — delimiter detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectDelimiter:
    def test_detects_comma(self, tmp_path):
        f = tmp_path / "comma.csv"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
        from app.import_wizard.csv_profiler import detect_delimiter
        assert detect_delimiter(str(f)) == ","

    def test_detects_semicolon(self, tmp_path):
        f = tmp_path / "semi.csv"
        f.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
        from app.import_wizard.csv_profiler import detect_delimiter
        assert detect_delimiter(str(f)) == ";"

    def test_detects_tab(self, tmp_path):
        f = tmp_path / "tab.tsv"
        f.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n", encoding="utf-8")
        from app.import_wizard.csv_profiler import detect_delimiter
        assert detect_delimiter(str(f)) == "\t"

    def test_missing_file_returns_comma(self, tmp_path):
        from app.import_wizard.csv_profiler import detect_delimiter
        assert detect_delimiter(str(tmp_path / "nope.csv")) == ","


# ─────────────────────────────────────────────────────────────────────────────
# csv_profiler — header detection
# ─────────────────────────────────────────────────────────────────────────────

class TestFindHeaderRowIndex:
    def test_first_row_is_header(self):
        from app.import_wizard.csv_profiler import _find_header_row_index
        rows = [
            ["Timestamp", "Voltage", "Current"],
            ["2024-01-01", "230.1", "100.2"],
            ["2024-01-02", "231.0", "101.0"],
        ]
        assert _find_header_row_index(rows) == 0

    def test_skips_metadata_rows(self):
        from app.import_wizard.csv_profiler import _find_header_row_index
        rows = [
            ["Station: TEST"],
            [""],
            ["Timestamp", "Voltage", "Current"],
            ["2024-01-01", "230.1", "100.2"],
        ]
        assert _find_header_row_index(rows) == 2

    def test_empty_rows_fallback(self):
        from app.import_wizard.csv_profiler import _find_header_row_index
        assert _find_header_row_index([]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# csv_profiler — profile_csv
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileCsv:
    def test_basic_profile(self):
        path = _write_csv(_sample_csv_rows())
        from app.import_wizard.csv_profiler import profile_csv
        model, warnings = profile_csv(path)
        os.unlink(path)
        assert model.column_names == ["Timestamp", "Voltage_A", "Current_A", "MW", "Freq"]
        assert len(model.preview_rows) == 4
        assert not any(w.severity == ValidationSeverity.ERROR for w in warnings)

    def test_header_row_index_is_zero(self):
        path = _write_csv(_sample_csv_rows())
        from app.import_wizard.csv_profiler import profile_csv
        model, _ = profile_csv(path)
        os.unlink(path)
        assert model.header_row_index == 0
        assert model.skipped_row_count == 0

    def test_skipped_rows_counted(self):
        rows = [
            ["Station: XYZ"],
            [""],
            ["Timestamp", "Voltage"],
            ["2024-01-01 00:00:00", "230.0"],
            ["2024-01-01 00:00:01", "230.1"],
        ]
        path = _write_csv(rows)
        from app.import_wizard.csv_profiler import profile_csv
        model, _ = profile_csv(path)
        os.unlink(path)
        assert model.header_row_index == 2
        assert model.skipped_row_count == 2

    def test_max_preview_rows_respected(self):
        rows = [["T", "V"]] + [["2024-01-01", str(i)] for i in range(100)]
        path = _write_csv(rows)
        from app.import_wizard.csv_profiler import profile_csv
        model, _ = profile_csv(path, max_preview_rows=10)
        os.unlink(path)
        assert len(model.preview_rows) <= 10

    def test_error_on_empty_file(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("", encoding="utf-8")
        from app.import_wizard.csv_profiler import profile_csv
        model, warnings = profile_csv(str(f))
        assert any(w.severity == ValidationSeverity.ERROR for w in warnings)

    def test_semicolon_delimiter(self):
        rows = [["Time", "Voltage", "Current"],
                ["2024-01-01 00:00:00", "230.0", "100.0"],
                ["2024-01-01 00:00:01", "230.1", "100.1"]]
        path = _write_csv(rows, delimiter=";")
        from app.import_wizard.csv_profiler import profile_csv
        model, _ = profile_csv(path, delimiter=";")
        os.unlink(path)
        assert model.column_names == ["Time", "Voltage", "Current"]

    def test_row_count_estimate_populated(self):
        rows = [["T", "V"]] + [["2024-01-01", str(i)] for i in range(50)]
        path = _write_csv(rows)
        from app.import_wizard.csv_profiler import profile_csv
        model, _ = profile_csv(path)
        os.unlink(path)
        assert model.row_count_estimate >= 0

    def test_duplicate_column_names_deduplicated(self):
        rows = [["V", "V", "V"], ["1", "2", "3"]]
        path = _write_csv(rows)
        from app.import_wizard.csv_profiler import profile_csv
        model, _ = profile_csv(path)
        os.unlink(path)
        assert len(set(model.column_names)) == len(model.column_names)

    def test_latin1_encoding(self, tmp_path):
        f = tmp_path / "latin.csv"
        f.write_bytes(b"Zeit,Spannung\n2024-01-01 00:00:00,230.0\n")
        from app.import_wizard.csv_profiler import profile_csv
        model, _ = profile_csv(str(f))
        assert "Zeit" in model.column_names or len(model.column_names) > 0


# ─────────────────────────────────────────────────────────────────────────────
# file_profiler — profile_import_file
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileImportFile:
    def test_csv_basic(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        os.unlink(path)
        assert isinstance(result, FileProfileResult)
        assert result.provider_type == "csv"
        assert not result.has_errors()
        assert result.raw_preview.column_names == ["Timestamp", "Voltage_A", "Current_A", "MW", "Freq"]

    def test_missing_file_returns_error(self, tmp_path):
        result = profile_import_file(str(tmp_path / "no_such_file.csv"))
        assert result.has_errors()
        assert any(m.code == "FILE_NOT_FOUND" for m in result.validation_messages)

    def test_csv_timestamp_candidates_detected(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        os.unlink(path)
        assert len(result.timestamp_candidates) >= 1
        assert result.timestamp_candidates[0].column_name == "Timestamp"

    def test_csv_column_mappings_populated(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        os.unlink(path)
        assert len(result.column_mappings) > 0
        names = [m.source_name for m in result.column_mappings]
        assert "Voltage_A" in names or "Current_A" in names

    def test_csv_delimiter_detected(self):
        path = _write_csv(_sample_csv_rows(), delimiter=";")
        result = profile_import_file(path, delimiter=";")
        os.unlink(path)
        assert result.delimiter == ";"

    def test_timestamp_candidates_sorted_by_confidence(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        os.unlink(path)
        confidences = [c.confidence for c in result.timestamp_candidates]
        assert confidences == sorted(confidences, reverse=True)

    def test_provider_type_is_csv_for_csv(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        os.unlink(path)
        assert result.provider_type == "csv"
        assert result.sheet_name is None

    def test_result_has_no_errors_for_valid_csv(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        os.unlink(path)
        error_msgs = [m for m in result.validation_messages if m.severity == ValidationSeverity.ERROR]
        assert error_msgs == []

    def test_excel_title_row_then_elapsed_time_header(self, tmp_path):
        path = _write_xlsx([
            [r"D:\Study_frequency\Loss of 2000MW.out", None, None, None],
            ["Time", "1 - KAWA FREQ", "2 - TIE LINE 1", "3 - TIE LINE 2"],
            [-0.002, 0.0, -0.673291, -0.673291],
            [0.008, 0.00000542871, -0.671219, -0.671219],
            [0.018, 0.00000524838, -0.669146, -0.669146],
        ], tmp_path)
        result = profile_import_file(path)
        assert not result.has_errors()
        assert result.raw_preview.header_row_index == 1
        assert result.raw_preview.column_names[:4] == [
            "Time",
            "1 - KAWA FREQ",
            "2 - TIE LINE 1",
            "3 - TIE LINE 2",
        ]
        assert result.timestamp_candidates
        assert result.timestamp_candidates[0].column_name == "Time"
        assert result.timestamp_candidates[0].detected_format == "elapsed_seconds"


# ─────────────────────────────────────────────────────────────────────────────
# file_profiler — populate_session
# ─────────────────────────────────────────────────────────────────────────────

class TestPopulateSession:
    def _make_session(self) -> ImportWizardSession:
        return ImportWizardSession(source_path="/tmp/test.csv", provider_type="csv")

    def test_populates_raw_preview(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        os.unlink(path)
        session = self._make_session()
        populate_session(session, result)
        assert session.raw_preview is not None
        assert session.raw_preview.column_names == result.raw_preview.column_names

    def test_populates_timestamp_candidates(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        os.unlink(path)
        session = self._make_session()
        populate_session(session, result)
        assert len(session.timestamp_candidates) == len(result.timestamp_candidates)

    def test_populates_column_mappings(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        os.unlink(path)
        session = self._make_session()
        populate_session(session, result)
        assert len(session.column_mappings) == len(result.column_mappings)

    def test_auto_selects_best_timestamp(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        os.unlink(path)
        session = self._make_session()
        populate_session(session, result)
        assert session.selected_timestamp_column is not None

    def test_messages_added_to_session(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        os.unlink(path)
        session = self._make_session()
        populate_session(session, result)
        # Session messages should be at least as many as result messages
        assert len(session.validation_messages) >= len(result.validation_messages)

    def test_delimiter_propagated(self):
        path = _write_csv(_sample_csv_rows(), delimiter=";")
        result = profile_import_file(path, delimiter=";")
        os.unlink(path)
        session = self._make_session()
        populate_session(session, result)
        assert session.delimiter == ";"


# ─────────────────────────────────────────────────────────────────────────────
# Integration — full pipeline round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipelineRoundTrip:
    def test_profile_and_populate_roundtrip(self):
        path = _write_csv(_sample_csv_rows())
        result = profile_import_file(path)
        session = ImportWizardSession(source_path=path, provider_type="csv")
        populate_session(session, result)
        os.unlink(path)

        assert session.raw_preview is not None
        assert session.selected_timestamp_column == "Timestamp"
        assert session.column_mappings
        from app.import_wizard.column_mapping import ParameterType
        types = {m.parameter_type for m in session.column_mappings}
        assert ParameterType.UNKNOWN not in types or len(types) > 1

    def test_metadata_rows_are_skipped(self):
        rows = [
            ["Station: ALPHA"],
            ["Recorded: 2024-01-01"],
            [""],
            ["Timestamp", "Voltage", "Current"],
            ["2024-01-01 00:00:00", "230.0", "100.0"],
            ["2024-01-01 00:00:01", "230.1", "100.1"],
        ]
        path = _write_csv(rows)
        result = profile_import_file(path)
        os.unlink(path)
        assert result.raw_preview.skipped_row_count >= 3
        assert "Timestamp" in result.raw_preview.column_names
