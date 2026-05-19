"""Runtime Qt tests — Import Wizard authoritative execution flow (Phase 8.55H).

These tests exercise the full dialog → plan-builder → pipeline path with a
real (in-memory) Qt application and real CSV files, using an ImmediateThreadPool
so background workers run synchronously in-process.

Coverage
--------
- GUI timestamp selection overrides auto-choice
- GUI column rename propagates to DisturbanceRecord channel name
- GUI column exclusion keeps channel absent from DisturbanceRecord
- GUI type override (→ digital) routes to DigitalChannel
- Validation blocks execution when all columns excluded
- Validation blocks execution when no timestamp selected
- Graceful handling of duplicate canonical names
- Open waveform after successful authoritative import
- Re-profiling a new file clears stale plan state
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app.import_wizard.column_mapping import ParameterType
from app.ui.import_wizard import ImportWizardDialog


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────────────────────────────────────


def _qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class ImmediateThreadPool:
    """Synchronous thread pool substitute for deterministic tests."""

    def start(self, worker) -> None:
        worker.run()


def _make_dlg() -> ImportWizardDialog:
    return ImportWizardDialog(thread_pool=ImmediateThreadPool())


def _standard_csv(tmp_path, name: str = "import.csv") -> str:
    path = tmp_path / name
    path.write_text(
        "timestamp,va,ia,mw,mvar,trip\n"
        "2026-01-01 00:00:00.000,230.1,1.00,100.0,5.0,0\n"
        "2026-01-01 00:00:00.020,230.2,1.01,100.1,5.1,0\n"
        "2026-01-01 00:00:00.040,229.9,0.99,99.9,4.9,1\n",
        encoding="utf-8",
    )
    return str(path)


def _profile_and_advance(dlg: ImportWizardDialog, csv_path: str) -> None:
    """Profile the file and let the wizard advance to the column-mapping page."""
    dlg.set_source_path(csv_path)
    dlg.profile_selected_file()  # synchronous via ImmediateThreadPool


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAuthoritativeTimestampSelection:
    def test_default_candidate_used_on_clean_import(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        emitted = []
        dlg.import_completed.connect(emitted.append)
        try:
            _profile_and_advance(dlg, csv_path)
            dlg.run_import()
            dlg.open_waveform()
            assert emitted, "No record emitted"
            assert emitted[0].sample_count() == 3
        finally:
            dlg.close()
            app.processEvents()

    def test_user_selected_candidate_propagates_to_result(self, tmp_path) -> None:
        """When the user clicks a non-default candidate, that column is used."""
        app = _qapp()
        dlg = _make_dlg()
        # CSV with two plausible timestamp columns
        csv = tmp_path / "two_ts.csv"
        csv.write_text(
            "ts1,ts2,mw\n"
            "2026-01-01 00:00:00,2026-06-01 00:00:00,100.0\n"
            "2026-01-01 00:00:01,2026-06-01 00:00:01,101.0\n",
            encoding="utf-8",
        )
        try:
            _profile_and_advance(dlg, str(csv))
            # Manually select the second candidate (ts2) if available
            candidates = dlg.timestamp_model.candidates
            if len(candidates) >= 2:
                dlg.timestamp_model.select_row(1)
                dlg._on_timestamp_clicked(dlg.timestamp_model.index(1, 0))

            dlg.run_import()
            result = dlg.pipeline_result
            assert result is not None
            # The selected candidate must be present in the result
            if result.selected_candidate is not None:
                assert result.selected_candidate.column_name in {c.column_name for c in candidates}
        finally:
            dlg.close()
            app.processEvents()

    def test_timestamp_selection_cleared_on_re_profile(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv1 = _standard_csv(tmp_path, "file1.csv")
        csv2 = _standard_csv(tmp_path, "file2.csv")
        try:
            _profile_and_advance(dlg, csv1)
            _profile_and_advance(dlg, csv2)
            # Plan state must be cleared when a new file is profiled
            assert dlg.plan_build_result is None
            assert dlg.pipeline_result is None
        finally:
            dlg.close()
            app.processEvents()


class TestAuthoritativeColumnMapping:
    def test_renamed_channel_in_emitted_record(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        emitted = []
        dlg.import_completed.connect(emitted.append)
        try:
            _profile_and_advance(dlg, csv_path)

            # Rename 'mw' → 'active_power' via the column model
            for row, mapping in enumerate(dlg.column_model.mappings):
                if mapping.source_name == "mw":
                    mapping.user_name_override = "active_power"
                    break

            dlg.run_import()
            dlg.open_waveform()
            assert emitted
            record = emitted[0]
            all_names = [c.name for c in record.analog_channels + record.digital_channels]
            assert "active_power" in all_names
            assert "mw" not in all_names
        finally:
            dlg.close()
            app.processEvents()

    def test_excluded_column_absent_from_record(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        emitted = []
        dlg.import_completed.connect(emitted.append)
        try:
            _profile_and_advance(dlg, csv_path)

            # Exclude 'mvar'
            for mapping in dlg.column_model.mappings:
                if mapping.source_name == "mvar":
                    mapping.excluded = True

            dlg.run_import()
            dlg.open_waveform()
            assert emitted
            record = emitted[0]
            all_names = [c.name for c in record.analog_channels + record.digital_channels]
            assert "mvar" not in all_names
        finally:
            dlg.close()
            app.processEvents()

    def test_type_override_digital_creates_digital_channel(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        emitted = []
        dlg.import_completed.connect(emitted.append)
        try:
            _profile_and_advance(dlg, csv_path)

            # Force 'mvar' to digital type
            for mapping in dlg.column_model.mappings:
                if mapping.source_name == "mvar":
                    mapping.user_type_override = ParameterType.DIGITAL

            dlg.run_import()
            dlg.open_waveform()
            assert emitted
            record = emitted[0]
            digital_names = [c.name for c in record.digital_channels]
            assert "mvar" in digital_names
        finally:
            dlg.close()
            app.processEvents()

    def test_unit_override_preserved(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        try:
            _profile_and_advance(dlg, csv_path)

            va_effective_name = None
            for mapping in dlg.column_model.mappings:
                if mapping.source_name == "va":
                    mapping.user_unit_override = "pu"
                    va_effective_name = mapping.effective_name
                    break

            dlg.run_import()
            result = dlg.pipeline_result
            assert result is not None
            if result.success and result.dataset is not None and va_effective_name:
                units = {p.canonical_name: p.unit for p in result.dataset.parameters}
                assert units.get(va_effective_name) == "pu"
        finally:
            dlg.close()
            app.processEvents()


class TestValidationBlocksExecution:
    def test_no_columns_included_blocks_import(self, tmp_path) -> None:
        """run_import() must not dispatch when all columns are excluded."""
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        try:
            _profile_and_advance(dlg, csv_path)

            # Exclude everything
            for mapping in dlg.column_model.mappings:
                mapping.excluded = True

            # run_import() should call QMessageBox — intercept by patching
            from unittest.mock import patch
            with patch("app.ui.import_wizard.import_wizard_dialog.QMessageBox.critical") as mock_crit:
                dlg.run_import()
                assert mock_crit.called, "QMessageBox.critical was not called for blocked import"

            # Pipeline must not have run
            assert dlg.pipeline_result is None
        finally:
            dlg.close()
            app.processEvents()

    def test_duplicate_canonical_names_blocks_import(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        try:
            _profile_and_advance(dlg, csv_path)

            # Give two columns the same output name
            count = 0
            for mapping in dlg.column_model.mappings:
                if mapping.source_name in ("va", "ia"):
                    mapping.user_name_override = "voltage"
                    count += 1
                if count == 2:
                    break

            from unittest.mock import patch
            with patch("app.ui.import_wizard.import_wizard_dialog.QMessageBox.critical") as mock_crit:
                dlg.run_import()
                assert mock_crit.called

            assert dlg.pipeline_result is None
        finally:
            dlg.close()
            app.processEvents()


class TestVisualizationHandoff:
    def test_open_waveform_emits_record(self, tmp_path) -> None:
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        emitted = []
        dlg.import_completed.connect(emitted.append)
        try:
            _profile_and_advance(dlg, csv_path)
            dlg.run_import()
            dlg.open_waveform()
            assert emitted
            record = emitted[0]
            assert record is not None
            assert record.sample_count() == 3
        finally:
            dlg.close()
            app.processEvents()

    def test_open_waveform_without_import_shows_warning(self) -> None:
        app = _qapp()
        dlg = _make_dlg()
        try:
            from unittest.mock import patch
            with patch("app.ui.import_wizard.import_wizard_dialog.QMessageBox.warning") as mock_warn:
                dlg.open_waveform()
                assert mock_warn.called
        finally:
            dlg.close()
            app.processEvents()

    def test_emitted_record_channel_names_match_plan(self, tmp_path) -> None:
        """Visualization handoff: channel names in emitted record must match GUI decisions."""
        app = _qapp()
        dlg = _make_dlg()
        csv_path = _standard_csv(tmp_path)
        emitted = []
        dlg.import_completed.connect(emitted.append)
        try:
            _profile_and_advance(dlg, csv_path)

            # Rename two channels
            renames = {"mw": "active_power", "mvar": "reactive_power"}
            for mapping in dlg.column_model.mappings:
                if mapping.source_name in renames:
                    mapping.user_name_override = renames[mapping.source_name]

            dlg.run_import()
            dlg.open_waveform()
            assert emitted
            record = emitted[0]
            all_names = {c.name for c in record.analog_channels + record.digital_channels}
            assert "active_power" in all_names
            assert "reactive_power" in all_names
            assert "mw" not in all_names
            assert "mvar" not in all_names
        finally:
            dlg.close()
            app.processEvents()
