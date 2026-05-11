"""Non-GUI unit tests for PowerwaveMainWindow utility functions.

Tests only module-level helpers (_build_provider_manager, _format_load_status)
that have no Qt widget dependencies. No QApplication is required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from app.providers import ProviderManager, ComtradeProvider, CsvProvider, ExcelProvider
from app.ui.main_window.main_window import _build_provider_manager, _format_load_status


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildProviderManager
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildProviderManager:
    def test_returns_provider_manager_instance(self) -> None:
        mgr = _build_provider_manager()
        assert isinstance(mgr, ProviderManager)

    def test_three_providers_registered(self) -> None:
        mgr = _build_provider_manager()
        assert len(mgr.available_providers()) == 3

    def test_comtrade_registered(self) -> None:
        mgr = _build_provider_manager()
        assert "comtrade" in mgr.available_providers()

    def test_csv_registered(self) -> None:
        mgr = _build_provider_manager()
        assert "csv" in mgr.available_providers()

    def test_excel_registered(self) -> None:
        mgr = _build_provider_manager()
        assert "excel" in mgr.available_providers()

    def test_comtrade_can_load_cfg(self) -> None:
        mgr = _build_provider_manager()
        assert mgr.find_provider(Path("test.cfg")) is not None

    def test_csv_can_load_csv(self) -> None:
        mgr = _build_provider_manager()
        assert mgr.find_provider(Path("test.csv")) is not None

    def test_excel_can_load_xlsx(self) -> None:
        mgr = _build_provider_manager()
        assert mgr.find_provider(Path("test.xlsx")) is not None

    def test_second_call_returns_independent_instance(self) -> None:
        mgr1 = _build_provider_manager()
        mgr2 = _build_provider_manager()
        assert mgr1 is not mgr2

    def test_comtrade_is_first_provider(self) -> None:
        mgr = _build_provider_manager()
        providers = mgr.available_providers()
        assert providers[0] == "comtrade"

    def test_csv_is_second_provider(self) -> None:
        mgr = _build_provider_manager()
        providers = mgr.available_providers()
        assert providers[1] == "csv"


# ─────────────────────────────────────────────────────────────────────────────
# TestFormatLoadStatus
# ─────────────────────────────────────────────────────────────────────────────


def _make_record(
    source_file: str = "/data/fault.cfg",
    station_name: str = "SUB1",
    n_analog: int = 4,
    n_digital: int = 2,
    sampling_rates: list[float] | None = None,
) -> MagicMock:
    """Build a minimal DisturbanceRecord mock for _format_load_status tests."""
    record = MagicMock()
    record.metadata.source_file = source_file
    record.metadata.station_name = station_name
    record.analog_channels = [MagicMock() for _ in range(n_analog)]
    record.digital_channels = [MagicMock() for _ in range(n_digital)]
    rates = sampling_rates if sampling_rates is not None else [1000.0]
    record.sampling_info.sampling_rates = rates
    return record


class TestFormatLoadStatus:
    def test_returns_str(self) -> None:
        record = _make_record()
        result = _format_load_status(record)
        assert isinstance(result, str)

    def test_contains_basename_not_full_path(self) -> None:
        record = _make_record(source_file="/some/deep/path/fault.cfg")
        result = _format_load_status(record)
        assert "fault.cfg" in result
        assert "/some/deep/path/" not in result

    def test_contains_analog_count(self) -> None:
        record = _make_record(n_analog=6)
        result = _format_load_status(record)
        assert "6 analog" in result

    def test_contains_digital_count(self) -> None:
        record = _make_record(n_digital=3)
        result = _format_load_status(record)
        assert "3 digital" in result

    def test_contains_sampling_rate(self) -> None:
        record = _make_record(sampling_rates=[3200.0])
        result = _format_load_status(record)
        assert "3200.0 Hz" in result

    def test_unknown_rate_when_no_sampling_rates(self) -> None:
        record = _make_record(sampling_rates=[])
        result = _format_load_status(record)
        assert "unknown" in result

    def test_zero_analog_channels(self) -> None:
        record = _make_record(n_analog=0)
        result = _format_load_status(record)
        assert "0 analog" in result

    def test_zero_digital_channels(self) -> None:
        record = _make_record(n_digital=0)
        result = _format_load_status(record)
        assert "0 digital" in result
