"""Unit tests for the provider system.

Covers: registration, duplicate rejection, discovery, load routing,
provider-not-found, validation, unregistration, and stub providers.

All tests use lightweight mock providers — no real file parsing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from app.models import (
    AnalogChannel,
    DisturbanceRecord,
    RecordingMetadata,
    SamplingInformation,
    TimingInformation,
)
from app.providers.base.base_provider import BaseProvider
from app.providers.base.exceptions import (
    DuplicateProviderError,
    ProviderError,
    ProviderLoadError,
    ProviderNotFoundError,
)
from app.providers.base.provider_manager import ProviderManager
from app.providers.comtrade.comtrade_provider import ComtradeProvider
from app.providers.csv.csv_provider import CsvProvider
from app.providers.excel.excel_provider import ExcelProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_record() -> DisturbanceRecord:
    """Return the simplest valid DisturbanceRecord for mock provider use."""
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="MOCK",
            recorder_name="MOCK_IED",
            source_file="mock.mock",
            provider_type="MOCK",
            nominal_frequency=50.0,
        ),
        waveform_data=pd.DataFrame({"time": [0.0], "VA": [1.0]}),
        analog_channels=[AnalogChannel(name="VA", unit="kV", index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation(
            sampling_rates=[50.0], samples_per_rate=[1]
        ),
        timing_info=TimingInformation(
            start_time=datetime(2024, 1, 1),
            trigger_time=datetime(2024, 1, 1),
        ),
    )


class _MockProvider(BaseProvider):
    """Handles .mock files; load() returns a valid DisturbanceRecord."""

    provider_name = "mock"

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() == ".mock"

    def load(self, path: Path) -> DisturbanceRecord:
        return _minimal_record()


class _MockProviderA(BaseProvider):
    """Handles .shared files — registered first."""

    provider_name = "mock_a"

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() == ".shared"

    def load(self, path: Path) -> DisturbanceRecord:
        return _minimal_record()


class _MockProviderB(BaseProvider):
    """Handles .shared files — registered second."""

    provider_name = "mock_b"

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() == ".shared"

    def load(self, path: Path) -> DisturbanceRecord:
        return _minimal_record()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestProviderRegistration:
    def test_register_single_provider(self) -> None:
        manager = ProviderManager()
        manager.register_provider(_MockProvider())
        assert "mock" in manager.available_providers()

    def test_available_providers_empty_by_default(self) -> None:
        assert ProviderManager().available_providers() == []

    def test_available_providers_preserves_insertion_order(self) -> None:
        manager = ProviderManager()
        manager.register_provider(_MockProviderA())
        manager.register_provider(_MockProviderB())
        assert manager.available_providers() == ["mock_a", "mock_b"]

    def test_duplicate_registration_raises_duplicate_error(self) -> None:
        manager = ProviderManager()
        manager.register_provider(_MockProvider())
        with pytest.raises(DuplicateProviderError):
            manager.register_provider(_MockProvider())

    def test_duplicate_error_is_provider_error_subclass(self) -> None:
        manager = ProviderManager()
        manager.register_provider(_MockProvider())
        with pytest.raises(ProviderError):
            manager.register_provider(_MockProvider())

    def test_registration_rejects_non_provider(self) -> None:
        manager = ProviderManager()
        with pytest.raises(ProviderError):
            manager.register_provider("not-a-provider")  # type: ignore[arg-type]

    def test_registration_rejects_plain_object(self) -> None:
        manager = ProviderManager()
        with pytest.raises(ProviderError):
            manager.register_provider(object())  # type: ignore[arg-type]

    def test_unregister_removes_provider(self) -> None:
        manager = ProviderManager()
        manager.register_provider(_MockProvider())
        manager.unregister_provider("mock")
        assert "mock" not in manager.available_providers()

    def test_unregister_nonexistent_raises_provider_error(self) -> None:
        manager = ProviderManager()
        with pytest.raises(ProviderError):
            manager.unregister_provider("nonexistent")

    def test_reregister_after_unregister_succeeds(self) -> None:
        manager = ProviderManager()
        manager.register_provider(_MockProvider())
        manager.unregister_provider("mock")
        manager.register_provider(_MockProvider())  # must not raise
        assert "mock" in manager.available_providers()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestProviderDiscovery:
    def test_find_provider_returns_correct_provider(self) -> None:
        manager = ProviderManager()
        manager.register_provider(_MockProvider())
        provider = manager.find_provider(Path("event.mock"))
        assert provider.provider_name == "mock"

    def test_find_provider_case_insensitive_suffix(self) -> None:
        manager = ProviderManager()
        manager.register_provider(_MockProvider())
        provider = manager.find_provider(Path("event.MOCK"))
        assert provider.provider_name == "mock"

    def test_find_provider_not_found_raises(self) -> None:
        manager = ProviderManager()
        with pytest.raises(ProviderNotFoundError):
            manager.find_provider(Path("event.xyz"))

    def test_find_provider_not_found_is_provider_error(self) -> None:
        manager = ProviderManager()
        with pytest.raises(ProviderError):
            manager.find_provider(Path("event.xyz"))

    def test_find_provider_first_registration_wins(self) -> None:
        """When two providers handle the same extension, the first registered wins."""
        manager = ProviderManager()
        manager.register_provider(_MockProviderA())
        manager.register_provider(_MockProviderB())
        provider = manager.find_provider(Path("event.shared"))
        assert provider.provider_name == "mock_a"

    def test_find_provider_after_unregister_finds_second(self) -> None:
        manager = ProviderManager()
        manager.register_provider(_MockProviderA())
        manager.register_provider(_MockProviderB())
        manager.unregister_provider("mock_a")
        provider = manager.find_provider(Path("event.shared"))
        assert provider.provider_name == "mock_b"


# ---------------------------------------------------------------------------
# Load routing
# ---------------------------------------------------------------------------


class TestProviderLoading:
    def test_load_returns_disturbance_record(self) -> None:
        manager = ProviderManager()
        manager.register_provider(_MockProvider())
        record = manager.load(Path("event.mock"))
        assert isinstance(record, DisturbanceRecord)

    def test_load_raises_not_found_for_unhandled_extension(self) -> None:
        manager = ProviderManager()
        with pytest.raises(ProviderNotFoundError):
            manager.load(Path("event.xyz"))

    def test_load_wraps_provider_exception_in_load_error(self) -> None:
        """ProviderManager.load() wraps NotImplementedError from stubs."""
        manager = ProviderManager()
        manager.register_provider(ComtradeProvider())
        with pytest.raises(ProviderLoadError):
            manager.load(Path("fault.cfg"))

    def test_load_error_preserves_original_cause(self) -> None:
        manager = ProviderManager()
        manager.register_provider(ComtradeProvider())
        with pytest.raises(ProviderLoadError) as exc_info:
            manager.load(Path("fault.cfg"))
        assert exc_info.value.__cause__ is not None

    def test_load_routes_through_correct_provider(self) -> None:
        """Only the matching provider is invoked."""
        manager = ProviderManager()
        manager.register_provider(_MockProviderA())
        manager.register_provider(_MockProvider())
        # .mock suffix → MockProvider, not MockProviderA
        record = manager.load(Path("data.mock"))
        assert isinstance(record, DisturbanceRecord)


# ---------------------------------------------------------------------------
# Stub providers
# ---------------------------------------------------------------------------


class TestComtradeProviderStub:
    def test_can_load_cfg(self) -> None:
        assert ComtradeProvider().can_load(Path("fault.cfg")) is True

    def test_can_load_comtrade(self) -> None:
        assert ComtradeProvider().can_load(Path("fault.comtrade")) is True

    def test_cannot_load_csv(self) -> None:
        assert ComtradeProvider().can_load(Path("fault.csv")) is False

    def test_cannot_load_xlsx(self) -> None:
        assert ComtradeProvider().can_load(Path("fault.xlsx")) is False

    def test_load_raises_provider_load_error_for_missing_file(self) -> None:
        with pytest.raises(ProviderLoadError):
            ComtradeProvider().load(Path("nonexistent_fault_file.cfg"))

    def test_provider_name(self) -> None:
        assert ComtradeProvider.provider_name == "comtrade"


class TestCsvProviderStub:
    def test_can_load_csv(self) -> None:
        assert CsvProvider().can_load(Path("data.csv")) is True

    def test_cannot_load_cfg(self) -> None:
        assert CsvProvider().can_load(Path("data.cfg")) is False

    def test_load_raises_provider_load_error_for_missing_file(self) -> None:
        with pytest.raises(ProviderLoadError):
            CsvProvider().load(Path("nonexistent_data_file.csv"))

    def test_provider_name(self) -> None:
        assert CsvProvider.provider_name == "csv"


class TestExcelProviderStub:
    def test_can_load_xlsx(self) -> None:
        assert ExcelProvider().can_load(Path("data.xlsx")) is True

    def test_can_load_xls(self) -> None:
        assert ExcelProvider().can_load(Path("data.xls")) is True

    def test_cannot_load_csv(self) -> None:
        assert ExcelProvider().can_load(Path("data.csv")) is False

    def test_load_raises_provider_load_error_for_missing_file(self) -> None:
        with pytest.raises(ProviderLoadError):
            ExcelProvider().load(Path("nonexistent_data_file.xlsx"))

    def test_provider_name(self) -> None:
        assert ExcelProvider.provider_name == "excel"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_provider_not_found_is_provider_error(self) -> None:
        assert issubclass(ProviderNotFoundError, ProviderError)

    def test_provider_load_error_is_provider_error(self) -> None:
        assert issubclass(ProviderLoadError, ProviderError)

    def test_duplicate_provider_error_is_provider_error(self) -> None:
        assert issubclass(DuplicateProviderError, ProviderError)

    def test_provider_error_is_exception(self) -> None:
        assert issubclass(ProviderError, Exception)
