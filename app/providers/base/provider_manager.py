from __future__ import annotations

from pathlib import Path

from app.models import DisturbanceRecord
from app.providers.base.base_provider import BaseProvider
from app.providers.base.exceptions import (
    ProviderError,
    ProviderLoadError,
    ProviderNotFoundError,
)
from app.providers.base.provider_registry import ProviderRegistry


class ProviderManager:
    """Orchestrates provider registration, discovery, and file loading.

    Typical usage::

        manager = ProviderManager()
        manager.register_provider(ComtradeProvider())
        manager.register_provider(CsvProvider())

        record = manager.load(Path("fault.cfg"))

    Provider discovery follows insertion order — the first registered provider
    whose ``can_load()`` returns True is selected.  This makes priority explicit
    and deterministic.
    """

    def __init__(self) -> None:
        self._registry = ProviderRegistry()

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def register_provider(self, provider: BaseProvider) -> None:
        """Register *provider* for use in the load workflow.

        Raises:
            ProviderError: if *provider* is not a BaseProvider subclass.
            DuplicateProviderError: if provider_name is already registered.
        """
        if not isinstance(provider, BaseProvider):
            raise ProviderError(
                f"Expected a BaseProvider subclass, got {type(provider).__name__!r}."
            )
        self._registry.register(provider)

    def unregister_provider(self, name: str) -> None:
        """Remove the provider registered under *name*.

        Raises:
            ProviderError: if *name* is not registered.
        """
        self._registry.unregister(name)

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def available_providers(self) -> list[str]:
        """Return registered provider names in insertion order."""
        return self._registry.names()

    def find_provider(self, path: Path) -> BaseProvider:
        """Return the first provider compatible with *path*.

        Raises:
            ProviderNotFoundError: if no registered provider can handle *path*.
        """
        provider = self._registry.find(path)
        if provider is None:
            raise ProviderNotFoundError(
                f"No registered provider can load '{path}'. "
                f"Registered providers: {self.available_providers()}"
            )
        return provider

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    def load(self, path: Path) -> DisturbanceRecord:
        """Discover the compatible provider and return a DisturbanceRecord.

        Discovery and loading are separated so callers can inspect which
        provider was selected before committing to load (e.g., for UI prompts).

        Raises:
            ProviderNotFoundError: if no provider is compatible with *path*.
            ProviderLoadError: if the selected provider raises during load().
        """
        provider = self.find_provider(path)
        try:
            return provider.load(path)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderLoadError(
                f"Provider '{provider.provider_name}' failed to load '{path}': {exc}"
            ) from exc
