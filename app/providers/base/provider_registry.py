from __future__ import annotations

from pathlib import Path

from app.providers.base.base_provider import BaseProvider
from app.providers.base.exceptions import DuplicateProviderError, ProviderError


class ProviderRegistry:
    """Ordered collection of registered providers.

    Maintains insertion order so that ``find()`` returns the first compatible
    provider in registration sequence — enabling priority-based resolution.

    Raises ``DuplicateProviderError`` when the same provider_name is registered
    twice.  Raises ``ProviderError`` when trying to remove a name that is not
    registered.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    # ------------------------------------------------------------------ #
    # Mutation                                                             #
    # ------------------------------------------------------------------ #

    def register(self, provider: BaseProvider) -> None:
        """Add *provider* to the registry.

        Raises:
            DuplicateProviderError: if provider_name is already registered.
        """
        if provider.provider_name in self._providers:
            raise DuplicateProviderError(
                f"Provider '{provider.provider_name}' is already registered."
            )
        self._providers[provider.provider_name] = provider

    def unregister(self, name: str) -> None:
        """Remove the provider registered under *name*.

        Raises:
            ProviderError: if *name* is not currently registered.
        """
        if name not in self._providers:
            raise ProviderError(f"Provider '{name}' is not registered.")
        del self._providers[name]

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def find(self, path: Path) -> BaseProvider | None:
        """Return the first registered provider whose can_load() accepts *path*.

        Returns None if no compatible provider exists.
        """
        for provider in self._providers.values():
            if provider.can_load(path):
                return provider
        return None

    def names(self) -> list[str]:
        """Return provider names in registration order."""
        return list(self._providers.keys())

    def get_all(self) -> list[BaseProvider]:
        """Return all registered providers in registration order."""
        return list(self._providers.values())

    def __len__(self) -> int:
        return len(self._providers)

    def __contains__(self, name: str) -> bool:
        return name in self._providers
