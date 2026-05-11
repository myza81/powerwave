from __future__ import annotations


class ProviderError(Exception):
    """Base class for all provider-layer errors."""


class ProviderNotFoundError(ProviderError):
    """No registered provider is compatible with the given path."""


class ProviderLoadError(ProviderError):
    """A provider raised an unexpected exception during load()."""


class DuplicateProviderError(ProviderError):
    """A provider with the same name is already registered."""
