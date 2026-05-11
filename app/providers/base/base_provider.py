from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.models import DisturbanceRecord


class BaseProvider(ABC):
    """Abstract contract that every ingestion provider must satisfy.

    Subclasses must:
    - Override ``provider_name`` with a unique lowercase identifier.
    - Implement ``can_load()`` to report compatibility with a file path.
    - Implement ``load()`` to parse the file and return a DisturbanceRecord.

    Providers must remain isolated from UI, visualization, and analytics systems.
    They must never return parser-specific structures — only DisturbanceRecord.
    """

    provider_name: str = "base"

    @abstractmethod
    def can_load(self, path: Path) -> bool:
        """Return True if this provider can handle the given file path."""

    @abstractmethod
    def load(self, path: Path) -> DisturbanceRecord:
        """Parse the file at *path* and return a normalized DisturbanceRecord."""
