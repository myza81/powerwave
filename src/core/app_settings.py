"""
src/core/app_settings.py

Application-wide settings singleton for PowerWave Analyst.

Persists user preferences to ~/.powerwave_analyst/settings.json.
Provides typed accessors for all configurable values.

Architecture: Core layer — no UI imports (LAW 1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ── Default values ─────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, dict[str, Any]] = {
    'calculation': {
        'nominal_frequency':             50,   # Hz — 50 or 60
        'rms_tolerance_ms':              10.0, # ms — RMS merger tolerance
        'pu_yrange':                     2.0,  # pu — symmetric ±range for PU Y-axis
        'comtrade_tz_offset_h':          0,    # UTC offset of COMTRADE timestamps (0=UTC, 8=MYT/SGT)
        'timestamp_grouping_threshold_h': 1.0, # h — files within this window share a canvas group
    },
    'display': {
        'theme':            'dark',
        'cursor_c1_colour': '#FFD700',   # C1 gold
        'cursor_c2_colour': '#00E5FF',   # C2 cyan
        'panel_font_size':  9,           # pt — Measurements panel font size
    },
    'pmu': {
        'default_timezone': 'SGT (UTC+8)',
    },
}

_CONFIG_DIR:  Path = Path.home() / '.powerwave_analyst'
_CONFIG_FILE: Path = _CONFIG_DIR / 'settings.json'


# ── Singleton ──────────────────────────────────────────────────────────────────

class AppSettings:
    """Application-wide settings singleton.

    All access is through class-methods — no instantiation needed.

    Example::

        from core.app_settings import AppSettings

        freq = AppSettings.get('calculation.nominal_frequency')   # → 50
        AppSettings.set('display.theme', 'light')
        AppSettings.save()
    """

    _data:   dict[str, dict[str, Any]] = {}
    _loaded: bool = False

    # ── Internal helpers ───────────────────────────────────────────────────────

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Load settings from disk on first access (lazy initialisation)."""
        if cls._loaded:
            return
        # Start from a clean copy of defaults
        cls._data = {
            section: dict(values)
            for section, values in _DEFAULTS.items()
        }
        # Overlay with on-disk values (unknown keys are silently ignored)
        if _CONFIG_FILE.exists():
            try:
                with _CONFIG_FILE.open('r', encoding='utf-8') as fh:
                    on_disk: dict = json.load(fh)
                for section, values in on_disk.items():
                    if section in cls._data and isinstance(values, dict):
                        cls._data[section].update(values)
            except (json.JSONDecodeError, OSError):
                pass   # corrupted file — silently fall back to defaults
        cls._loaded = True

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Return the setting at ``section.key``.

        Args:
            key:     Dot-separated path, e.g. ``'calculation.nominal_frequency'``.
            default: Value returned when the key does not exist.

        Returns:
            The stored value, or ``default`` if absent.
        """
        cls._ensure_loaded()
        parts = key.split('.', 1)
        if len(parts) != 2:
            return default
        section, name = parts
        return cls._data.get(section, {}).get(name, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Store a value at ``section.key`` in memory.

        Call :meth:`save` afterwards to persist to disk.

        Args:
            key:   Dot-separated path, e.g. ``'display.theme'``.
            value: New value.
        """
        cls._ensure_loaded()
        parts = key.split('.', 1)
        if len(parts) != 2:
            return
        section, name = parts
        if section not in cls._data:
            cls._data[section] = {}
        cls._data[section][name] = value

    @classmethod
    def save(cls) -> None:
        """Write current in-memory settings to ``~/.powerwave_analyst/settings.json``."""
        cls._ensure_loaded()
        try:
            _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with _CONFIG_FILE.open('w', encoding='utf-8') as fh:
                json.dump(cls._data, fh, indent=2)
        except OSError:
            pass   # non-fatal — settings just won't persist this session

    @classmethod
    def reset_to_defaults(cls) -> None:
        """Reset all settings to factory defaults in memory (does not save)."""
        cls._data = {
            section: dict(values)
            for section, values in _DEFAULTS.items()
        }

    @classmethod
    def snapshot(cls) -> dict[str, dict[str, Any]]:
        """Return a deep copy of all settings (used to populate the dialog).

        Returns:
            ``{section: {key: value, ...}, ...}``
        """
        cls._ensure_loaded()
        return {s: dict(v) for s, v in cls._data.items()}

    @classmethod
    def apply_snapshot(cls, data: dict[str, dict[str, Any]]) -> None:
        """Replace in-memory settings with ``data`` and persist to disk.

        Args:
            data: A ``{section: {key: value}}`` dict (from a modified snapshot).
        """
        cls._ensure_loaded()
        for section, values in data.items():
            if section in cls._data and isinstance(values, dict):
                cls._data[section].update(values)
        cls.save()
