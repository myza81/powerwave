"""
src/core/app_state.py

Central application signal hub.  A single module-level instance (``app_state``)
is imported wherever signals need to be emitted or connected.

Architecture: Cross-cutting infrastructure — may be imported by any layer.
              Never import from ui/ here (no circular upward dependency).
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class AppState(QObject):
    """Central signal hub for inter-layer communication (LAW 6).

    All cross-layer events flow through this singleton.  ui/ emits and
    connects here; engine/ and parsers/ never touch it directly.
    """

    # Emitted by MainWindow after a file is parsed; carries the
    # DisturbanceRecord so WaveformPanel and ChannelCanvas can update.
    record_loaded = pyqtSignal(object)          # object = DisturbanceRecord

    # Emitted when a measurement cursor moves; receivers update readouts.
    cursor_moved = pyqtSignal(int, float)       # (cursor_id, time_seconds)

    # Emitted when the user toggles a channel's visibility checkbox.
    channel_toggled = pyqtSignal(int, bool)     # (channel_id, visible)

    # Emitted when the user switches view mode (instantaneous / RMS overlays).
    view_mode_changed = pyqtSignal(str)         # 'instantaneous'|'rms_half'|'rms_full'


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this instance wherever signals must be emitted or connected:
#   from src.core.app_state import app_state
app_state: AppState = AppState()
