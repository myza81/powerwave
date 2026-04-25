"""
src/ui/theme_palette.py

Colour palettes for PowerWave Analyst's dark and light themes.

Usage::

    from ui.theme_palette import current, get

    p = current()                  # palette for the saved theme
    p = get('light')               # explicit theme
    bg = p['bg_toolbar']           # '#E6E6F0' (light) or '#2D2D2D' (dark)

Architecture: Presentation layer (ui/) — reads core.AppSettings only.
"""

from __future__ import annotations

from core.app_settings import AppSettings

# ── Colour dictionaries ────────────────────────────────────────────────────────

DARK: dict[str, str] = {
    'bg_app':      '#1E1E2A',   # main window
    'bg_panel':    '#1E1E1E',   # panels / tables
    'bg_canvas':   '#1E1E1E',   # waveform plot area
    'bg_toolbar':  '#2D2D2D',   # toolbars / button bars
    'bg_sidebar':  '#252535',   # menus / nav lists / side panels
    'bg_input':    '#2A2A3A',   # spin-boxes / combos / line-edits
    'bg_item':     '#252525',   # tree / list widget background
    'bg_hover':    '#2A2A4A',   # item hover (also used as Δ-row highlight)
    'bg_selected': '#3A3A5A',   # selected item background
    'bg_header':   '#3A3A3A',   # column / section headers
    'bg_dialog':   '#1E1E2E',   # dialogs
    'bg_scroll':   '#222222',   # scroll-area viewport fill
    'text':        '#CCCCCC',   # standard text
    'text_bright': '#FFFFFF',   # high-emphasis text / selected
    'text_dim':    '#AAAAAA',   # secondary / placeholder text
    'text_input':  '#EEEEEE',   # text inside input widgets
    'text_accent': '#AADDFF',   # section-header label colour
    'border':      '#555566',   # main widget borders
    'border_dim':  '#3A3A4A',   # subtle / divider borders
    'accent':      '#6688FF',   # focus highlight / accent chrome
    'sep':         '#555555',   # inline separator widget background
    'sep_line':    '#444444',   # QFrame HLine colour
}

LIGHT: dict[str, str] = {
    'bg_app':      '#F2F2F8',
    'bg_panel':    '#FFFFFF',
    'bg_canvas':   '#F5F5FF',   # waveform plot area (light)
    'bg_toolbar':  '#E6E6F0',
    'bg_sidebar':  '#EAEAF4',
    'bg_input':    '#FFFFFF',
    'bg_item':     '#F5F5FC',
    'bg_hover':    '#D8D8EE',
    'bg_selected': '#3A3A5A',   # keep dark-blue so white text stays readable
    'bg_header':   '#DCDCEC',
    'bg_dialog':   '#F4F4FA',
    'bg_scroll':   '#EBEBF5',
    'text':        '#2A2A3A',
    'text_bright': '#111111',
    'text_dim':    '#666677',
    'text_input':  '#111111',
    'text_accent': '#224499',
    'border':      '#AAAACC',
    'border_dim':  '#CCCCDD',
    'accent':      '#3355BB',
    'sep':         '#AAAAAA',
    'sep_line':    '#CCCCCC',
}


# ── Public helpers ─────────────────────────────────────────────────────────────

def get(theme: str) -> dict[str, str]:
    """Return the colour palette for *theme* (``'dark'`` or ``'light'``)."""
    return LIGHT if theme == 'light' else DARK


def current() -> dict[str, str]:
    """Return the palette for the currently saved theme (reads AppSettings)."""
    return get(AppSettings.get('display.theme', 'dark'))
