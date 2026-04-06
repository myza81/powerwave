"""
src/utils/channel_ordering.py

Shared channel row ordering for LabelPanel and ChannelCanvas.

Provides:
  - Row height constants (single source of truth for both panels)
  - get_ordered_rows() — builds the canonical row list used by both panels

Row order within each bay: analogue channels first, digital channels last.
Bay headers are emitted only for non-empty bay names.
Only visible channels with data are included.

Architecture: Shared utility — imports models/ only (LAW 1).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Union

from models.channel import AnalogueChannel, DigitalChannel
from models.disturbance_record import DisturbanceRecord

# ── Row height constants (single source of truth) ─────────────────────────────

ROW_HEIGHT_BAY_HEADER: int = 20   # px — bay group header / canvas spacer
ROW_HEIGHT_ANALOGUE: int   = 80   # px — analogue waveform row
ROW_HEIGHT_DIGITAL: int    = 12   # px — compact digital channel row

# ── Row type aliases ──────────────────────────────────────────────────────────

BayHeaderRow = dict   # {'type': 'bay_header', 'bay_name': str}
AnalogueRow  = dict   # {'type': 'analogue',   'channel': AnalogueChannel}
DigitalRow   = dict   # {'type': 'digital',    'channel': DigitalChannel}
ChannelRow   = Union[BayHeaderRow, AnalogueRow, DigitalRow]


def get_ordered_rows(record: DisturbanceRecord) -> list[ChannelRow]:
    """Return the canonical ordered row list for LabelPanel and ChannelCanvas.

    Order within each bay: analogue channels first, digital channels last.
    Bay headers are emitted only when bay_name is non-empty.
    Only visible channels with data are included.

    Bay order is determined by first appearance across analogue channels,
    then digital-only bays are appended.

    Args:
        record: The DisturbanceRecord to build rows for.

    Returns:
        List of row dicts, each with a 'type' key:
          {'type': 'bay_header', 'bay_name': str}
          {'type': 'analogue',   'channel': AnalogueChannel}
          {'type': 'digital',    'channel': DigitalChannel}
    """
    bay_analogue: dict[str, list[AnalogueChannel]] = defaultdict(list)
    bay_digital:  dict[str, list[DigitalChannel]]  = defaultdict(list)
    bay_order:    list[str]                         = []

    for ch in record.analogue_channels:
        if not ch.visible or not ch.raw_data.size:
            continue
        bay = ch.bay_name or ''
        if bay not in bay_order:
            bay_order.append(bay)
        bay_analogue[bay].append(ch)

    for ch in record.digital_channels:
        if not ch.visible or not ch.data.size:
            continue
        bay = ch.bay_name or ''
        if bay not in bay_order:
            bay_order.append(bay)
        bay_digital[bay].append(ch)

    rows: list[ChannelRow] = []

    for bay in bay_order:
        a_chs = bay_analogue[bay]
        d_chs = bay_digital[bay]

        if not a_chs and not d_chs:
            continue

        # Emit bay header only for named bays
        if bay:
            rows.append({'type': 'bay_header', 'bay_name': bay})

        for ch in a_chs:
            rows.append({'type': 'analogue', 'channel': ch})

        for ch in d_chs:
            rows.append({'type': 'digital', 'channel': ch})

    return rows
