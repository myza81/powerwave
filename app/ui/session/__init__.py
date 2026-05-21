"""Session workspace UI package — Phase 9B/9D/9E."""
from __future__ import annotations

from app.ui.session.channel_tree_widget import ChannelTreeWidget
from app.ui.session.legend_widget import ChannelLegendWidget, generate_source_variant_colour
from app.ui.session.session_canvas_controller import SessionCanvasController
from app.ui.session.session_panel import SessionPanel
from app.ui.session.source_row_widget import SourceRowWidget

__all__ = [
    "ChannelLegendWidget",
    "ChannelTreeWidget",
    "SessionCanvasController",
    "SessionPanel",
    "SourceRowWidget",
    "generate_source_variant_colour",
]
