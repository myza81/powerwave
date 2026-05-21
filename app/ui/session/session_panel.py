"""SessionPanel — dockable Event Analysis Session workspace (Phase 9B/9C).

Phase 9C additions:
  - "Align All to Reference" toolbar button emits auto_align_requested("all").
  - offset_reset_requested signal re-emitted from source rows.
  - refresh_source_row / refresh_all pass alignment_notes to SourceRowWidget.

Architecture:
  SessionPanel(QDockWidget) owns a scrollable list of SourceRowWidgets.
  Each SourceRowWidget wraps one SessionSource and emits typed signals.
  SessionPanel re-emits those signals without modification so callers can
  connect directly to the panel rather than each row.

Performance guards:
  - add_source_row / remove_source_row target individual rows only.
  - refresh_all preserves expansion state across full redraws.
  - No full panel rebuild is triggered by single-source updates.
  - Signal blocking guards in SourceRowWidget prevent storms during refresh.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.sessions.session_models import PanelConfig, SessionSource, SourceQualityMetrics
from app.ui.session.source_row_widget import SourceRowWidget


class SessionPanel(QDockWidget):
    """Dockable workspace for the active EventAnalysisSession.

    Pure Qt view/controller — no session business logic.
    All mutations flow through signals; the caller (PowerwaveMainWindow)
    owns the EventAnalysisSession and acts on these signals.
    """

    # Signals required by spec (Phase 9B)
    source_add_requested = pyqtSignal()
    source_remove_requested = pyqtSignal(str)              # source_id
    offset_changed = pyqtSignal(str, float)                # source_id, offset_s
    auto_align_requested = pyqtSignal(str)                 # source_id or 'all'
    channel_visibility_changed = pyqtSignal(str, str, bool)
    channel_colour_change_requested = pyqtSignal(str, str)
    channel_panel_changed = pyqtSignal(str, str, str)
    session_cleared = pyqtSignal()

    # Phase 9C additions
    offset_reset_requested = pyqtSignal(str)               # source_id
    source_active_changed = pyqtSignal(str, bool)          # source_id, is_active

    # Phase 9E addition
    set_as_reference_requested = pyqtSignal(str)           # source_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Event Analysis Session", parent)
        self._source_rows: dict[str, SourceRowWidget] = {}
        # Expansion state is owned here to survive row removal/re-addition
        self._expansion_state: dict[str, bool] = {}
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def add_source_row(
        self,
        source: SessionSource,
        analog_channels: list,
        digital_channels: list,
        panels: list[PanelConfig],
        metrics: SourceQualityMetrics | None = None,
        alignment_notes: str = "",
    ) -> None:
        """Add a new source row. Safe to call while the panel is visible."""
        if source.source_id in self._source_rows:
            return
        row = SourceRowWidget(
            source,
            analog_channels,
            digital_channels,
            panels,
            metrics,
            parent=self._sources_widget,
        )
        # Apply notes to badge tooltip if provided
        if alignment_notes:
            row._apply_alignment(source, notes=alignment_notes)
        self._wire_row(row)
        # Insert before the trailing stretch (always the last item)
        insert_pos = max(0, self._sources_layout.count() - 1)
        self._sources_layout.insertWidget(insert_pos, row)
        self._source_rows[source.source_id] = row
        # Restore expansion state if we've seen this source before
        if self._expansion_state.get(source.source_id, False):
            row.set_expanded(True)

    def remove_source_row(self, source_id: str) -> None:
        """Remove a source row. Preserves expansion state for potential re-add."""
        row = self._source_rows.pop(source_id, None)
        if row is None:
            return
        self._expansion_state[source_id] = row.is_expanded()
        self._sources_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()

    def refresh_source_row(
        self,
        source_id: str,
        source: SessionSource,
        metrics: SourceQualityMetrics | None,
        panels: list[PanelConfig],
        alignment_notes: str = "",
    ) -> None:
        """Refresh a single source row. Does NOT rebuild the entire panel."""
        row = self._source_rows.get(source_id)
        if row is not None:
            row.refresh(source, metrics, panels, alignment_notes=alignment_notes)

    def refresh_all(self, session) -> None:
        """Full sync from EventAnalysisSession. Preserves expansion state."""
        panels = session.list_panels()
        session_ids = {s.source_id for s in session.list_sources()}

        # Remove rows no longer in the session
        stale = [sid for sid in self._source_rows if sid not in session_ids]
        for sid in stale:
            self.remove_source_row(sid)

        # Add new / refresh existing rows
        for source in session.list_sources():
            sid = source.source_id
            try:
                metrics: SourceQualityMetrics | None = (
                    session.get_source_quality_metrics(sid)
                )
            except Exception:  # noqa: BLE001
                metrics = None

            # Fetch alignment notes if the session supports it (Phase 9C)
            try:
                notes: str = session.get_alignment_notes(sid)
            except AttributeError:
                notes = ""

            analog_channels = [
                ch
                for ch in session.list_analog_channels(active_only=False)
                if ch.source_id == sid
            ]
            digital_channels = [
                ch
                for ch in session.list_digital_channels(active_only=False)
                if ch.source_id == sid
            ]

            if sid in self._source_rows:
                self.refresh_source_row(sid, source, metrics, panels, alignment_notes=notes)
            else:
                self.add_source_row(
                    source,
                    analog_channels,
                    digital_channels,
                    panels,
                    metrics,
                    alignment_notes=notes,
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ── Toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._add_btn = QPushButton("+ Add Source")
        self._add_btn.setToolTip("Add a new source to the session via Import Wizard")
        self._add_btn.clicked.connect(self.source_add_requested)
        toolbar.addWidget(self._add_btn)

        self._clear_btn = QPushButton("Clear Session")
        self._clear_btn.setToolTip("Remove all sources and start a new session")
        self._clear_btn.clicked.connect(self._on_clear_session)
        toolbar.addWidget(self._clear_btn)

        # Phase 9C: Align All to Reference
        self._align_all_btn = QPushButton("Align All")
        self._align_all_btn.setToolTip(
            "Auto-align all active sources by trigger detection.\n"
            "Each source's first step event is shifted to t=0."
        )
        self._align_all_btn.clicked.connect(
            lambda: self.auto_align_requested.emit("all")
        )
        toolbar.addWidget(self._align_all_btn)

        toolbar.addStretch()
        outer.addLayout(toolbar)

        # ── Scroll area containing source rows ──
        self._sources_widget = QWidget()
        self._sources_layout = QVBoxLayout(self._sources_widget)
        self._sources_layout.setContentsMargins(0, 0, 0, 0)
        self._sources_layout.setSpacing(4)
        self._sources_layout.addStretch()  # rows pushed upward

        scroll = QScrollArea()
        scroll.setWidget(self._sources_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        outer.addWidget(scroll)

        self.setWidget(container)
        self.setMinimumWidth(340)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _wire_row(self, row: SourceRowWidget) -> None:
        """Connect a source row's signals to this panel's re-emission signals."""
        row.offset_changed.connect(self.offset_changed)
        row.offset_reset_requested.connect(self.offset_reset_requested)
        row.auto_align_requested.connect(self.auto_align_requested)
        row.channel_visibility_changed.connect(self.channel_visibility_changed)
        row.channel_colour_change_requested.connect(
            self.channel_colour_change_requested
        )
        row.channel_panel_changed.connect(self.channel_panel_changed)
        row.remove_requested.connect(self.source_remove_requested)
        row.active_changed.connect(self.source_active_changed)
        row.set_as_reference_requested.connect(self.set_as_reference_requested)

    def _on_clear_session(self) -> None:
        source_ids = list(self._source_rows.keys())
        for sid in source_ids:
            self.remove_source_row(sid)
        self._expansion_state.clear()
        self.session_cleared.emit()
