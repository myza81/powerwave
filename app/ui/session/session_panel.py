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


def _fmt_readout(value: float, unit: str) -> str:
    """Format a numeric readout value with SI magnitude prefix (K / M)."""
    abs_v = abs(value)
    suffix = f" {unit}" if unit else ""
    if abs_v >= 1_000_000:
        return f"{value / 1_000_000:.3f} M{unit}"
    if abs_v >= 1_000:
        return f"{value / 1_000:.3f} K{unit}"
    return f"{value:.3f}{suffix}"
from PyQt6.QtWidgets import (
    QDockWidget,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.sessions.session_models import PanelConfig, SessionSource, SourceQualityMetrics
from app.sessions.timing_compatibility import (
    SessionTimingAssessment,
    TimingCompatibilityLevel,
    assess_session_timing_compatibility,
)
from app.ui.dialogs import confirm_destructive_action
from app.ui.session.calculated_signal_row_widget import CalculatedSignalRowWidget
from app.ui.session.source_row_widget import SourceRowWidget
from app.ui.session.timing_details_dialog import TimingDetailsDialog


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
    offset_edit_finished = pyqtSignal(str)                  # source_id (Phase 3B)
    auto_align_requested = pyqtSignal(str)                 # source_id or 'all'
    channel_visibility_changed = pyqtSignal(str, str, bool)
    channel_colour_change_requested = pyqtSignal(str, str)
    channel_panel_changed = pyqtSignal(str, str, str)
    new_panel_requested = pyqtSignal(str, str)           # source_id, ch_name
    session_cleared = pyqtSignal()

    # Phase 9C additions
    offset_reset_requested = pyqtSignal(str)               # source_id
    source_active_changed = pyqtSignal(str, bool)          # source_id, is_active

    # Phase 9E addition
    set_as_reference_requested = pyqtSignal(str)           # source_id

    # Phase 3B additions — Calculated Signals section
    calculated_signal_visibility_changed = pyqtSignal(str, bool)     # calc_id, visible
    calculated_signal_recalculate_requested = pyqtSignal(str)        # calc_id
    calculated_signal_recalculate_all_requested = pyqtSignal()
    calculated_signal_delete_requested = pyqtSignal(str)              # calc_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Event Analysis Session", parent)
        self._source_rows: dict[str, SourceRowWidget] = {}
        # Expansion state is owned here to survive row removal/re-addition
        self._expansion_state: dict[str, bool] = {}
        self._calc_rows: dict[str, CalculatedSignalRowWidget] = {}
        self._timing_assessment: SessionTimingAssessment | None = None
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

    def update_channel_colour(
        self, source_id: str, channel_name: str, colour_hex: str
    ) -> None:
        """Update the colour swatch for one channel without rebuilding any row."""
        row = self._source_rows.get(source_id)
        if row is not None:
            row.update_channel_colour(channel_name, colour_hex)

    def update_channel_panel(
        self, source_id: str, channel_name: str, panel_id: str
    ) -> None:
        """Select a specific panel in a channel's combo (e.g. after new panel created)."""
        row = self._source_rows.get(source_id)
        if row is not None:
            row.update_channel_panel(channel_name, panel_id)

    def update_crosshair_readouts(self, t: float, values: list) -> None:
        """Update the Readout column from a crosshair hover event.

        *values* is a list of (source_id, channel_name, label, value, unit, color) tuples
        emitted by SessionCanvasWidget.crosshair_values_changed.
        """
        if abs(t) < 1.0:
            self._time_lbl.setText(f"t = {t * 1000:,.3f} ms")
        else:
            self._time_lbl.setText(f"t = {t:,.3f} s")
        for source_id, channel_name, _label, value, unit, _color in values:
            row = self._source_rows.get(source_id)
            if row is None:
                continue
            if value is None:
                row.set_channel_readout(channel_name, "—")
            elif isinstance(value, str):
                row.set_channel_readout(channel_name, value)
            else:
                row.set_channel_readout(channel_name, _fmt_readout(value, unit))

    def clear_crosshair_readouts(self) -> None:
        """Clear all Readout column values across every source row."""
        self._time_lbl.setText("t = —")
        for row in self._source_rows.values():
            row.clear_readouts()

    def refresh_all_panel_choices(self, panels) -> None:
        """Refresh panel dropdowns in every source row after panels change."""
        for row in self._source_rows.values():
            row._channel_tree.update_panel_choices(panels)

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

    def refresh_calculated_signals(self, session) -> None:
        """Full sync of the Calculated Signals section from EventAnalysisSession.

        Session-owned display state only -- never triggers a calculation.
        Adds rows for new entries, refreshes existing ones in place, and
        removes rows for calc_ids no longer in the session (deleted).
        """
        entries = {e.definition.calc_id: e for e in session.list_calculated_signals()}

        stale_ids = [cid for cid in self._calc_rows if cid not in entries]
        for cid in stale_ids:
            row = self._calc_rows.pop(cid)
            self._calc_signals_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()

        for calc_id, entry in entries.items():
            if calc_id not in self._calc_rows:
                row = CalculatedSignalRowWidget(calc_id, parent=self._calc_signals_widget)
                self._wire_calc_row(row)
                insert_pos = max(0, self._calc_signals_layout.count() - 1)
                self._calc_signals_layout.insertWidget(insert_pos, row)
                self._calc_rows[calc_id] = row
            self._calc_rows[calc_id].refresh(
                entry.definition.name, entry.result, entry.is_visible
            )

        self._calc_signals_group.setVisible(bool(entries))

    def refresh_timing_assessment(self, session) -> None:
        """Recompute and display the session-wide timing-reference
        compatibility banner (Sprint 1B). Read-only: assess_session_timing_
        compatibility() never mutates the session, any source, or any
        record -- this method only updates the banner's own display state.
        """
        assessment = assess_session_timing_compatibility(session)
        self._timing_assessment = assessment

        if not assessment.has_warning:
            self._timing_banner.setVisible(False)
            return

        icon = "⚠" if assessment.level is not TimingCompatibilityLevel.UNKNOWN else "❔"
        self._timing_banner_label.setText(f"{icon} {assessment.summary}")
        self._timing_banner.setVisible(True)

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

        self._time_lbl = QLabel("t = —")
        self._time_lbl.setStyleSheet(
            "color: #888888; font-family: Menlo, Consolas, 'Courier New'; font-size: 11px;"
        )
        self._time_lbl.setToolTip("Current crosshair time")
        toolbar.addWidget(self._time_lbl)

        outer.addLayout(toolbar)

        # ── Timing-reference compatibility warning (Sprint 1B) ──
        # Persistent, not a transient status-bar message or a Python
        # warnings.warn() -- stays visible for as long as the underlying
        # condition exists. Hidden by default; refresh_timing_assessment()
        # shows it only when the session's active sources' timing
        # references require review.
        self._timing_banner = QFrame()
        self._timing_banner.setFrameShape(QFrame.Shape.StyledPanel)
        self._timing_banner.setStyleSheet(
            "QFrame { background: #4a3b1a; border: 1px solid #b06000; border-radius: 4px; }"
        )
        timing_banner_layout = QHBoxLayout(self._timing_banner)
        timing_banner_layout.setContentsMargins(8, 4, 8, 4)
        self._timing_banner_label = QLabel("")
        self._timing_banner_label.setWordWrap(True)
        self._timing_banner_label.setStyleSheet("color: #FFD27F; font-size: 11px;")
        timing_banner_layout.addWidget(self._timing_banner_label, stretch=1)
        self._timing_details_btn = QPushButton("Details")
        self._timing_details_btn.setToolTip(
            "Show each source's timing reference, current session offsets, "
            "and why this warning appeared"
        )
        self._timing_details_btn.clicked.connect(self._on_timing_details_clicked)
        timing_banner_layout.addWidget(self._timing_details_btn)
        self._timing_banner.setVisible(False)
        outer.addWidget(self._timing_banner)

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

        # ── Calculated Signals section (Phase 3B) ──
        # A dedicated section, not a fake SourceRowWidget: calculated
        # signals are session-owned derived curves, never SessionSources.
        self._calc_signals_group = QGroupBox("Calculated Signals")
        calc_outer = QVBoxLayout(self._calc_signals_group)
        calc_outer.setContentsMargins(4, 4, 4, 4)
        calc_outer.setSpacing(4)

        calc_toolbar = QHBoxLayout()
        calc_toolbar.addStretch()
        self._recalc_all_btn = QPushButton("Recalculate All")
        self._recalc_all_btn.setToolTip("Recalculate every calculated signal in this session")
        self._recalc_all_btn.clicked.connect(
            self.calculated_signal_recalculate_all_requested
        )
        calc_toolbar.addWidget(self._recalc_all_btn)
        calc_outer.addLayout(calc_toolbar)

        self._calc_signals_widget = QWidget()
        self._calc_signals_layout = QVBoxLayout(self._calc_signals_widget)
        self._calc_signals_layout.setContentsMargins(0, 0, 0, 0)
        self._calc_signals_layout.setSpacing(2)
        self._calc_signals_layout.addStretch()
        calc_outer.addWidget(self._calc_signals_widget)

        self._calc_signals_group.setVisible(False)  # hidden until a signal exists
        outer.addWidget(self._calc_signals_group)

        self.setWidget(container)
        self.setMinimumWidth(340)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _wire_calc_row(self, row: CalculatedSignalRowWidget) -> None:
        row.visibility_changed.connect(self.calculated_signal_visibility_changed)
        row.recalculate_requested.connect(self.calculated_signal_recalculate_requested)
        row.delete_requested.connect(self.calculated_signal_delete_requested)

    def _wire_row(self, row: SourceRowWidget) -> None:
        """Connect a source row's signals to this panel's re-emission signals."""
        row.offset_changed.connect(self.offset_changed)
        row.offset_edit_finished.connect(self.offset_edit_finished)
        row.offset_reset_requested.connect(self.offset_reset_requested)
        row.auto_align_requested.connect(self.auto_align_requested)
        row.channel_visibility_changed.connect(self.channel_visibility_changed)
        row.channel_colour_change_requested.connect(
            self.channel_colour_change_requested
        )
        row.channel_panel_changed.connect(self.channel_panel_changed)
        row.new_panel_requested.connect(self.new_panel_requested)
        row.remove_requested.connect(self.source_remove_requested)
        row.active_changed.connect(self.source_active_changed)
        row.set_as_reference_requested.connect(self.set_as_reference_requested)

    def _on_timing_details_clicked(self) -> None:
        if self._timing_assessment is None:
            return
        dlg = TimingDetailsDialog(self._timing_assessment, parent=self)
        dlg.exec()

    def _on_clear_session(self) -> None:
        if self._source_rows or self._calc_rows:
            if not confirm_destructive_action(
                self,
                title="Clear current session?",
                message=(
                    "All loaded sources, alignment settings, calculated "
                    "signals, and layout changes in this session will be "
                    "removed.\n\n"
                    "This action cannot be undone."
                ),
            ):
                return
        source_ids = list(self._source_rows.keys())
        for sid in source_ids:
            self.remove_source_row(sid)
        for calc_id in list(self._calc_rows.keys()):
            row = self._calc_rows.pop(calc_id)
            self._calc_signals_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._calc_signals_group.setVisible(False)
        self._expansion_state.clear()
        self.session_cleared.emit()
