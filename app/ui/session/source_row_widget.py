"""SourceRowWidget — collapsible per-source control row for the Session Panel.

Phase 9C additions over Phase 9B:
  - Reset button resets offset to 0.000 s via offset_reset_requested signal.
  - _apply_alignment() accepts alignment_notes for badge tooltip.
  - refresh() accepts alignment_notes kwarg passed through to badge.
  - auto-align tooltip no longer says "Phase 9C".

Signal contract (unchanged from Phase 9B except new offset_reset_requested):
  offset_changed(source_id, offset_s)       — spinbox changed by user
  offset_reset_requested(source_id)         — Reset button clicked
  auto_align_requested(source_id)           — Auto button clicked
  channel_visibility_changed(...)           — from channel tree
  channel_colour_change_requested(...)      — from channel tree
  channel_panel_changed(...)                — from channel tree
  remove_requested(source_id)              — Remove button clicked
  active_changed(source_id, is_active)      — active checkbox changed
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.sessions.session_models import PanelConfig, SessionSource, SourceQualityMetrics
from app.ui.session.channel_tree_widget import ChannelTreeWidget


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers (no Qt coupling)
# ─────────────────────────────────────────────────────────────────────────────


def _format_rate(hz: float) -> str:
    if hz <= 0:
        return "— Hz"
    if hz >= 1000:
        return f"{hz:,.0f} Hz"
    if hz >= 1:
        return f"{hz:.1f} Hz"
    return f"{hz:.4f} Hz"


def _quality_tooltip(metrics: SourceQualityMetrics) -> str:
    uniformity = "Uniform" if metrics.time_is_uniform else "Non-uniform (jitter detected)"
    return (
        f"Source: {metrics.source_id}\n"
        f"Samples: {metrics.sample_count:,}   "
        f"Rate: {metrics.inferred_sample_rate_hz:.2f} Hz   "
        f"Stability: {metrics.sample_rate_stability * 100:.0f}%\n"
        f"Missing: {metrics.missing_data_pct:.1f}%    "
        f"Duplicates: {metrics.duplicate_timestamp_pct:.1f}%\n"
        f"Interpolated: {metrics.interpolated_pct:.1f}%   "
        f"Resample ratio: {metrics.resampling_ratio:.4f}×\n"
        f"Time uniformity: {uniformity}"
    )


def _alignment_badge_text(source: SessionSource) -> str:
    method = source.alignment_method
    if method == "none":
        return ""
    conf = source.alignment_confidence
    if conf is not None:
        level = "High" if conf >= 0.75 else ("Medium" if conf >= 0.40 else "Low")
        return f"● {level} ({conf:.2f})"
    return f"● {method}"


def _alignment_badge_style(source: SessionSource) -> str:
    method = source.alignment_method
    if method == "none":
        return ""
    conf = source.alignment_confidence
    if conf is not None:
        if conf >= 0.75:
            return "color: #22aa44; font-weight: bold;"
        if conf >= 0.40:
            return "color: #cc8800; font-weight: bold;"
        return "color: #cc3333; font-weight: bold;"
    return "color: #888888;"


# ─────────────────────────────────────────────────────────────────────────────
# Widget
# ─────────────────────────────────────────────────────────────────────────────


class SourceRowWidget(QWidget):
    """Collapsible control row for one EventAnalysisSession source.

    Expansion state is not stored internally — the SessionPanel owns it via
    is_expanded() / set_expanded() to survive panel-level refresh cycles.
    """

    offset_changed = pyqtSignal(str, float)              # source_id, offset_s
    offset_reset_requested = pyqtSignal(str)             # source_id
    auto_align_requested = pyqtSignal(str)               # source_id
    set_as_reference_requested = pyqtSignal(str)         # source_id
    channel_visibility_changed = pyqtSignal(str, str, bool)
    channel_colour_change_requested = pyqtSignal(str, str)
    channel_panel_changed = pyqtSignal(str, str, str)
    remove_requested = pyqtSignal(str)                   # source_id
    active_changed = pyqtSignal(str, bool)               # source_id, is_active

    def __init__(
        self,
        source: SessionSource,
        analog_channels: list,
        digital_channels: list,
        panels: list[PanelConfig],
        metrics: SourceQualityMetrics | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source_id = source.source_id
        self._sample_interval_s: float = 0.001   # fallback; updated from metrics
        self._updating = False
        self._build_ui(source, analog_channels, digital_channels, panels)
        if metrics is not None:
            self._apply_metrics(metrics)
        self._apply_alignment(source)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def source_id(self) -> str:
        return self._source_id

    def is_expanded(self) -> bool:
        return self._expand_btn.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self._expand_btn.setChecked(expanded)
        self._channel_tree.setVisible(expanded)
        self._expand_btn.setText("Channels ▲" if expanded else "Channels ▼")

    def refresh(
        self,
        source: SessionSource,
        metrics: SourceQualityMetrics | None,
        panels: list[PanelConfig],
        alignment_notes: str = "",
    ) -> None:
        """Refresh display fields without rebuilding the widget tree."""
        self._updating = True
        try:
            self._name_label.setText(f"<b>{source.display_name}</b>")
            self._active_cb.setChecked(source.is_active)
            self._provider_badge.setText(source.provider_type.upper())
            self._offset_spin.setValue(source.time_offset_s)
        finally:
            self._updating = False
        self._apply_alignment(source, notes=alignment_notes)
        if metrics is not None:
            self._apply_metrics(metrics)
        self._channel_tree.update_panel_choices(panels)

    # ─────────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(
        self,
        source: SessionSource,
        analog_channels: list,
        digital_channels: list,
        panels: list[PanelConfig],
    ) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(2)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Raised)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(6, 5, 6, 5)
        frame_layout.setSpacing(4)

        frame_layout.addLayout(self._build_row1(source, analog_channels, digital_channels))
        frame_layout.addLayout(self._build_row2())
        frame_layout.addLayout(self._build_row3(source))
        frame_layout.addLayout(self._build_row4(source))

        outer.addWidget(frame)

        self._channel_tree = ChannelTreeWidget(source.source_id, parent=self)
        self._channel_tree.populate(analog_channels, digital_channels, panels)
        self._channel_tree.setVisible(False)
        self._channel_tree.channel_visibility_changed.connect(
            self.channel_visibility_changed
        )
        self._channel_tree.channel_colour_change_requested.connect(
            self.channel_colour_change_requested
        )
        self._channel_tree.channel_panel_changed.connect(self.channel_panel_changed)
        outer.addWidget(self._channel_tree)

    def _build_row1(
        self,
        source: SessionSource,
        analog_channels: list,
        digital_channels: list,
    ) -> QHBoxLayout:
        """Row 1: active checkbox, name, provider badge, counts, rate."""
        row = QHBoxLayout()
        row.setSpacing(6)

        self._active_cb = QCheckBox()
        self._active_cb.setChecked(source.is_active)
        self._active_cb.setToolTip("Include source in session canvas")
        self._active_cb.stateChanged.connect(self._on_active_changed)
        row.addWidget(self._active_cb)

        self._name_label = QLabel(f"<b>{source.display_name}</b>")
        self._name_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._name_label.setToolTip("Right-click for alignment options")
        self._name_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._name_label.customContextMenuRequested.connect(self._show_source_context_menu)
        row.addWidget(self._name_label)

        self._provider_badge = QLabel(source.provider_type.upper())
        self._provider_badge.setStyleSheet(
            "background: #3a5a8a; color: white; border-radius: 3px;"
            " padding: 1px 5px; font-size: 10px;"
        )
        row.addWidget(self._provider_badge)

        n_a = len(analog_channels)
        n_d = len(digital_channels)
        self._count_label = QLabel(f"{n_a}A + {n_d}D")
        self._count_label.setToolTip(
            f"{n_a} analog channel(s), {n_d} digital channel(s)"
        )
        row.addWidget(self._count_label)

        self._rate_label = QLabel("— Hz")
        self._rate_label.setToolTip("Inferred sample rate")
        row.addWidget(self._rate_label)

        return row

    def _build_row2(self) -> QHBoxLayout:
        """Row 2: quality bar + uniformity label + ⓘ tooltip button."""
        row = QHBoxLayout()
        row.setSpacing(5)

        row.addWidget(QLabel("Quality:"))

        self._quality_bar = QProgressBar()
        self._quality_bar.setRange(0, 100)
        self._quality_bar.setValue(0)
        self._quality_bar.setMaximumHeight(10)
        self._quality_bar.setMaximumWidth(90)
        self._quality_bar.setTextVisible(False)
        self._quality_bar.setToolTip("Sample rate stability (0–100%)")
        row.addWidget(self._quality_bar)

        self._uniformity_label = QLabel("")
        self._uniformity_label.setToolTip("Time-axis uniformity of this source")
        row.addWidget(self._uniformity_label)

        self._info_btn = QPushButton("ⓘ")
        self._info_btn.setFixedSize(20, 20)
        self._info_btn.setFlat(True)
        self._info_btn.setToolTip("No quality data yet")
        row.addWidget(self._info_btn)

        row.addStretch()
        return row

    def _build_row3(self, source: SessionSource) -> QHBoxLayout:
        """Row 3: offset spinbox, fine arrows, Reset, confidence badge, Auto."""
        row = QHBoxLayout()
        row.setSpacing(4)

        row.addWidget(QLabel("Offset:"))

        self._fine_left_btn = QPushButton("←")
        self._fine_left_btn.setFixedWidth(24)
        self._fine_left_btn.setToolTip("Nudge left by one native sample interval")
        self._fine_left_btn.clicked.connect(self._on_fine_left)
        row.addWidget(self._fine_left_btn)

        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setRange(-9999.999, 9999.999)
        self._offset_spin.setDecimals(3)
        self._offset_spin.setSingleStep(0.001)
        self._offset_spin.setSuffix(" s")
        self._offset_spin.setFixedWidth(115)
        self._offset_spin.setKeyboardTracking(True)
        self._offset_spin.setValue(source.time_offset_s)
        self._offset_spin.valueChanged.connect(self._on_offset_spin_changed)
        row.addWidget(self._offset_spin)

        self._fine_right_btn = QPushButton("→")
        self._fine_right_btn.setFixedWidth(24)
        self._fine_right_btn.setToolTip("Nudge right by one native sample interval")
        self._fine_right_btn.clicked.connect(self._on_fine_right)
        row.addWidget(self._fine_right_btn)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setFixedWidth(44)
        self._reset_btn.setToolTip("Reset offset to 0.000 s and clear alignment method")
        self._reset_btn.clicked.connect(
            lambda: self.offset_reset_requested.emit(self._source_id)
        )
        row.addWidget(self._reset_btn)

        self._alignment_badge = QLabel("")
        self._alignment_badge.setToolTip("Alignment method and confidence")
        row.addWidget(self._alignment_badge)

        self._auto_align_btn = QPushButton("Auto")
        self._auto_align_btn.setFixedWidth(40)
        self._auto_align_btn.setToolTip(
            "Auto-align by trigger detection.\n"
            "Shifts this source so its first step event lands at t=0."
        )
        self._auto_align_btn.clicked.connect(
            lambda: self.auto_align_requested.emit(self._source_id)
        )
        row.addWidget(self._auto_align_btn)

        row.addStretch()
        return row

    def _build_row4(self, source: SessionSource) -> QHBoxLayout:
        """Row 4: expand/collapse channels toggle + remove button."""
        row = QHBoxLayout()
        row.setSpacing(4)

        self._expand_btn = QPushButton("Channels ▼")
        self._expand_btn.setCheckable(True)
        self._expand_btn.setChecked(False)
        self._expand_btn.clicked.connect(self._on_expand_toggled)
        row.addWidget(self._expand_btn)

        row.addStretch()

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setToolTip(f"Remove {source.display_name} from session")
        self._remove_btn.clicked.connect(
            lambda: self.remove_requested.emit(self._source_id)
        )
        row.addWidget(self._remove_btn)

        return row

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_metrics(self, metrics: SourceQualityMetrics) -> None:
        self._quality_bar.setValue(int(metrics.sample_rate_stability * 100))
        self._rate_label.setText(_format_rate(metrics.inferred_sample_rate_hz))
        if metrics.inferred_sample_rate_hz > 0:
            self._sample_interval_s = 1.0 / metrics.inferred_sample_rate_hz
        self._uniformity_label.setText(
            "Uniform" if metrics.time_is_uniform else "Non-uniform"
        )
        self._info_btn.setToolTip(_quality_tooltip(metrics))

    def _apply_alignment(self, source: SessionSource, notes: str = "") -> None:
        self._alignment_badge.setText(_alignment_badge_text(source))
        self._alignment_badge.setStyleSheet(_alignment_badge_style(source))
        tooltip = notes if notes else "Alignment method and confidence"
        self._alignment_badge.setToolTip(tooltip)

    def _on_active_changed(self, _state) -> None:
        if self._updating:
            return
        self.active_changed.emit(self._source_id, self._active_cb.isChecked())

    def _on_offset_spin_changed(self, value: float) -> None:
        if self._updating:
            return
        self.offset_changed.emit(self._source_id, value)

    def _on_fine_left(self) -> None:
        step = self._sample_interval_s if self._sample_interval_s > 0 else 0.001
        self._offset_spin.setValue(self._offset_spin.value() - step)

    def _on_fine_right(self) -> None:
        step = self._sample_interval_s if self._sample_interval_s > 0 else 0.001
        self._offset_spin.setValue(self._offset_spin.value() + step)

    def _on_expand_toggled(self, checked: bool) -> None:
        self._channel_tree.setVisible(checked)
        self._expand_btn.setText("Channels ▲" if checked else "Channels ▼")

    def _show_source_context_menu(self, pos: QPoint) -> None:
        """Right-click context menu on the source name label."""
        menu = QMenu(self)
        ref_act = menu.addAction("Set as Reference")
        ref_act.setToolTip(
            "Set this source's offset to 0.0 s;\n"
            "adjust all other sources relative to it."
        )
        chosen = menu.exec(self._name_label.mapToGlobal(pos))
        if chosen == ref_act:
            self.set_as_reference_requested.emit(self._source_id)
