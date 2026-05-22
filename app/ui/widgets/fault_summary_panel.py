"""Fault characterisation summary panel.

A compact dock strip showing the classified fault type, involved phases,
and symmetrical component ratios computed at the fault midpoint.

Layout (two rows inside a single frame):
  Row 1: Fault type label + description + severity colour
  Row 2: Phase indicators (A/B/C circles) | V₁ V₂ V₀ in pu | Unbalance %

Call ``load_fault(result)`` after classification completes.
Call ``clear_fault()`` when a record is unloaded or no fault found.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.analytics.fault.fault_classifier import FaultCharacterisation, FaultType


class FaultSummaryPanel(QWidget):
    """Compact fault characterisation readout.

    Public API:
        load_fault(result: FaultCharacterisation) — populate panel
        clear_fault()                              — reset to idle
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FaultSummaryPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("QFrame { background: #1A1A1A; border-top: 1px solid #333333; }")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(3)

        # ── Row 1: Fault type heading ─────────────────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        self._lbl_type = QLabel("No fault characterised")
        self._lbl_type.setStyleSheet(
            "color: #888888; font-size: 12px; font-weight: bold; background: transparent;"
        )
        row1.addWidget(self._lbl_type)

        self._lbl_desc = QLabel("")
        self._lbl_desc.setStyleSheet("color: #AAAAAA; font-size: 11px; background: transparent;")
        row1.addWidget(self._lbl_desc)
        row1.addStretch()

        self._lbl_time = QLabel("")
        self._lbl_time.setStyleSheet("color: #666666; font-size: 10px; background: transparent;")
        row1.addWidget(self._lbl_time)
        outer.addLayout(row1)

        # ── Row 2: Phase indicators + sequence values ─────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(12)

        # Phase circles: A / B / C
        phase_frame = QWidget()
        phase_frame.setStyleSheet("background: transparent;")
        phase_layout = QHBoxLayout(phase_frame)
        phase_layout.setContentsMargins(0, 0, 0, 0)
        phase_layout.setSpacing(6)

        self._phase_labels: dict[str, QLabel] = {}
        for ph in ("A", "B", "C"):
            lbl = QLabel(f"● {ph}")
            lbl.setStyleSheet("color: #555555; font-size: 13px; background: transparent;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            phase_layout.addWidget(lbl)
            self._phase_labels[ph] = lbl

        self._lbl_ground = QLabel("")
        self._lbl_ground.setStyleSheet("color: #888888; font-size: 12px; background: transparent;")
        phase_layout.addWidget(self._lbl_ground)

        row2.addWidget(phase_frame)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setStyleSheet("color: #3C3C3C;")
        divider.setFixedWidth(1)
        row2.addWidget(divider)

        # Sequence component labels
        seq_frame = QWidget()
        seq_frame.setStyleSheet("background: transparent;")
        seq_layout = QHBoxLayout(seq_frame)
        seq_layout.setContentsMargins(0, 0, 0, 0)
        seq_layout.setSpacing(16)

        self._lbl_v1 = self._make_seq_label("V₁:", "1.00 pu")
        self._lbl_v2 = self._make_seq_label("V₂:", "—")
        self._lbl_v0 = self._make_seq_label("V₀:", "—")
        self._lbl_unbal = self._make_seq_label("Unbal:", "—")
        self._lbl_depth = self._make_seq_label("Depth:", "—")

        for pair_widget in (self._lbl_v1, self._lbl_v2, self._lbl_v0,
                             self._lbl_unbal, self._lbl_depth):
            seq_layout.addWidget(pair_widget)

        row2.addWidget(seq_frame)
        row2.addStretch()
        outer.addLayout(row2)

        root.addWidget(frame)

    def _make_seq_label(self, prefix: str, initial: str) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lbl_key = QLabel(prefix)
        lbl_key.setStyleSheet("color: #888888; font-size: 10px; background: transparent;")
        lbl_val = QLabel(initial)
        lbl_val.setStyleSheet("color: #CCCCCC; font-size: 11px; font-family: monospace; background: transparent;")
        lbl_val.setObjectName("seq_val")
        lay.addWidget(lbl_key)
        lay.addWidget(lbl_val)
        return w

    def _seq_val(self, widget: QWidget) -> QLabel:
        return widget.findChild(QLabel, "seq_val")  # type: ignore[return-value]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def load_fault(self, result: FaultCharacterisation) -> None:
        """Populate the panel with a fault characterisation result."""
        ft = result.fault_type
        color = ft.color

        # Row 1
        self._lbl_type.setText(ft.value)
        self._lbl_type.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold; background: transparent;"
        )
        self._lbl_desc.setText(f"— {ft.description}")
        if abs(result.t_fault_mid) < 1.0:
            t_text = f"t = {result.t_fault_mid * 1000:.2f} ms"
        else:
            t_text = f"t = {result.t_fault_mid:.4f} s"
        self._lbl_time.setText(t_text)

        # Phase circles
        faulted = set(ft.faulted_phases)
        for ph, lbl in self._phase_labels.items():
            if ph in faulted:
                lbl.setStyleSheet(
                    f"color: {color}; font-size: 13px; font-weight: bold; background: transparent;"
                )
            else:
                lbl.setStyleSheet("color: #444444; font-size: 13px; background: transparent;")

        # Ground indicator
        if ft.is_ground_fault:
            self._lbl_ground.setText("⏚")
            self._lbl_ground.setStyleSheet(
                f"color: {color}; font-size: 14px; background: transparent;"
            )
        else:
            self._lbl_ground.setText("")

        # Sequence values
        self._seq_val(self._lbl_v1).setText(f"{result.v1_pu:.3f} pu")
        self._seq_val(self._lbl_v2).setText(f"{result.v2_pu:.3f} pu")
        self._seq_val(self._lbl_v0).setText(f"{result.v0_pu:.3f} pu")
        self._seq_val(self._lbl_unbal).setText(f"{result.unbalance_pct:.1f}%")
        self._seq_val(self._lbl_depth).setText(f"{result.fault_depth_pct:.1f}%")

        # Colour V2/V0 if elevated
        v2_color = "#FFAA33" if result.v2_pu > 0.05 else "#CCCCCC"
        v0_color = "#FF5555" if result.v0_pu > 0.08 else "#CCCCCC"
        self._seq_val(self._lbl_v2).setStyleSheet(
            f"color: {v2_color}; font-size: 11px; font-family: monospace; background: transparent;"
        )
        self._seq_val(self._lbl_v0).setStyleSheet(
            f"color: {v0_color}; font-size: 11px; font-family: monospace; background: transparent;"
        )

    def clear_fault(self) -> None:
        """Reset the panel to its idle state."""
        self._lbl_type.setText("No fault characterised")
        self._lbl_type.setStyleSheet(
            "color: #888888; font-size: 12px; font-weight: bold; background: transparent;"
        )
        self._lbl_desc.setText("")
        self._lbl_time.setText("")
        self._lbl_ground.setText("")
        for lbl in self._phase_labels.values():
            lbl.setStyleSheet("color: #555555; font-size: 13px; background: transparent;")
        for w in (self._lbl_v1, self._lbl_v2, self._lbl_v0, self._lbl_unbal, self._lbl_depth):
            self._seq_val(w).setText("—")
            self._seq_val(w).setStyleSheet(
                "color: #CCCCCC; font-size: 11px; font-family: monospace; background: transparent;"
            )
