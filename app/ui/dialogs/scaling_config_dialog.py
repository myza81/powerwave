"""Scaling Configuration Dialog — Phase 5B.

Allows the operator to set session-level transformer ratios and per-unit base
values. All fields are optional; leaving a field at its default means no scaling
is applied for that dimension.

Usage::

    from app.ui.dialogs.scaling_config_dialog import ScalingConfigDialog
    from app.analytics.scaling import GlobalScalingConfig

    dlg = ScalingConfigDialog(parent=self)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        config = dlg.get_config()
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from app.analytics.scaling import GlobalScalingConfig, VoltageReference


class ScalingConfigDialog(QDialog):
    """Modal dialog for configuring session-level engineering scaling parameters.

    All fields are pre-populated from *initial_config* (defaults to the zero
    config: PT=CT=1.0, no per-unit bases set).  The operator can accept to
    commit or cancel to discard.
    """

    def __init__(
        self,
        parent=None,
        *,
        initial_config: GlobalScalingConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scaling Configuration")
        self.setMinimumWidth(380)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )

        cfg = initial_config or GlobalScalingConfig()
        self._build_ui(cfg)

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self, cfg: GlobalScalingConfig) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(self._build_transformer_group(cfg))
        layout.addWidget(self._build_per_unit_group(cfg))
        layout.addWidget(self._build_note_label())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_transformer_group(self, cfg: GlobalScalingConfig) -> QGroupBox:
        group = QGroupBox("Transformer Ratios")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._pt_spin = self._ratio_spinbox(cfg.pt_ratio, max_val=100_000.0)
        form.addRow("PT Ratio:", self._pt_spin)

        self._ct_spin = self._ratio_spinbox(cfg.ct_ratio, max_val=10_000.0)
        form.addRow("CT Ratio:", self._ct_spin)

        return group

    def _build_per_unit_group(self, cfg: GlobalScalingConfig) -> QGroupBox:
        group = QGroupBox("Per-Unit Base Values")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._vbase_spin = self._base_spinbox(
            cfg.voltage_base_kv,
            max_val=1_500.0,
            suffix=" kV",
        )
        form.addRow("Voltage Base L-L:", self._vbase_spin)

        self._ibase_spin = self._base_spinbox(
            cfg.current_base_ka,
            max_val=1_000.0,
            suffix=" kA",
            decimals=4,
            step=0.001,
        )
        form.addRow("Current Base:", self._ibase_spin)

        self._vref_combo = QComboBox()
        self._vref_labels = [
            ("Phase-to-Ground", VoltageReference.PHASE_TO_GROUND),
            ("Phase-to-Phase", VoltageReference.PHASE_TO_PHASE),
            ("Unknown", VoltageReference.UNKNOWN),
        ]
        for label, _ in self._vref_labels:
            self._vref_combo.addItem(label)
        current_ref = cfg.voltage_reference
        for i, (_, ref) in enumerate(self._vref_labels):
            if ref == current_ref:
                self._vref_combo.setCurrentIndex(i)
                break
        form.addRow("Voltage Reference:", self._vref_combo)

        return group

    @staticmethod
    def _build_note_label() -> QLabel:
        label = QLabel(
            "Leave Voltage/Current Base at 0.00 to disable per-unit scaling."
        )
        label.setWordWrap(True)
        label.setStyleSheet("color: gray; font-size: 10px;")
        return label

    # ─────────────────────────────────────────────────────────────────────────
    # Spinbox helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ratio_spinbox(value: float, *, max_val: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.001, max_val)
        spin.setDecimals(4)
        spin.setSingleStep(0.1)
        spin.setValue(float(value))
        return spin

    @staticmethod
    def _base_spinbox(
        value: float | None,
        *,
        max_val: float,
        suffix: str = "",
        decimals: int = 3,
        step: float = 1.0,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, max_val)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setSpecialValueText("Not set")
        spin.setValue(float(value) if value is not None and value > 0 else 0.0)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    # ─────────────────────────────────────────────────────────────────────────
    # Result
    # ─────────────────────────────────────────────────────────────────────────

    def get_config(self) -> GlobalScalingConfig:
        """Return a GlobalScalingConfig built from the current dialog state."""
        vbase_raw = self._vbase_spin.value()
        ibase_raw = self._ibase_spin.value()
        _, vref = self._vref_labels[self._vref_combo.currentIndex()]
        return GlobalScalingConfig(
            pt_ratio=self._pt_spin.value(),
            ct_ratio=self._ct_spin.value(),
            voltage_base_kv=vbase_raw if vbase_raw > 0 else None,
            current_base_ka=ibase_raw if ibase_raw > 0 else None,
            voltage_reference=vref,
        )
