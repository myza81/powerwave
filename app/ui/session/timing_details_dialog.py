"""TimingDetailsDialog — read-only view of session timing-reference
compatibility (Sprint 1B).

Pure presentation: renders a SessionTimingAssessment computed elsewhere
(app.sessions.timing_compatibility). Never triggers a repair, offset
change, timezone conversion, or recalculation -- there is deliberately no
"Fix" action here; the engineer decides what, if anything, to do.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.sessions.timing_compatibility import (
    SessionTimingAssessment,
    TimingCompatibilityLevel,
    TimingReferenceClass,
)

_REFERENCE_LABELS: dict[TimingReferenceClass, str] = {
    TimingReferenceClass.ABSOLUTE_AWARE: "Absolute datetime (timezone-aware)",
    TimingReferenceClass.ABSOLUTE_NAIVE: "Absolute datetime (timezone unspecified)",
    TimingReferenceClass.ELAPSED_UNANCHORED: "Elapsed time (no absolute anchor)",
    TimingReferenceClass.RECONSTRUCTED: "Reconstructed timing (anchor and interval assumed)",
    TimingReferenceClass.SAMPLE_INDEX: "Sample index (no time unit)",
    TimingReferenceClass.UNKNOWN: "Unrecognized timing reference",
}


def reference_class_label(reference_class: TimingReferenceClass) -> str:
    return _REFERENCE_LABELS.get(reference_class, reference_class.value)


class TimingDetailsDialog(QDialog):
    """Read-only details view: each active source's timing reference and
    current session offset, plus the factual pairwise assessment.
    """

    def __init__(
        self, assessment: SessionTimingAssessment, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Timing Reference Details")
        self.setMinimumSize(580, 420)
        self._build_ui(assessment)

    def _build_ui(self, assessment: SessionTimingAssessment) -> None:
        layout = QVBoxLayout(self)

        summary = QLabel(assessment.summary)
        summary.setWordWrap(True)
        bold = summary.font()
        bold.setBold(True)
        summary.setFont(bold)
        layout.addWidget(summary)

        layout.addWidget(QLabel("Sources"))
        self._source_table = QTableWidget(len(assessment.source_profiles), 4)
        self._source_table.setHorizontalHeaderLabels(
            ["Source", "Timing reference", "Session offset", "Alignment method"]
        )
        self._source_table.verticalHeader().setVisible(False)
        self._source_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, profile in enumerate(assessment.source_profiles):
            self._source_table.setItem(row, 0, QTableWidgetItem(profile.display_name))
            self._source_table.setItem(
                row, 1, QTableWidgetItem(reference_class_label(profile.reference_class))
            )
            self._source_table.setItem(
                row, 2, QTableWidgetItem(f"{profile.time_offset_s:+.3f} s")
            )
            self._source_table.setItem(row, 3, QTableWidgetItem(profile.alignment_method))
        self._source_table.resizeColumnsToContents()
        layout.addWidget(self._source_table)

        non_compatible = [
            r for r in assessment.pair_results
            if r.level is not TimingCompatibilityLevel.COMPATIBLE
        ]
        if non_compatible:
            layout.addWidget(QLabel("Assessment"))
            for result in non_compatible:
                item_lbl = QLabel(f"• {result.message}")
                item_lbl.setWordWrap(True)
                layout.addWidget(item_lbl)

        if assessment.manual_alignment_present:
            note = QLabel(
                "This comparison currently relies on manually chosen session "
                "offsets. Powerwave does not verify that manual alignment is "
                "correct -- it only records that it was applied."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: #b06000;")
            layout.addWidget(note)

        layout.addStretch(0)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)
