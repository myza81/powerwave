"""Diagnostics panel widget for the Import Wizard complete page.

DiagnosticsPanel renders an ImportDiagnosticsSummary as a structured,
engineering-readable plain-text view.  No custom painting or heavy widgets.

Usage
-----
    panel = DiagnosticsPanel()
    panel.set_summary(build_import_diagnostics(pipeline_result))
    panel.plain_text()          # for tests
"""
from __future__ import annotations

from PyQt6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

from app.import_wizard.diagnostics_summary import ImportDiagnosticsSummary


# ─────────────────────────────────────────────────────────────────────────────
# Text renderer (pure function — no Qt, easy to unit-test)
# ─────────────────────────────────────────────────────────────────────────────


def render_diagnostics_text(summary: ImportDiagnosticsSummary) -> str:
    """Format an ImportDiagnosticsSummary as engineering-readable plain text."""
    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("─" * max(len(title), 40))

    def row(label: str, value: str) -> None:
        lines.append(f"  {label:<30}{value}")

    # ── Status line ───────────────────────────────────────────────────────────
    status_icon = "✓" if summary.success else "✗"
    status_word = "Import successful" if summary.success else "Import failed"
    lines.append(f"{status_icon} {status_word}")
    if summary.source_file_name:
        lines.append(f"  Source: {summary.source_file_name}  ({summary.provider_type.upper()})")
    if summary.import_duration_s is not None:
        lines.append(f"  Duration: {summary.import_duration_s:.2f} s")

    # ── Data summary ──────────────────────────────────────────────────────────
    section("DATA SUMMARY")
    row("Rows imported:", f"{summary.normalized_rows:,}")
    if summary.dropped_rows > 0:
        row("Rows dropped:", f"{summary.dropped_rows:,}  ← data loss")
    if summary.duplicate_timestamps > 0:
        row("Duplicate timestamps:", f"{summary.duplicate_timestamps:,}")
    if summary.invalid_value_count > 0:
        row("Invalid values:", f"{summary.invalid_value_count:,}")
    completeness = summary.data_completeness_pct
    if completeness is not None and summary.total_rows > 0:
        row("Data completeness:", f"{completeness:.1f}%")
    row("Analog channels:", str(summary.analog_channels))
    row("Digital channels:", str(summary.digital_channels))
    if summary.excluded_column_count > 0:
        row("Excluded columns:", str(summary.excluded_column_count))
    if summary.user_overridden_count > 0:
        row("User overrides applied:", str(summary.user_overridden_count))

    # ── Timestamp ─────────────────────────────────────────────────────────────
    section("TIMESTAMP")
    if summary.timestamp_column:
        row("Column:", summary.timestamp_column)
    row("Strategy:", summary.timestamp_strategy)
    if summary.timestamp_format:
        row("Format:", summary.timestamp_format)
        row("Format source:", summary.timestamp_format_source)
    if summary.timestamp_confidence is not None:
        conf_pct = f"{summary.timestamp_confidence * 100:.0f}%"
        row(
            "Detection confidence:",
            f"{summary.timestamp_confidence_label} ({conf_pct})",
        )
    if summary.repair_actions:
        row("Repair actions:", "")
        for action in summary.repair_actions:
            lines.append(f"    • {action}")

    # ── Channel classification ────────────────────────────────────────────────
    section("CHANNEL CLASSIFICATION")
    row("Confidence:", summary.classification_confidence_label)
    if summary.low_confidence_columns:
        names = ", ".join(summary.low_confidence_columns)
        row("Low-confidence channels:", names)

    # ── Validation messages ───────────────────────────────────────────────────
    n_errors = len(summary.errors)
    n_warnings = len(summary.warnings)
    n_infos = len(summary.infos)
    total_msgs = n_errors + n_warnings + n_infos
    if total_msgs > 0:
        section(f"VALIDATION  ({n_errors} error(s), {n_warnings} warning(s), {n_infos} info(s))")
        for msg in summary.errors:
            col = f" [{msg.affected_column}]" if msg.affected_column else ""
            lines.append(f"  ✗ ERROR   {msg.code}{col}: {msg.message}")
        for msg in summary.warnings:
            col = f" [{msg.affected_column}]" if msg.affected_column else ""
            lines.append(f"  ⚠ WARNING {msg.code}{col}: {msg.message}")
        for msg in summary.infos:
            col = f" [{msg.affected_column}]" if msg.affected_column else ""
            lines.append(f"  ℹ INFO    {msg.code}{col}: {msg.message}")

    # ── Export guidance ───────────────────────────────────────────────────────
    if summary.export_guidance:
        section("EXPORT")
        for tip in summary.export_guidance:
            lines.append(f"  {tip}")

    # ── Large-file guidance ───────────────────────────────────────────────────
    if summary.large_file_guidance:
        section("PERFORMANCE GUIDANCE")
        for tip in summary.large_file_guidance:
            lines.append(f"  {tip}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Qt widget
# ─────────────────────────────────────────────────────────────────────────────


class DiagnosticsPanel(QWidget):
    """Read-only engineering diagnostics panel.

    Wraps a QPlainTextEdit to display a formatted ImportDiagnosticsSummary.
    Keeps the same widget style as the rest of the Import Wizard.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlaceholderText(
            "Import diagnostics will appear here after a successful import."
        )
        layout.addWidget(self._text)

    def set_summary(self, summary: ImportDiagnosticsSummary) -> None:
        """Render the diagnostics summary into the panel."""
        self._text.setPlainText(render_diagnostics_text(summary))

    def set_failure_text(self, text: str) -> None:
        """Show a plain failure/error message (no structured summary available)."""
        self._text.setPlainText(text)

    def clear(self) -> None:
        """Clear the panel content."""
        self._text.clear()

    def plain_text(self) -> str:
        """Return the current text content (for testing)."""
        return self._text.toPlainText()
