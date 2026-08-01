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

from html import escape

from PyQt6.QtWidgets import QTextEdit, QVBoxLayout, QWidget

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


def render_diagnostics_html(summary: ImportDiagnosticsSummary) -> str:
    """Format an ImportDiagnosticsSummary as a compact rich-text report."""
    status_color = "#1B5E20" if summary.success else "#B71C1C"
    status_word = "Import successful" if summary.success else "Import failed"
    status_icon = "✓" if summary.success else "✗"

    def esc(value: object) -> str:
        return escape(str(value))

    def section(title: str) -> str:
        return (
            "<div style='margin-top: 12px; margin-bottom: 6px; "
            "font-weight: 700; color: #333333; letter-spacing: 0.3px;'>"
            f"{esc(title)}</div>"
        )

    def row(label: str, value: str, *, alert: bool = False) -> str:
        value_color = "#B71C1C" if alert else "#222222"
        value_weight = "700" if alert else "400"
        return (
            "<tr>"
            "<td style='padding: 2px 18px 2px 0; color: #555555; white-space: nowrap;'>"
            f"{esc(label)}</td>"
            f"<td style='padding: 2px 0; color: {value_color}; font-weight: {value_weight};'>"
            f"{esc(value)}</td>"
            "</tr>"
        )

    html: list[str] = [
        "<div style='font-family: -apple-system, BlinkMacSystemFont, "
        "&quot;Segoe UI&quot;, sans-serif; font-size: 13px; line-height: 1.35;'>",
        f"<div style='font-weight: 700; color: {status_color};'>"
        f"{esc(status_icon)} {esc(status_word)}</div>",
    ]
    if summary.source_file_name:
        html.append(
            "<div style='color: #444444;'>"
            f"Source: {esc(summary.source_file_name)} "
            f"<span style='color: #777777;'>({esc(summary.provider_type.upper())})</span>"
            "</div>"
        )
    if summary.import_duration_s is not None:
        html.append(f"<div style='color: #444444;'>Duration: {summary.import_duration_s:.2f} s</div>")

    html.append(section("Data Summary"))
    html.append("<table cellspacing='0' cellpadding='0'>")
    html.append(row("Rows imported", f"{summary.normalized_rows:,}"))
    if summary.dropped_rows > 0:
        html.append(row("Rows dropped", f"{summary.dropped_rows:,} data loss", alert=True))
    if summary.duplicate_timestamps > 0:
        html.append(row("Duplicate timestamps", f"{summary.duplicate_timestamps:,}", alert=True))
    if summary.invalid_value_count > 0:
        html.append(row("Invalid values", f"{summary.invalid_value_count:,}", alert=True))
    completeness = summary.data_completeness_pct
    if completeness is not None and summary.total_rows > 0:
        html.append(row("Data completeness", f"{completeness:.1f}%"))
    html.append(row("Analog channels", str(summary.analog_channels)))
    html.append(row("Digital channels", str(summary.digital_channels)))
    if summary.excluded_column_count > 0:
        html.append(row("Excluded columns", str(summary.excluded_column_count)))
    if summary.user_overridden_count > 0:
        html.append(row("User overrides applied", str(summary.user_overridden_count)))
    html.append("</table>")

    html.append(section("Timestamp"))
    html.append("<table cellspacing='0' cellpadding='0'>")
    if summary.timestamp_column:
        html.append(row("Column", summary.timestamp_column))
    html.append(row("Strategy", summary.timestamp_strategy))
    if summary.timestamp_format:
        html.append(row("Format", summary.timestamp_format))
        html.append(row("Format source", summary.timestamp_format_source))
    if summary.timestamp_confidence is not None:
        conf_pct = f"{summary.timestamp_confidence * 100:.0f}%"
        html.append(row("Detection confidence", f"{summary.timestamp_confidence_label} ({conf_pct})"))
    html.append("</table>")
    if summary.repair_actions:
        html.append("<div style='margin-top: 4px; color: #555555;'>Repair actions</div><ul style='margin-top: 2px;'>")
        html.extend(f"<li>{esc(action)}</li>" for action in summary.repair_actions)
        html.append("</ul>")

    html.append(section("Channel Classification"))
    html.append("<table cellspacing='0' cellpadding='0'>")
    html.append(row("Confidence", summary.classification_confidence_label))
    if summary.low_confidence_columns:
        html.append(row("Low-confidence channels", ", ".join(summary.low_confidence_columns), alert=True))
    html.append("</table>")

    n_errors = len(summary.errors)
    n_warnings = len(summary.warnings)
    n_infos = len(summary.infos)
    if n_errors + n_warnings + n_infos > 0:
        html.append(section(f"Validation ({n_errors} error(s), {n_warnings} warning(s), {n_infos} info(s))"))
        html.append("<ul style='margin-top: 2px;'>")
        for msg in summary.errors:
            col = f" [{msg.affected_column}]" if msg.affected_column else ""
            html.append(f"<li style='color: #B71C1C; font-weight: 700;'>ERROR {esc(msg.code)}{esc(col)}: {esc(msg.message)}</li>")
        for msg in summary.warnings:
            col = f" [{msg.affected_column}]" if msg.affected_column else ""
            html.append(f"<li style='color: #B71C1C;'>WARNING {esc(msg.code)}{esc(col)}: {esc(msg.message)}</li>")
        for msg in summary.infos:
            col = f" [{msg.affected_column}]" if msg.affected_column else ""
            html.append(f"<li style='color: #555555;'>INFO {esc(msg.code)}{esc(col)}: {esc(msg.message)}</li>")
        html.append("</ul>")

    if summary.export_guidance:
        html.append(section("Export"))
        html.append("<ul style='margin-top: 2px;'>")
        html.extend(f"<li>{esc(tip)}</li>" for tip in summary.export_guidance)
        html.append("</ul>")

    if summary.large_file_guidance:
        html.append(section("Performance Guidance"))
        html.append("<ul style='margin-top: 2px;'>")
        html.extend(f"<li>{esc(tip)}</li>" for tip in summary.large_file_guidance)
        html.append("</ul>")

    html.append("</div>")
    return "".join(html)


# ─────────────────────────────────────────────────────────────────────────────
# Qt widget
# ─────────────────────────────────────────────────────────────────────────────


class DiagnosticsPanel(QWidget):
    """Read-only engineering diagnostics panel.

    Wraps a QTextEdit to display a formatted ImportDiagnosticsSummary.
    Keeps the same widget style as the rest of the Import Wizard.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlaceholderText(
            "Import diagnostics will appear here after a successful import."
        )
        layout.addWidget(self._text)
        self._plain_text_cache = ""

    def set_summary(self, summary: ImportDiagnosticsSummary) -> None:
        """Render the diagnostics summary into the panel."""
        self._plain_text_cache = render_diagnostics_text(summary)
        self._text.setHtml(render_diagnostics_html(summary))

    def set_failure_text(self, text: str) -> None:
        """Show a plain failure/error message (no structured summary available)."""
        self._plain_text_cache = text
        self._text.setPlainText(text)

    def clear(self) -> None:
        """Clear the panel content."""
        self._plain_text_cache = ""
        self._text.clear()

    def plain_text(self) -> str:
        """Return the current text content (for testing)."""
        return self._plain_text_cache
