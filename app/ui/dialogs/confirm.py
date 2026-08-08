"""Sprint 1D — a single, reusable confirmation prompt for destructive
session actions (Remove Source, Clear Session, New/Open Session, Delete
Calculated Signal), so callers don't hand-roll QMessageBox logic.

Follows the same Yes/No-with-No-default convention already used by the
Import Wizard's own discard confirmation
(app/ui/import_wizard/import_wizard_dialog.py::_confirm_discard_current_session).
"""
from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox, QWidget


def confirm_destructive_action(parent: QWidget | None, *, title: str, message: str) -> bool:
    """Ask the user to confirm an irreversible action.

    Returns True only if the user explicitly chose Yes. No is the default
    button, so pressing Enter or Escape (or closing the dialog) leaves the
    action unconfirmed.
    """
    response = QMessageBox.question(
        parent,
        title,
        message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return response == QMessageBox.StandardButton.Yes
