"""Runtime-test Qt cleanup helpers."""
from __future__ import annotations

import sys

import pytest
from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication


@pytest.fixture()
def runtime_qapp():
    """Return QApplication and drain runtime worker/widget state after a test."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    QThreadPool.globalInstance().waitForDone(5000)
    for widget in list(QApplication.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    app.processEvents()
    app.sendPostedEvents()
    app.processEvents()
