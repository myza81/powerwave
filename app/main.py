import os
import sys

import pyqtgraph as pg
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from app.ui.main_window import PowerwaveMainWindow

# Must be called before any pg widget is instantiated (VIEWPORT_RENDERING_POLICY §1)
_USE_OPENGL = os.environ.get("POWERWAVE_USE_OPENGL", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
pg.setConfigOptions(
    useOpenGL=_USE_OPENGL,
    antialias=False,
    foreground="w",
    background="#1E1E1E",
)


def main() -> int:
    QApplication.setFont(QFont("Arial"))
    app = QApplication(sys.argv)
    window = PowerwaveMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
