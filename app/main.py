import sys

from PyQt6.QtWidgets import QApplication, QMainWindow


class PowerwaveMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Powerwave")
        self.resize(1200, 800)


def main() -> int:
    app = QApplication(sys.argv)
    window = PowerwaveMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
