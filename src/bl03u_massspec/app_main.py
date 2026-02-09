import sys

from PyQt6 import QtWidgets

from .massspec_func import massspec_func


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    ui = massspec_func()
    ui.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
