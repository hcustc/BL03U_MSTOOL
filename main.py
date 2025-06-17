import sys
from login_dialog import LoginDialog
from massspec_func import massspec_func
import numpy as np
from massspec import Ui_MainWindow
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # login = LoginDialog()
    ui = massspec_func()
    ui.show()
    sys.exit(app.exec())
    # if login.exec() == QtWidgets.QDialog.DialogCode.Accepted:
    #     ui = massspec_func()
    #     ui.show()
    #     sys.exit(app.exec())
    # else:
    #     sys.exit()