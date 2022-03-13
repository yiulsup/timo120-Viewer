import posix_ipc as pi
import numpy as np
import cv2
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, QTimer


class timo(QMainWindow):
    def __init__(self):
        super(timo, self).__init__()
        uic.loadUi("./UI/timo.ui", self)
        self.show()


app = QApplication(sys.argv)
window = timo()
app.exec_()
