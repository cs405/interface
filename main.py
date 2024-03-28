import sys
from PyQt6.QtWidgets import QApplication
from PyQt6 import uic
from event_detection import *
from img_caption import *
from Chinese_correction import *
from ocr_test import *
from grounding_caption import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = uic.loadUi("./interface.ui")

    # 初始第一页
    ui.tabWidget.setCurrentIndex(0)

    # 五个页面
    event_detection(ui)
    img_caption(ui)
    Chinese_correction(ui)
    grounding_caption(ui)
    ocr(ui)

    ui.show()
    sys.exit(app.exec())
