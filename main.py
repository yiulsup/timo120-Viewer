import posix_ipc as pi
import numpy as np
import cv2
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, QTimer, Qt

mq = pi.MessageQueue("/QUEUE", pi.O_CREAT | pi.O_RDWR)

class timo_thread(QThread):
    def __init__(self, parent):
        super(timo_thread, self).__init__()
        self.parent = parent
        self.message = np.zeros(10800)

    def run(self):
        self.first = 0
        self.cnt = 0
        self.YOLO_net = cv2.dnn.readNet("yolov2-tiny.weights", "yolov2-tiny.cfg")

        classes = []
        with open("yolo.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.YOLO_net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.YOLO_net.getUnconnectedOutLayers()]

        while True:
            self.m = 0
            while True:
                self.m = mq.receive()
                self.a = np.asarray(self.m)

                if self.a[0] == b'1234' and self.first == 0:
                    self.first = 1

                elif self.first == 1:
                    if self.a[0] == b'1234' and self.cnt == 0:
                        continue
                    self.message[self.cnt] = self.a[0]

                    self.cnt = self.cnt + 1
                    if self.cnt == 10800:
                        self.cnt = 0
                        frame = self.message.reshape(90, 120)

                        max = np.max(frame)
                        min = np.min(frame)


                        nfactor = 255 / (max - min)
                        pTemp = frame - min
                        nTemp = pTemp * nfactor
                        frame = nTemp
                        frame = np.array(frame).astype('uint8')
                        rgbImage = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        rgbImage = cv2.resize(rgbImage, (160, 160))

                        h, w, c = rgbImage.shape

                        # YOLO 입력
                        blob = cv2.dnn.blobFromImage(rgbImage, 0.00392, (160, 160), (0, 0, 0), True, crop=False)
                        self.YOLO_net.setInput(blob)
                        outs = self.YOLO_net.forward(self.output_layers)

                        class_ids = []
                        confidences = []
                        boxes = []

                        for out in outs:

                            for detection in out:

                                scores = detection[5:]
                                class_id = np.argmax(scores)
                                confidence = scores[class_id]

                                if confidence > 0.5:
                                    # Object detected
                                    center_x = int(detection[0] * w)
                                    center_y = int(detection[1] * h)
                                    dw = int(detection[2] * w)
                                    dh = int(detection[3] * h)
                                    # Rectangle coordinate
                                    x = int(center_x - dw / 2)
                                    y = int(center_y - dh / 2)
                                    boxes.append([x, y, dw, dh])
                                    confidences.append(float(confidence))
                                    class_ids.append(class_id)

                        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

                        for i in range(len(boxes)):
                            if i in indexes:
                                x, y, w, h = boxes[i]
                                label = str(classes[class_ids[i]])
                                score = confidences[i]

                                cv2.rectangle(rgbImage, (x, y), (x + w, y + h), (0, 0, 255), 5)
                                cv2.putText(rgbImage, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

                        convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                        convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
                        pixmap = QPixmap(convertToQtFormat)
                        resizeImage = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
                        QApplication.processEvents()
                        self.parent.image.setPixmap(resizeImage)
                        self.parent.image.setScaledContents(1)
                        self.parent.image.show()

                else:
                    continue


class timo(QMainWindow):
    def __init__(self):
        super(timo, self).__init__()
        uic.loadUi("./UI/timo120-viewer.ui", self)
        self.show()
        self.sWidget.setCurrentIndex(0)
        self.pView.clicked.connect(self.View)
        self.pSet.clicked.connect(self.Set)
        self.pClose.clicked.connect(self.close)
        self.aView.triggered.connect(self.View)
        self.aSet.triggered.connect(self.Set)
        self.aClose.triggered.connect(self.close)


        self.th = timo_thread(self)
        self.th.start()

    def View(self):
        self.sWidget.setCurrentIndex(1)

    def Set(self):
        self.sWidget.setCurrentIndex(2)

app = QApplication(sys.argv)
window = timo()
app.exec_()

