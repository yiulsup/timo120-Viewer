import posix_ipc as pi
import numpy as np
import cv2
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindows
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, QTimer

message = np.zeros(10800)

mq = pi.MessageQueue("/QUEUE", pi.O_CREAT|pi.O_RDWR)
cnt = 0
first = 0

while True:
    m = mq.receive()
    a = np.asarray(m)

    icnt = 0
    
    if a[0] == b'1234' and first == 0:
        first = 1
            
    elif first == 1:
        if a[0] == b'1234' and cnt == 0:
            continue
        message[cnt] = a[0]
    
        cnt = cnt + 1
        if cnt == 10800:
            cnt = 0
            frame = message.reshape(90, 120) 
            max = np.max(frame)
            min = np.min(frame)
            nfactor = 255 / (max - min)
            pTemp = frame - min
            nTemp = pTemp * nfactor
            frame = nTemp
            frame = np.array(frame).astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow('d', frame)
            cv2.waitKey(1) 
            #print(message)
            
    else:
        continue
        
