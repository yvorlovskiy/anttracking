import sys
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi


class Window(QDialog):

    def __init__(self):
        super(Window, self).__init__()
        loadUi('window.ui', self)

        self.image = None  # loaded image depends on update_frame
        self.cropped_cap = None
        self.processedImage = None
        self.startButton.clicked.connect(self.start_webcam)  # start button
        self.stopButton.clicked.connect(self.stop_webcam)  # stop button
        self.cannyButton.toggled.connect(self.canny_webcam)
        self.cannyButton.setCheckable(True)
        self.canny_Enabled = False
        self.height = 480
        self.width = 440


    def canny_webcam(self, status):
        if status:
            self.canny_Enabled = True
            self.cannyButton.setText('Stop')
        else:
            self.canny_Enabled = False
            self.cannyButton.setText('Canny')





    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)  # sets frame dimensions based on label

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)  # uses function update frame
        self.timer.start(5)

    def update_frame(self):
        ret, self.image = self.capture.read()

        cropped_cap = self.image[100:300,100:300]
        self.image = cv2.flip(self.image, 1) # original image from the cam
        self.displayImage(self.image, 1)  # creates function display image

        if (self.canny_Enabled):
            gray=cv2.cvtColor(self.cropped_cap, cv2.COLOR_BGR2GRAY) if len(self.cropped_cap.shape)>=3 else self.cropped_cap
            self.processedImage = cv2.Canny(gray, 100, 200)
            self.displayImage(self.processedImage, 2)


    def stop_webcam(self):
        self.timer.stop()

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape)==3: # (0) rows, (1) columns, (2) channels
            if img.shape[2]==4: # r, g, b, and alpha channels
                qformat=QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window==1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)
        if window==2:
            self.processedLabel.setPixmap(QPixmap.fromImage(outImage))
            self.processedLabel .setScaledContents(True)





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.setWindowTitle('Interface')
    window.show()
    sys.exit(app.exec_())
