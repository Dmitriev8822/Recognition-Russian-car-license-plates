import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Камера")
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        self.setCentralWidget(QWidget(self))
        layout = QVBoxLayout(self.centralWidget())
        layout.addWidget(self.video_label)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000/30)  # 30 fps

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(frame))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())