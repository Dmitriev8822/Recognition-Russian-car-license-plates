import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter

import os
from YOLO.yolov8 import main as nn  # Подразумевается, что в данном файле содержится функция nn

FPS = 30

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Видео")

        # Создаем виджеты для отображения видео
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(QWidget(self))

        # Организуем виджеты в слой
        layout = QVBoxLayout(self.centralWidget())
        layout.addWidget(self.video_label)

        # Открытие видеофайла
        self.video_path = "videos/test.mp4"
        self.cap = cv2.VideoCapture(self.video_path)

        # Настройка таймера обновления кадров
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        fps = 1000 // FPS
        self.timer.start(fps)

    def update_frame(self):
        # Считываем кадр из видеофайла
        ret, frame = self.cap.read()
        if ret:
            processed_frame = nn(frame)

            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = QImage(processed_frame, processed_frame.shape[1], processed_frame.shape[0],
                         QImage.Format_RGB888)

            self.video_label.setPixmap(QPixmap.fromImage(img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())