import os
import cv2
from time import time
# from LPRN.LPRN_main import test as lpr
from YOLO.yolov8 import main as nn


image_path = os.path.join(rf'C:\Users\racco\Downloads\ru4566893.jpg')
image = cv2.imread(image_path)
nn(image)

