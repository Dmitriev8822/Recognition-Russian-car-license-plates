import os
import cv2
from time import time
# from LPRN.LPRN_main import test as lpr
from YOLO.yolov8 import main as nn

# image = '34c91b1bdfece916d5acabe0322b26e0' + '.jpg'
# image_path = os.path.join(rf'C:\Users\Dima\Desktop\my_dataset\{image}')
images = [os.path.join(r'C:\Users\Dima\Desktop\my_dataset', image) for image in os.listdir(r'C:\Users\Dima\Desktop\my_dataset')]
for image_path in images:
    image = cv2.imread(image_path)
    nn(image)

