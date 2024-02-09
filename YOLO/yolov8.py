import os
from ultralytics import YOLO
import cv2
import numpy as np
from time import time

from LPRN.LPRNet_main import main as lpr

# image_path = os.path.join('..', 'images', '1.jpg')
model_path = os.path.join('YOLO', 'yolov8t4.pt')
model = YOLO(model_path)


def main(image):
    predicts = list()
    ts = time()

    threshold = 0.5

    results = model(image)[0]

    for cnt, result in enumerate(results.boxes.data.tolist()):
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            rectangle_coords = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]

            lpr_image = image[int(y1):int(y2), int(x1):int(x2)]
            lpr_image = cv2.resize(lpr_image, (94, 24))

            # filename = os.path.join(data_path, "TEMP{:04d}.jpg".format(cnt + 1))
            # filename = os.path.join(data_path, "TEMPTEMP.jpg")
            # cv2.imwrite(filename, lpr_image)

            ts = time()
            predict = str(lpr(lpr_image))
            predicts.append(predict)
            tf = time()
            print('predict:', predict)
            print(f'Image processed {round(tf - ts, 2)} sec. (LPRNet)')

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 70, 255), 2)
            cv2.putText(image, f'LP({round(score * 100)}%): {predict}', (int(x1), int(y1) - 15), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 70, 255), 2)

    tf = time()
    print(f'Image processed {round(tf - ts, 2)} sec. (YOLO)')

    # clear_temp_folder()

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image


if __name__ == '__main__':
    image_path = os.path.join('..', 'images', '1.jpg')
    image = cv2.imread(image_path)
    main(image)
