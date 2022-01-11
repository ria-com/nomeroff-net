import os
import glob
import cv2
from _paths import current_dir

from nomeroff_net import pipeline

if __name__ == '__main__':
    number_plate_short_detection_and_reading = pipeline("number_plate_short_detection_and_reading", image_loader="opencv")

    root_dir = os.path.join(current_dir, '../images/*')
    images = number_plate_short_detection_and_reading(glob.glob(root_dir))

    for img in images:
        cv2.imshow("Display window", img)
        k = cv2.waitKey(0)
