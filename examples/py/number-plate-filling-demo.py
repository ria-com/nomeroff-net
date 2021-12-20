# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import all necessary libraries.
import sys
import glob
import matplotlib.image as mpimg
import cv2
import copy

# nomeroff_net path
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net.pipes.number_plate_localizators.yolo_v5_detector import Detector

detector = Detector()
detector.load()

root_dir = '../images/*'

imgs = [mpimg.imread(img_path) for img_path in glob.glob(root_dir)]

for img in imgs:
    target_boxes = detector.detect_bbox(copy.deepcopy(img))

    # draw rect and 4 points
    for target_box in target_boxes:
        cv2.rectangle(img,
                      (int(target_box[0]), int(target_box[1])),
                      (int(target_box[2]), int(target_box[3])),
                      (0, 0, 0),
                      -1)
    cv2.imshow("Display window", img)
    k = cv2.waitKey(0)
