# Specify device
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import all necessary libraries.
import sys
import cv2

# nomeroff_net path
NOMEROFF_NET_DIR = os.path.abspath('../../')

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from nomeroff_net.pipes.number_plate_localizators.yolo_v5_detector import Detector
detector = Detector()
detector.load()

from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points import (np_points_craft,
                                                                                get_cv_zone_rgb,
                                                                                convert_cv_zones_rgb_to_bgr,
                                                                                reshape_points)
np_points_craft = np_points_craft()
np_points_craft.load()

from nomeroff_net.pipes.number_plate_classificators.options_detector import OptionsDetector
from nomeroff_net.text_detectors.eu import eu

# load models
optionsDetector = OptionsDetector()
optionsDetector.load("latest")

text_detector = eu
text_detector.load("latest")

# Detect numberplate
img_path = '../images/example2.jpeg'
img = cv2.imread(img_path)
img = img[..., ::-1]

target_boxes = detector.detect_bbox(img)
all_points = np_points_craft.detect(img,
                                  target_boxes,
                                  [5, 2, 0])

# cut zones
zones = convert_cv_zones_rgb_to_bgr([get_cv_zone_rgb(img, reshape_points(rect, 1)) for rect in all_points])

# predict zones attributes
region_ids, count_lines = optionsDetector.predict(zones)
region_names = optionsDetector.get_region_labels(region_ids)

# find text with postprocessing by standart
text_arr = text_detector.predict(zones)
print(text_arr)
