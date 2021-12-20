# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import all necessary libraries.
import sys
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import copy

# nomeroff_net path
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

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
from nomeroff_net.pipes.number_plate_text_readers.text_detector import TextDetector

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

# Initialize text detector.
text_detector = TextDetector({
    "eu_ua_2004_2015": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004"],
        "model_path": "latest"
    },
    "eu_ua_1995": {
        "for_regions": ["eu_ua_1995"],
        "model_path": "latest"
    },
    "eu": {
        "for_regions": ["eu"],
        "model_path": "latest"
    },
    "ru": {
        "for_regions": ["ru", "eu-ua-ordlo-lpr", "eu-ua-ordlo-dpr"],
        "model_path": "latest"
    },
    "kz": {
        "for_regions": ["kz"],
        "model_path": "latest"
    },
    "ge": {
        "for_regions": ["ge"],
        "model_path": "latest"
    },
    "su": {
        "for_regions": ["su"],
        "model_path": "latest"
    }
})

root_dir = '../images/*'

imgs = [mpimg.imread(img_path) for img_path in glob.glob(root_dir)]

for img in imgs:
    target_boxes = detector.detect_bbox(copy.deepcopy(img))

    all_points = np_points_craft.detect(img, target_boxes)
    all_points = [ps for ps in all_points if len(ps)]
    print(all_points)

    # cut zones
    to_show_zones = [get_cv_zone_rgb(img, reshape_points(rect, 1)) for rect in all_points]
    zones = convert_cv_zones_rgb_to_bgr(to_show_zones)
    for zone, points in zip(to_show_zones, all_points):
        plt.axis("off")
        plt.imshow(zone)
        plt.show()

    # find standart
    region_ids, count_lines = optionsDetector.predict(zones)
    region_names = optionsDetector.get_region_labels(region_ids)
    print(region_names)
    print(count_lines)

    # find text with postprocessing by standart
    text_arr = text_detector.predict(zones, region_names, count_lines)
    print(text_arr)

    # draw rect and 4 points
    for target_box, points in zip(target_boxes, all_points):
        cv2.rectangle(img,
                      (int(target_box[0]), int(target_box[1])),
                      (int(target_box[2]), int(target_box[3])),
                      (0, 120, 255),
                      3)
    cv2.imshow("Display window", img)
    k = cv2.waitKey(0)
