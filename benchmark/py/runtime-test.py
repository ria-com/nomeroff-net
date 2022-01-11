# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import all necessary libraries.
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

import time
from turbojpeg import TurboJPEG
jpeg = TurboJPEG()

# nomeroff_net path
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

# for best speed install PyTurboJPEG
# pip3 install -U git+https://github.com/lilohuang/PyTurboJPEG.git

from nomeroff_net.pipes.number_plate_localizators.yolo_v5_detector import Detector

detector = Detector()
detector.load()

from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points import (np_points_craft,
                                                                                convert_cv_zones_rgb_to_bgr,
                                                                                get_cv_zone_rgb,
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


def test(dir_name, fname):
    start_time = time.time()
    img_path = os.path.join(dir_name, fname)
    with open(img_path, 'rb') as in_file:
        img = jpeg.decode(in_file.read())
    image_load_time = time.time() - start_time

    start_time = time.time()
    target_boxes = detector.detect_bbox(img)
    detect_bbox_time = time.time() - start_time

    start_time = time.time()
    all_points = np_points_craft.detect(img, target_boxes)
    all_points = [ps for ps in all_points if len(ps)]
    craft_time = time.time() - start_time

    start_time = time.time()

    zones = convert_cv_zones_rgb_to_bgr([get_cv_zone_rgb(img, reshape_points(rect, 1)) for rect in all_points])

    perspective_align_time = time.time() - start_time

    start_time = time.time()
    region_ids, count_lines = optionsDetector.predict(zones)
    region_names = optionsDetector.get_region_labels(region_ids)
    classification_time = time.time() - start_time

    start_time = time.time()
    _ = text_detector.predict(zones, region_names, count_lines)
    ocr_time = time.time() - start_time
    return image_load_time, detect_bbox_time, craft_time, perspective_align_time, classification_time, ocr_time


def main():
    N = 10
    j = 0

    image_load_time_all = 0
    detect_bbox_time_all = 0
    craft_time_all = 0
    perspective_align_time_all = 0
    classification_time_all = 0
    ocr_time_all = 0

    start_process_time = time.time()
    root_dir = '../images/'
    for i in np.arange(N):
        print("pass {}".format(i))
        for dir_name, subdir_list, file_list in os.walk(root_dir):
            for file_name in file_list:
                image_load_time, \
                    detect_bbox_time, \
                    craft_time, \
                    perspective_align_time, \
                    classification_time, \
                    ocr_time = test(dir_name, file_name)
                image_load_time_all += image_load_time
                detect_bbox_time_all += detect_bbox_time
                craft_time_all += craft_time
                perspective_align_time_all += perspective_align_time
                classification_time_all += classification_time
                ocr_time_all += ocr_time
                j += 1
        i += 1
    end_time = time.time() - start_process_time

    print(f"Processed {j} photos")
    print(f"Time {end_time}")
    print(f"One photo process {end_time/j} seconds")
    print()
    print(f"image_load_time_all {image_load_time_all}; {image_load_time_all/j} per one photo")
    print(f"detect_bbox_time_all {detect_bbox_time_all}; {detect_bbox_time_all/j} per one photo")
    print(f"craft_time_all {craft_time_all}; {craft_time_all/j} per one photo")
    print(f"perspective_align_time_all {perspective_align_time_all}; {perspective_align_time_all/j} per one photo")
    print(f"classification_time_all {classification_time_all}; {classification_time_all/j} per one photo")
    print(f"ocr_time_all {ocr_time_all}; {ocr_time_all/j} per one photo")


if __name__ == "__main__":
    main()
