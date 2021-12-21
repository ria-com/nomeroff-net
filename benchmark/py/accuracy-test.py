# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU

# Import all necessary libraries.
import sys
import matplotlib.pyplot as plt
from termcolor import colored
import cv2
import numpy as np

# nomeroff_net path
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points import (np_points_craft,
                                                                                get_cv_zone_rgb,
                                                                                convert_cv_zones_rgb_to_bgr,
                                                                                reshape_points)
np_points_craft = np_points_craft()
np_points_craft.load()

from nomeroff_net.pipes.number_plate_localizators.yolo_v5_detector import Detector
detector = Detector()
detector.load()

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


def test(dir_name, fname, y, min_bbox_acc=0.5, verbose=0):
    n_good = 0
    n_bad = 0
    img_path = os.path.join(dir_name, fname)
    if verbose == 1:
        print(colored(f"__________ \t\t {img_path} \t\t __________", "blue"))
    img = cv2.imread(img_path)
    img = img[..., ::-1]

    target_boxes = detector.detect_bbox(img)

    all_points = np_points_craft.detect(img, target_boxes, [5, 2, 0])
    # for  images/14.jpeg bug
    all_points = [ps for ps in all_points if len(ps)]

    print('ll_points')
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
        # draw
        cv2.rectangle(img,
                      (int(target_box[0]), int(target_box[1])),
                      (int(target_box[2]), int(target_box[3])),
                      (0, 120, 255),
                      3)
        cv2.polylines(img, np.array([points], np.int32), True, (255, 120, 255), 3)
    plt.imshow(img)
    plt.show()

    for y_text in y:
        if y_text in text_arr:
            print(colored(f"OK: TEXT:{y_text} \t\t\t RESULTS:{text_arr} \n\t\t\t\t\t in PATH:{img_path}", 'green'))
            n_good += 1
        else:
            print(colored(f"NOT OK: TEXT:{y_text} \t\t\t RESULTS:{text_arr} \n\t\t\t\t\t in PATH:{img_path} ", 'red'))
            n_bad += 1
    return n_good, n_bad


def main():
    dir_name = "../images"

    test_data = {
        "31.jpeg": ["AI5255EI"],
        "1.jpeg": ["AT6883CM"],
        "2.jpeg": ["AT1515CK"],
        "3.jpeg": ["BX0578CE"],
        "4.jpeg": ["AC4249CB"],
        "5.jpeg": ["BC3496HC"],
        "6.jpeg": ["BC3496HC"],
        "7.jpeg": ["AO1306CH"],
        "8.jpeg": ["AE1077CO"],
        "9.jpeg": ["AB3391AK"],
        "10.jpeg": ["BE7425CB"],
        "11.jpeg": ["BE7425CB"],
        "12.jpeg": ["AB0680EA"],
        "13.jpeg": ["AB0680EA"],
        "14.jpeg": ["BM1930BM"],
        "15.jpeg": ["AI1382HB"],
        "16.jpeg": ["AB7333BH"],
        "17.jpeg": ["AB7642CT"],
        "18.jpeg": ["AC4921CB"],
        "19.jpeg": ["BC9911BK"],
        "20.jpeg": ["BC7007AK"],
        "21.jpeg": ["AB5649CI"],
        "22.jpeg": ["AX2756EK"],
        "23.jpeg": ["AA7564MX"],
        "24.jpeg": ["AM5696CK"],
        "25.jpeg": ["AM5696CK"],
    }


    g_good = 0
    g_bad = 0
    for file_name in test_data.keys():
        num_good, num_bad = test(dir_name, file_name, test_data[file_name], verbose=1)
        g_good += num_good
        g_bad += num_bad
    total = g_good + g_bad
    print(f"TOTAL GOOD: {g_good/total}")
    print(f"TOTAL BED: {g_bad/total}")


if __name__ == "__main__":
    main()

