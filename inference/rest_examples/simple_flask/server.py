"""
Flask REST API

RUN: python3 ./server.py

REQUEST '/version' location: curl 127.0.0.1:8888/version
REQUEST '/detect' location: curl --header "Content-Type: application/json" \
                                 --request GET --data \
                                 '{"path": "../../../examples/images/example1.jpeg"}' 127.0.0.1:8888/detect
"""

# Specify device
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from flask import Flask
from flask import request
from flask_wtf.csrf import CSRFProtect

# Import all necessary libraries.
import sys
import traceback
import cv2
import ujson
import copy

# nomeroff_net path
NOMEROFF_NET_DIR = os.path.abspath('../../../')
sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net import __version__
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

csrf = CSRFProtect()
app = Flask(__name__)


@app.route('/version', methods=['GET'])
def version():
    return __version__


@app.route('/detect', methods=['GET'])
def detect():
    data = request.get_json()
    img_path = data['path']
    try:
        img = cv2.imread(img_path)
        img = img[..., ::-1]
        print("img", img_path, img.shape)

        target_boxes = detector.detect_bbox(copy.deepcopy(img))
        all_points = np_points_craft.detect(img, target_boxes)
        all_points = [ps for ps in all_points if len(ps)]

        # cut zones
        rgb_zones = [get_cv_zone_rgb(img, reshape_points(rect, 1)) for rect in all_points]
        zones = convert_cv_zones_rgb_to_bgr(rgb_zones)

        # find standart
        region_ids, count_lines = optionsDetector.predict(zones)
        region_names = optionsDetector.get_region_labels(region_ids)

        # find text with postprocessing by standart
        text_arr = text_detector.predict(zones, region_names, count_lines)
        return ujson.dumps(dict(res=text_arr, img_path=img_path))
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        return ujson.dumps(dict(error=str(e), img_path=img_path))


if __name__ == '__main__':
    csrf.init_app(app)
    app.run(debug=False,
            port=os.environ.get("PORT", 8888),
            host='0.0.0.0',
            threaded=False,
            processes=1)
