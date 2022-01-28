"""
Tornado REST API

RUN: python3 ./server.py

REQUEST '/version' location: curl 127.0.0.1:8888/version
REQUEST '/detect' location: curl --header "Content-Type: application/json" \
                                 --request POST --data '{"path": "../images/example1.jpeg"}' 127.0.0.1:8888/detect
"""

# Specify device
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tornado.ioloop
import tornado.web
from tornado.web import Application
import tornado.web

# Import all necessary libraries.
import sys
import traceback
import torch
import logging
import cv2
import json
import copy

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

from NomeroffNet.YoloV5Detector import Detector

detector = Detector()
detector.load()

from NomeroffNet.BBoxNpPoints import (NpPointsCraft,
                                      getCvZoneRGB,
                                      convertCvZonesRGBtoBGR,
                                      reshapePoints)

npPointsCraft = NpPointsCraft()
npPointsCraft.load()

from NomeroffNet.OptionsDetector import OptionsDetector
from NomeroffNet.TextDetector import TextDetector

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

# Initialize text detector.
textDetector = TextDetector({
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

hn = logging.NullHandler()
hn.setLevel(logging.DEBUG)
logging.getLogger("tornado.access").addHandler(hn)
logging.getLogger("tornado.access").propagate = False


class GetVersion(tornado.web.RequestHandler):
    def initialize(self):
        self.thread = None

    def get(self):
        version = "v2.4"
        self.write(json.dumps(version))


class GetMask(tornado.web.RequestHandler):
    def initialize(self):
        self.thread = None

    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        img_path = data['path']
        try:
            img = cv2.imread(img_path)
            img = img[..., ::-1]
            print("img", img_path, img.shape)
            target_boxes = detector.detect_bbox(copy.deepcopy(img))
            with torch.no_grad():
                all_points = npPointsCraft.detect(img, target_boxes)
            all_points = [ps for ps in all_points if len(ps)]
            rgb_zones = [getCvZoneRGB(img, reshapePoints(rect, 1)) for rect in all_points]
            zones = convertCvZonesRGBtoBGR(rgb_zones)
            # find standart
            region_ids, count_lines = optionsDetector.predict(zones)
            region_names = optionsDetector.getRegionLabels(region_ids)
            # find text with postprocessing by standart
            text_arr = textDetector.predict(zones, region_names, count_lines)
            res = json.dumps(dict(res=text_arr, img_path=img_path))
            self.write(json.dumps(res))
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)
            res = json.dumps(dict(error=str(e), img_path=img_path))
            self.write(json.dumps(res))


if __name__ == '__main__':
    app = Application([
            (r"/detect", GetMask),
            (r"/version", GetVersion),
        ])

    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
