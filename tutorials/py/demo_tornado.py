# Specify device
import os

#os.environ["LRU_CACHE_CAPACITY"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tornado.ioloop
import tornado.web
from tornado.web import Application, RequestHandler
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
        "for_regions": ["ru", "eu-ua-fake-lnr", "eu-ua-fake-dnr"],
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
        version = "v2.0"
        self.write(json.dumps(version))


class GetMask(tornado.web.RequestHandler):
    def initialize(self):
        self.thread = None

    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        img_path = data['path']
        try:
            print("img_path", img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("img", img_path, img.shape)
            targetBoxes = detector.detect_bbox(copy.deepcopy(img))
            
            #zones = []
            #for targetBox in targetBoxes:
            #    x = int(min(targetBox[0], targetBox[2]))
            #    w = int(abs(targetBox[2]-targetBox[0]))
            #    y = int(min(targetBox[1], targetBox[3]))
            #    h = int(abs(targetBox[3]-targetBox[1]))
            #    image_part = img[y:y + h, x:x + w]
            #    zones.append(image_part)
            with torch.no_grad():
                all_points = npPointsCraft.detect(img, targetBoxes)
            all_points = [ps for ps in all_points if len(ps)]
            toShowZones = [getCvZoneRGB(img, reshapePoints(rect, 1)) for rect in all_points]
            zones = convertCvZonesRGBtoBGR(toShowZones)
            # find standart
            regionIds, countLines = optionsDetector.predict(zones)
            regionNames = optionsDetector.getRegionLabels(regionIds)
            # find text with postprocessing by standart
            textArr = textDetector.predict(zones, regionNames, countLines)
            res = json.dumps(dict(res=textArr, img_path=img_path))
            self.write(json.dumps(res))
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)
            res = json.dumps(dict(error=str(e), img_path=img_path))
            self.write(json.dumps(res))


if __name__ == '__main__':
    app = Application([
            (r"/d", GetMask),
            (r"/v", GetVersion),
        ])

    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
