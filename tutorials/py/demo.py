# Specify device
import os

os.environ["LRU_CACHE_CAPACITY"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from flask import Flask
from flask import request

# Import all necessary libraries.
import sys
import traceback
import matplotlib.image as mpimg
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

app = Flask(__name__)


@app.route('/v', methods=['GET'])
def version():
    return "v2.0"


@app.route('/d', methods=['POST'])
def detect():
    data = request.get_json()
    img_path = data['path']
    try:
        print("img_path", img_path)
    
        img = mpimg.imread(img_path)
        print("img", img_path, img.shape)

        targetBoxes = detector.detect_bbox(copy.deepcopy(img))
        targetBoxes = targetBoxes
 
        all_points = npPointsCraft.detect(img, targetBoxes)
        all_points = [ps for ps in all_points if len(ps)]
 
        # cut zones
        toShowZones = [getCvZoneRGB(img, reshapePoints(rect, 1)) for rect in all_points]
        zones = convertCvZonesRGBtoBGR(toShowZones)
 
        # find standart
        regionIds, countLines = optionsDetector.predict(zones)
        regionNames = optionsDetector.getRegionLabels(regionIds)
    
        # find text with postprocessing by standart
        textArr = textDetector.predict(zones, regionNames, countLines)
        return json.dumps(dict(res=textArr, img_path=img_path))
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        return json.dumps(dict(error=str(e), img_path=img_path))


if __name__ == '__main__':
    app.run(debug=False, 
	    port=os.environ.get("PORT", 8888), 
	    host='0.0.0.0',
	    threaded = False, 
	    processes=1)
