# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # For CPU inference
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

# Import all necessary libraries.
import numpy as np
import sys
import glob
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from turbojpeg import TurboJPEG
jpeg = TurboJPEG()

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

# for best speed install PyTurboJPEG
# pip3 install -U git+git://github.com/lilohuang/PyTurboJPEG.git

from NomeroffNet.YoloV5Detector import Detector

detector = Detector()
detector.load()

from NomeroffNet.BBoxNpPoints import NpPointsCraft, convertCvZonesRGBtoBGR, getCvZoneRGB, reshapePoints

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

async def test(dirName, fname):
    start_time = time.time()
    img_path = os.path.join(dirName, fname)
    with open(img_path, 'rb') as in_file:
        img = jpeg.decode(in_file.read())
    #img = cv2.imread(img_path)
    image_load_time = time.time() - start_time

    start_time = time.time()
    targetBoxes = detector.detect_bbox(img)
    detect_bbox_time = time.time() - start_time

    start_time = time.time()
    all_points = npPointsCraft.detect(img, targetBoxes)
    all_points = [ps for ps in all_points if len(ps)]
    craft_time = time.time() - start_time

    start_time = time.time()

    zones = convertCvZonesRGBtoBGR([getCvZoneRGB(img, reshapePoints(rect,1)) for rect in all_points])

    perspective_align_time = time.time() - start_time

    start_time = time.time()
    regionIds, stateIds, countLines = optionsDetector.predict(zones)
    regionNames = optionsDetector.getRegionLabels(regionIds)
    classification_time = time.time() - start_time

    start_time = time.time()
    textArr = textDetector.predict(zones, regionNames, countLines)
    ocr_time = time.time() - start_time
    return (image_load_time, detect_bbox_time, craft_time, perspective_align_time, classification_time, ocr_time)


N = 10

i = 0
j = 0

image_load_time_all        = 0
detect_bbox_time_all       = 0
craft_time_all             = 0
perspective_align_time_all = 0
classification_time_all    = 0
ocr_time_all               = 0

start_time = time.time()
rootDir = 'images/'
for i in np.arange(N):
    print("pass {}".format(i))
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            image_load_time, \
                detect_bbox_time, \
                craft_time, \
                perspective_align_time, \
                classification_time, \
                ocr_time = await test(dirName, fname)
            image_load_time_all        += image_load_time
            detect_bbox_time_all       += detect_bbox_time
            craft_time_all             += craft_time
            perspective_align_time_all += perspective_align_time
            classification_time_all    += classification_time
            ocr_time_all               += ocr_time
            j += 1
            #print(i, j)
    i += 1
end_time = time.time() - start_time

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