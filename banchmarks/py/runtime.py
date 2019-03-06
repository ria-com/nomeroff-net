import os
import sys
import json
import matplotlib.image as mpimg
from termcolor import colored
import cv2
import time
import numpy as np
import warnings
import asyncio
warnings.filterwarnings('ignore')

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../../')

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')

MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')
MASK_RCNN_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/mask_rcnn_numberplate_0700.h5")
OPTIONS_MODEL_PATH =  os.path.join(NOMEROFF_NET_DIR, "models/numberplate_options_2019_03_05.h5")

# If you use gpu version tensorflow please change model to gpu version named like *-gpu.pb
mode =  "cpu" if  "NN_MODE" not in os.environ else os.environ["NN_MODE"] if os.environ["NN_MODE"]=="gpu" else "cpu"
OCR_NP_UKR_TEXT =  os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_ua_12-{}.h5".format(mode))
OCR_NP_EU_TEXT =  os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_eu_2-{}.h5".format(mode))
OCR_NP_RU_TEXT =  os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_ru_3-{}.h5".format(mode))

sys.path.append(NOMEROFF_NET_DIR)

from NomeroffNet import  filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessingAsync

nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
nnet.loadModel(MASK_RCNN_MODEL_PATH)

rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load(OPTIONS_MODEL_PATH)

# Initialize text detector.
textDetector = TextDetector({
    "eu_ua_2004_2015": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004"],
        "model_path": OCR_NP_UKR_TEXT
    },
    "eu": {
        "for_regions": ["eu", "eu_ua_1995"],
        "model_path": OCR_NP_EU_TEXT
    },
    "ru": {
        "for_regions": ["ru"],
        "model_path": OCR_NP_RU_TEXT
    }
})


async def test(dirName, fname, max_img_w=1280):
    img_path = os.path.join(dirName, fname)
    img = mpimg.imread(img_path)

    # corect size for better speed
    img_w = img.shape[1]
    img_h = img.shape[0]
    img_w_r = 1
    img_h_r = 1
    if img_w > max_img_w:
        resized_img = cv2.resize(img, (max_img_w, int(max_img_w/img_w*img_h)))
        img_w_r = img_w/max_img_w
        img_h_r = img_h/(max_img_w/img_w*img_h)
    else:
        resized_img = img

    NP = nnet.detect([resized_img])

    # Generate image mask.
    cv_img_masks = await filters.cv_img_mask_async(NP)

    # Detect points.
    arrPoints = await rectDetector.detectAsync(cv_img_masks,  outboundHeightOffset=3-img_w_r)
    arrPoints[..., 1:2] = arrPoints[..., 1:2]*img_h_r
    arrPoints[..., 0:1] = arrPoints[..., 0:1]*img_w_r

    # cut zones
    zones = await rectDetector.get_cv_zonesBGR_async(img, arrPoints)

    # find standart
    regionIds, stateIds = optionsDetector.predict(zones)
    regionNames = optionsDetector.getRegionLabels(regionIds)

    # find text with postprocessing by standart
    textArr = textDetector.predict(zones, regionNames)
    textArr = await textPostprocessingAsync(textArr, regionNames)
    return textArr

async def runAll():
    print("START")
    N = 10
    i = 0
    j = 0
    start_time = time.time()
    rootDir = '../images/'
    for i in np.arange(N):
        for dirName, subdirList, fileList in os.walk(rootDir):
            for fname in fileList:
                await test(dirName, fname)
                j += 1
        i += 1
        print(i/N)
    end_time = time.time() - start_time
    print("Processed {} photos".format(j))
    print("Time {}".format(end_time))
    print("One photo process {} seconds".format(end_time/j))

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(runAll())

runAll()