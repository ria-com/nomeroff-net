# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import os
import sys
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
from termcolor import colored
import warnings
import time
warnings.filterwarnings('ignore')

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  filters
from NomeroffNet import  RectDetector
from NomeroffNet import  TextDetector
from NomeroffNet import  OptionsDetector
from NomeroffNet.DetectronDetector import  Detector
from NomeroffNet import  textPostprocessing
from NomeroffNet import  textPostprocessingAsync

rectDetector = RectDetector()

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
    }
})

nnet = Detector()
nnet.loadModel(NOMEROFF_NET_DIR)

def test(dirName, fname):
    img_path = os.path.join(dirName, fname)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv_imgs_masks = nnet.detect_mask([img])
    
    for cv_img_masks in cv_imgs_masks:
        #print(np.array(cv_img_masks).shape)
        # Detect points.
        arrPoints = rectDetector.detect(cv_img_masks)

        # cut zones
        zones = rectDetector.get_cv_zonesBGR(img, arrPoints, 64, 295)

        # find standart
        regionIds, stateIds, countLines = optionsDetector.predict(zones)
        regionNames = optionsDetector.getRegionLabels(regionIds)

        # find text with postprocessing by standart  
        textArr = textDetector.predict(zones, regionNames, countLines)
        textArr = textPostprocessing(textArr, regionNames)
        return textArr
    
N = 10

i = 0
j = 0
start_time = time.time()
rootDir = '../images/'
for i in np.arange(N):
    print("pass {}".format(i))
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            test(dirName, fname)
            j += 1
            #print(i, j)
    i += 1
end_time = time.time() - start_time

print(f"Processed {j} photos")
print(f"Time {end_time}")
print(f"One photo process {end_time/j} seconds")