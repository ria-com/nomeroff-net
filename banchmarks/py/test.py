# Specify device
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import os
import sys
import json
import numpy as np
import glob
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
from NomeroffNet import  textPostprocessing
from NomeroffNet import  textPostprocessingAsync
from NomeroffNet.DetectronDetector import  Detector

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

from NomeroffNet.Base import OCR

import cv2
import numpy as np

def test(dirName, fname, y, verbose=0):
    nGood = 0
    nBad  = 0
    img_path = os.path.join(dirName, fname)
    if verbose==1:
        print(colored(f"__________ \t\t {img_path} \t\t __________", "blue"))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv_imgs_masks = nnet.detect_mask([img])
    
    for cv_img_masks in cv_imgs_masks:
        for mask in cv_img_masks:
            plt.imshow(mask)
            plt.show()
            
        # Detect points.
        arrPoints = rectDetector.detect(cv_img_masks)
        
        if verbose:
            filters.draw_box(img, arrPoints, (0, 255, 0), 3)
            plt.imshow(img)
            plt.show()

        # cut zones
        zones = rectDetector.get_cv_zonesBGR(img, arrPoints)
        toShowZones = rectDetector.get_cv_zonesRGB(img, arrPoints)
        if verbose:
            for zone, points in zip(toShowZones, arrPoints):
                plt.imshow(zone)
                plt.show()

        # find standart
        regionIds, stateIds, lines = optionsDetector.predict(zones)
        regionNames = optionsDetector.getRegionLabels(regionIds)
        if verbose:
            print(regionNames)

        # find text with postprocessing by standart  
        textArr = textDetector.predict(zones, regionNames, lines)
        textArr = textPostprocessing(textArr, regionNames)
        if verbose:
            print(textArr)

        for yText in y:
            if yText in textArr:
                print(colored(f"OK: TEXT:{yText} \t\t\t RESULTS:{textArr} \n\t\t\t\t\t in PATH:{img_path}", 'green'))
                nGood += 1
            else:
                print(colored(f"NOT OK: TEXT:{yText} \t\t\t RESULTS:{textArr} \n\t\t\t\t\t in PATH:{img_path} ", 'red'))
                nBad += 1
    return nGood, nBad

dirName = "../images"

testData = {
    "0.jpeg": ["AI5255EI"],
    "1.jpeg": ["HH7777CC"],
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

gGood = 0
gBad = 0
i = 0
for fileName in testData.keys():
    nGood, nBad = test(dirName, fileName, testData[fileName], verbose=1)
    gGood += nGood
    gBad += nBad
    i += 1
total = gGood + gBad
print(f"TOTAL GOOD: {gGood/total}")
print(f"TOTAL BED: {gBad/total}")

