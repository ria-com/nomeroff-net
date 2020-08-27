# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

# Import all necessary libraries.
import numpy as np
import sys
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  filters
from NomeroffNet import  RectDetector
from NomeroffNet import  TextDetector
from NomeroffNet import  OptionsDetector
from NomeroffNet import  Detector
from NomeroffNet import  textPostprocessing

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

# Initialize npdetector with default configuration file.
nnet = Detector()
nnet.loadModel(NOMEROFF_NET_DIR)


# Walking through the ./examples/images/ directory and checking each of the images for license plates.
rootDir = '../images/*'

imgs = [mpimg.imread(img_path) for img_path in glob.glob(rootDir)]
        
cv_imgs_masks = nnet.detect_mask(imgs)

for img, cv_img_masks in zip(imgs, cv_imgs_masks):    
    # Detect points.
    arrPoints = rectDetector.detect(cv_img_masks)
            
    # Detect points.
    arrPoints = rectDetector.detect(cv_img_masks)

    # cut zones
    zones = rectDetector.get_cv_zonesBGR(img, arrPoints)
    toShowZones = rectDetector.get_cv_zonesRGB(img, arrPoints)
    for zone, points in zip(toShowZones, arrPoints):
        plt.axis("off")
        plt.imshow(zone)
        plt.show()

    # find standart
    regionIds, stateIds, countLines = optionsDetector.predict(zones)
    regionNames = optionsDetector.getRegionLabels(regionIds)
    print(regionNames)
    print(countLines)

    # find text with postprocessing by standart  
    textArr = textDetector.predict(zones, regionNames, countLines)
    textArr = textPostprocessing(textArr, regionNames)
    print(textArr)
    
