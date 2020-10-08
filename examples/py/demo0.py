# # Specify device
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 

# Import all necessary libraries.
import os
import numpy as np
import sys
import matplotlib.image as mpimg

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  Detector
from NomeroffNet import  filters
from NomeroffNet import  RectDetector
from NomeroffNet import  OptionsDetector
from NomeroffNet import  TextDetector
from NomeroffNet import  textPostprocessing

# load models
rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

textDetector = TextDetector.get_static_module("eu")()
textDetector.load("latest")

nnet = Detector()
nnet.loadModel(NOMEROFF_NET_DIR)

# Detect numberplate
img_path = '../images/example2.jpeg'
img = mpimg.imread(img_path)
NP = nnet.detect_mask([img])

# Generate image mask.
cv_imgs_masks = nnet.detect_mask([img])
    
for cv_img_masks in cv_imgs_masks:
    # Detect points.
    arrPoints = rectDetector.detect(cv_img_masks)
    
    # cut zones
    zones = rectDetector.get_cv_zonesBGR(img, arrPoints, 64, 295)

    # find standart
    regionIds, stateIds, countLines = optionsDetector.predict(zones)
    regionNames = optionsDetector.getRegionLabels(regionIds)
    
    # find text with postprocessing by standart  
    textArr = textDetector.predict(zones)
    textArr = textPostprocessing(textArr, regionNames)
    print(textArr)