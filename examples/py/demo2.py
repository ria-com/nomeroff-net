# Import all necessary libraries.
import os
import sys
import json
import matplotlib.image as mpimg
import cv2

# Load default configuration file.
NOMEROFF_NET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, "Mask_RCNN/")
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, "logs/")

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import Detector, filters, RectDetector

# Initialize rect detector with default configuration.
rectDetector = RectDetector()
# Initialize npdetector with default configuration.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
# Load weights in keras format.
nnet.loadModel("latest")

print("START RECOGNIZING")
rootDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../images/')

# Walking through the ./examples/images/ directory and checking each of the images for license plates.
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        img_path = os.path.join(dirName, fname)
        img = mpimg.imread(img_path)
        NP = nnet.detect([img])
        
        # Generate image mask.
        cv_img_masks = filters.cv_img_mask(NP) 
       
        res = []
        # Detect points.
        arrPoints = rectDetector.detect(cv_img_masks)
            
        # draw
        filters.draw_box(img, arrPoints, (0, 255, 0), 3)
        
        # show
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()