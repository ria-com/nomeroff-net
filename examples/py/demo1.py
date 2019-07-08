# Import all necessary libraries.
import sys
import os
import json
import cv2
import matplotlib.image as mpimg

# Load default configuration file.
NOMEROFF_NET_DIR = "../../"
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, "Mask_RCNN/")
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, "logs/")

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import Detector, filters

# Initialize the detector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)

# Load weights in keras format.
nnet.loadModel("latest")

# Walking through the ./examples/images/ directory and checking each of the images for license plates.
rootDir = '../images/'

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        img_path = os.path.join(dirName, fname)
        img = mpimg.imread(img_path)
        
        np = nnet.detect([img])
    
        # Generate splashs.
        splashs = filters.color_splash(img, np)
        for splash in splashs:
            cv2.imshow('image',splash)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


