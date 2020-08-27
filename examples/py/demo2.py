# Import all necessary libraries.
import os
import numpy as np
import glob
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import Detector, filters, RectDetector

# Import license plate recognition tools.
from NomeroffNet import  Detector
from NomeroffNet import  filters
from NomeroffNet import  RectDetector

# load models
rectDetector = RectDetector()

nnet = Detector()
nnet.loadModel(NOMEROFF_NET_DIR)

# Walking through the ./examples/images/ directory and checking each of the images for license plates.
rootDir = '../images/*'

imgs = [mpimg.imread(img_path) for img_path in glob.glob(rootDir)]
        
cv_imgs_masks = nnet.detect_mask(imgs)

for img, cv_img_masks in zip(imgs, cv_imgs_masks):    
    # Detect points.
    arrPoints = rectDetector.detect(cv_img_masks)

    filters.draw_box(img, arrPoints, (0, 255, 0), 3)
    plt.axis("off")
    plt.imshow(img)
    plt.show()