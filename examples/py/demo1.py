# Import all necessary libraries.
import os
import numpy as np
import sys
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  Detector
from NomeroffNet import  filters

# load model
nnet = Detector()
nnet.loadModel(NOMEROFF_NET_DIR)

# Walking through the ./examples/images/ directory and checking each of the images for license plates.
rootDir = '../images/*'

imgs = [mpimg.imread(img_path) for img_path in glob.glob(rootDir)]
        
cv_imgs_masks = nnet.detect_mask(imgs)

for img, cv_img_masks in zip(imgs, cv_imgs_masks):
    # Generate splashs.
    splashs = filters.color_splash(img, cv_img_masks)
    for splash in splashs:
        plt.imshow(splash)
        plt.axis("off")
        plt.show()
