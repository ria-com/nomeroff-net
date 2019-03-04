# Import all necessary libraries.
import sys
import os
import json
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')

# Load default configuration file.
NOMEROFF_NET_DIR = "../../"
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, "Mask_RCNN/")

MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, "logs/")
MASK_RCNN_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/mask_rcnn_numberplate_0700.pb")

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import Detector, filters

# Initialize the detector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)

# Load weights in keras format.
nnet.loadModel(MASK_RCNN_MODEL_PATH)

# Walking through the ./examples/images/ directory and checking each of the images for license plates.
rootDir = '../images/'

i = 0
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        img_path = os.path.join(dirName, fname)
        img = mpimg.imread(img_path)

        np = nnet.detect([img])

        # Generate splashs.
        splashs = filters.color_splash(img, np)
        for splash in splashs:
            mpimg.imsave(os.path.join(dirName, "{}.png".format(i)), splash)
            i += 1