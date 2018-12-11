# Import all necessary libraries.
import sys
import os
import json
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

# Load default configuration file.
with open('../config/default.json') as data_file:
    config = json.load(data_file)

sys.path.append(os.path.abspath(config["NOMEROFF_NET"]["ROOT"]))

# Import license plate recognition tools.
from NomeroffNet import Detector, filters

# Initialize the detector with default configuration file.
nnet = Detector(config)

# Load weights in keras format.
nnet.loadModel()

# Walking through the ./examples/images/ directory and checking each of the images for license plates.
rootDir = 'images/'

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        img_path = os.path.join(dirName, fname)

        np = nnet.detect([img_path])

        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.show()

        # Generate masks.
        masks = filters.mask(np)
        for mask in masks:
            plt.imshow(mask)
            plt.show()