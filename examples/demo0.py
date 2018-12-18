# **NOTE**: Before executing this example please clone "default.json.example" to "default.json" in ./config
# Import all necessary libraries.
import os
import sys
import json
import matplotlib.image as mpimg
# Load default configuration file.
with open('../config/default.json') as data_file:
    config = json.load(data_file)

sys.path.append(os.path.abspath(config["NOMEROFF_NET"]["ROOT"]))


# Import license plate recognition tools.
from NomeroffNet import Detector

# Initialize npdetector with default configuration file.
nnet = Detector(config)

# Load weights in keras format.
nnet.loadModel()

# Import license plate recognition tools.
from NomeroffNet import  filters, RectDetector, TextDetector

# Initialize rect detector with default configuration file.
rectDetector = RectDetector(config["RECT_DETECTOR"])

# Initialize text detector.
textDetector = TextDetector(config["TEXT_DETECTOR"])

# Detect numberplate
img_path = '../examples/ok/example1.jpeg'
img = mpimg.imread(img_path)
NP = nnet.detect([img])

# Generate image mask.
cv_img_masks = filters.cv_img_mask(NP)
for img_mask in cv_img_masks:
    # Detect points.
    points = rectDetector.detect(img_mask, outboundOffset=3, fixRectangleAngle=3)
    # Split on zones
    zones = rectDetector.get_cv_zones(img, points)
    for zone in zones:
        text = textDetector.detect(zone)
        print('Detected numberplate: %s'%text)