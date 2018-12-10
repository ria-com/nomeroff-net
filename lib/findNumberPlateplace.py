import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

MASK_RCNN_DIR = os.path.abspath("/var/www/Mask_RCNN/");

ROOT_DIR = os.path.abspath("/var/www/nomeroff-net")
MODEL_DIR = os.path.join(ROOT_DIR, "samples/logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "samples/images")
NP_MODEL_PATH = os.path.join(ROOT_DIR, "lib/models/mask_rcnn_numberplate_0100.h5")

class_names = ['BG', 'NUMBERPLATE']

sys.path.append(MASK_RCNN_DIR)

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn import utils

class InferenceConfig(Config):
    NAME = "numberplate"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    DETECTION_MIN_CONFIDENCE = 0.5

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(NP_MODEL_PATH, by_name=True)

def detect(image_path, result_path):
    image = skimage.io.imread(image_path)
    r = model.detect([image], verbose=1)
    gray = skimage.color.gray2rgb(mask) * 255
    skimage.io.imsave(result_path, gray)