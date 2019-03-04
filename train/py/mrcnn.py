import os
import cv2
import numpy as np
import sys
import json
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, "logs")

DATASET_NAME = "mrcnn"
VERSION = "2019_2_20"
MASK_RCNN_FROZEN_PATH = os.path.join(NOMEROFF_NET_DIR, "models/", 'numberplate_{}_{}.pb'.format(DATASET_NAME, VERSION))

# Import license plate recognition tools.
from NomeroffNet import  Detector
from NomeroffNet.Base import convert_keras_to_freeze_pb

CONFIG = {
    "GPU_COUNT": 1,
    "IMAGES_PER_GPU": 1,
    "WEIGHTS": "coco",
    "EPOCHS": 100,
    "CLASS_NAMES": ["BG", "numberplate"],
    "NAME": "numberplate",
    "DATASET_DIR": "../datasets/mrcnn",
    "LAYERS": "all",
    "NUM_CLASSES": 2
}

# Initialize npdetector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR, CONFIG)

nnet.train()
#nnet.loadModel(MASK_RCNN_MODEL_PATH)

model = nnet.getKerasModel()
convert_keras_to_freeze_pb(model,MASK_RCNN_FROZEN_MODEL_PATH)