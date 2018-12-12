import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, config):
        self.MASK_RCNN_DIR = os.path.abspath(config["MASK_RCNN"]["DIR"])
        self.ROOT = os.path.abspath(config["NOMEROFF_NET"]["ROOT"])
        self.LOG_DIR = os.path.join(self.ROOT, config["NOMEROFF_NET"]["LOG_DIR"])
        self.MODEL_PATH = os.path.join(self.ROOT, config["NOMEROFF_NET"]["MODEL_PATH"])

        self.CLASS_NAMES = config["NOMEROFF_NET"]["CLASS_NAMES"]
        self.NN_MASK_RCNN_CONFIG = config["NN_MASK_RCNN"]
        sys.path.append(self.MASK_RCNN_DIR)
        sys.path.append(self.ROOT)

    def loadModel(self, verbose = 0):
        import mrcnn.model as modellib
        from .mrcnn import InferenceConfig

        config = InferenceConfig(self.NN_MASK_RCNN_CONFIG)
        self.MODEL = modellib.MaskRCNN(mode="inference", model_dir=self.LOG_DIR, config=config)
        self.MODEL.load_weights(self.MODEL_PATH, by_name=True)

    def detectFromFile(self, image_paths, verbose = 0):
        images = [skimage.io.imread(image_path) for image_path in image_paths]
        return self.detect(images, verbose=verbose)

    def detect(self, images, verbose = 0):
        return self.MODEL.detect(images, verbose=verbose)