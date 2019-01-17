import os
import cv2
import sys
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Detector:
    def __init__(self, mask_rcnn_dir, log_dir, mask_rcnn_config = None):
        self.MASK_RCNN_DIR = mask_rcnn_dir
        self.LOG_DIR = log_dir

        self.CLASS_NAMES = ["BG", "NUMBERPLATE"]

        DEFAULT_MASK_RCNN_CONFIG = {
          "NAME": "numberplate",
          "GPU_COUNT": 1,
          "IMAGES_PER_GPU": 1,
          "NUM_CLASSES": 2,
          "DETECTION_MIN_CONFIDENCE": 0.7,
          "IMAGE_MAX_DIM" = 1024, # work ?
          "IMAGE_RESIZE_MODE" = "square" # work ?
        }
        self.NN_MASK_RCNN_CONFIG = mask_rcnn_config or DEFAULT_MASK_RCNN_CONFIG
        sys.path.append(self.MASK_RCNN_DIR)

    def loadModel(self, model_path, verbose = 0):
        import mrcnn.model as modellib
        from .mrcnn import InferenceConfig

        config = InferenceConfig(self.NN_MASK_RCNN_CONFIG)
        self.MODEL = modellib.MaskRCNN(mode="inference", model_dir=self.LOG_DIR, config=config)
        self.MODEL.load_weights(model_path, by_name=True)

    def normalize(self, images):
        res = []
        for image in images:
            # delete 4 chanel
            res.append(image[..., :3])
        return res;

    def detectFromFile(self, image_paths, verbose = 0):
        images = [mpimg.imread(image_path) for image_path in image_paths]
        return self.detect(images), verbose=verbose)

    def detect(self, images, verbose = 0):
        return self.MODEL.detect(self.normalize(images), verbose=verbose)