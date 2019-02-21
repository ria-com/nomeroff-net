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

        DEFAULT_MASK_RCNN_CONFIG = {
          "NAME": "numberplate",
          "GPU_COUNT": 1,
          "IMAGES_PER_GPU": 1,
          "NUM_CLASSES": 2,
          "DETECTION_MIN_CONFIDENCE": 0.7,
          "IMAGE_MAX_DIM": 1024, # work ?
          "IMAGE_RESIZE_MODE": "square" # work ?
        }
        self.NN_MASK_RCNN_CONFIG = mask_rcnn_config or DEFAULT_MASK_RCNN_CONFIG
        sys.path.append(self.MASK_RCNN_DIR)

        from .mrcnn import InferenceConfig
        self.CONFIG = InferenceConfig(self.NN_MASK_RCNN_CONFIG)

    def loadModel(self, model_path, verbose = 0):
        import mrcnn.model as modellib
        self.MODEL = modellib.MaskRCNN(mode="inference", model_dir=self.LOG_DIR, config=self.CONFIG)
        self.MODEL.load_weights(model_path, by_name=True)

    def normalize(self, images):
        res = []
        for image in images:
            # delete 4 chanel
            res.append(image[..., :3])
        return res;

    def detectFromFile(self, image_paths, verbose = 0):
        images = [mpimg.imread(image_path) for image_path in image_paths]
        return self.detect(images, verbose=verbose)

    def detect(self, images, verbose = 0):
        return self.MODEL.detect(self.normalize(images), verbose=verbose)

    ############################################################
    #  Training
    ############################################################
    def train(self, augmentation=None, verbose=1):
        from .mrcnn import Dataset
        import mrcnn.model as modellib
        from mrcnn import utils
        from imgaug import augmenters as iaa
        if verbose:
            self.CONFIG.display()
        model = modellib.MaskRCNN(mode="training", config=self.CONFIG, model_dir=self.LOG_DIR)

        # Select weights file to load
        assert self.CONFIG.WEIGHTS and type(self.CONFIG.WEIGHTS) is str

        if self.CONFIG.WEIGHTS.lower() == "coco":
            weights_path = os.path.join(self.LOG_DIR, "mask_rcnn_coco.h5")
            # Download weights file
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
        elif self.CONFIG.WEIGHTS.lower() == "last":
            # Find last trained weights
            weights_path = model.find_last()
        elif self.CONFIG.WEIGHTS.lower() == "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = self.CONFIG.WEIGHTS

        # Load weights
        if verbose:
            print("Loading weights ", weights_path)
        if self.CONFIG.WEIGHTS.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)

        # Training dataset.
        if verbose:
            print("Prepare train data")
        dataset_train = Dataset()
        dataset_train.load_numberplate("train", self.CONFIG)
        dataset_train.prepare()

        # Validation dataset
        if verbose:
            print("Prepare validation data")
        dataset_val = Dataset()
        dataset_val.load_numberplate("val", self.CONFIG)
        dataset_val.prepare()

        # Image augmentation
        # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
        augmentation_default = iaa.SomeOf((0, 2), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180),
                       iaa.Affine(rotate=270)]),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0))
        ])
        augmentation = augmentation or augmentation_default

        # *** This training schedule is an example. Update to your needs ***
        # Since we're using a very small dataset, and starting from
        # COCO trained weights, we don't need to train too long. Also,
        # no need to train all layers, just the heads should do it.
        if verbose:
            print("Training network")
        model.train(dataset_train, dataset_val,
                    learning_rate=self.CONFIG.LEARNING_RATE,
                    epochs=self.CONFIG.EPOCHS,
                    augmentation=augmentation,
                    layers=self.CONFIG.LAYERS )