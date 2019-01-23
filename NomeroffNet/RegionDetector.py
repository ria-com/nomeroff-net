import cv2
import numpy as np
from keras.models import load_model
from .Base import ImgClassificator
from keras.models import Model, Input

class RegionDetector(ImgClassificator):
    def __init__(self):
        ImgClassificator.__init__(self)
        # input
        self.HEIGHT         = 64
        self.WEIGHT         = 295

        # outputs
        self.CLASS_LABELS = ["xx-unknown", "eu-ua-2015", "eu-ua-2004", "eu-ua-1995", "eu", "xx-transit"]

class SquireRegionDetector(ImgClassificator):
    def __init__(self):
        ImgClassificator.__init__(self)
        # input
        self.HEIGHT         = 100
        self.WEIGHT         = 200

        # outputs
        self.CLASS_LABELS = ["xx-unknown-squire", "eu-ua-2015-squire", "eu-ua-2004-squire", "eu-ua-1995-squire", "eu-squire", "xx-transit-squire"]

    def load(self, path_to_model, verbose = 0):
        self.MODEL = load_model(path_to_model)

        #self.MODEL.layers.pop(0)
        #newInput = Input(shape=(self.HEIGHT, self.WEIGHT, self.COLOR_CHANNELS))
        #newOutputs = self.MODEL(newInput)
        #self.MODEL = Model(newInput, newOutputs)

        if verbose:
            self.MODEL.summary()

class SquireDetector(ImgClassificator):
    def __init__(self):
        ImgClassificator.__init__(self)
        # input
        self.HEIGHT         = 100
        self.WEIGHT         = 200

        # outputs
        self.CLASS_LABELS = ["not_squire", "squire"]