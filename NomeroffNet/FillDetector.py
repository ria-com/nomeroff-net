import cv2
import numpy as np
from keras.models import load_model
from .Base import ImgClassificator

class FillDetector(ImgClassificator):
    def __init__(self):
        ImgClassificator.__init__(self)
        # input
        self.HEIGHT         = 64
        self.WEIGHT         = 295

        # outputs
        self.CLASS_LABELS = ["filled", "not_filled", "not_number"]

    def normalize(self, img):
        img = img / 255.
        img = cv2.resize(img, (295, 64))
        return img