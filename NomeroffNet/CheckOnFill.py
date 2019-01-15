import cv2
import numpy as np
from keras.models import load_model
from .Base import ImgClassificator

class RegionDetector(ImgClassificator):
    def __init__(self):
        self.MODEL = None
        self.CLASS_LABELS = ["not_number", "filled", "not_filled"]
        ImgClassificator.__init__(self)