import cv2
import numpy as np
from keras.models import load_model

class StandartDetector():
    def __init__(self, config):
        self.MODEL = None
        self.CLASS_LABELS = ["euro", "ukr2015", "ukr2004"]
        self.__dict__ = config

    def load(self, path_to_model, verbose = 0):
        self.MODEL = load_model(path_to_model)
        if verbose:
            self.MODEL.summary()

    def getLabels(self, index):
        return self.CLASS_LABELS[index]

    def normalize(self, img):
        img = img / 255.
        img = cv2.resize(img, (295, 64))
        return img

    def predict(self, img):
        predicted = self.MODEL.predict(np.array([img]))
        return int(np.argmax(predicted))