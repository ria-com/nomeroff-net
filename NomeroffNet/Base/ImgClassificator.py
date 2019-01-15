import cv2
import numpy as np
from keras.models import load_model

class ImgClassificator():
    def __init__(self):
        pass

    def isLoaded(self):
        if self.MODEL == None:
            return False
        return True

    def load(self, path_to_model, verbose = 0):
        self.MODEL = load_model(path_to_model)
        if verbose:
            self.MODEL.summary()

    def getLabels(self, index):
        return self.CLASS_LABELS[index]

    def normalize(self, img):
        img = img / 255.
        img = cv2.resize(img, (64, 295))
        return img

    def predict(self, img):
        img = self.normalize(img)
        predicted = self.MODEL.predict(np.array([img]))
        return int(np.argmax(predicted))