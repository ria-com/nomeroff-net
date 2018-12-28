import re
import cv2
import numpy as np
from keras.models import load_model

class TextPostprocessing():
    def __init__(self, config):
        self.__dict__ = config

    def to_upper_case():

    def process(text, reg_exp, toUpperCase=True, deleteWhiteSpases=True):
        if type(text) is not str:
            raise ValueError("text is not str")

        # converting all letters to upper case
        if toUpperCase == True:
            text = text.upper()

        # delete all whitespaces
        if deleteWhiteSpases == True:
            text = re.sub(r"\s", "", text)


