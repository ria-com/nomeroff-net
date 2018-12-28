from PIL import Image
import pytesseract
import cv2
import os
import re
import sys
import re
import importlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import postprocess

class TextDetector:
    def __init__(self, config):
        self.__dict__ = config

    def detect(self, cv_img, standart="NPPatterns", options = None):
        # set default options
        if options == None:
            options = self.DEFAULT_OPTIONS

        # load the cv_img as a PIL/Pillow cv_img, apply OCR, and then delete the temporary file
        text = pytesseract.image_to_string(Image.fromarray(cv_img), lang=options["LANG"], config=" --psm {0} --oem {1}".format(options["PSM"], options["OEM"]))

        if standart in dir(postprocess):
           PostprocessManager = getattr(getattr(postprocess, standart), standart)
        else:
           PostprocessManager = getattr(getattr(postprocess, "NPPatterns"), "NPPatterns")
        postprocessManager = PostprocessManager()
        return postprocessManager.find(text)