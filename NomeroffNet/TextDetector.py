from PIL import Image
import pytesseract
import cv2
import os
import re

class TextDetector:
    def __init__(self, config):
        self.__dict__ = config

    def filter(self, text):
        return re.sub(r"\W", "", text)

    def detect(self, cv_img, options = None):
        # set default options
        if options == None:
            options = self.DEFAULT_OPTIONS

        # color masks
        if "red" == options["COLOR"]:
            blue, green, cv_img = cv2.split(cv_img)
        elif "green" == options["COLOR"]:
            blue, cv_img, red = cv2.split(cv_img)
        elif "blue" == options["COLOR"]:
            cv_img, green, red = cv2.split(cv_img)
        else:
            pass
            #cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # preprocessing cv_img
        if "thresh" in options["PREPROCESS"]:
            # check to see if we should apply thresholding to preprocess the cv_img
        	cv_img = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if "blur" in options["PREPROCESS"]:
            # make a check to see if median blurring should be done to remove noise
            cv_img = cv2.medianBlur(cv_img, 5)

        # load the cv_img as a PIL/Pillow cv_img, apply OCR, and then delete the temporary file
        text = pytesseract.image_to_string(Image.fromarray(cv_img), lang=options["LANG"], config=" --psm {0} --oem {1}".format(options["PSM"], options["OEM"]))
        return self.filter(text)
