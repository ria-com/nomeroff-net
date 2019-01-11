from PIL import Image
import pytesseract
import cv2
import os
import re
import sys
import re
import importlib

class tesseract:
    def __init__(self):
        self.CONFIG = "--psm 6 --oem 2"
        self.LANG = "eng"

    def detect(self, img,  lang = None, config = None):
        lang = lang or self.LANG
        config = config or self.CONFIG

        # load the cv_img as a PIL/Pillow cv_img, apply OCR, and then delete the temporary file
        text = pytesseract.image_to_string(Image.fromarray(img), lang=lang, config=config)
        return text