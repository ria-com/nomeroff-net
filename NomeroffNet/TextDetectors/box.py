from Base import OCR
from tools import split

class box(OCR):
    def __init__(self):
        OCR.__init__(self)

    def normalize(self, img):
        img = split(img)
        return OCR.normalize(self, img)
