import numpy as np
from PIL import Image
from .base import BaseImageLoader


class PillowImageLoader(BaseImageLoader):
    def load(self, img_path):
        im = Image.open(img_path)
        img = np.asarray(im)
        return img
