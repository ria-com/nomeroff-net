import cv2
from .base import BaseImageLoader


class OpencvImageLoader(BaseImageLoader):
    def load(self, img_path):
        img = cv2.imread(img_path)
        img = img[..., ::-1]
        return img
