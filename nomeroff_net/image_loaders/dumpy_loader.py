from .base import BaseImageLoader


class DumpyImageLoader(BaseImageLoader):

    def load(self, img):
        return img
