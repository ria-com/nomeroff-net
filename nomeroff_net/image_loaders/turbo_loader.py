from turbojpeg import TurboJPEG
from turbojpeg import TJPF_RGB
from .base import BaseImageLoader


class TurboImageLoader(BaseImageLoader):
    def __init__(self, **kwargs):
        self.jpeg = TurboJPEG(**kwargs)

    def load(self, img_path):
        with open(img_path, 'rb') as in_file:
            img = self.jpeg.decode(in_file.read())
            img = img[..., ::-1]
        return img
