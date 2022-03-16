"""
python3 -m nomeroff_net.image_loaders.turbo_loader
"""
import os
from turbojpeg import TurboJPEG
from turbojpeg import TJPF_RGB
from .base import BaseImageLoader


class TurboImageLoader(BaseImageLoader):
    def __init__(self, **kwargs):
        self.jpeg = TurboJPEG(**kwargs)

    def load(self, img_path):
        with open(img_path, 'rb') as in_file:
            img = self.jpeg.decode(in_file.read(), TJPF_RGB)
        return img


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_file = os.path.join(current_dir, "../../data/examples/oneline_images/example1.jpeg")

    image_loader = TurboImageLoader()
    loaded_img = image_loader.load(img_file)
