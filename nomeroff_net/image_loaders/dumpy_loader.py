"""
python3 -m nomeroff_net.image_loaders.dumpy_loader
"""
import os
from .base import BaseImageLoader


class DumpyImageLoader(BaseImageLoader):

    def load(self, img):
        return img


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_file = os.path.join(current_dir, "../../data/examples/oneline_images/example1.jpeg")

    image_loader = DumpyImageLoader()
    loaded_img = image_loader.load(img_file)
