"""
python3 -m nomeroff_net.image_loaders.pillow_loader
"""
import os
import numpy as np
from PIL import Image
from .base import BaseImageLoader


class PillowImageLoader(BaseImageLoader):
    def load(self, img_path):
        im = Image.open(img_path)
        img = np.asarray(im)
        return img


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_file = os.path.join(current_dir, "../../data/examples/oneline_images/example1.jpeg")

    image_loader = PillowImageLoader()
    loaded_img = image_loader.load(img_file)
