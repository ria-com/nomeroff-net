"""
python3 -m nomeroff_net.image_loaders.opencv_loader
"""
import os
import cv2
from .base import BaseImageLoader


class OpencvImageLoader(BaseImageLoader):
    def load(self, img_path):
        img = cv2.imread(img_path)
        img = img[..., ::-1]
        return img


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_file = os.path.join(current_dir, "../../data/examples/oneline_images/example1.jpeg")

    image_loader = OpencvImageLoader()
    loaded_img = image_loader.load(img_file)
