from .base import BaseImageLoader
from .opencv_loader import OpencvImageLoader
from .pillow_loader import PillowImageLoader
from .turbo_loader import TurboImageLoader
from .dumpy_loader import DumpyImageLoader

image_loaders_map = {
    "opencv": OpencvImageLoader,
    "cv2": OpencvImageLoader,
    "pillow": PillowImageLoader,
    "turbo": TurboImageLoader,
}
