"""
python3 nomeroff_net/image_loaders/base.py
"""
from abc import abstractmethod


class BaseImageLoader(object):

    @abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError("load not implemented")


if __name__ == "__main__":
    base_image_loader = BaseImageLoader()
