from abc import abstractmethod


class BaseImageLoader(object):

    @abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError("load not implemented")
