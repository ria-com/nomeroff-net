from mrcnn.config import Config

class InferenceConfig(Config):
    def __init__(self, config):
        self.__dict__ = config
        super(InferenceConfig, self).__init__()