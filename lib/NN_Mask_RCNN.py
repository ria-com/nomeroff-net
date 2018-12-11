from mrcnn.config import Config

class InferenceConfig(Config):
    def __init__(self, config):
        self.NAME = config["NAME"]
        self.GPU_COUNT = config["GPU_COUNT"]
        self.IMAGES_PER_GPU = config["IMAGES_PER_GPU"]
        self.NUM_CLASSES = config["NUM_CLASSES"]
        self.DETECTION_MIN_CONFIDENCE = config["DETECTION_MIN_CONFIDENCE"]

        super(InferenceConfig, self).__init__()