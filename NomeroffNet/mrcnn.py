import os
import json
import numpy as np

from mrcnn.config import Config
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from pycocotools import mask as maskUtils

############################################################
#  Configurations
############################################################
class InferenceConfig(Config):
    def __init__(self, config):
         """
             Configuration for training on the  dataset.
             Derives from the base Config class and overrides some values.
             https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py.
         """
         self.__dict__ = config
         super(InferenceConfig, self).__init__()