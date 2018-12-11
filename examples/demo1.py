import sys
import os
from configparser import ConfigParser

sys.path.append("../")

import json
with open('../config/default.json') as data_file:
    config = json.load(data_file)

from NomeroffNet import Detector, filters

nnet = Detector(config)
nnet.loadModel()
np = nnet.detect(["/var/www/py/nomeroff-net/examples/images/250024318.jpeg"])
masks = filters.mask(np)

print(masks)