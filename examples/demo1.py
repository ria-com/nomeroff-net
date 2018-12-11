import sys
import os
import json

with open('../config/default.json') as data_file:
    config = json.load(data_file)

sys.path.append(os.path.abspath(config["NOMEROFF_NET"]["ROOT"]))

from NomeroffNet import Detector, filters

nnet = Detector(config)
nnet.loadModel()
np = nnet.detect(["/var/www/py/nomeroff-net/examples/images/250024318.jpeg"])
masks = filters.mask(np)

print(masks)