import sys
import os
from konfig import Config

sys.path.append("../")

config = Config('../config/default.ini')

from lib.NN import NNNET
import lib.filters as filters

nnnet = NNNET(config)

nnnet.loadModel()
np = nnnet.detect(["/var/www/py/nomeroff-net/examples/images/250024318.jpeg"])
masks = filters.mask(np)

print(masks)