import os
import sys
import warnings
warnings.filterwarnings('ignore')

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../../')

DATASET_NAME = "options"
VERSION = "2019_2_20"

LOG_DIR = os.path.join(NOMEROFF_NET_DIR, "logs/")
PATH_TO_DATASET = os.path.join(NOMEROFF_NET_DIR, "datasets/", DATASET_NAME)
RESULT_PATH = os.path.join(NOMEROFF_NET_DIR, "models/", 'numberplate_{}_{}.h5'.format(DATASET_NAME, VERSION))

FROZEN_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/", 'numberplate_{}_{}.pb'.format(DATASET_NAME, VERSION))

sys.path.append(NOMEROFF_NET_DIR)

from NomeroffNet import OptionsDetector
from NomeroffNet.Base import convert_keras_to_freeze_pb

# definde your parameters
# definde your parameters
class MyNumberClassificator(OptionsDetector):
    def __init__(self):
        OptionsDetector.__init__(self)
        # outputs 1
        self.CLASS_STATE = ["BACKGROUND", "FILLED", "NOT_FILLED"]

        # outputs 2
        self.CLASS_REGION = ["xx-unknown", "eu-ua-2015", "eu-ua-2004", "eu-ua-1995", "eu", "xx-transit"]

        self.EPOCHS           = 2
        self.BATCH_SIZE       = 10

        self.HEIGHT         = 64
        self.WEIGHT         = 295

# initialize region detector.
numberClassificator = MyNumberClassificator()
numberClassificator.prepare(PATH_TO_DATASET)

# train
model = numberClassificator.train(LOG_DIR)

numberClassificator.test()

numberClassificator.save(RESULT_PATH)
#model = numberClassificator.load(RESULT_PATH)
