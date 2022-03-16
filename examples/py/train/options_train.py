import os
import sys
import warnings
warnings.filterwarnings('ignore')

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../../../')

DATASET_NAME = "options_test"
VERSION = "2020_08_20_tensorflow_v2"

LOG_DIR = os.path.join(NOMEROFF_NET_DIR, "logs/")
PATH_TO_DATASET = os.path.join(NOMEROFF_NET_DIR, "datasets/", DATASET_NAME)
RESULT_PATH = os.path.join(NOMEROFF_NET_DIR, "models/", 'numberplate_{}_{}.h5'.format(DATASET_NAME, VERSION))

sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net import OptionsDetector

# definde your parameters
class MyNpClassificator(OptionsDetector):
    def __init__(self):
        OptionsDetector.__init__(self)
        # outputs 1
        self.CLASS_STATE = ["garbage", "filled", "not filled", "empty"]
        
        # outputs 2
        self.CLASS_REGION = ["xx-unknown", "eu-ua-2015", "eu-ua-2004", "eu-ua-1995", "eu", "xx-transit", "ru", "kz", "eu-ua-ordlo-dnr", "eu-ua-ordlo-lnr", "ge"]
        
        # output 3
        self.CLASS_COUNT_LINE = ["0", "1", "2"]
        
        self.EPOCHS           = 100
        self.BATCH_SIZE       = 1 # 64
        
        self.HEIGHT         = 64
        self.WEIGHT         = 295

# initialize region detector.
npClassificator = MyNpClassificator()
npClassificator.prepare(PATH_TO_DATASET, verbose=1)

# train
model = npClassificator.train(LOG_DIR, cnn="simple")

npClassificator.test()

npClassificator.save(RESULT_PATH)