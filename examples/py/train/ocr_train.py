import os
import sys
import warnings
warnings.filterwarnings('ignore')

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../../../')

DATASET_NAME = "test"
VERSION = "test"
PATH_TO_DATASET = os.path.join(NOMEROFF_NET_DIR, "datasets/ocr/", DATASET_NAME)
RESULT_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/", 'anpr_ocr_{}_{}.h5'.format(DATASET_NAME, VERSION))

sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net.Base import OCR

class Test(OCR):
    def __init__(self):
        OCR.__init__(self)
        # only for usage model
        # in train generate automaticly
        self.letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "E", "H", "I", "K", "M", "O", "P", "T", "X"]
        
        self.EPOCHS = 50
        
ocrTextDetector = Test()

model = ocrTextDetector.prepare(PATH_TO_DATASET, use_aug=0)
model = ocrTextDetector.train(is_random=0)

ocrTextDetector.test(verbose=True)

ocrTextDetector.save(RESULT_MODEL_PATH, verbose=True)

# Train with aug
ocrTextDetector = Test()
ocrTextDetector.EPOCHS = 50

model = ocrTextDetector.prepare(PATH_TO_DATASET, use_aug=True)
model = ocrTextDetector.train(load_last_weights=True)

ocrTextDetector.test(verbose=True)
ocrTextDetector.save(RESULT_MODEL_PATH, verbose=True)