import os
import sys
import warnings
warnings.filterwarnings('ignore')

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../../')

DATASET_NAME = "ua"
VERSION = "12"
MODE = "cpu"
PATH_TO_DATASET = os.path.join(NOMEROFF_NET_DIR, "datasets/ocr/", DATASET_NAME)
RESULT_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/", 'anpr_ocr_{}_{}-{}.h5'.format(DATASET_NAME, VERSION, MODE))

FROZEN_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/", 'anpr_ocr_{}_{}-{}.pb'.format(DATASET_NAME, VERSION, MODE))

sys.path.append(NOMEROFF_NET_DIR)

from NomeroffNet.Base import OCR, convert_keras_to_freeze_pb

class ua(OCR):
    def __init__(self):
        OCR.__init__(self)
        # only for usage model
        # in train generate automaticly
        self.letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "E", "H", "I", "K", "M", "O", "P", "T", "X"]

        self.EPOCHS = 1

ocrTextDetector = ua()
model = ocrTextDetector.prepare(PATH_TO_DATASET, aug_count=0)

model = ocrTextDetector.train(mode=MODE)

ocrTextDetector.test(verbose=True)

ocrTextDetector.save(RESULT_MODEL_PATH, verbose=True)
#model = ocrTextDetector.load(RESULT_MODEL_PATH)
