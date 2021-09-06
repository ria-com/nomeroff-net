# Specify device
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import all necessary libraries.
import sys
import cv2

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../../')

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet.YoloV5Detector import Detector
detector = Detector()
detector.load()

from NomeroffNet.TextDetectors.eu import eu
from NomeroffNet.TextPostprocessing import textPostprocessing

# load models
textDetector = eu
textDetector.load("latest")

# Detect numberplate
img_path = '../images/example2.jpeg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

targetBoxes = detector.detect_bbox(img)

zones = []
regionNames = []
for targetBox in targetBoxes:
    x = int(min(targetBox[0], targetBox[2]))
    w = int(abs(targetBox[2]-targetBox[0]))
    y = int(min(targetBox[1], targetBox[3]))
    h = int(abs(targetBox[3]-targetBox[1]))

    image_part = img[y:y + h, x:x + w]
    zones.append(image_part)
    regionNames.append('eu')

# find text with postprocessing by standart
textArr = textDetector.predict(zones)
textArr = textPostprocessing(textArr, regionNames)
print(textArr)
