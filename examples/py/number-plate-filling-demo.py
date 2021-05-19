# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # For CPU inference

# Import all necessary libraries.
import sys
import glob
import matplotlib.image as mpimg
import cv2
import copy

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

from NomeroffNet.YoloV5Detector import Detector

detector = Detector()
detector.load()

rootDir = '../images/*'

imgs = [mpimg.imread(img_path) for img_path in glob.glob(rootDir)]

for img in imgs:
    targetBoxes = detector.detect_bbox(copy.deepcopy(img))
    targetBoxes = targetBoxes

    # draw rect and 4 points
    for targetBox in targetBoxes:
        cv2.rectangle(img,
                      (int(targetBox[0]), int(targetBox[1])),
                      (int(targetBox[2]), int(targetBox[3])),
                      (0, 0, 0),
                      -1)
    cv2.imshow("Display window", img)
    k = cv2.waitKey(0)
