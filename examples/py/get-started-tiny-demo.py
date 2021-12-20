# Specify device
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import all necessary libraries.
import sys
import cv2

# nomeroff_net path
NOMEROFF_NET_DIR = os.path.abspath('../../')

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from nomeroff_net.pipes.number_plate_localizators.yolo_v5_detector import Detector
detector = Detector()
detector.load()

from nomeroff_net.text_detectors.eu import eu
from nomeroff_net.pipes.number_plate_text_readers.text_postprocessing import text_postprocessing
from nomeroff_net.tools.image_processing import crop_image

# load models
text_detector = eu()
text_detector.load("latest")

# Detect numberplate
img_path = '../images/example2.jpeg'
img = cv2.imread(img_path)
img = img[..., ::-1]

target_boxes = detector.detect_bbox(img)

zones = []
region_names = []
for target_box in target_boxes:
    image_part, (x, w, y, h) = crop_image(img, target_box)
    zones.append(image_part)
    region_names.append('eu')

# find text with postprocessing by standart
text_arr = text_detector.predict(zones)
text_arr = text_postprocessing(text_arr, region_names)
print(text_arr)
