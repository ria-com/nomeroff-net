"""
The example demonstrates license plate number detection.
python3 examples/py/inference/get-started-demo.py
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
nomeroff_net_dir = os.path.join(current_dir, "../../../")
sys.path.append(nomeroff_net_dir)

import glob
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

if __name__ == '__main__':
    number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="opencv")

    result = number_plate_detection_and_reading(glob.glob(
        os.path.join(nomeroff_net_dir, './data/examples/oneline_images/*')))

    (images, images_bboxs,
     images_points, images_zones, region_ids,
     region_names, count_lines,
     confidences, texts) = unzip(result)

    # (['AC4921CB'], ['RP70012', 'JJF509'])
    print(texts)
