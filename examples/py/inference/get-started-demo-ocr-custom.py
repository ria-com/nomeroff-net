"""
The example demonstrates license plate number detection.
python3 examples/py/inference/get-started-demo-ocr-custom.py
"""

import os
import matplotlib.pyplot as plt
from _paths import nomeroff_net_dir

from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
from nomeroff_net.pipes.number_plate_classificators.options_detector import CLASS_REGION_ALL

number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading",
                                              # setup ocrs (presets key is detector file name in dir ./nomeroff_net/text_detectors/ )
                                              presets={
                                                "ru": {
                                                    "for_regions": CLASS_REGION_ALL,
                                                    "model_path": "latest"
                                                },
                                              },
                                              default_label="ru",
                                              default_lines_count=1,
                                              # if you not need detect region or count lines
                                              off_number_plate_classification=True,
                                              image_loader="opencv")

result = number_plate_detection_and_reading([
    os.path.join(nomeroff_net_dir, './data/examples/oneline_images/20190525.jpg'),
])

(images, images_bboxs,
 images_points, images_zones, region_ids,
 region_names, count_lines,
 confidences, texts) = unzip(result)

print(texts)
