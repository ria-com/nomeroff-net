"""
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python3 number-plate-recognition-trt.py
"""

import os
import sys
import faulthandler


faulthandler.enable()

current_dir = os.path.dirname(os.path.realpath(__file__))
nomeroff_net_dir = os.path.join(current_dir, "../../../")
sys.path.append(nomeroff_net_dir)

from glob import glob
import pycuda.autoinit

from nomeroff_net.tools import unzip
from nomeroff_net.pipelines.number_plate_detection_and_reading_trt import NumberPlateDetectionAndReadingTrt


if __name__ == '__main__':
    number_plate_detection_and_reading_trt = NumberPlateDetectionAndReadingTrt(
        task="number_plate_detection_and_reading_trt",
        image_loader="opencv",

        # numberplate detector trt paths
        path_to_model=os.path.join(nomeroff_net_dir,
                                   "./data/model_repository/yolov5s/1/model.engine"),
        plugin_lib=os.path.join(nomeroff_net_dir,
                                "./data/model_repository/yolov5s/1/libmyplugins.so"),

        # numberplate classification trt paths
        path_to_classification_model=os.path.join(nomeroff_net_dir,
                                                  "./data/model_repository/numberplate_options/1/model.trt"),
        options={
            "class_region": [
                    "military",
                    "eu_ua_2015",
                    "eu_ua_2004",
                    "eu_ua_1995",
                    "eu",
                    "xx_transit",
                    "ru",
                    "kz",
                    "eu-ua-fake-dpr",
                    "eu-ua-fake-lpr",
                    "ge",
                    "by",
                    "su",
                    "kg",
                    "am"
                ],
            "count_lines": [
                    1,
                    2,
                    3
                ],
        },

        # numberplate text recognition trt paths
        presets={
            "eu_ua_2004_2015": {
                "for_regions": ["eu_ua_2015", "eu_ua_2004"],
                "model_path": os.path.join(nomeroff_net_dir,
                                           "./data/model_repository/ocr-eu_ua_2004_2015/1/model.trt")
            },
            "eu_ua_1995": {
                "for_regions": ["eu_ua_1995"],
                "model_path": os.path.join(nomeroff_net_dir,
                                           "./data/model_repository/ocr-eu_ua_1995/1/model.trt")
            },
            "eu": {
                "for_regions": ["eu", "xx_unknown"],
                "model_path": os.path.join(nomeroff_net_dir,
                                           "./data/model_repository/ocr-eu/1/model.trt")
            },
            "ru": {
                "for_regions": ["ru", "eu-ua-ordlo-lpr", "eu-ua-ordlo-dpr"],
                "model_path": os.path.join(nomeroff_net_dir,
                                           "./data/model_repository/ocr-ru/1/model.trt")
            },
            "kz": {
                "for_regions": ["kz"],
                "model_path": os.path.join(nomeroff_net_dir,
                                           "./data/model_repository/ocr-kz/1/model.trt")
            },
            "ge": {
                "for_regions": ["ge"],
                "model_path": os.path.join(nomeroff_net_dir,
                                           "./data/model_repository/ocr-ge/1/model.trt")
            },
            "su": {
                "for_regions": ["su"],
                "model_path": os.path.join(nomeroff_net_dir,
                                           "./data/model_repository/ocr-su/1/model.trt")
            }
        },
        default_label="eu",
    )

    result = number_plate_detection_and_reading_trt(glob(os.path.join(nomeroff_net_dir,
                                                                      './data/examples/oneline_images/*')))

    (images, images_bboxs, 
     images_points, images_zones, region_ids, 
     region_names, count_lines, 
     confidences, texts) = unzip(result)
    
    print(texts)
