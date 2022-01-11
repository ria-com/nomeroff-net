import os
from glob import glob
import pycuda.autoinit

from _paths import current_dir
from nomeroff_net.tools import unzip
from nomeroff_net.pipelines.number_plate_detection_and_reading_trt import NumberPlateDetectionAndReadingTrt

if __name__ == '__main__':
    number_plate_detection_and_reading_trt = NumberPlateDetectionAndReadingTrt(
        task="number_plate_detection_and_reading_trt",
        image_loader="opencv",
        path_to_model="../../inference/convertors/yolo2tensorrt/yolov5s-2021-12-14.engine",
        plugin_lib="../../inference/convertors/yolo2tensorrt/libmyplugins.so",
        path_to_classification_model="../../data/model_repository/numberplate_options/1/model.onnx",
        prisets={
            "eu_ua_2004_2015": {
                "for_regions": ["eu_ua_2015", "eu_ua_2004"],
                "model_path": "../../data/model_repository/ocr-eu_ua_2004_2015/1/model.onnx"
            },
            "eu_ua_1995": {
                "for_regions": ["eu_ua_1995"],
                "model_path": "../../data/model_repository/ocr-eu_ua_1995/1/model.onnx"
            },
            "eu": {
                "for_regions": ["eu"],
                "model_path": "../../data/model_repository/ocr-eu/1/model.onnx"
            },
            "ru": {
                "for_regions": ["ru", "eu-ua-ordlo-lpr", "eu-ua-ordlo-dpr"],
                "model_path": "../../data/model_repository/ocr-ru/1/model.onnx"
            },
            "kz": {
                "for_regions": ["kz"],
                "model_path": "../../data/model_repository/ocr-kz/1/model.onnx"
            },
            "ge": {
                "for_regions": ["ge"],
                "model_path": "../../data/model_repository/ocr-ge/1/model.onnx"
            },
            "su": {
                "for_regions": ["su"],
                "model_path": "../../data/model_repository/ocr-su/1/model.onnx"
            }
        },
        default_label="eu"
    )

    result = number_plate_detection_and_reading_trt(glob(os.path.join(current_dir, '../images/*')))

    (images, images_bboxs,
     images_points, images_zones, region_ids,
     region_names, count_lines,
     confidences, texts) = unzip(result)

    print(texts)
