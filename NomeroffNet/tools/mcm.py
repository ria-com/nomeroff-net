# import python modules
import os

from modelhub_client import (ModelHub,
                             models_example)

models = {
    "numberplate_orientation": {
        "application": "OrientationDetector",
        "url": "https://nomeroff.net.ua"
               "/models/orientation/torch/numberplate_orientations_2021_07_12_pytorch_lightning.ckpt",
        "orientations": ["0°", "90°", "180°", "270°"],
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2021-05-12.zip",
    },
    "numberplate_options": {
        "application": "OptionsDetector",
        "url": "https://nomeroff.net.ua/models/options/torch/numberplate_options_2021_05_23.pt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOptionsDataset-2021-07-08.zip",
        "class_region": [
            "xx_unknown",
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
    },
    "yolov5": {
        "application": "Detector",
        "url": "https://nomeroff.net.ua/models/object_detection/yolov5s-2021-07-28.pt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2021-05-12.zip",
        "repo": "https://github.com/ultralytics/yolov5.git",
    },
    "yolov5engine": {
        "application": "Detector",
        "url": "https://nomeroff.net.ua/models/object_detection/yolov5s-2021-05-14.engine",
    },
    "craft_mlt": {
        "application": "NpPointsCraft",
        "url": "https://nomeroff.net.ua/models/scene_text_detection/craft_mlt_25k_2020-02-16.pth",
        "repo": "https://github.com/clovaai/CRAFT-pytorch.git",
    },
    "craft_refiner": {
        "application": "NpPointsCraft",
        "url": "https://nomeroff.net.ua/models/scene_text_detection/craft_refiner_CTW1500_2020-02-16.pth",
        "repo": "https://github.com/clovaai/CRAFT-pytorch.git",
    },
    "EuUaFrom2004": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/ua/tf2/anpr_ocr_ua_2021_01_15_tensorflow_v2.h5",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrUa-2020-12-21.zip",
    },
    "EuUa1995": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/ua-1995/tf2/anpr_ocr_ua-1995_2021_01_12_tensorflow_v2.h5",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrUa-1995-2021-01-12.zip"
    },
    "Eu": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/eu/tf2/anpr_ocr_eu_2020_10_08_tensorflow_v2.3.h5",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrEu-2020-10-09.zip"
    },
    "Ru": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/ru/tf2/anpr_ocr_ru_2020_10_12_tensorflow_v2.3.h5",
        "datasets": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2020-10-12.zip",
    },
    "Kz": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/kz/tf2/anpr_ocr_kz_2020_08_26_tensorflow_v2.h5",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrKz-2019-04-26.zip"
    },
    "Ge": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/ge/tf2/anpr_ocr_ge_2020_08_21_tensorflow_v2.h5",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrGe-2019-07-06.zip"
    },
    "By": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/by/tf2/anpr_ocr_by_2020_10_09_tensorflow_v2.3.h5",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrBy-2020-10-09.zip",
    },
    "Su": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/su/anpr_ocr_su_2020_11_27_tensorflow_v2.3.h5",
        "datasets": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrSu-2020-11-27.zip"
    },
    "Kg": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/kg/tf2/anpr_ocr_kg_2020_12_31_tensorflow_v2.3.h5",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrKg-2020-12-31.zip",
    },
    "Am": {
        "application": "TextDetector",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrAm-2021-05-20-all-draft.zip",
    }
}

# initial
local_storage = os.environ.get('LOCAL_STORAGE', os.path.join(os.path.dirname(__file__), "../../data"))
modelhub = ModelHub(models=models,
                    local_storage=local_storage)


def get_mode() -> str:
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    if len(devices):
        return "gpu"
    return "cpu"


def get_mode_torch() -> str:
    import torch
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"
