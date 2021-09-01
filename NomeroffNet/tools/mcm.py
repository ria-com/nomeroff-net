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
        "url": "https://nomeroff.net.ua/models/options/torch/numberplate_options_2021_08_13_pytorch_lightning.ckpt",
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
        "url": "https://nomeroff.net.ua/models/ocr/ua/torch/anpr_ocr_eu_2004_2015_2021_08_25_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrUa-2021-08-25.zip",
    },
    "EuUa1995": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/ua-1995/torch/anpr_ocr_eu_1995_2021_08_25_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrUa-1995-2021-08-25.zip",
    },
    "Eu": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/eu/torch/anpr_ocr_eu_2021_08_30_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrEu-2020-10-09.zip"
    },
    "Ru": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/ru/torch/anpr_ocr_ru_2021_09_01_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2021-09-01.zip",
    },
    "Kz": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/kz/torch/anpr_ocr_kz_2021_09_01_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrKz-2019-04-26.zip"
    },
    "Ge": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/ge/torch/anpr_ocr_ge_2021_08_30_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrGe-2019-07-06.zip"
    },
    "By": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/by/torch/anpr_ocr_by_2021_08_29_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrBy-2021-08-27.zip",
    },
    "Su": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/su/torch/anpr_ocr_su_2021_08_30_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrSu-2021-08-27.zip"
    },
    "Kg": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/kg/torch/anpr_ocr_kg_2021_08_30_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrKg-2020-12-31.zip",
    },
    "Am": {
        "application": "TextDetector",
        "url": "https://nomeroff.net.ua/models/ocr/am/torch/anpr_ocr_am_2021_08_30_pytorch_lightning.ckpt",
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
