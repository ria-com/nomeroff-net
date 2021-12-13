# import python modules
import os

from modelhub_client import ModelHub

models = {
    "numberplate_orientation": {
        "application": "OrientationDetector",
        "url": "https://nomeroff.net.ua"
               "/models/orientation/torch/numberplate_orientations_2021_07_12_pytorch_lightning.ckpt",
        "orientations": ["0°", "90°", "180°", "270°"],
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2021-05-12.zip",
    },
    "numberplate_orientations": {
        "application": "InverseDetector",
        "url": "https://nomeroff.net.ua/models/inverse/torch/numberplate_inverse_2021_09_12_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOrientationDataset-2021-09-12.zip",
        "orientations": [
            "0",
            "180"
        ],
    },
    "numberplate_options": {
        "application": "OptionsDetector",
        "url": "https://nomeroff.net.ua/models/options/torch/resnet18/numberplate_options_2021_11_25_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOptionsDataset-2021-11-25.zip",
        "class_region": [
            "military-ua",
            "eu-ua-2015",
            "eu-ua-2004",
            "eu-ua-1995",
            "eu",
            "xx-transit",
            "ru",
            "kz",
            "eu-ua-ordlo-dpr",
            "eu-ua-ordlo-lpr",
            "ge",
            "by",
            "su",
            "kg",
            "am",
        ],
        "count_lines": [
            1,
            2,
            3,
        ],
    },
    "numberplate_options_uacustom": {
        "application": "OptionsDetector",
        "url": "https://nomeroff.net.ua" +
               "/models/options/torch/resnet18/numberplate_options_2021_11_24_uacustom_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOptionsDataset-2021-11-25.zip",
        "class_region": [
            "eu-ua-2015",
            "eu-ua-2004",
            "eu-ua-1995",
            "eu",
            "xx-transit",
            "eu-ua-ordlo-dpr",
            "eu-ua-ordlo-lpr",
            "ge",
            "su",
        ],
        "count_lines": [
            1,
            2,
            3,
        ],
    },
    "yolov5": {
        "application": "Detector",
        "url": "https://nomeroff.net.ua/models/object_detection/yolov5s-2021-12-12.pt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2021-12-12.zip",
        "repo": "https://github.com/ultralytics/yolov5.git",
        #"commit_id": "27bf4282d3d5879f0f4f7492400675ba93a3db1b",
    },
    "yolov5engine": {
        "application": "Detector",
        "url": "https://nomeroff.net.ua/models/object_detection/yolov5s-2021-07-28.engine",
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
        "url": "https://nomeroff.net.ua/models/ocr/eu/torch/anpr_ocr_eu_2021_09_27_pytorch_lightning.ckpt",
        "dataset": "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrEu-2021-09-27.zip"
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


def get_mode_torch() -> str:
    import torch
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"
