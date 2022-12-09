import os

from modelhub_client import ModelHub

model_config_urls = [
    # numberplate classification
    "http://models.vsp.net.ua/config_model/nomeroff-net-np-classification/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ua-np-classification/model-1.json",

    # ocr
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-am/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-by/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-eu/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-eu_ua_1995/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-eu_ua_from_2004/model-6.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-ge/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-kg/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-kz/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-ru/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-ru-military/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-su/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-md/model-1.json",

    # object detection
    "http://models.vsp.net.ua/config_model/nomeroff-net-yolov5/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-yolox/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-yolov5_brand_np/model-2.json",

    # text localization
    "http://models.vsp.net.ua/config_model/craft-mlt/model-1.json",
    "http://models.vsp.net.ua/config_model/craft-refiner/model-1.json",
]

# initial
local_storage = os.environ.get('LOCAL_STORAGE', os.path.join(os.path.dirname(__file__), "../../data"))
modelhub = ModelHub(model_config_urls=model_config_urls,
                    local_storage=local_storage)


def get_mode_torch() -> str:
    import torch
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def get_device_torch() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
