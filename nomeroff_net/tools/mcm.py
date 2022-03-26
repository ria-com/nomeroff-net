import os

from modelhub_client import ModelHub

model_config_urls = [
    "http://models.vsp.net.ua/config_model/nomeroff-net-np-classification/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-am/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-base/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-by/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-eu/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-eu_ua_1995/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-eu_ua_from_2004/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-ge/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-kg/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-kz/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-ru/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-ru-military/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ocr-su/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-ua-np-classification/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-yolov5/model-1.json",
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
