import torch
import numpy as np
from typing import List

from nomeroff_net.tools.mcm import (modelhub, get_device_torch)

# download and append to path yolo repo
info = modelhub.download_repo_for_model("yolov5")
repo_path = info["repo_path"]


class Detector(object):
    """

    """
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self, numberplate_classes=None) -> None:
        self.model = None
        self.numberplate_classes = ["numberplate"]
        if numberplate_classes is not None:
            self.numberplate_classes = numberplate_classes
        self.device = get_device_torch()

    def load_model(self, weights: str, device: str = '') -> None:
        device = device or self.device
        model = torch.hub.load(repo_path, 'custom', path=weights, source="local")
        model.to(device)
        if device != 'cpu':  # half precision only supported on CUDA
            model.half()  # to FP16

        self.model = model
        self.device = device

    def load(self, path_to_model: str = "latest") -> None:
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("yolov5")
            path_to_model = model_info["path"]
            self.numberplate_classes = model_info.get("classes", self.numberplate_classes)
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model, self.get_classname(), "numberplate_options")
            path_to_model = model_info["path"]
        elif path_to_model.startswith("modelhub://"):
            path_to_model = path_to_model.split("modelhub://")[1]
            model_info = modelhub.download_model_by_name(path_to_model)
            self.numberplate_classes = model_info.get("classes", self.numberplate_classes)
            path_to_model = model_info["path"]
        self.load_model(path_to_model)

    @torch.no_grad()
    def predict(self, imgs: List[np.ndarray], min_accuracy: float = 0.5) -> np.ndarray:
        model_outputs = self.model(imgs)
        model_outputs = [[[item["xmin"], item["ymin"], item["xmax"], item["ymax"], item["confidence"], item["class"]]
                         for item in img_item.to_dict(orient="records")
                         if item["confidence"] > min_accuracy]
                         for img_item in model_outputs.pandas().xyxy]
        return np.array(model_outputs)
