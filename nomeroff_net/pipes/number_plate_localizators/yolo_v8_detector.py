import torch
import numpy as np
from typing import List
from nomeroff_net.pipes.number_plate_localizators.yolo_v5_detector import Detector as YoloDetector
from nomeroff_net.tools.mcm import (modelhub, get_device_torch)


class Detector(YoloDetector):
    """

    """
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self, numberplate_classes=None, yolo_model_type='yolov8') -> None:
        self.model = None
        self.numberplate_classes = ["numberplate"]
        if numberplate_classes is not None:
            self.numberplate_classes = numberplate_classes
        self.device = get_device_torch()
        self.yolo_model_type = yolo_model_type

    def load_model(self, weights: str, device: str = '') -> None:
        from ultralytics import YOLO

        device = device or self.device
        # model = torch.hub.load(repo_path, 'custom', path=weights, source="local")
        model = YOLO(weights)
        model.to(device)
        # if device != 'cpu':  # half precision only supported on CUDA
        #     model.half()  # to FP16
        self.model = model
        self.device = device

    def load(self, path_to_model: str = "latest") -> None:
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name(self.yolo_model_type)
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

    def convert_model_outputs_to_array(self, model_outputs):
        return [self.convert_model_output_to_array(model_output) for model_output in model_outputs]

    @staticmethod
    def convert_model_output_to_array(result):
        model_output = []
        for item, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy()):
            model_output.append([item[0], item[1], item[2], item[3], conf, int(cls)])
        return model_output

    @torch.no_grad()
    def predict(self, imgs: List[np.ndarray], min_accuracy: float = 0.4) -> np.ndarray:
        model_outputs = self.model(imgs, conf=min_accuracy, verbose=False, save=False, save_txt=False, show=False,
                                   iou=0.7, agnostic_nms=True)
        result = self.convert_model_outputs_to_array(model_outputs)
        return np.array(result)
