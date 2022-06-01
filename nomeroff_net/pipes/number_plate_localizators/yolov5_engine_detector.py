"""
An example that uses TensorRT's Python api to make inferences.
"""
import pycuda.autoinit
import torch
from nomeroff_net.tools.mcm import modelhub
from nomeroff_net.pipes.number_plate_localizators.yolo_v5_detector import Detector as YoloDetector

# download and append to path yolo repo
info = modelhub.download_repo_for_model("yolov5")
repo_path = info["repo_path"]


class Detector(YoloDetector):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self) -> None:
        self.model = None

    def load_model(self, weights: str, *args) -> None:
        self.model = torch.hub.load(repo_path, 'custom',  path=weights, source="local")
