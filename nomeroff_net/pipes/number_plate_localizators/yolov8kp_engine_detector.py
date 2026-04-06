"""
An example that uses TensorRT's Python api to make inferences.
"""
from nomeroff_net.pipes.number_plate_localizators.yolo_kp_detector import Detector as YoloDetector


class Detector(YoloDetector):
    """
    description: A YOLOv8 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def load_model(self, weights: str, device: str = "") -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights, task="pose")
        self.device = device or self.device
