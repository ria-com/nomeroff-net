"""
An example that uses TensorRT's Python api to make inferences.
"""
from nomeroff_net.pipes.number_plate_localizators.yolo_v8_detector import Detector as YoloDetector


class Detector(YoloDetector):
    """
    description: A YOLOv8 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def load_model(self, weights: str, *args) -> None:
        from ultralytics import YOLO

        model = YOLO(weights)
        self.model = model
