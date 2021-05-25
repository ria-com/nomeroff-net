# load default packages
import os
import sys
import pathlib
import torch
import numpy as np
from typing import List

# download and append to path yolo repo
NOMEROFF_NET_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "../")
YOLOV5_DIR = os.environ.get("YOLOV5_DIR", os.path.join(NOMEROFF_NET_DIR, 'yolov5'))
YOLOV5_URL = "https://github.com/ultralytics/yolov5.git"

if not os.path.exists(YOLOV5_DIR):
    from git import Repo
    Repo.clone_from(YOLOV5_URL, YOLOV5_DIR)
sys.path.append(YOLOV5_DIR)

# load yolo packages
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device, load_classifier, time_synchronized

# load NomerooffNet packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Base')))
from mcm.mcm import download_latest_model
from mcm.mcm import get_mode_torch


class Detector(object):
    """

    """
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.half = False

    def load_model(self, weights: str, device: str = 'cuda') -> None:
        device = select_device(device)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        half = device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()  # to FP16
        
        self.model = model
        self.device = device
        self.half = half

    def load(self, path_to_model: str = "latest") -> None:
        if path_to_model == "latest":
            model_info = download_latest_model(self.get_classname(), "yolov5x", ext="pt", mode=get_mode_torch())
            path_to_model = model_info["path"]
        device = "cpu"
        if get_mode_torch() == "gpu":
            device = "cuda"
        self.load_model(path_to_model, device)

    def detect_bbox(self, img: np.ndarray, img_size: int = 640, stride: int = 32, min_accuracy: float = 0.5) -> List:
        """
        TODO: input img in BGR format, not RGB; To Be Implemented in release 2.2
        """
        # normalize
        img_shape = img.shape
        img = letterbox(img, img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred)
        res = []
        for i, det in enumerate(pred): 
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_shape).round()
                res.append(det.cpu().detach().numpy())
        if len(res):
            return [[x1, y1, x2, y2, acc, b] for x1, y1, x2, y2, acc, b in res[0] if acc > min_accuracy]
        else:
            return []
