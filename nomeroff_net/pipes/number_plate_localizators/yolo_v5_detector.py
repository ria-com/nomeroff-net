import os
import torch
import numpy as np
from typing import List, Tuple

from nomeroff_net.tools import (modelhub, get_mode_torch)

# download and append to path yolo repo
modelhub.download_repo_for_model("yolov5")

# load yolo packages
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device


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

    def load_model(self, weights: str, device: str = '') -> None:
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
            model_info = modelhub.download_model_by_name("yolov5")
            path_to_model = model_info["path"]
        device = "cpu"
        if get_mode_torch() == "gpu":
            device = os.environ.get("CUDA_VISIBLE_DEVICES", '0')
        self.load_model(path_to_model, device)

    @staticmethod
    def scale_predicted_coords(img, pred, orig_img_shape):
        res = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_img_shape).round()
                res.append(det.cpu().detach().numpy())
        return res

    def normalize_img(self, img, img_size, stride):
        """
        TODO: auto=False if pipeline batch size > 1
        """
        img = letterbox(img, img_size, stride=stride, auto=False)[0]
        img = img.transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.unsqueeze(0)
        return img

    def normalize_imgs(self, imgs: List[np.ndarray], img_size: int = (640, 640), stride: int = 32):
        return [self.normalize_img(img, img_size, stride) for img in imgs]

    def postprocessing(self,
                       preds: torch.Tensor,
                       imgs: List[np.ndarray],
                       orig_img_shapes: List[Tuple],
                       min_accuracy: float = 0.5):
        res = []
        for pred, img, orig_img_shape in zip(preds, imgs, orig_img_shapes):
            pred = non_max_suppression(pred)
            predicted_coords = self.scale_predicted_coords(img, pred, orig_img_shape)
            if len(predicted_coords):
                res.append([[x1, y1, x2, y2, acc, b]
                            for x1, y1, x2, y2, acc, b in predicted_coords[0]
                            if acc > min_accuracy])
            else:
                res.append([])
        return res

    def detect_bbox(self,
                    img: np.ndarray,
                    img_size: int = 640,
                    stride: int = 32,
                    min_accuracy: float = 0.5) -> List:
        """
        TODO: input img in BGR format, not RGB; To Be Implemented in release 2.2
        """
        orig_img_shapes = [img.shape]
        input_tensor = self.normalize_img(img, img_size, stride)
        preds = self.model([input_tensor])
        return self.postprocessing(preds, [img], orig_img_shapes, min_accuracy)[0]

    def detect(self,
               imgs: List[np.ndarray],
               img_size: int = 640,
               stride: int = 32,
               min_accuracy: float = 0.5) -> List:
        orig_img_shapes = [img.shape for img in imgs]
        input_tensors = self.normalize_imgs(imgs, img_size, stride)
        preds = self.model(torch.cat(input_tensors, dim=0))
        return self.postprocessing(preds, imgs, orig_img_shapes, min_accuracy)
