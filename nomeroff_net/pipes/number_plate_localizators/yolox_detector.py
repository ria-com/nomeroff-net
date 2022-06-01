import torch
import numpy as np
from typing import List, Tuple

from nomeroff_net.tools import (modelhub, get_mode_torch)

# download and append to path yolo repo
modelhub.download_repo_for_model("yoloxv5s")

# load yolox packages
from .yolox_exps.yolox_s import Exp as DefaultExp
from yolox.utils import fuse_model, postprocess
from yolox.data.data_augment import ValTransform as DefaultTransform


class Detector(object):
    """

    """
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self, exp_class=DefaultExp, half=True, fuse=False,
                 legacy=False, transform_class=DefaultTransform) -> None:
        self.model = None
        self.fuse = fuse
        self.device = ""
        self.half = half
        self.exp = exp_class()
        self.clasees = ["numberplate"]

        self.trt_file = None
        self.decoder = None
        self.confthre = self.exp.test_conf
        self.num_classes = self.exp .num_classes
        self.nmsthre = self.exp.nmsthre
        self.preproc = transform_class(legacy=legacy)

    def load_model(self, weights: str, device: str = get_mode_torch()) -> None:
        self.model = self.exp.get_model()
        self.model.eval()

        # load the model state dict
        ckpt = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])

        if device == "gpu":
            self.model.cuda()
            if self.half:
                self.model.half()  # to FP16
        self.device = device

        if self.fuse:
            self.model = fuse_model(self.model)

    def load(self, path_to_model: str = "latest", device: str = get_mode_torch()) -> None:
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("yoloxv5s")
            path_to_model = model_info["path"]

        self.load_model(path_to_model, device)

    def normalize_img(self, img):
        """
        TODO: auto=False if pipeline batch size > 1
        """
        img, _ = self.preproc(img, None, self.exp.test_size)
        return img

    def normalize_imgs(self, imgs: List[np.ndarray], **_):
        if len(imgs) == 1:
            normalized_imgs = self.normalize_img(imgs[0])
            normalized_imgs = np.expand_dims(normalized_imgs, axis=0)
        else:
            normalized_imgs = np.zeros((len(imgs), 3, *self.exp.test_size))
            for i, img in enumerate(imgs):
                normalized_imgs[i] = self.normalize_img(img)
        input_tensors = torch.from_numpy(normalized_imgs)
        input_tensors = input_tensors.type(torch.FloatTensor)
        if self.device == "gpu":
            input_tensors = input_tensors.cuda()
            if self.half:
                input_tensors = input_tensors.half()  # to FP16
        return input_tensors

    def postprocessing(self,
                       preds: torch.Tensor,
                       imgs: List[np.ndarray],
                       orig_img_shapes: List[Tuple],
                       min_accuracy: float = 0.5,
                       **_):
        res = []
        for pred, img, orig_img_shape in zip(preds, imgs, orig_img_shapes):
            pred = pred.cpu().numpy()
            ratio = min([img.shape[1]/orig_img_shape[0], img.shape[2]/orig_img_shape[1]])
            if len(pred):
                res.append([[x1/ratio, y1/ratio, x2/ratio, y2/ratio, acc, b]
                            for x1, y1, x2, y2, acc, b, *_ in pred
                            if acc > min_accuracy])
            else:
                res.append([])
        return res

    @torch.no_grad()
    def forward(self, input_tensors):
        preds = self.model(input_tensors)
        if self.decoder is not None:
            preds = self.decoder(preds, dtype=preds.type())
        preds = postprocess(
            preds, self.num_classes, self.confthre,
            self.nmsthre, class_agnostic=True
        )
        return preds

    @torch.no_grad()
    def detect_bbox(self, img: np.ndarray, min_accuracy: float = 0.5) -> List:
        orig_img_shapes = [img.shape]
        normalized_img = self.normalize_img(img)
        input_tensor = torch.from_numpy(normalized_img)
        input_tensor = input_tensor.type(torch.FloatTensor)
        if self.device == "gpu":
            input_tensor = input_tensor.cuda()
            if self.half:
                input_tensor = input_tensor.half()  # to FP16

        input_tensor = input_tensor.unsqueeze(0)

        preds = self.forward(input_tensor)
        return self.postprocessing(preds, input_tensor, orig_img_shapes, min_accuracy)[0]

    @torch.no_grad()
    def detect(self, imgs: List[np.ndarray], min_accuracy: float = 0.5) -> List:
        orig_img_shapes = [img.shape for img in imgs]
        input_tensors = self.normalize_imgs(imgs)
        preds = self.forward(input_tensors)
        return self.postprocessing(preds, input_tensors, orig_img_shapes, min_accuracy=min_accuracy)
