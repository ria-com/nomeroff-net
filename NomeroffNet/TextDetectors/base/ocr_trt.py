# import modules
from typing import List, Any, Dict

import onnxruntime
import numpy as np
import torch

from NomeroffNet.tools import (modelhub,
                               get_mode_torch)
from NomeroffNet.data_modules.data_loaders import normalize
from NomeroffNet.tools.ocr_tools import decode_batch
from .ocr import OCR


mode_torch = get_mode_torch()


class OcrTrt(OCR):
    def __init__(self) -> None:
        OCR.__init__(self)
        self.ort_session = None
        self.input_name = None

    def is_loaded(self) -> bool:
        if self.ort_session is None:
            return False
        return True

    def load_model(self, path_to_model):
        self.ort_session = onnxruntime.InferenceSession(path_to_model)
        self.input_name = self.ort_session.get_inputs()[0].name
        return self.ort_session

    def load(self, path_to_model: str = "latest", options: Dict = None) -> onnxruntime.InferenceSession:
        """
        TODO: describe method
        """
        if options is None:
            options = dict()
        self.__dict__.update(options)

        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name(self.get_classname())
            path_to_model = model_info["path"]
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model,
                                                        self.get_classname(),
                                                        self.get_classname())
            path_to_model = model_info["path"]
        self.create_model()
        return self.load_model(path_to_model)

    def predict(self, imgs: List, return_acc: bool = False) -> Any:
        xs = []
        for img in imgs:
            x = normalize(img,
                          width=self.width,
                          height=self.height)
            xs.append(x)
        pred_texts = []
        net_out_value = []
        if bool(xs):
            xs = np.moveaxis(np.array(xs), 3, 1)
            ort_inputs = {
                self.input_name: np.random.randn(
                    len(xs), self.color_channels,
                    self.height, self.width
                ).astype(np.float32)
            }
            net_out_value = self.ort_session.run(None, ort_inputs)
            pred_texts = decode_batch(torch.Tensor(net_out_value), self.label_converter)
        pred_texts = [pred_text.upper() for pred_text in pred_texts]
        if return_acc:
            if len(net_out_value):
                net_out_value = np.array(net_out_value)
                net_out_value = net_out_value.reshape((net_out_value.shape[1],
                                                       net_out_value.shape[0],
                                                       net_out_value.shape[2]))
            return pred_texts, net_out_value
        return pred_texts
