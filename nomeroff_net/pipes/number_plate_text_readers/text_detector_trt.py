import copy
from typing import Dict
import numpy as np

from nomeroff_net.tools.mcm import modelhub
from nomeroff_net.pipes.number_plate_text_readers.text_detector import TextDetector
from .base.ocr import OCR
from .base.ocr_trt import OcrTrt


class TextDetectorTrt(TextDetector):
    def __init__(self,
                 presets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 **kwargs) -> None:
        TextDetector.__init__(self, presets, default_label, default_lines_count, load_models=False, **kwargs)
        self.detectors = []
        for detector_name in self.detectors_names:
            model_conf = copy.deepcopy(modelhub.models[detector_name])
            model_conf.update(self.presets[detector_name])

            detector = OcrTrt()
            OCR.__init__(
                detector,
                model_name=detector_name,
                letters=model_conf["letters"],
                linear_size=model_conf["linear_size"],
                max_text_len=model_conf["max_text_len"],
                height=model_conf["height"],
                width=model_conf["width"],
                color_channels=model_conf["color_channels"],
                hidden_size=model_conf["hidden_size"],
                backbone=model_conf["backbone"],
            )
            detector.init_label_converter()
            detector.load(self.presets[detector_name]['model_path'])
            self.detectors.append(detector)

    def forward(self, predicted):
        for key in predicted.keys():
            xs = np.array(predicted[key]["xs"])
            predicted[key]["ys"] = self.detectors[int(key)].forward(xs)
        return predicted

    @staticmethod
    def get_static_module(name: str) -> object:
        detector = TextDetector.get_static_module(name)
        trt_detector_class = type(f"{detector.__class__.get_classname()}_trt",
                                   (OcrTrt, detector.__class__),
                                   dict())
        trt_detector = trt_detector_class()
        detector.__class__.__init__(trt_detector)
        return trt_detector
