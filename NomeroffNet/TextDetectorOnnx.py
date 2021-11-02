import sys
import os
from typing import Dict

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from TextDetector import TextDetector
from TextDetectors.base.ocr_trt import OcrTrt


class TextDetectorOnnx(TextDetector):
    def __init__(self,
                 prisets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1) -> None:
        TextDetector.__init__(self, prisets, default_label, default_lines_count)
        for i, detector_class in enumerate(self.detectors):
            onnx_detector_class = type(f"{detector_class.get_classname()}_onnx",
                                       (OcrTrt, detector_class),
                                       dict())
            self.detectors[i] = onnx_detector_class()
            detector_class.__init__(self.detectors[i])
        for detector, detector_name in zip(self.detectors, self.detectors_names):
            detector.load(self.prisets[detector_name]['model_path'])

    def load(self):
        pass

    @staticmethod
    def get_static_module(name: str) -> object:
        detector = TextDetector.get_static_module(name)
        onnx_detector_class = type(f"{detector.__class__.get_classname()}_onnx",
                                   (OcrTrt, detector.__class__),
                                   dict())
        onnx_detector = onnx_detector_class()
        detector.__class__.__init__(onnx_detector)
        return onnx_detector
