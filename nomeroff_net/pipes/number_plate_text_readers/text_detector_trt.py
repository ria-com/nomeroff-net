from typing import Dict

from nomeroff_net.pipes.number_plate_text_readers.text_detector import TextDetector
from .base.ocr_trt import OcrTrt


class TextDetectorTrt(TextDetector):
    def __init__(self,
                 presets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1) -> None:
        TextDetector.__init__(self, presets, default_label, default_lines_count, load_models=False)
        for i, detector_class in enumerate(self.detectors):
            trt_detector_class = type(f"{detector_class.get_classname()}_trt",
                                      (OcrTrt, detector_class),
                                      dict())
            self.detectors[i] = trt_detector_class()
            detector_class.__init__(self.detectors[i])
        for detector, detector_name in zip(self.detectors, self.detectors_names):
            detector.load(self.presets[detector_name]['model_path'])

    @staticmethod
    def get_static_module(name: str) -> object:
        detector = TextDetector.get_static_module(name)
        trt_detector_class = type(f"{detector.__class__.get_classname()}_trt",
                                   (OcrTrt, detector.__class__),
                                   dict())
        trt_detector = trt_detector_class()
        detector.__class__.__init__(trt_detector)
        return trt_detector
