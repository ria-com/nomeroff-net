import os
import sys
import onnxruntime
from typing import List, Dict, Tuple
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from tools import (modelhub,
                   get_mode_torch)
from OptionsDetector import OptionsDetector
from data_modules.data_loaders import normalize


class OptionsDetectorOnnx(OptionsDetector):
    """
    TODO: describe class
    """
    def __init__(self, options: Dict = None) -> None:
        OptionsDetector.__init__(self, options)
        self.ort_session = None
        self.input_name = None

    def load_model(self, path_to_model):
        self.ort_session = onnxruntime.InferenceSession(path_to_model)
        self.input_name = self.ort_session.get_inputs()[0].name
        return self.ort_session

    def is_loaded(self) -> bool:
        if self.ort_session is None:
            return False
        return True

    def load(self, path_to_model: str = "latest", options: Dict = None) -> onnxruntime.InferenceSession:
        """
        TODO: describe method
        """
        if options is None:
            options = dict()
        self.__dict__.update(options)

        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("numberplate_options_onnx")
            path_to_model = model_info["path"]
            self.class_region = model_info["class_region"]
            self.count_lines = model_info["count_lines"]
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model,
                                                        self.get_classname(),
                                                        "numberplate_options_onnx")
            path_to_model = model_info["path"]
        self.create_model()
        return self.load_model(path_to_model)

    def predict(self, imgs: List[np.ndarray], return_acc: bool = False) -> Tuple:
        """
        Predict options(region, count lines) by numberplate images
        """
        region_ids, count_lines, confidences, predicted = self.predict_with_confidence(imgs)
        if return_acc:
            return region_ids, count_lines, predicted
        return region_ids, count_lines

    def predict_with_confidence(self, imgs: List[np.ndarray]) -> Tuple:
        """
        Predict options(region, count lines) with confidence by numberplate images
        """
        xs = [normalize(img) for img in imgs]
        predicted = [[], []]
        if bool(xs):
            xs = np.moveaxis(np.array(xs), 3, 1)
            ort_inputs = {
                self.input_name: np.random.randn(
                    len(xs),
                    self.color_channels,
                    self.height,
                    self.width
                ).astype(np.float32)
            }
            predicted = self.ort_session.run(None, ort_inputs)

        confidences = []
        region_ids = []
        count_lines = []
        for region, count_line in zip(predicted[0], predicted[1]):
            region_ids.append(int(np.argmax(region)))
            count_lines.append(int(np.argmax(count_line)))
            region = region.tolist()
            count_line = count_line.tolist()
            region_confidence = region[int(np.argmax(region))]
            count_lines_confidence = count_line[int(np.argmax(count_line))]
            confidences.append([region_confidence, count_lines_confidence])
        count_lines = self.getCountLinesLabels(count_lines)
        return region_ids, count_lines, confidences, predicted
