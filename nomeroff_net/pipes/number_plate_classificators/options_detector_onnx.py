import onnxruntime
import numpy as np

from typing import List, Dict, Tuple

from nomeroff_net.tools import modelhub
from nomeroff_net.pipes.number_plate_classificators.options_detector import OptionsDetector
from nomeroff_net.tools.image_processing import normalize_img


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
        xs = [normalize_img(img) for img in imgs]
        predicted = [[], []]
        if bool(xs):
            xs = np.moveaxis(np.array(xs), 3, 1)
            predicted = self.ort_session.run(None, {self.input_name: xs})

        confidences, region_ids, count_lines = self.unzip_predicted(predicted)
        count_lines = self.get_count_lines_labels(count_lines)
        return region_ids, count_lines, confidences, predicted
