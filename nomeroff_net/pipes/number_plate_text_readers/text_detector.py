import cv2
import numpy as np
import warnings
import copy
from typing import List, Dict, Tuple
from torch import no_grad
from .base.ocr import OCR
from nomeroff_net.tools.mcm import modelhub
from nomeroff_net.tools.errors import TextDetectorError
from nomeroff_net.tools.image_processing import convert_cv_zones_rgb_to_bgr


class TextDetector(object):
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self,
                 presets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 load_models=True,
                 option_detector_width=0,
                 option_detector_height=0,
                 off_number_plate_classification=True) -> None:
        if presets is None:
            presets = {}
        self.presets = presets

        self.detectors_map = {}
        self.detectors = []
        self.detectors_names = []

        self.option_detector_width = option_detector_width
        self.option_detector_height = option_detector_height

        self.default_label = default_label
        self.default_lines_count = default_lines_count
        self.off_number_plate_classification = off_number_plate_classification

        i = 0
        for preset_name in self.presets:
            preset = self.presets[preset_name]
            for region in preset["for_regions"]:
                self.detectors_map[region.replace("-", '_')] = i
            _label = preset_name
            if modelhub.models.get(_label, None) is None:
                raise TextDetectorError("Text detector {} not exists.".format(_label))
            self.detectors_names.append(_label)
            i += 1

        if load_models:
            self.load()

    def load(self):
        """
        TODO: support reloading
        """
        self.detectors = []
        for i, detector_name in enumerate(self.detectors_names):
            model_conf = copy.deepcopy(modelhub.models[detector_name])
            model_conf.update(self.presets[detector_name])
            detector = OCR(model_name=detector_name, letters=model_conf["letters"],
                           linear_size=model_conf["linear_size"], max_text_len=model_conf["max_text_len"],
                           height=model_conf["height"], width=model_conf["width"],
                           color_channels=model_conf["color_channels"],
                           hidden_size=model_conf["hidden_size"], backbone=model_conf["backbone"])
            detector.load(self.presets[detector_name]['model_path'])
            detector.init_label_converter()
            self.detectors.append(detector)

    def define_predict_classes(self,
                               zones: List[np.ndarray],
                               labels: List[int] = None,
                               lines: List[int] = None) -> Tuple:
        if labels is None:
            labels = []
        if lines is None:
            lines = []

        while len(labels) < len(zones):
            labels.append(self.default_label)
        while len(lines) < len(zones):
            lines.append(self.default_lines_count)
        return labels, lines

    def define_order_detector(
            self,
            zones: List[np.ndarray],
            labels: List[int] = None) -> Dict:
        predicted = {}
        i = 0
        for zone, label in zip(zones, labels):
            if label not in self.detectors_map.keys():
                warnings.warn(f"Label '{label}' not in {self.detectors_map.keys()}! "
                              f"Label changed on default '{self.default_label}'.")
                label = self.default_label
            detector = self.detectors_map[label]
            if detector not in predicted.keys():
                predicted[detector] = {"zones": [], "order": []}
            predicted[detector]["zones"].append(zone)
            predicted[detector]["order"].append(i)
            i += 1
        return predicted

    def get_avalible_module(self) -> List[str]:
        return self.detectors_names

    def preprocess(self,
                   zones: List[np.ndarray],
                   labels: List[str] = None,
                   lines: List[int] = None):
        labels, lines = self.define_predict_classes(zones, labels, lines)
        predicted = self.define_order_detector(zones, labels)
        for key in predicted.keys():
            if self.off_number_plate_classification:
                zones = convert_cv_zones_rgb_to_bgr(predicted[key]["zones"])
                predicted[key]["xs"] = self.detectors[int(key)].preprocess(zones)
            elif (self.option_detector_width != self.detectors[key].width or
                    self.option_detector_height != self.detectors[key].height):
                zones = [np.moveaxis(cv2.resize(np.moveaxis(zone, 0, 2),
                                    (self.detectors[int(key)].width, self.detectors[int(key)].height)), 2, 0)
                         for zone in zones]
                predicted[key]["xs"] = self.detectors[int(key)].preprocess(zones, need_preprocess=False)
            else:
                predicted[key]["xs"] = self.detectors[int(key)].preprocess(predicted[key]["zones"],
                                                                           need_preprocess=False)
        return predicted

    @no_grad()
    def forward(self, predicted):
        for key in predicted.keys():
            xs = predicted[key]["xs"]
            predicted[key]["ys"] = self.detectors[int(key)].forward(xs)
        return predicted

    def postprocess(self, predicted):
        res_all, order_all = [], []
        for key in predicted.keys():
            predicted[key]["ys"] = self.detectors[int(key)].postprocess(predicted[key]["ys"])
            res_all = res_all + predicted[key]["ys"]
            order_all = order_all + predicted[key]["order"]
        return [x for _, x in sorted(zip(order_all, res_all), key=lambda pair: pair[0])]

    def predict(self,
                zones: List[np.ndarray],
                labels: List[str] = None,
                lines: List[int] = None,
                return_acc: bool = False) -> List:

        labels, lines = self.define_predict_classes(zones, labels, lines)
        predicted = self.define_order_detector(zones, labels)

        res_all, scores, order_all = [], [], []
        for key in predicted.keys():
            if return_acc:
                buff_res, acc = self.detectors[int(key)].predict(predicted[key]["zones"], return_acc=return_acc)
                res_all = res_all + buff_res
                scores = scores + list(acc)
            else:
                res_all = res_all + self.detectors[int(key)].predict(predicted[key]["zones"], return_acc=return_acc)
            order_all = order_all + predicted[key]["order"]

        if return_acc:
            return [
                [x for _, x in sorted(zip(order_all, res_all), key=lambda pair: pair[0])],
                [x for _, x in sorted(zip(order_all, scores), key=lambda pair: pair[0])]
            ]
        return [x for _, x in sorted(zip(order_all, res_all), key=lambda pair: pair[0])]

    @staticmethod
    def get_static_module(name: str, **kwargs) -> object:
        model_conf = copy.deepcopy(modelhub.models[name])
        model_conf.update(**kwargs)
        detector = OCR(model_name=name, letters=model_conf["letters"],
                       linear_size=model_conf["linear_size"], max_text_len=model_conf["max_text_len"],
                       height=model_conf["height"], width=model_conf["width"],
                       color_channels=model_conf["color_channels"],
                       hidden_size=model_conf["hidden_size"], backbone=model_conf["backbone"])
        detector.init_label_converter()
        return detector

    def get_acc(self, predicted: List, decode: List, regions: List[str]) -> List[List[float]]:
        acc = []
        for i, region in enumerate(regions):
            if self.detectors_map.get(region, None) is None or len(decode[i]) == 0:
                acc.append([0.])
            else:
                detector = self.detectors[int(self.detectors_map[region])]
                _acc = detector.get_acc([predicted[i]], [decode[i]])
                acc.append([float(_acc)])
        return acc

    def get_module(self, name: str) -> object:
        ind = self.detectors_names.index(name)
        return self.detectors[ind]
