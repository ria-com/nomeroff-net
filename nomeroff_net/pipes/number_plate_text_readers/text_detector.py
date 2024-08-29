import numpy as np
import warnings
import copy
import torch
from typing import List, Dict, Tuple
from torch import no_grad
from .base.ocr import OCR, device_torch
from .multiple_postprocessing import multiple_postprocessing_mapping
from nomeroff_net.tools.mcm import modelhub
from nomeroff_net.tools.errors import TextDetectorError
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import split_numberplate
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
                 multiline_splitter="",
                 off_number_plate_classification=True) -> None:
        if presets is None:
            presets = {}
        self.presets = presets

        self.detectors_map = {}
        self.detectors = []
        self.detectors_names = []

        self.option_detector_width = option_detector_width
        self.option_detector_height = option_detector_height

        self.multiline_splitter = multiline_splitter
        self.default_label = default_label
        self.default_lines_count = int(default_lines_count)
        self.off_number_plate_classification = off_number_plate_classification

        for preset_name in self.presets:
            if preset_name in self.detectors_names:
                detector_id = self.detectors_names.index(preset_name)
            else:
                detector_id = len(self.detectors_names)
                self.detectors_names.append(preset_name)

            preset = self.presets[preset_name]
            for count_lines in preset.get("for_count_lines", [1]):
                for region in preset["for_regions"]:
                    self.detectors_map[(int(count_lines), region.replace("-", '_'))] = detector_id
            if modelhub.models.get(preset_name, None) is None:
                raise TextDetectorError("Text detector {} not exists.".format(preset_name))

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
            labels: List[int] = None,
            lines: List[int] = None,
            processed_zones: List[np.ndarray] = None
    ) -> Dict:
        if processed_zones is None:
            processed_zones = [None for _ in zones]
        if len(zones) != len(processed_zones):
            raise TextDetectorError("len(zones) != len(processed_zones) !!!")
        predicted = {}
        zone_id = 0
        for zone, label, count_line, p_zone in zip(zones, labels, lines, processed_zones):
            count_line = int(count_line)
            if (count_line, label) not in self.detectors_map.keys():
                warnings.warn(f"Label '{label}' not in {self.detectors_map.keys()}! "
                              f"Label changed on default '{self.default_label}'.")
                label = self.default_label
                count_line = self.default_lines_count
            detector = self.detectors_map[(count_line, label)]
            if detector not in predicted.keys():
                predicted[detector] = {
                    "zones": [],
                    "order": [],
                    "xs": [],
                    "count_line": [],
                    "label": [],
                }
            if count_line > 1:
                parts = split_numberplate(zone, count_line)
            else:
                parts = [zone]
            if (self.option_detector_width != self.detectors[detector].width or
                self.option_detector_height != self.detectors[detector].height or
                count_line == 2 or self.off_number_plate_classification):

                parts = convert_cv_zones_rgb_to_bgr(parts)
                xs = self.detectors[detector].normalize(parts)
                xs = np.moveaxis(np.array(xs), 3, 1)
            else:
                xs = [p_zone]
            predicted[detector]["zones"].extend(parts)
            predicted[detector]["order"].extend([zone_id for _ in parts])
            predicted[detector]["count_line"].extend([count_line for _ in parts])
            predicted[detector]["label"].extend([label for _ in parts])
            predicted[detector]["xs"].extend(xs)

            zone_id += 1
        return predicted

    def get_avalible_module(self) -> List[str]:
        return self.detectors_names

    def preprocess(self,
                   orig_zones: List[np.ndarray],
                   zones: List[np.ndarray],
                   labels: List[str] = None,
                   lines: List[int] = None):
        labels, lines = self.define_predict_classes(zones, labels, lines)
        predicted = self.define_order_detector(orig_zones, labels, lines, zones)
        return predicted

    @no_grad()
    def forward(self, predicted):
        for key in predicted.keys():
            xs = predicted[key]["xs"]

            # to tensor
            xs = np.array(xs)
            xs = torch.tensor(xs)
            xs = xs.to(device_torch)

            predicted[key]["ys"] = self.detectors[int(key)].forward(xs)
        return predicted

    def postprocess(self, predicted):
        mapping = {}
        for key in predicted.keys():
            predicted[key]["ys"] = self.detectors[int(key)].postprocess(predicted[key]["ys"])
            for text, zone_id, count_line, label in zip(predicted[key]["ys"],
                                                        predicted[key]["order"],
                                                        predicted[key]["count_line"],
                                                        predicted[key]["label"]):
                if zone_id in mapping:
                    mapping[zone_id]["text"] += self.multiline_splitter + text
                else:
                    mapping[zone_id] = {
                        "order": zone_id,
                        "text": text,
                        "count_line": count_line,
                        "label": label,
                    }
        res_all = []
        for item in mapping.values():
            post = multiple_postprocessing_mapping.get(item["label"], multiple_postprocessing_mapping["default"])
            text = post.postprocess_multiline_text(item["text"], item["count_line"])
            res_all.append(text)
        order_all = [item["order"] for item in mapping.values()]

        return [x for _, x in sorted(zip(order_all, res_all), key=lambda pair: pair[0])]

    def predict(self,
                zones: List[np.ndarray],
                labels: List[str] = None,
                lines: List[int] = None,
                return_acc: bool = False) -> List:

        labels, lines = self.define_predict_classes(zones, labels, lines)
        predicted = self.define_order_detector(zones, labels, lines)

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
