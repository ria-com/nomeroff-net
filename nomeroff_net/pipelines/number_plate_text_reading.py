from torch import no_grad
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
from nomeroff_net.tools import unzip
from nomeroff_net.pipes.number_plate_text_readers.text_detector import TextDetector

DEFAULT_PRESETS = {
    "eu_ua_2004_2015_efficientnet_b2": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "eu_ua_1995_efficientnet_b2": {
        "for_regions": ["eu_ua_1995"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "eu_ua_custom_efficientnet_b2": {
        "for_regions": ["eu_ua_custom"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "xx_transit_efficientnet_b2": {
        "for_regions": ["xx_transit"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "eu_efficientnet_b2": {
        "for_regions": ["eu", "xx_unknown"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "ru": {
        "for_regions": ["ru", "eu_ua_ordlo_lpr", "eu_ua_ordlo_dpr"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "kz": {
        "for_regions": ["kz"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "kg": {  # "kg_shufflenet_v2_x2_0"
        "for_regions": ["kg"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "ge": {
        "for_regions": ["ge"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "su_efficientnet_b2": {
        "for_regions": ["su"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "am": {
        "for_regions": ["am"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "by": {
        "for_regions": ["by"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "eu_2lines_efficientnet_b2": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004", "eu_ua_1995", "eu_ua_custom", "xx_transit",
                        "eu", "xx_unknown", "ru", "eu_ua_ordlo_lpr", "eu_ua_ordlo_dpr", "kz",
                        "kg", "ge", "am", "by"],
        "for_count_lines": [2, 3],
        "model_path": "latest"
    },
    "su_2lines_efficientnet_b2": {
        "for_regions": ["su", "military"],
        "for_count_lines": [2, 3],
        "model_path": "latest"
    }
}


class NumberPlateTextReading(Pipeline):
    """
    Number Plate Text Reading Pipeline
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 presets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 class_detector=TextDetector,
                 option_detector_width=0,
                 option_detector_height=0,
                 off_number_plate_classification=True,
                 multiline_splitter="",
                 **kwargs):
        if presets is None:
            presets = DEFAULT_PRESETS
        super().__init__(task, image_loader, **kwargs)
        self.detector = class_detector(presets, default_label, default_lines_count,
                                       option_detector_width=option_detector_width,
                                       option_detector_height=option_detector_height,
                                       off_number_plate_classification=off_number_plate_classification,
                                       multiline_splitter=multiline_splitter)

    def sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images, labels, lines, preprocessed_np = unzip(inputs)
        images = [self.image_loader.load(item) for item in images]
        return unzip([images, labels, lines, preprocessed_np])

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        images, labels, lines, preprocessed_np = unzip(inputs)
        model_inputs = self.detector.preprocess(images, preprocessed_np, labels, lines)
        model_outputs = self.detector.forward(model_inputs)
        model_outputs = self.detector.postprocess(model_outputs)
        return unzip([images, model_outputs, labels])

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        images, model_outputs, labels = unzip(inputs)
        return unzip([model_outputs, images])
