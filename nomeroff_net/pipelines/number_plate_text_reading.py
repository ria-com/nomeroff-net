from torch import no_grad
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
from nomeroff_net.tools import unzip
from nomeroff_net.pipes.number_plate_text_readers.text_detector import TextDetector
from nomeroff_net.pipes.number_plate_text_readers.text_postprocessing import text_postprocessing

DEFAULT_PRISETS = {
    "eu_ua_2004_2015": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004"],
        "model_path": "latest"
    },
    "eu_ua_1995": {
        "for_regions": ["eu_ua_1995"],
        "model_path": "latest"
    },
    "eu": {
        "for_regions": ["eu", "xx_transit", "xx_unknown"],
        "model_path": "latest"
    },
    "ru": {
        "for_regions": ["ru", "eu_ua_ordlo_lpr", "eu_ua_ordlo_dpr"],
        "model_path": "latest"
    },
    "kz": {
        "for_regions": ["kz"],
        "model_path": "latest"
    },
    "kg": {
        "for_regions": ["kg"],
        "model_path": "latest"
    },
    "ge": {
        "for_regions": ["ge"],
        "model_path": "latest"
    },
    "su": {
        "for_regions": ["su"],
        "model_path": "latest"
    },
    "am": {
        "for_regions": ["am"],
        "model_path": "latest"
    },
    "by": {
        "for_regions": ["by"],
        "model_path": "latest"
    },
}


class NumberPlateTextReading(Pipeline):
    """
    Number Plate Text Reading Pipeline
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 prisets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 class_detector=TextDetector,
                 **kwargs):
        if prisets is None:
            prisets = DEFAULT_PRISETS
        super().__init__(task, image_loader, **kwargs)
        self.detector = class_detector(prisets, default_label, default_lines_count)

    def sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images, labels, lines = unzip(inputs)
        images = [self.image_loader.load(item) for item in images]
        return unzip([images, labels, lines])

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        images, labels, lines = unzip(inputs)
        model_inputs = self.detector.preprocess(images, labels, lines)
        model_outputs = self.detector.forward(model_inputs)
        model_outputs = self.detector.postprocess(model_outputs)
        return unzip([images, model_outputs, labels])

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        images, model_outputs, labels = unzip(inputs)
        outputs = text_postprocessing(model_outputs, labels)
        return unzip([outputs, images])
