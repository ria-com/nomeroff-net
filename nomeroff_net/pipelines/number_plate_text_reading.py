from torch import no_grad
from typing import Any, Dict, Optional
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
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
        "for_regions": ["eu"],
        "model_path": "latest"
    },
    "ru": {
        "for_regions": ["ru", "eu-ua-ordlo-lpr", "eu-ua-ordlo-dpr"],
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
    Number Plate Localization
    """

    def __init__(self,
                 task,
                 image_loader: Optional[BaseImageLoader],
                 prisets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 **kwargs):
        if prisets is None:
            prisets = DEFAULT_PRISETS
        super().__init__(task, image_loader, **kwargs)
        self.detector = TextDetector(prisets, default_label, default_lines_count)

    def sanitize_parameters(self):
        return {}, {}, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images, labels, lines = inputs
        images = [self.image_loader.load(item) for item in images]
        predicted, res_all, order_all = self.detector.preprocess(images, labels, lines)
        return predicted, res_all, order_all, images

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        return self.detector.forward(inputs)

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        model_outputs, labels, res_all, order_all = inputs
        outputs = self.detector.postprocess(model_outputs, res_all, order_all)
        outputs = text_postprocessing(outputs, labels)
        return outputs

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        images, labels, lines = inputs
        model_inputs, res_all, order_all, images = self.preprocess([images, labels, lines], **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess([model_outputs, labels, res_all, order_all], **postprocess_params)
        return outputs, images
