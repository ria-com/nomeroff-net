from torch import no_grad
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
from nomeroff_net.pipes.number_plate_classificators.options_detector import OptionsDetector
from nomeroff_net.tools import unzip


class NumberPlateClassification(Pipeline):
    """
    Number Plate Classification Pipeline
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model="latest",
                 options=None,
                 class_detector=OptionsDetector,
                 **kwargs):
        super().__init__(task, image_loader, **kwargs)
        self.detector = class_detector(options=options)
        self.detector.load(path_to_model, options=options)

    def sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        return self.detector.preprocess(images)

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        model_output = self.detector.forward(inputs)
        model_output = [p.cpu().numpy() for p in model_output]
        model_output = [*model_output, inputs]
        return unzip(model_output)

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        unziped_inputs = unzip(inputs)
        processed_np = [np for np in unziped_inputs[2]]
        confidences, region_ids, count_lines = self.detector.unzip_predicted(unziped_inputs)
        count_lines = self.detector.custom_count_lines_id_to_all_count_lines(count_lines)
        region_names = self.detector.get_region_labels(region_ids)
        return unzip([region_ids, region_names, count_lines, confidences, inputs, processed_np])
