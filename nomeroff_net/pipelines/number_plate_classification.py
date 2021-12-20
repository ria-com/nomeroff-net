from torch import no_grad
from typing import Any, Dict, Optional
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
from nomeroff_net.pipes.number_plate_classificators.options_detector import OptionsDetector


class NumberPlateClassification(Pipeline):
    """
    Number Plate Localization
    """

    def __init__(self,
                 task,
                 image_loader: Optional[BaseImageLoader],
                 path_to_model="latest",
                 options=None,
                 **kwargs):
        super().__init__(task, image_loader, **kwargs)
        self.detector = OptionsDetector(options=options)
        self.detector.load(path_to_model, options=options)

    def sanitize_parameters(self):
        return {}, {}, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        model_inputs = self.detector.preprocess(images)
        return model_inputs, images

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        return self.detector.model(inputs)

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        model_outputs, images = inputs
        predicted = [p.cpu().numpy() for p in model_outputs]
        confidences, region_ids, count_lines = self.detector.unzip_predicted(predicted)
        count_lines = self.detector.custom_count_lines_id_to_all_count_lines(count_lines)
        region_names = self.detector.get_region_labels(region_ids)
        return region_ids, region_names, count_lines, confidences, predicted, images

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        model_inputs, images = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess([model_outputs, images], **postprocess_params)
        return outputs
