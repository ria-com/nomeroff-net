from torch import no_grad
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
from nomeroff_net.tools import unzip
from nomeroff_net.pipes.number_plate_localizators.yolox_detector import Detector


class NumberPlateLocalizationX(Pipeline):
    """
    Number Plate Localization
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model="latest",
                 **kwargs):
        super().__init__(task, image_loader, **kwargs)
        self.detector = Detector()
        self.detector.load(path_to_model)

    def sanitize_parameters(self, img_size=None, stride=None, min_accuracy=None, **kwargs):
        parameters = {}
        postprocess_parameters = {}
        if img_size is not None:
            parameters["img_size"] = img_size
        if stride is not None:
            parameters["stride"] = stride
        if min_accuracy is not None:
            postprocess_parameters["min_accuracy"] = min_accuracy
        return {}, parameters, postprocess_parameters

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        return images

    @no_grad()
    def forward(self, images: Any, **forward_parameters: Dict) -> Any:
        model_inputs = self.detector.normalize_imgs(images, **forward_parameters)
        model_outputs = self.detector.forward(model_inputs)
        return unzip([model_outputs, model_inputs, images])

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        model_outputs, images, orig_images = unzip(inputs)
        orig_img_shapes = [img.shape for img in orig_images]
        output = self.detector.postprocessing(model_outputs,
                                              images,
                                              orig_img_shapes,
                                              **postprocess_parameters)
        return unzip([output, orig_images])
