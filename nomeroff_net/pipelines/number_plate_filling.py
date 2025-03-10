import cv2
import numpy as np
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline, empty_method
from nomeroff_net.tools import unzip
from .number_plate_localization import NumberPlateLocalization


class NumberPlateFilling(Pipeline):
    """
    Number Plate Localization
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model="latest",
                 **kwargs):
        self.number_plate_localization = NumberPlateLocalization(
            "number_plate_localization",
            image_loader,
            path_to_model=path_to_model)

        self.pipelines = [
            self.number_plate_localization,
        ]
        super().__init__(task, image_loader, **kwargs)

    def sanitize_parameters(self, **kwargs):
        forward_parameters = {}
        for pipeline in self.pipelines:
            for dict_params in pipeline.sanitize_parameters(**kwargs):
                forward_parameters.update(dict_params)
        return {}, forward_parameters, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        return inputs

    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        images_bboxs, images = unzip(self.number_plate_localization(inputs, **forward_parameters))

        filled_images = []
        for bboxs, image in zip(images_bboxs, images):
            image = image.astype(np.uint8)
            for bbox in bboxs:
                cntr = bbox[-1]
                cntr = np.array(cntr, dtype=np.int32)
                cv2.drawContours(image, [cntr], -1, (0, 0, 0), -1)
            filled_images.append(image)
        return filled_images

    @empty_method
    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        return inputs
