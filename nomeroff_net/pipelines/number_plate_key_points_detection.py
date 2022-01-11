from torch import no_grad
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
from nomeroff_net.tools import unzip
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points import NpPointsCraft


class NumberPlateKeyPointsDetection(Pipeline):
    """
    Number Plate Localization
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 mtl_model_path: str = "latest",
                 refiner_model_path: str = "latest",
                 **kwargs):
        super().__init__(task, image_loader, **kwargs)
        self.detector = NpPointsCraft()
        self.detector.load(mtl_model_path, refiner_model_path)

    def sanitize_parameters(self, quality_profile=None, **kwargs):
        forward_parameters = {}
        if quality_profile is not None:
            forward_parameters["quality_profile"] = quality_profile
        return {}, forward_parameters, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images, images_target_boxes = unzip(inputs)
        images = [self.image_loader.load(item) for item in images]
        return unzip([images, images_target_boxes])

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        images, images_target_boxes = unzip(inputs)
        images_points, images_mline_boxes = self.detector.detect_mline_many(images, images_target_boxes,
                                                                            **forward_parameters)
        return unzip([images_points, images_mline_boxes])

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        images_points, images_mline_boxes = unzip(inputs)
        return unzip([images_points, images_mline_boxes])
