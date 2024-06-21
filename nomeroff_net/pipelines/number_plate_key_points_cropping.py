from torch import no_grad
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
from nomeroff_net.tools import unzip


class NumberPlateKeyPointsCropping(Pipeline):
    """
    Number Plate Localization
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 **kwargs):
        super().__init__(task, image_loader, **kwargs)

    def sanitize_parameters(self, quality_profile=None, **kwargs):
        forward_parameters = {}
        if quality_profile is not None:
            forward_parameters["quality_profile"] = quality_profile
        return {}, forward_parameters, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        return inputs
