from torch import no_grad
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
from nomeroff_net.tools import unzip
from nomeroff_net.tools.mcm import get_device_torch


class NumberPlateUpscaling(Pipeline):
    """
    Number Plate Localization
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 **kwargs):
        from upscaler import HAT

        super().__init__(task, image_loader, **kwargs)
        device_torch = get_device_torch()
        self.model = HAT(tile_size=320, num_gpu=int(device_torch == "cuda"))

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        return images

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        images, images_points = unzip(inputs)
        new_images = []
        new_images_points = []
        for i, (img, points) in enumerate(zip(images, images_points)):
            is_need_resize = img.shape[0] < 50
            if is_need_resize:
                img = self.model.run(img)
                points = points*4
            new_images.append(img)
            new_images_points.append(points)
        return unzip([new_images, new_images_points])

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        return inputs
