from torch import no_grad
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
from nomeroff_net.tools import unzip
from nomeroff_net.pipes.number_plate_localizators.yolov5_engine_detector import Detector


class NumberPlateLocalizationTrt(Pipeline):
    """
    Number Plate Localization Tensorrt
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 engine_file_path: str,
                 plugin_lib: str,
                 **kwargs):
        super().__init__(task, image_loader, **kwargs)
        self.detector = Detector()
        self.detector.load_model(engine_file_path=engine_file_path, plugin_lib=plugin_lib)

    def sanitize_parameters(self, img_size=None, stride=None, min_accuracy=None, **kwargs):
        postprocess_parameters = {}
        if min_accuracy is not None:
            postprocess_parameters["min_accuracy"] = min_accuracy
        return {}, {}, postprocess_parameters

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        batch_input_image = self.detector.yolov5_wrapper.prepare_batch_input_image(images)
        return unzip([images, batch_input_image])

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        images, batch_input_image = unzip(inputs)
        detected_images_bboxs = self.detector.yolov5_wrapper.infer(batch_input_image)
        return unzip([images, detected_images_bboxs])

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        images, detected_images_bboxs = unzip(inputs)
        detected_images_bboxs = self.detector.postprocessing(detected_images_bboxs, **postprocess_parameters)
        return unzip([images, detected_images_bboxs])
