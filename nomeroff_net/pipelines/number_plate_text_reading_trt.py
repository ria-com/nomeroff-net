from typing import Dict, Optional, Union, Any
from nomeroff_net.image_loaders import BaseImageLoader
from .number_plate_text_reading import NumberPlateTextReading
from nomeroff_net.tools import unzip
from nomeroff_net.pipes.number_plate_text_readers.text_detector_trt import TextDetectorTrt


class NumberPlateTextReadingTrt(NumberPlateTextReading):
    """
    Number Plate Text Reading ONNX Pipeline
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 presets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 **kwargs):
        NumberPlateTextReading.__init__(self, task, image_loader, presets, default_label,
                                        default_lines_count, class_detector=TextDetectorTrt, **kwargs)

    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        images, labels, lines = unzip(inputs)
        model_outputs = []
        for image, label, line in zip(images, labels, lines):
            model_inputs = self.detector.preprocess([image], [label], [line])
            model_output = self.detector.forward(model_inputs)
            model_output = self.detector.postprocess(model_output)
            model_outputs.append(model_output[0])
        return unzip([images, model_outputs, labels])
