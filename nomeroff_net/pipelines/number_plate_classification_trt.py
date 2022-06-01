from typing import Optional, Union, Dict, Any
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipes.number_plate_classificators.options_detector_trt import OptionsDetectorTrt
from .number_plate_classification import NumberPlateClassification
from nomeroff_net.tools import unzip


class NumberPlateClassificationTrt(NumberPlateClassification):
    """
    Number Plate Classification ONNX Pipeline
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model="latest",
                 options: Dict = None,
                 **kwargs):
        NumberPlateClassification.__init__(self, task, image_loader,
                                           path_to_model, options,
                                           class_detector=OptionsDetectorTrt, **kwargs)

    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        model_outputs = []
        for inp in inputs:
            model_output = self.detector.forward([inp])
            model_output = unzip(model_output)[0]
            model_outputs.append(model_output)
        return model_outputs
