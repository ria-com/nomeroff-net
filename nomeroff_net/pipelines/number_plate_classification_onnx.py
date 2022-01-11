from typing import Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipes.number_plate_classificators.options_detector_onnx import OptionsDetector
from number_plate_classification import NumberPlateClassification


class NumberPlateClassificationOnnx(NumberPlateClassification):
    """
    Number Plate Classification ONNX Pipeline
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model="latest",
                 options=None,
                 **kwargs):
        NumberPlateClassification.__init__(self, task, image_loader,
                                           path_to_model, options,
                                           class_detector=OptionsDetector, **kwargs)
