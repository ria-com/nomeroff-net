from typing import Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.number_plate_text_reading import NumberPlateTextReading
from nomeroff_net.pipes.number_plate_text_readers.text_detector_onnx import TextDetector


class NumberPlateTextReadingOnnx(NumberPlateTextReading):
    """
    Number Plate Text Reading ONNX Pipeline
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 prisets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 class_detector=TextDetector,
                 **kwargs):
        NumberPlateTextReading.__init__(self, task, image_loader, prisets, default_label,
                                        default_lines_count, class_detector,
                                        class_detector=TextDetector, **kwargs)
