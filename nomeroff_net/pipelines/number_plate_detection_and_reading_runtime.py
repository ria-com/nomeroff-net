from typing import Dict, Optional, List, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.number_plate_detection_and_reading import NumberPlateDetectionAndReading
from nomeroff_net.pipelines.base import RuntimePipeline


class NumberPlateDetectionAndReadingRuntime(NumberPlateDetectionAndReading, RuntimePipeline):
    """
    Number Plate Detection and reading runtime
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model: str = "latest",
                 mtl_model_path: str = "latest",
                 refiner_model_path: str = "latest",
                 path_to_classification_model: str = "latest",
                 prisets: Dict = None,
                 classification_options: List = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 **kwargs):
        NumberPlateDetectionAndReading.__init__(
            self, task, image_loader, path_to_model,
            mtl_model_path, refiner_model_path,
            path_to_classification_model, prisets,
            classification_options, default_label,
            default_lines_count, **kwargs)
        RuntimePipeline.__init__(
            self, self.pipelines)

