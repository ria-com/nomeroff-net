from typing import Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline, CompositePipeline
from .number_plate_localization_trt import NumberPlateLocalizationTrt
from .number_plate_key_points_detection import NumberPlateKeyPointsDetection
from .number_plate_text_reading_trt import NumberPlateTextReadingTrt
from .number_plate_classification_trt import NumberPlateClassificationTrt
from .number_plate_detection_and_reading import NumberPlateDetectionAndReading


class NumberPlateDetectionAndReadingTrt(NumberPlateDetectionAndReading):
    """
    Number Plate Detection And Reading TensorRT
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model: str,
                 path_to_classification_model: str,
                 presets: Dict,
                 mtl_model_path: str = "latest",
                 refiner_model_path: str = "latest",
                 classification_options: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 **kwargs):

        self.number_plate_key_points_detection = NumberPlateKeyPointsDetection(
            "number_plate_key_points_detection",
            image_loader=None,
            mtl_model_path=mtl_model_path,
            refiner_model_path=refiner_model_path)

        self.number_plate_localization = NumberPlateLocalizationTrt(
            "number_plate_localization",
            image_loader=None,
            engine_file_path=path_to_model)
        self.number_plate_classification = NumberPlateClassificationTrt(
            "number_plate_classification",
            image_loader=None,
            path_to_model=path_to_classification_model,
            options=classification_options)
        self.number_plate_text_reading = NumberPlateTextReadingTrt(
            "number_plate_text_reading",
            image_loader=None,
            presets=presets,
            default_label=default_label,
            default_lines_count=default_lines_count,
        )
       
        self.pipelines = [
            self.number_plate_classification,
            self.number_plate_text_reading,
            self.number_plate_key_points_detection,
            self.number_plate_localization,
        ]
        Pipeline.__init__(self, task, image_loader, **kwargs)
        CompositePipeline.__init__(self, self.pipelines)
