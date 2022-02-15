from typing import Any, Dict, Optional, List, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline, CompositePipeline, empty_method
from nomeroff_net.pipelines.number_plate_localization import NumberPlateLocalization
from nomeroff_net.pipelines.number_plate_key_points_detection_v2 import NumberPlateKeyPointsDetectionV2
from nomeroff_net.pipelines.number_plate_text_reading import NumberPlateTextReading
from nomeroff_net.pipelines.number_plate_classification import NumberPlateClassification
from nomeroff_net.tools.image_processing import crop_number_plate_zones_from_images, group_by_image_ids
from nomeroff_net.tools import unzip


class NumberPlateDetectionAndReading(Pipeline, CompositePipeline):
    """
    Number Plate Localization
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
        self.number_plate_localization = NumberPlateLocalization(
            "number_plate_localization",
            image_loader=None,
            path_to_model=path_to_model)
        self.number_plate_key_points_detection = NumberPlateKeyPointsDetectionV2(
            "number_plate_key_points_detection",
            image_loader=None,
            mtl_model_path=mtl_model_path,
            refiner_model_path=refiner_model_path)
        self.number_plate_classification = NumberPlateClassification(
            "number_plate_classification",
            image_loader=None,
            path_to_model=path_to_classification_model,
            options=classification_options)
        self.number_plate_text_reading = NumberPlateTextReading(
            "number_plate_text_reading",
            image_loader=None,
            prisets=prisets,
            default_label=default_label,
            default_lines_count=default_lines_count,
        )
        self.pipelines = [
            self.number_plate_localization,
            self.number_plate_key_points_detection,
            self.number_plate_classification,
            self.number_plate_text_reading,
        ]
        Pipeline.__init__(self, task, image_loader, **kwargs)
        CompositePipeline.__init__(self, self.pipelines)

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        return images

    def forward_detection_np(self, inputs: Any, **forward_parameters: Dict):
        images_bboxs, images = unzip(self.number_plate_localization(inputs, **forward_parameters))
        images_points, images_mline_boxes = unzip(self.number_plate_key_points_detection(unzip([images, images_bboxs]),
                                                                                         **forward_parameters))
        zones, image_ids = crop_number_plate_zones_from_images(images, images_points)
        (region_ids, region_names, count_lines,
         confidences, predicted) = unzip(self.number_plate_classification(zones, **forward_parameters))
        return (region_ids, region_names, count_lines, confidences,
                predicted, zones, image_ids, images_bboxs, images,
                images_points, images_mline_boxes)

    def forward_recognition_np(self, region_ids, region_names,
                               count_lines, confidences,
                               zones, image_ids,
                               images_bboxs, images,
                               images_points, **forward_parameters):
        texts, _ = unzip(self.number_plate_text_reading(unzip([zones,
                                                               region_names,
                                                               count_lines]), **forward_parameters))
        (region_ids, region_names, count_lines, confidences, texts, zones) = \
            group_by_image_ids(image_ids, (region_ids, region_names, count_lines, confidences, texts, zones))
        return unzip([images, images_bboxs,
                      images_points, zones,
                      region_ids, region_names,
                      count_lines, confidences, texts])

    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        """
        TODO: split into two methods so that there is no duplication of code
        """
        (region_ids, region_names,
         count_lines, confidences, predicted,
         zones, image_ids,
         images_bboxs, images,
         images_points, images_mline_boxes) = self.forward_detection_np(inputs, **forward_parameters)
        return self.forward_recognition_np(region_ids, region_names,
                                           count_lines, confidences,
                                           zones, image_ids,
                                           images_bboxs, images,
                                           images_points, **forward_parameters)

    @empty_method
    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        return inputs
