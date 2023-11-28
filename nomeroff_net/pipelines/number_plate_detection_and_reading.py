"""number plate detection and reading pipeline

Examples:
    >>> from nomeroff_net import pipeline
    >>> from nomeroff_net.tools import unzip
    >>> number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="opencv")
    >>> results = number_plate_detection_and_reading(['./data/examples/oneline_images/example1.jpeg', './data/examples/oneline_images/example2.jpeg'])
    >>> (images, images_bboxs, images_points, images_zones, region_ids,region_names, count_lines, confidences, texts) = unzip(results)
    >>> print(texts)
    (['AC4921CB'], ['RP70012', 'JJF509'])
"""
from typing import Any, Dict, Optional, List, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline, CompositePipeline, empty_method
from .number_plate_localization import NumberPlateLocalization as DefaultNumberPlateLocalization
from .number_plate_key_points_detection import NumberPlateKeyPointsDetection
from .number_plate_text_reading import NumberPlateTextReading
from.number_plate_classification import NumberPlateClassification
from nomeroff_net.tools.image_processing import crop_number_plate_zones_from_images, group_by_image_ids
from nomeroff_net.tools import unzip


class NumberPlateDetectionAndReading(Pipeline, CompositePipeline):
    """
    Number Plate Detection And Reading Class
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model: str = "latest",
                 mtl_model_path: str = "latest",
                 refiner_model_path: str = "latest",
                 path_to_classification_model: str = "latest",
                 presets: Dict = None,
                 off_number_plate_classification: bool = False,
                 classification_options: List = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 number_plate_localization_class: Pipeline = DefaultNumberPlateLocalization,
                 number_plate_localization_detector=None,
                 **kwargs):
        """
        init NumberPlateDetectionAndReading Class
        Args:
            image_loader (): image_loader
            path_to_model (): path_to_model
            mtl_model_path (): mtl_model_path
            refiner_model_path (): refiner_model_path
            path_to_classification_model (): path_to_classification_model
            presets (): presets
            off_number_plate_classification (): off_number_plate_classification
            classification_options (): classification_options
            default_label (): default_label
            default_lines_count (): default_lines_count
            number_plate_localization_class (): number_plate_localization_class
            number_plate_localization_detector (): number_plate_localization_detector

        """
        self.default_label = default_label
        self.default_lines_count = default_lines_count
        self.number_plate_localization = number_plate_localization_class(
            "number_plate_localization",
            image_loader=None,
            path_to_model=path_to_model,
            detector=number_plate_localization_detector
        )
        self.number_plate_key_points_detection = NumberPlateKeyPointsDetection(
            "number_plate_key_points_detection",
            image_loader=None,
            mtl_model_path=mtl_model_path,
            refiner_model_path=refiner_model_path)
        self.number_plate_classification = None
        option_detector_width = 0
        option_detector_height = 0
        if not off_number_plate_classification:
            self.number_plate_classification = NumberPlateClassification(
                "number_plate_classification",
                image_loader=None,
                path_to_model=path_to_classification_model,
                options=classification_options)
            option_detector_width = self.number_plate_classification.detector.width
            option_detector_height = self.number_plate_classification.detector.height
        self.number_plate_text_reading = NumberPlateTextReading(
            "number_plate_text_reading",
            image_loader=None,
            presets=presets,
            option_detector_width=option_detector_width,
            option_detector_height=option_detector_height,
            default_label=default_label,
            default_lines_count=default_lines_count,
            off_number_plate_classification=off_number_plate_classification,
        )
        self.pipelines = [
            self.number_plate_localization,
            self.number_plate_key_points_detection,
            self.number_plate_text_reading,
        ]
        if self.number_plate_classification is not None:
            self.pipelines.append(self.number_plate_classification)
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
        if self.number_plate_classification is None or not len(zones):
            region_ids = [-1 for _ in zones]
            region_names = [self.default_label for _ in zones]
            count_lines = [self.default_lines_count for _ in zones]
            confidences = [-1 for _ in zones]
            predicted = [-1 for _ in zones]
            preprocessed_np = [None for _ in zones]
        else:
            (region_ids, region_names, count_lines,
             confidences, predicted, preprocessed_np) = unzip(self.number_plate_classification(zones,
                                                                                               **forward_parameters))
        return (region_ids, region_names, count_lines, confidences,
                predicted, zones, image_ids, images_bboxs, images,
                images_points, images_mline_boxes, preprocessed_np)

    def forward_recognition_np(self, region_ids, region_names,
                               count_lines, confidences,
                               zones, image_ids,
                               images_bboxs, images,
                               images_points, preprocessed_np, **forward_parameters):
        number_plate_text_reading_res = unzip(
            self.number_plate_text_reading(unzip([zones,
                                                  region_names,
                                                  count_lines, preprocessed_np]), **forward_parameters))
        if len(number_plate_text_reading_res):
            texts, _ = number_plate_text_reading_res
        else:
            texts = []
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
         images_points, images_mline_boxes, preprocessed_np) = self.forward_detection_np(inputs, **forward_parameters)
        return self.forward_recognition_np(region_ids, region_names,
                                           count_lines, confidences,
                                           zones, image_ids,
                                           images_bboxs, images,
                                           images_points, preprocessed_np, **forward_parameters)

    @empty_method
    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        return inputs
