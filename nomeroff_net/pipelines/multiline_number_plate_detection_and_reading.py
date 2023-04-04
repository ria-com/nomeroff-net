"""
multiline number plate detection and reading pipeline

Examples:
    >>> import os
    >>> import matplotlib.pyplot as plt
    >>> from glob import glob
    >>> from nomeroff_net import pipeline
    >>> from nomeroff_net.tools import unzip

    >>> multiline_number_plate_detection_and_reading = pipeline("multiline_number_plate_detection_and_reading", image_loader="opencv")
    >>> result = multiline_number_plate_detection_and_reading(glob(os.path.join(nomeroff_net_dir, './data/examples/multiline_images/*')))
    >>> (images, images_bboxs, images_points, images_zones, region_ids, region_names, count_lines, confidences, texts) = unzip(result)
    >>> print(texts)
    (['AI7007ET'], ['241VBZ10'])
"""
from typing import Any, Dict, Optional, List, Union
from nomeroff_net.image_loaders import BaseImageLoader
from .number_plate_detection_and_reading import NumberPlateDetectionAndReading
from nomeroff_net.pipes.number_plate_multiline_extractors.multiline_np_extractor \
    import convert_multiline_images_to_one_line


class MultilineNumberPlateDetectionAndReading(NumberPlateDetectionAndReading):
    """
    Multiline Number Plate Detection and reading Pipeline
    """
    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model: str = "latest",
                 mtl_model_path: str = "latest",
                 refiner_model_path: str = "latest",
                 path_to_classification_model: str = "latest",
                 presets: Dict = None,
                 classification_options: List = None,
                 off_number_plate_classification: bool = False,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 **kwargs):
        """

        """
        NumberPlateDetectionAndReading.__init__(
            self,
            task=task,
            image_loader=image_loader,
            path_to_model=path_to_model,
            mtl_model_path=mtl_model_path,
            refiner_model_path=refiner_model_path,
            off_number_plate_classification=off_number_plate_classification,
            path_to_classification_model=path_to_classification_model,
            presets=presets,
            classification_options=classification_options,
            default_label=default_label,
            default_lines_count=default_lines_count,
            **kwargs)

    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        (region_ids, region_names,
         count_lines, confidences, predicted,
         zones, image_ids,
         images_bboxs, images,
         images_points, images_mline_boxes, preprocessed_np) = self.forward_detection_np(inputs, **forward_parameters)
        zones = convert_multiline_images_to_one_line(
            image_ids,
            images,
            zones,
            images_mline_boxes,
            images_bboxs,
            count_lines,
            region_names)
        return self.forward_recognition_np(region_ids, region_names,
                                           count_lines, confidences,
                                           zones, image_ids,
                                           images_bboxs, images,
                                           images_points, preprocessed_np, **forward_parameters)
