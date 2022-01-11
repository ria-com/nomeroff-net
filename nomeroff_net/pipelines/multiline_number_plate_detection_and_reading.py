from typing import Any, Dict, Optional, List
from nomeroff_net.tools import unzip
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.number_plate_detection_and_reading import NumberPlateDetectionAndReading
from nomeroff_net.tools.image_processing import crop_number_plate_zones_from_images, group_by_image_ids
from nomeroff_net.pipes.number_plate_multiline_extractors.multiline_np_extractor \
    import convert_multiline_images_to_one_line


class MultilineNumberPlateDetectionAndReadingRuntime(NumberPlateDetectionAndReading):
    """
    Number Plate Detection and reading runtime
    """

    def __init__(self,
                 task,
                 image_loader: Optional[BaseImageLoader],
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

    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        images_bboxs, images = unzip(self.number_plate_localization(inputs, **forward_parameters))
        images_points, images_mline_boxes = unzip(self.number_plate_key_points_detection(unzip([images, images_bboxs]),
                                                                                         **forward_parameters))
        zones, image_ids = crop_number_plate_zones_from_images(images, images_points)
        (region_ids, region_names, count_lines,
         confidences, predicted) = unzip(self.number_plate_classification(zones, **forward_parameters))
        zones = convert_multiline_images_to_one_line(
            image_ids,
            images,
            zones,
            images_mline_boxes,
            images_bboxs,
            count_lines,
            region_names)
        texts, _ = unzip(self.number_plate_text_reading(unzip([zones,
                                                               region_names,
                                                               count_lines]), **forward_parameters))
        (region_ids, region_names, count_lines, confidences, texts, zones) = \
            group_by_image_ids(image_ids, (region_ids, region_names, count_lines, confidences, texts, zones))
        return unzip([images, images_bboxs,
                      images_points, zones,
                      region_ids, region_names,
                      count_lines, confidences, texts])
