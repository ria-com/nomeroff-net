from torch import no_grad
from typing import Any, Dict, Optional, List, Union
from nomeroff_net.tools import unzip
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline, empty_method
from nomeroff_net.tools.image_processing import crop_number_plate_zones_from_images, group_by_image_ids
from nomeroff_net.pipes.number_plate_localizators.yolo_v5_detector import Detector
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points import NpPointsCraft
from nomeroff_net.pipes.number_plate_classificators.options_detector import OptionsDetector
from nomeroff_net.pipes.number_plate_text_readers.text_detector import TextDetector
from .number_plate_text_reading import DEFAULT_PRESETS


class NumberPlateDetectionAndReadingV2(Pipeline):
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
                 presets: Dict = None,
                 classification_options: List = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 class_detector=OptionsDetector,
                 ocr_class_detector=TextDetector,
                 **kwargs):
        super().__init__(task, image_loader, **kwargs)
        self.localization_detector = Detector()
        self.localization_detector.load(path_to_model)

        self.key_points_detector = NpPointsCraft()
        self.key_points_detector.load(mtl_model_path, refiner_model_path)

        self.option_detector = class_detector(options=classification_options)
        self.option_detector.load(path_to_classification_model, options=classification_options)

        if presets is None:
            presets = DEFAULT_PRESETS
        self.ocr_detector = ocr_class_detector(presets, default_label, default_lines_count)
        Pipeline.__init__(self, task, image_loader, **kwargs)

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        return images

    def sanitize_parameters(self, img_size=None, stride=None, min_accuracy=None, quality_profile=None, **kwargs):
        params = {}
        if img_size is not None:
            params["img_size"] = img_size
        if stride is not None:
            params["stride"] = stride
        if min_accuracy is not None:
            params["min_accuracy"] = min_accuracy
        if quality_profile is not None:
            params["quality_profile"] = quality_profile
        return {}, params, {}

    def forward_detection_np(self, images: Any, **forward_parameters: Dict):
        images_target_boxes = self.localization_detector.predict(images)

        images_points, images_mline_boxes = self.key_points_detector.detect(
            unzip([images, images_target_boxes]),
            **forward_parameters)

        zones, image_ids = crop_number_plate_zones_from_images(images, images_points)
        zones_model_input = self.option_detector.preprocess(zones)
        options_output = self.option_detector.forward(zones_model_input)
        options_output = [p.cpu().numpy() for p in options_output]
        confidences, region_ids, count_lines = self.option_detector.unzip_predicted(options_output)
        count_lines = self.option_detector.custom_count_lines_id_to_all_count_lines(count_lines)
        region_names = self.option_detector.get_region_labels(region_ids)

        return (region_ids, region_names, count_lines, confidences,
                options_output, zones, image_ids, images_target_boxes, images,
                images_points, images_mline_boxes)

    def forward_recognition_np(self, region_ids, region_names,
                               count_lines, confidences,
                               zones, image_ids,
                               images_bboxs, images,
                               images_points, **_):
        model_inputs = self.ocr_detector.preprocess(zones, region_names, count_lines)
        model_outputs = self.ocr_detector.forward(model_inputs)
        model_outputs = self.ocr_detector.postprocess(model_outputs)
        texts = model_outputs

        (region_ids, region_names, count_lines, confidences, texts, zones) = \
            group_by_image_ids(image_ids, (region_ids, region_names, count_lines, confidences, texts, zones))
        return [images, images_bboxs,
                images_points, zones,
                region_ids, region_names,
                count_lines, confidences, texts]

    @no_grad()
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
