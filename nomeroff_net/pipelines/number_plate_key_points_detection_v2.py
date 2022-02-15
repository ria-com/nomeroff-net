import cv2
import numpy as np
from torch import no_grad
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline, empty_method
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points import NpPointsCraft
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import (
    add_coordinates_offset,
    make_rect_variants,
    detect_best_perspective,
    normalize_perspective_images,
    addopt_rect_to_bbox,
    split_boxes,
    normalize_rect,
)
from nomeroff_net.tools import unzip
from nomeroff_net.tools.image_processing import (distance,
                                                 get_cv_zone_rgb,
                                                 crop_image,
                                                 minimum_bounding_rectangle,
                                                 reshape_points)


class NumberPlateKeyPointsDetectionV2(Pipeline):
    """
    Number Plate Localization
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 mtl_model_path: str = "latest",
                 refiner_model_path: str = "latest",
                 **kwargs):
        super().__init__(task, image_loader, **kwargs)
        self.detector = NpPointsCraft()
        self.detector.load(mtl_model_path, refiner_model_path)

    def sanitize_parameters(self, quality_profile=None, **kwargs):
        forward_parameters = {}
        if quality_profile is not None:
            forward_parameters["quality_profile"] = quality_profile
        return {}, forward_parameters, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        canvas_size = preprocess_parameters.get("canvas_size", 300)
        mag_ratio = preprocess_parameters.get("mag_ratio", 1.0)

        res = []
        for image_id, (image, target_boxes) in enumerate(inputs):
            for target_box in target_boxes:
                image_part, (x0, w0, y0, h0) = crop_image(image, target_box)
                if h0 / w0 > 3.5:
                    image_part = cv2.rotate(image_part, cv2.ROTATE_90_CLOCKWISE)
                x, ratio_h, ratio_w = self.detector.preprocessing(image_part, canvas_size, mag_ratio)
                image_shape = image_part.shape
                res.append([x, image, ratio_h, ratio_w, target_box, image_id, (x0, w0, y0, h0), image_shape])
        return res

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        return [[*self.detector.forward(x[0]), *x[1:]] for x in inputs]

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        quality_profile = postprocess_parameters.get("quality_profile", [3, 1, 0, 0])
        text_threshold = postprocess_parameters.get("text_threshold", 0.6)
        link_threshold = postprocess_parameters.get("link_threshold", 0.7)
        low_text = postprocess_parameters.get("low_text", 0.4)

        all_points = []
        all_mline_boxes = []
        all_image_ids = []
        for score_text, score_link, image, ratio_h, ratio_w, target_box, image_id, (x0, w0, y0, h0), image_shape \
                in inputs:
            all_image_ids.append(image_id)
            bboxes, ret_score_text = self.detector.postprocessing(
                score_text, score_link, text_threshold,
                link_threshold, low_text, ratio_w, ratio_h)
            dimensions = [{'dx': distance(poly[0], poly[1]), 'dy': distance(poly[1], poly[2])}
                          for poly in bboxes]
            np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions)
            multiline_rects = [bboxes[i] for i in np_bboxes_idx]
            target_points = []

            if len(np_bboxes_idx) == 1:
                target_points = bboxes[np_bboxes_idx[0]]

            if len(np_bboxes_idx) > 1:
                target_points = minimum_bounding_rectangle(np.concatenate(multiline_rects, axis=0))

            if len(np_bboxes_idx) > 0:
                target_points = normalize_rect(target_points)
                target_points = addopt_rect_to_bbox(target_points, image_shape, 7, 12, 0, 12)
            local_propably_points, mline_boxes = target_points, multiline_rects
            all_mline_boxes.append(mline_boxes)
            propably_points = add_coordinates_offset(local_propably_points, x0, y0)
            if len(propably_points):
                target_points_variants = make_rect_variants(propably_points, quality_profile)
                if len(target_points_variants):
                    target_points_variants = make_rect_variants(propably_points, quality_profile)
                    if len(target_points_variants) > 1:
                        img_parts = [get_cv_zone_rgb(image, reshape_points(rect, 1)) for rect in target_points_variants]
                        idx = detect_best_perspective(normalize_perspective_images(img_parts))
                        points = target_points_variants[idx]
                    else:
                        points = target_points_variants[0]
                    all_points.append(points)
                else:
                    all_points.append([
                        [x0, y0 + h0],
                        [x0, y0],
                        [x0 + w0, y0],
                        [x0 + w0, y0 + h0]
                    ])

        n = max(all_image_ids) + 1
        images_points = [[] for _ in range(n)]
        images_mline_boxes = [[] for _ in range(n)]
        for point, mline_box, image_id in zip(all_points, all_mline_boxes, all_image_ids):
            images_points[image_id].append(point)
            images_mline_boxes[image_id].append(point)
        return unzip([images_points, images_mline_boxes])
