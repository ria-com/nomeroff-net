import os
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from typing import List, Dict, Tuple, Any

from nomeroff_net.tools.mcm import (modelhub, get_mode_torch)
from nomeroff_net.tools.image_processing import (distance,
                                                 get_cv_zone_rgb,
                                                 crop_image,
                                                 minimum_bounding_rectangle,
                                                 reshape_points)

from .bbox_np_points_tools import (
    copy_state_dict,
    add_coordinates_offset,
    make_rect_variants,
    detect_best_perspective,
    normalize_perspective_images,
    get_det_boxes,
    addopt_rect_to_bbox,
    split_boxes,
    filter_boxes,
    normalize_rect,
)

# load CRAFT packages
from craft_text_detector import image_utils
from craft_text_detector import craft_utils
from craft_text_detector.models.craftnet import CraftNet
from craft_text_detector.models.refinenet import RefineNet


class NpPointsCraft(object):
    """
    np_points_craft Class
    git clone https://github.com/clovaai/CRAFT-pytorch.git
    """

    def __init__(self):
        self.is_cuda = False
        self.is_poly = False
        self.net = None
        self.refine_net = None

    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def load(self,
             mtl_model_path: str = "latest",
             refiner_model_path: str = "latest") -> None:
        """
        TODO: describe method
        """
        if mtl_model_path == "latest":
            model_info = modelhub.download_model_by_name("craft_mlt")
            mtl_model_path = model_info["path"]
        if refiner_model_path == "latest":
            model_info = modelhub.download_model_by_name("craft_refiner")
            refiner_model_path = model_info["path"]
        device = "cpu"
        if get_mode_torch() == "gpu":
            device = "cuda"
        self.load_model(device, True, mtl_model_path, refiner_model_path)

    def load_model(self,
                   device: str = "cuda",
                   is_refine: bool = True,
                   trained_model: str = None,
                   refiner_model: str = None) -> None:
        """
        TODO: describe method
        """
        is_cuda = device == "cuda"
        self.is_cuda = is_cuda

        # load net
        self.net = CraftNet()  # initialize

        model = copy_state_dict(torch.load(trained_model, map_location='cpu'))
        self.net.load_state_dict(model)

        if is_cuda:
            self.net = self.net.cuda()
            cudnn.benchmark = False

        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if is_refine:
            self.refine_net = RefineNet()
            self.refine_net.load_state_dict(copy_state_dict(torch.load(refiner_model, map_location='cpu')))
            if is_cuda:
                self.refine_net = self.refine_net.cuda()

            self.refine_net.eval()
            self.is_poly = True

    @staticmethod
    def preprocessing_craft(image, canvas_size, mag_ratio):
        # resize
        img_resized, target_ratio, size_heatmap = image_utils.resize_aspect_ratio(
            image,
            canvas_size,
            interpolation=cv2.INTER_LINEAR)
        ratio_h = ratio_w = 1 / target_ratio
        x = image_utils.normalizeMeanVariance(img_resized)
        return x, ratio_h, ratio_w

    @staticmethod
    def craft_postprocessing(score_text: np.ndarray, score_link: np.ndarray, text_threshold: float,
                             link_threshold: float, low_text: float, ratio_w: float, ratio_h: float):
        # Post-processing
        boxes = get_det_boxes(score_text, score_link, text_threshold, link_threshold, low_text)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = image_utils.cvt2HeatmapImg(render_img)
        return boxes, ret_score_text

    @torch.no_grad()
    def forward(self, x: np.ndarray) -> Tuple[Any, Any]:
        """
        TODO: describe function
        """
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if self.is_cuda:
            x = x.cuda()

        y, feature = self.net(x)
        y_refiner = self.refine_net(y, feature)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        return score_text, score_link

    def detect(self,
               inputs,
               canvas_size: int = 300,
               mag_ratio: float = 1.0,
               quality_profile: List = None,
               text_threshold: float = 0.6,
               link_threshold: float = 0.7,
               low_text: float = 0.4
               ):
        preprocessed_data = self.preprocess(inputs, canvas_size, mag_ratio)
        model_outputs = self.forward_batch(preprocessed_data)
        return self.postprocess(model_outputs, quality_profile, text_threshold, link_threshold, low_text)

    @torch.no_grad()
    def forward_batch(self, inputs: Any, **_) -> Any:
        return [[*self.forward(x[0]), *x[1:]] for x in inputs]

    def preprocess(self, inputs: Any, canvas_size: int = 300, mag_ratio: float = 1.0, **_) -> Any:
        res = []
        for image_id, (image, target_boxes) in enumerate(inputs):
            for target_box in target_boxes:
                image_part, (x0, w0, y0, h0) = crop_image(image, target_box)
                if h0 / w0 > 3.5:
                    image_part = cv2.rotate(image_part, cv2.ROTATE_90_CLOCKWISE)
                x, ratio_h, ratio_w = self.preprocessing_craft(image_part, canvas_size, mag_ratio)
                res.append([x, image, ratio_h, ratio_w, target_box, image_id, (x0, w0, y0, h0), image_part])
        return res

    def postprocess(self, inputs: Any,
                    quality_profile: List = None,
                    text_threshold: float = 0.6,
                    link_threshold: float = 0.7,
                    low_text: float = 0.4,
                    in_zone_only: bool = False,
                    **_) -> Any:
        if quality_profile is None:
            quality_profile = [1, 0, 0, 0]

        all_points = []
        all_mline_boxes = []
        all_image_ids = []
        all_count_lines = []
        all_image_parts = []
        for score_text, score_link, image, ratio_h, ratio_w, target_box, image_id, (x0, w0, y0, h0), image_part \
                in inputs:
            all_image_parts.append(image_part)
            image_shape = image_part.shape
            all_image_ids.append(image_id)
            bboxes, ret_score_text = self.craft_postprocessing(
                score_text, score_link, text_threshold,
                link_threshold, low_text, ratio_w, ratio_h)
            dimensions = [{'dx': distance(poly[0], poly[1]), 'dy': distance(poly[1], poly[2])}
                          for poly in bboxes]
            np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions)
            multiline_rects = [bboxes[i] for i in np_bboxes_idx]

            probably_count_lines = 1
            target_points = []
            if len(np_bboxes_idx) == 1:
                target_points = bboxes[np_bboxes_idx[0]]
            if len(np_bboxes_idx) > 1:
                started_boxes = np.concatenate([bboxes[i] for i in np_bboxes_idx], axis=0)
                target_points = minimum_bounding_rectangle(np.concatenate(multiline_rects, axis=0))
                np_bboxes_idx, garbage_bboxes_idx, probably_count_lines = filter_boxes(bboxes, dimensions,
                                                                                       target_points, np_bboxes_idx)
                filtred_boxes = np.concatenate([bboxes[i] for i in np_bboxes_idx], axis=0)
                if len(started_boxes) > len(filtred_boxes):
                    target_points = minimum_bounding_rectangle(started_boxes)
            if len(np_bboxes_idx) > 0:
                target_points = normalize_rect(target_points)
                target_points = addopt_rect_to_bbox(target_points, image_shape, 7, 12, 0, 12)
            all_count_lines.append(probably_count_lines)

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
                    if in_zone_only:
                        for i in range(len(points)):
                            points[i][0] = x0 if points[i][0] < x0 else points[i][0]
                            points[i][1] = y0 if points[i][1] < y0 else points[i][1]
                            points[i][0] = x0 + w0 if points[i][0] > x0 + w0 else points[i][0]
                            points[i][1] = y0 + h0 if points[i][1] > y0 + h0 else points[i][1]
                    all_points.append(points)
                else:
                    all_points.append([
                        [x0, y0 + h0],
                        [x0, y0],
                        [x0 + w0, y0],
                        [x0 + w0, y0 + h0]
                    ])
        if len(all_image_ids):
            n = max(all_image_ids) + 1
        else:
            n = 1
        images_points = [[] for _ in range(n)]
        images_mline_boxes = [[] for _ in range(n)]
        for point, mline_box, image_id in zip(all_points, all_mline_boxes, all_image_ids):
            images_points[image_id].append(point)
            images_mline_boxes[image_id].append(mline_box)
        return images_points, images_mline_boxes
