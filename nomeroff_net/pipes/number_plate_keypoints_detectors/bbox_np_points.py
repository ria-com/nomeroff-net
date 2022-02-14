import os
import time
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
    normalize_rect,
)
info = modelhub.download_repo_for_model("craft_mlt")
CRAFT_DIR = info["repo_path"]

# load CRAFT packages
from craft_mlt import imgproc
from craft_mlt import craft_utils
from craft_mlt.craft import CRAFT
from craft_mlt.refinenet import RefineNet


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
                   trained_model: str = os.path.join(CRAFT_DIR, 'weights/craft_mlt_25k.pth'),
                   refiner_model: str = os.path.join(CRAFT_DIR, 'weights/craft_refiner_CTW1500.pth')) -> None:
        """
        TODO: describe method
        """
        is_cuda = device == "cuda"
        self.is_cuda = is_cuda

        # load net
        self.net = CRAFT()  # initialize

        print('Loading weights from checkpoint (' + trained_model + ')')
        if is_cuda:
            model = torch.load(trained_model)
            self.net.load_state_dict(copy_state_dict(model))
        else:
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
            print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
            if is_cuda:
                self.refine_net.load_state_dict(copy_state_dict(torch.load(refiner_model)))
                self.refine_net = self.refine_net.cuda()
            else:
                self.refine_net.load_state_dict(copy_state_dict(torch.load(refiner_model, map_location='cpu')))

            self.refine_net.eval()
            self.is_poly = True

    def detect_by_image_path(self,
                             image_path: str,
                             target_boxes: List[Dict],
                             quality_profile: List = None) -> Tuple[List[Dict], Any]:
        """
        TODO: describe method
        """
        if quality_profile is None:
            quality_profile = [1, 0, 0, 0]
        image = imgproc.loadImage(image_path)
        for target_box in target_boxes:
            x = min(target_box['x1'], target_box['x2'])
            w = abs(target_box['x2'] - target_box['x1'])
            y = min(target_box['y1'], target_box['y2'])
            h = abs(target_box['y2'] - target_box['y1'])

            image_part = image[y:y + h, x:x + w]
            points = self.detect_in_bbox(image_part)
            propably_points = add_coordinates_offset(points, x, y)
            target_box['points'] = []
            target_box['img_parts'] = []
            if len(propably_points):
                target_points_variants = make_rect_variants(propably_points, quality_profile)
                if len(target_points_variants) > 1:
                    img_parts = [get_cv_zone_rgb(image, reshape_points(rect, 1)) for rect in target_points_variants]
                    normalized_perspective_img = normalize_perspective_images(img_parts)
                    idx = detect_best_perspective(normalized_perspective_img)
                    target_box['points'] = target_points_variants[idx]
                    target_box['img_parts'] = img_parts
                else:
                    target_box['points'] = target_points_variants[0]
        return target_boxes, image

    def detect(self, image: np.ndarray, target_boxes: List, quality_profile: List = None) -> List:
        """
        TODO: describe method
        """
        points, mline_boxes = self.detect_mline(image, target_boxes, quality_profile)
        return points

    def detect_mline_many(self,
                          images: List[np.ndarray],
                          images_target_boxes: List,
                          quality_profile: List = None,
                          **_) -> Tuple:
        images_points = []
        images_mline_boxes = []
        for image, target_boxes in zip(images, images_target_boxes):
            points, mline_boxes = self.detect_mline(image, target_boxes, quality_profile)
            images_points.append(points)
            images_mline_boxes.append(mline_boxes)
        return images_points, images_mline_boxes

    def detect_mline(self, image: np.ndarray, target_boxes: List, quality_profile: List = None) -> Tuple:
        """
        TODO: describe method
        """
        if quality_profile is None:
            quality_profile = [3, 1, 0, 0]
        all_points = []
        all_mline_boxes = []
        for target_box in target_boxes:
            image_part, (x, w, y, h) = crop_image(image, target_box)

            if h / w > 3.5:
                image_part = cv2.rotate(image_part, cv2.ROTATE_90_CLOCKWISE)
            local_propably_points, mline_boxes = self.detect_in_bbox(image_part)
            all_mline_boxes.append(mline_boxes)
            propably_points = add_coordinates_offset(local_propably_points, x, y)
            if len(propably_points):
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
                    [x, y + h],
                    [x, y],
                    [x + w, y],
                    [x + w, y + h]
                ])
        return all_points, all_mline_boxes

    def detect_in_bbox(self,
                       image: np.ndarray,
                       low_text=0.4,
                       link_threshold=0.7,
                       text_threshold=0.6,
                       canvas_size=300,
                       mag_ratio=1.0):
        """
        TODO: describe method
        """
        bboxes, np_bboxes_idx, multiline_rects = self.detect_probably_multiline_zones(
            image, low_text, link_threshold,
            text_threshold, canvas_size, mag_ratio)

        target_points = []

        if len(np_bboxes_idx) == 1:
            target_points = bboxes[np_bboxes_idx[0]]

        if len(np_bboxes_idx) > 1:
            target_points = minimum_bounding_rectangle(np.concatenate(multiline_rects, axis=0))

        if len(np_bboxes_idx) > 0:
            target_points = normalize_rect(target_points)
            target_points = addopt_rect_to_bbox(target_points, image.shape, 7, 12, 0, 12)
        return target_points, multiline_rects

    @staticmethod
    def preprocessing(image, canvas_size, mag_ratio):
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image,
            canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        x = imgproc.normalizeMeanVariance(img_resized)
        return x, ratio_h, ratio_w

    @staticmethod
    def postprocessing(score_text: np.ndarray, score_link: np.ndarray, text_threshold: float,
                       link_threshold: float, low_text: float, ratio_w: float, ratio_h: float):
        # Post-processing
        boxes = get_det_boxes(score_text, score_link, text_threshold, link_threshold, low_text)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)
        return boxes, ret_score_text

    @torch.no_grad()
    def forward(self, x: np.ndarray, cuda: bool) -> Tuple[Any, Any]:
        """
        TODO: describe function
        """
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        y, feature = self.net(x)
        y_refiner = self.refine_net(y, feature)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        return score_text, score_link

    def detect_probably_multiline_zones(self,
                                        image,
                                        low_text=0.4,
                                        link_threshold=0.7,
                                        text_threshold=0.6,
                                        canvas_size=300,
                                        mag_ratio=1.0):
        """
        TODO: describe method
        """
        x, ratio_h, ratio_w = self.preprocessing(image, canvas_size, mag_ratio)
        score_text, score_link = self.forward(x, self.is_cuda)
        bboxes, ret_score_text = self.postprocessing(
            score_text, score_link, text_threshold,
            link_threshold, low_text, ratio_w, ratio_h)

        dimensions = []
        for poly in bboxes:
            dimensions.append({'dx': distance(poly[0], poly[1]), 'dy': distance(poly[1], poly[2])})

        np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions)
        multiline_rects = [bboxes[i] for i in np_bboxes_idx]
        return bboxes, np_bboxes_idx, multiline_rects
