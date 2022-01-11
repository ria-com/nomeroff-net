import os
import math
import time
import cv2
import torch
import collections
import numpy as np
import torch.backends.cudnn as cudnn

from collections import OrderedDict
from torch.autograd import Variable
from typing import List, Dict, Tuple, Any, Union

from nomeroff_net.tools.mcm import (modelhub, get_mode_torch)
from nomeroff_net.tools.image_processing import (fline,
                                                 distance,
                                                 linear_line_matrix,
                                                 get_y_by_matrix,
                                                 find_distances,
                                                 get_cv_zone_rgb,
                                                 fix_clockwise2,
                                                 find_min_x_idx,
                                                 crop_image,
                                                 detect_intersection,
                                                 minimum_bounding_rectangle,
                                                 reshape_points)

info = modelhub.download_repo_for_model("craft_mlt")
CRAFT_DIR = info["repo_path"]

# load CRAFT packages
from craft_mlt import imgproc
from craft_mlt import craft_utils
from craft_mlt.craft import CRAFT
from craft_mlt.refinenet import RefineNet


def copy_state_dict(state_dict: Dict) -> OrderedDict:
    """
    Craft routines
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def get_det_boxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8),
                                                                          connectivity=4)

    det = []
    mapper = []
    for k in range(1, n_labels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det


@torch.no_grad()
def test_net(net: CRAFT, image: np.ndarray, text_threshold: float,
             link_threshold: float, low_text: float, cuda: bool,
             canvas_size: int, refine_net: RefineNet = None,
             mag_ratio: float = 1.5) -> Tuple[Any, Any]:
    """
    TODO: describe function
    """

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image,
                                                                          canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    # Post-processing
    boxes = get_det_boxes(score_text, score_link, text_threshold, link_threshold, low_text)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    return boxes, ret_score_text


def split_boxes(bboxes: List[Union[np.ndarray, np.ndarray]], dimensions: List[Dict],
                similarity_range: int = 0.5) -> Tuple[List[int], List[int]]:
    """
    TODO: describe function
    """
    np_bboxes_idx = []
    garbage_bboxes_idx = []
    max_dy = 0
    if len(bboxes):
        max_dy = max([dimension['dy'] for dimension in dimensions])
    for i, (bbox, dimension) in enumerate(zip(bboxes, dimensions)):
        if max_dy * similarity_range <= dimension['dy']:
            np_bboxes_idx.append(i)
        else:
            garbage_bboxes_idx.append(i)
    return np_bboxes_idx, garbage_bboxes_idx


def detect_intersection_norm_dd(matrix1: np.ndarray, matrix2: np.ndarray, d1: float, d2: float) -> np.ndarray:
    """
    TODO: describe function
    """
    x = np.array([matrix1[:2], matrix2[:2]])
    c0 = matrix1[2] - d1 * (matrix1[0] ** 2 + matrix1[1] ** 2) ** 0.5
    c1 = matrix2[2] - d2 * (matrix2[0] ** 2 + matrix2[1] ** 2) ** 0.5
    y = np.array([c0, c1])
    return np.linalg.solve(x, y)


def detect_distance_from_point_to_line(matrix: List[np.ndarray],
                                       point: Union) -> float:
    """
    Определение растояния от точки к линии
    https://ru.onlinemschool.com/math/library/analytic_geometry/p_line1/
    """
    a = matrix[0]
    b = matrix[1]
    c = matrix[2]
    x = point[0]
    y = point[1]
    return abs(a * x + b * y - c) / math.sqrt(a ** 2 + b ** 2)


def fix_clockwise(target_points: List) -> List:
    """
    TODO: describe function
    """
    stat1 = fline(target_points[0], target_points[1])
    stat2 = fline(target_points[0], target_points[2])
    if target_points[0][0] == target_points[1][0] and (target_points[0][1] > target_points[1][1]):
        stat1[2] = -stat1[2]

    if stat2[2] < stat1[2]:
        target_points = np.array([target_points[0], target_points[3], target_points[2], target_points[1]])
    return target_points


def make_offsets(
        bbox: Tuple,
        distanses_offset_left_max_percentage: float,
        offset_top_max_percentage: float,
        offset_right_max_percentage: float,
        offset_bottom_max_percentage: float):

    distanses_offset_left_percentage = distanses_offset_left_max_percentage
    offset_top_percentage = offset_top_max_percentage
    offset_right_percentage = offset_right_max_percentage
    offset_bottom_percentage = offset_bottom_max_percentage

    k = bbox[1] / bbox[0]

    if k < 2:
        offset_top_percentage = offset_top_percentage / 2
        offset_bottom_percentage = offset_bottom_percentage / 2

    if k < 1:
        offset_top_percentage = 0
        offset_bottom_percentage = 0

    offsets = [
        distanses_offset_left_percentage,
        offset_top_percentage,
        offset_right_percentage,
        offset_bottom_percentage
    ]
    return offsets


def addopt_rect_to_bbox_make_points(
        distanses: List,
        bbox: Tuple,
        distanses_offset_left_max_percentage: float,
        offset_top_max_percentage: float,
        offset_right_max_percentage: float,
        offset_bottom_max_percentage: float):
    points = []
    offsets = make_offsets(bbox,
                           distanses_offset_left_max_percentage,
                           offset_top_max_percentage,
                           offset_right_max_percentage,
                           offset_bottom_max_percentage)

    cnt = len(distanses)
    for i in range(cnt):
        i_next = i + 1
        if i_next == cnt:
            i_next = 0
        offsets[i] = distanses[i_next]['d'] * offsets[i] / 100
    for i in range(cnt):
        i_prev = i
        i_next = i + 1
        if i_next == cnt:
            i_next = 0
        offset1 = offsets[i_prev]
        offset2 = offsets[i_next]
        points.append(
            detect_intersection_norm_dd(distanses[i_prev]['matrix'], distanses[i_next]['matrix'], offset1, offset2))
    return points


def addopt_rect_to_bbox(target_points: List,
                        bbox: Tuple,
                        distanses_offset_left_max_percentage: float,
                        offset_top_max_percentage: float,
                        offset_right_max_percentage: float,
                        offset_bottom_max_percentage: float) -> np.ndarray:
    """
    TODO: describe function
    """
    distanses = find_distances(target_points)
    points = addopt_rect_to_bbox_make_points(
        distanses,
        bbox,
        distanses_offset_left_max_percentage,
        offset_top_max_percentage,
        offset_right_max_percentage,
        offset_bottom_max_percentage)

    points = reshape_points(points, 3)

    distanses = find_distances(points)

    if distanses[3]['coef'][2] == 90:
        return np.array(points)

    h = bbox[0]
    w = bbox[1]

    matrix_left = linear_line_matrix([0, 0], [0, h])
    matrix_right = linear_line_matrix([w, 0], [w, h])

    p_left_top = detect_intersection(matrix_left, distanses[1]['matrix'])
    p_left_bottom = detect_intersection(matrix_left, distanses[3]['matrix'])
    p_right_top = detect_intersection(matrix_right, distanses[1]['matrix'])
    p_right_bottom = detect_intersection(matrix_right, distanses[3]['matrix'])

    offset_left_bottom = distance(points[0], p_left_bottom)
    offset_left_top = distance(points[1], p_left_top)
    offset_right_top = distance(points[2], p_right_top)
    offset_right_bottom = distance(points[3], p_right_bottom)

    over_left_top = points[1][0] < 0
    over_left_bottom = points[0][0] < 0
    if not over_left_top and not over_left_bottom:
        if offset_left_top > offset_left_bottom:
            points[0] = p_left_bottom
            left_distance = detect_distance_from_point_to_line(distanses[0]['matrix'], p_left_bottom)
            points[1] = detect_intersection_norm_dd(distanses[0]['matrix'], distanses[1]['matrix'], left_distance, 0)
        else:
            points[1] = p_left_top
            left_distance = detect_distance_from_point_to_line(distanses[0]['matrix'], p_left_top)
            points[0] = detect_intersection_norm_dd(distanses[3]['matrix'], distanses[0]['matrix'], 0, left_distance)

    over_right_top = points[2][0] > w
    over_right_bottom = points[3][0] > w
    if not over_right_top and not over_right_bottom:
        if offset_right_top > offset_right_bottom:
            points[3] = p_right_bottom
            right_distance = detect_distance_from_point_to_line(distanses[2]['matrix'], p_right_bottom)
            points[2] = detect_intersection_norm_dd(distanses[1]['matrix'], distanses[2]['matrix'], 0, right_distance)
        else:
            points[2] = p_right_top
            right_distance = detect_distance_from_point_to_line(distanses[2]['matrix'], p_right_top)
            points[3] = detect_intersection_norm_dd(distanses[2]['matrix'], distanses[3]['matrix'], right_distance, 0)

    return np.array(points)


def add_coordinates_offset(points: List or np.ndarray, x: float, y: float) -> List:
    """
    TODO: describe function
    """
    return [[point[0] + x, point[1] + y] for point in points]


def normalize_rect(rect: List) -> np.ndarray or List:
    """
    TODO: describe function
    """
    rect = fix_clockwise2(rect)
    min_x_idx = find_min_x_idx(rect)
    rect = reshape_points(rect, min_x_idx)
    coef_ccw = fline(rect[0], rect[3])
    angle_ccw = round(coef_ccw[2], 2)
    d_bottom = distance(rect[0], rect[3])
    d_left = distance(rect[0], rect[1])
    k = d_bottom / d_left
    if not round(rect[0][0], 4) == round(rect[1][0], 4):
        if d_bottom < d_left:
            k = d_left / d_bottom
            if k > 1.5 or angle_ccw < 0 or angle_ccw > 45:
                rect = reshape_points(rect, 3)
        else:
            if k < 1.5 and (angle_ccw < 0 or angle_ccw > 45):
                rect = reshape_points(rect, 3)
    return rect


def prepare_image_text(img: np.ndarray) -> np.ndarray:
    """
    сперва переведём изображение из RGB в чёрно серый
    значения пикселей будут от 0 до 255
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.normalize(gray_image, None, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    (thresh, black_and_white_image) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return black_and_white_image


def detect_best_perspective(bw_images: List[np.ndarray]) -> int:
    """
    TODO: describe function
    """
    res = []
    idx = 0
    diff = 1000000
    diff_cnt = 0
    for i, img in enumerate(bw_images):
        s = np.sum(img, axis=0)
        img_stat = collections.Counter(s)
        img_stat_dict = OrderedDict(img_stat.most_common())
        max_stat = max(img_stat_dict, key=int)
        max_stat_count = img_stat_dict[max_stat]
        min_stat = min(img_stat_dict, key=int)
        min_stat_count = img_stat_dict[min_stat]
        res.append({'max': max_stat, 'min': min_stat, 'maxCnt': max_stat_count, 'minCnt': min_stat_count})

        if min_stat < diff:
            idx = i
            diff = min_stat
        if min_stat == diff and max_stat_count + min_stat_count > diff_cnt:
            idx = i
            diff_cnt = max_stat_count + min_stat_count
    return idx


def add_point_offset(point: List, x: float, y: float) -> List:
    """
    TODO: describe function
    """
    return [point[0] + x, point[1] + y]


def add_point_offsets(points: List, dx: float, dy: float) -> List:
    """
    TODO: describe function
    """
    return [
        add_point_offset(points[0], -dx, -dy),
        add_point_offset(points[1], dx, dy),
        add_point_offset(points[2], dx, dy),
        add_point_offset(points[3], -dx, -dy),
    ]


def make_rect_variants(propably_points: List, quality_profile: List = None) -> List:
    """
    TODO: describe function
    """
    if quality_profile is None:
        quality_profile = [3, 1, 0, 0]

    steps = quality_profile[0]
    steps_plus = quality_profile[1]
    steps_minus = quality_profile[2]
    step = 1
    if len(quality_profile) > 3:
        step_adaptive = quality_profile[3] > 0
    else:
        step_adaptive = False

    distanses = find_distances(propably_points)

    point_centre_left = [propably_points[0][0] + (propably_points[1][0] - propably_points[0][0]) / 2,
                         propably_points[0][1] + (propably_points[1][1] - propably_points[0][1]) / 2]

    if distanses[3]["matrix"][1] == 0:
        return [propably_points]
    point_bottom_left = [point_centre_left[0], get_y_by_matrix(distanses[3]["matrix"], point_centre_left[0])]
    dx = propably_points[0][0] - point_bottom_left[0]
    dy = propably_points[0][1] - point_bottom_left[1]

    dx_step = dx / steps
    dy_step = dy / steps

    if step_adaptive:
        d_max = distance(point_centre_left, propably_points[0])
        dd = math.sqrt(dx ** 2 + dy ** 2)
        steps_all = int(d_max / dd)

        step = int((steps_all * 2) / steps)
        if step < 1:
            step = 1
        steps_minus = steps_all + steps_minus * step
        steps_plus = steps_all + steps_plus * step

    points_arr = []
    for i in range(-steps_minus, steps + steps_plus + 1, step):
        points_arr.append(add_point_offsets(propably_points, i * dx_step, i * dy_step))
    return points_arr


def normalize_perspective_images(images: List or np.ndarray) -> List[np.ndarray]:
    """
    TODO: describe function
    """
    new_images = []
    for img in images:
        new_images.append(prepare_image_text(img))
    return new_images


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
                          quality_profile: List = None) -> Tuple:
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

        bboxes, score_text = test_net(self.net, image, text_threshold, link_threshold, low_text,
                                      self.is_cuda, canvas_size, self.refine_net, mag_ratio)
        dimensions = []
        for poly in bboxes:
            dimensions.append({'dx': distance(poly[0], poly[1]), 'dy': distance(poly[1], poly[2])})

        np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions)

        target_points = []

        if len(np_bboxes_idx) == 1:
            target_points = bboxes[np_bboxes_idx[0]]

        if len(np_bboxes_idx) > 1:
            target_points = minimum_bounding_rectangle(np.concatenate([bboxes[i] for i in np_bboxes_idx], axis=0))

        if len(np_bboxes_idx) > 0:
            target_points = normalize_rect(target_points)
            target_points = addopt_rect_to_bbox(target_points, image.shape, 7, 12, 0, 12)
        return target_points, [bboxes[i] for i in np_bboxes_idx]

    def detect_probably_multiline_zones(self,
                                        image,
                                        low_text=0.4,
                                        link_threshold=0.7,
                                        text_threshold=0.6,
                                        canvas_size=300,
                                        mag_ratio=1.0,
                                        debug=False):
        """
        TODO: describe method
        """
        t = time.time()
        bboxes, score_text = test_net(self.net, image, text_threshold, link_threshold, low_text,
                                      self.is_cuda, self.is_poly, canvas_size, self.refine_net, mag_ratio)
        if debug:
            print("elapsed time : {}s".format(time.time() - t))

        dimensions = []
        for poly in bboxes:
            dimensions.append({'dx': distance(poly[0], poly[1]), 'dy': distance(poly[1], poly[2])})

        np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions)

        return [bboxes[i] for i in np_bboxes_idx]
