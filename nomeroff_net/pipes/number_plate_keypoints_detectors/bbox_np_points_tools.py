import cv2
import math
import numpy as np
import collections
from typing import List, Dict, Tuple, Union, Any
from collections import OrderedDict

from nomeroff_net.tools.image_processing import (fline,
                                                 distance,
                                                 linear_line_matrix,
                                                 get_y_by_matrix,
                                                 find_distances,
                                                 fix_clockwise2,
                                                 find_min_x_idx,
                                                 detect_intersection,
                                                 reshape_points)

try:
    from .cpp_bindings.cpp_bindings import find_char_boxes, find_word_boxes
    CPP_BIND_AVAILABLE = True
except Exception as e:
    CPP_BIND_AVAILABLE = False


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


def get_det_boxes(textmap, linkmap, text_threshold, link_threshold, low_text, use_cpp_bindings=True):
    """
    get det boxes
    """
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

    if CPP_BIND_AVAILABLE and use_cpp_bindings:
        det, mapper = find_word_boxes(textmap, labels, n_labels, stats,
                                      text_threshold, fast_mode=True, rotated_box=True)
    else:
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


def filter_boxes(bboxes: List[Union[np.ndarray, np.ndarray]], dimensions: List[Dict],
                 target_points: Any,
                 np_bboxes_idx: List[int], filter_range: int = 0.7) -> Tuple[List[int], List[int], int]:
    """
    TODO: describe function
    """
    target_points = normalize_rect(target_points)
    dy = distance(target_points[0], target_points[1])
    new_np_bboxes_idx = []
    garbage_bboxes_idx = []
    max_dy = 0
    if len(bboxes):
        max_dy = max([dimension['dy'] for dimension in dimensions])
    for i, (bbox, dimension) in enumerate(zip(bboxes, dimensions)):
        if i in np_bboxes_idx:
            coef = dimension['dy']/max_dy
            if coef > filter_range:
                new_np_bboxes_idx.append(i)
            else:
                boxify_factor = dimension['dx']/dimension['dy']
                dx_offset = round(dimension['dx']/2)
                if bbox[0][0] <= dx_offset and 0.7 < boxify_factor < 1.7:
                    garbage_bboxes_idx.append(i)
                else:
                    new_np_bboxes_idx.append(i)
        else:
            garbage_bboxes_idx.append(i)

    probably_count_lines = round(dy/max_dy)
    probably_count_lines = 1 if probably_count_lines < 1 else probably_count_lines
    probably_count_lines = 3 if probably_count_lines > 3 else probably_count_lines
    return new_np_bboxes_idx, garbage_bboxes_idx, probably_count_lines

