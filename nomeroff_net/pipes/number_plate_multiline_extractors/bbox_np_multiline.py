import cv2
import numpy as np
import importlib

from typing import List, Any, Tuple, Dict
from nomeroff_net.np_multiline import default

from nomeroff_net.tools.image_processing import (distance,
                                                 linear_line_matrix,
                                                 get_y_by_matrix,
                                                 normalize,
                                                 build_perspective)

multiline_plugins = {}


def mean_rect_distance(rect: List, start_idx: int) -> np.ndarray:
    idxs = [(idx, idx - 4)[idx > 3] for idx in range(start_idx, len(rect) + start_idx)]
    return np.mean([distance(rect[idxs[0]], rect[idxs[1]]), distance(rect[idxs[2]], rect[idxs[3]])])


def find_width(rect: List) -> np.ndarray:
    return mean_rect_distance(rect, 0)


def find_hieght(rect: List) -> np.ndarray:
    return mean_rect_distance(rect, 1)


def to_pretty_point(points: List) -> List:
    sorted_on_x = sorted([point for point in points], key=lambda x: x[0])
    res = [[0, 0], [0, 0], [0, 0], [0, 0]]
    if sorted_on_x[0][1] < sorted_on_x[1][1]:
        res[0], res[3] = sorted_on_x[0], sorted_on_x[1]
    else:
        res[0], res[3] = sorted_on_x[1], sorted_on_x[0]

    if sorted_on_x[2][1] < sorted_on_x[3][1]:
        res[1], res[2] = sorted_on_x[2], sorted_on_x[3]
    else:
        res[1], res[2] = sorted_on_x[3], sorted_on_x[2]
    return res


def rotate_to_pretty(points: List) -> Tuple:
    points = to_pretty_point(points)
    w = find_width(points)
    h = find_hieght(points)
    return points, w, h


def get_cv_zones_rgb_lite(img: np.ndarray, rects: list) -> List:
    dsts = []
    for rect in rects:
        rect, w, h = rotate_to_pretty(rect)
        dst = build_perspective(img, rect, int(w), int(h))
        dsts.append(dst)
    return dsts


def target_resize(zone: np.ndarray, target_zone_value: float) -> np.ndarray:
    height = target_zone_value
    k = target_zone_value / zone.shape[0]
    width = int(zone.shape[1] * k)
    dim = (width, height)
    # resize image
    return cv2.resize(zone, dim, interpolation=cv2.INTER_AREA)


def calc_diff(p1: List[int], p2: List[int]) -> Tuple[int, int]:
    return p1[0]-p2[0], p1[1]-p2[1]


def apply_new_point(point: List[int], dx: int, dy: int) -> List[int]:
    return [point[0]+dx, point[1]+dy]


class MultilineConverter:
    def __init__(self,
                 image_part: np.ndarray,
                 rects: List,
                 target_points: List,
                 line_dispersion: float = 0.3,
                 bg_fill: int = 223) -> None:
        self.image_part = image_part
        self.rects = rects
        self.target_points = target_points
        self.dx, self.dy = calc_diff(target_points[0], target_points[3])
        self.line_dispersion = line_dispersion
        self.bg_fill = bg_fill
        self.probably_lines = self.detect_line_count()

    def detect_line_count(self) -> Dict:
        lines = {}
        zones_cnt = len(self.rects)
        current_line = 1
        idx = 0
        lines[current_line] = [
            {
                'idx': idx,
                'h': distance(self.rects[idx][3], self.rects[idx][0]),
                'y': self.rects[idx][0][1],
                'x': self.rects[idx][0][0],
                'matrix': linear_line_matrix(self.rects[idx][0], apply_new_point(self.rects[idx][0], self.dx, self.dy))}
        ]
        if zones_cnt > 1:
            for idx in range(1, len(self.rects)):
                rect = self.rects[idx - 1]
                rect_next = self.rects[idx]
                h = distance(rect[3], rect[0])
                y = rect[0][1]
                x = rect[0][0]
                dy = h * self.line_dispersion
                matrix = linear_line_matrix(rect[0], apply_new_point(rect[0], self.dx, self.dy))
                y_next = get_y_by_matrix(matrix, rect_next[0][0])
                if not (y_next - dy < rect_next[0][1] < y_next + dy):
                    current_line = current_line + 1
                if current_line not in lines.keys():
                    lines[current_line] = []
                lines[current_line].append({'idx': idx, 'h': h, 'y': y, 'x': x, 'matrix': matrix})
        return lines

    def is_multiline(self) -> bool:
        return len(self.probably_lines.keys()) > 1

    def covert_to_1_line(self, region_name: str) -> np.ndarray:
        # Fix region_name
        region_name = region_name.replace('-', '_')
        if self.is_multiline():
            cmd = self.load_region_plugin(region_name)
            return self.merge_lines(cmd)
        else:
            return self.image_part

    @staticmethod
    def load_region_plugin(region_name: str) -> str:
        if region_name in multiline_plugins:
            mod = multiline_plugins[region_name]
        else:
            try:
                module = 'nomeroff_net.np_multiline.%s' % region_name
                mod = importlib.import_module(module)
            except ModuleNotFoundError:
                print('Module {} is absent!'.format(region_name))
                mod = default
            multiline_plugins[region_name] = mod
        return mod

    def merge_lines(self, cmd: Any) -> np.ndarray:
        # NP config
        np_config_default = {
            "padding-left-coef": 1.5,
            "padding-right-coef": 6,
            "padding-zones-coef": 4,
            "padding-top-coef": 8,
            "padding-bottom-coef": 8,
        }

        # Detect zones
        img_zones = get_cv_zones_rgb_lite(self.image_part, self.rects)
        img_zones, np_config = cmd.prepare_multiline_rects(self.rects, img_zones, self.probably_lines)

        if np_config is None:
            np_config = {}
        padding_left_coef = np_config.get('padding-left-coef', np_config_default["padding-left-coef"])
        padding_right_coef = np_config.get('padding-right-coef', np_config_default["padding-right-coef"])
        padding_zones_coef = np_config.get('padding-zones-coef', np_config_default["padding-zones-coef"])
        padding_top_coef = np_config.get('padding-top-coef', np_config_default["padding-top-coef"])
        padding_bottom_coef = np_config.get('padding-bottom-coef', np_config_default["padding-bottom-coef"])

        target_zone_value = max([img_zone.shape[0] for img_zone in img_zones])

        res_zone = []
        bg_fill = np.max([np.max(zone) for zone in img_zones])
        cnt = len(img_zones)
        for idx, zone in enumerate(img_zones):
            if target_zone_value != zone.shape[0]:
                zone = target_resize(zone, target_zone_value)
            res_zone.append(zone)
            if idx < cnt-1:
                res_zone.append(np.ones((target_zone_value,
                                         int(target_zone_value / padding_zones_coef),
                                         3), dtype="uint8") * bg_fill)

        # Show result
        temp = np.hstack(res_zone)
        temp = normalize(temp)

        res_zone = [
            np.ones((target_zone_value, int(target_zone_value / padding_left_coef)), dtype="uint8") * self.bg_fill,
            temp,
            np.ones((target_zone_value, int(target_zone_value / padding_right_coef)), dtype="uint8") * self.bg_fill
        ]
        temp = np.hstack(res_zone)

        res_zone = [
            np.ones((int(target_zone_value / padding_top_coef), temp.shape[1]), dtype="uint8") * self.bg_fill,
            temp,
            np.ones((int(target_zone_value / padding_bottom_coef), temp.shape[1]), dtype="uint8") * self.bg_fill
        ]
        return np.vstack(res_zone)
