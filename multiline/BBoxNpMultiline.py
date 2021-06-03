import cv2
import numpy as np
import importlib
from typing import List, Any, Tuple, Dict
from NomeroffNet.NpMultiline import default
from tools import (fline,
                   distance,
                   linearLineMatrix,
                   getYByMatrix,
                   findDistances,
                   buildPerspective,
                   getCvZoneRGB,
                   getMeanDistance,
                   reshapePoints,
                   getCvZonesRGB,
                   convertCvZonesRGBtoBGR,
                   getCvZonesBGR)

multiline_plugins = {}


def meanRectDistance(rect: List, start_idx: int) -> np.ndarray:
    idxs = [(idx, idx - 4)[idx > 3] for idx in range(start_idx, len(rect) + start_idx)]
    return np.mean([distance(rect[idxs[0]], rect[idxs[1]]), distance(rect[idxs[2]], rect[idxs[3]])])


def findWidth(rect: List) -> np.ndarray:
    return meanRectDistance(rect, 0)


def findHieght(rect: List) -> np.ndarray:
    return meanRectDistance(rect, 1)


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
    w = findWidth(points)
    h = findHieght(points)
    return points, w, h


def get_cv_zonesRGBLite(img: np.ndarray, rects: list) -> List:
    dsts = []
    for rect in rects:
        rect, w, h = rotate_to_pretty(rect)
        dst = buildPerspective(img, rect, w, h)
        dsts.append(dst)
    return dsts


def normalize(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img -= np.amin(img)
    img = img.astype(np.float32)
    img *= 255 / np.amax(img)
    img = img.astype("uint8")
    return img


def normalize_color(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    color_min = np.amin(img)
    img -= color_min
    color_max = np.amax(img)
    img *= 255/color_max
    img = img.astype(np.uint8)
    return img


def target_resize(zone: np.ndarray, target_zone_value: float) -> np.ndarray:
    height = target_zone_value
    k = target_zone_value / zone.shape[0]
    print(k)
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
        self.imagePart = image_part
        self.rects = rects
        self.targetPoints = target_points
        self.dx, self.dy = calc_diff(target_points[0], target_points[3])
        self.line_dispersion = line_dispersion
        self.bg_fill = bg_fill
        self.probablyLines = self.detect_line_count()

    def detect_line_count(self) -> Dict:
        # print('self.rects')
        # print(self.rects)
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
                'matrix': linearLineMatrix(self.rects[idx][0], apply_new_point(self.rects[idx][0], self.dx, self.dy))}
        ]
        if zones_cnt > 1:
            for idx in range(1, len(self.rects)):
                print('idx {}'.format(idx))
                rect = self.rects[idx - 1]
                rect_next = self.rects[idx]
                h = distance(rect[3], rect[0])
                y = rect[0][1]
                x = rect[0][0]
                dy = h * self.line_dispersion
                matrix = linearLineMatrix(rect[0], apply_new_point(rect[0], self.dx, self.dy))
                y_next = getYByMatrix(matrix, rect_next[0][0])
                if y_next - dy < rect_next[0][1] < y_next + dy:
                    pass
                else:
                    current_line = current_line + 1
                if current_line not in lines.keys():
                    lines[current_line] = []
                lines[current_line].append({'idx': idx, 'h': h, 'y': y, 'x': x, 'matrix': matrix})
        return lines

    def is_multiline(self) -> bool:
        return len(self.probablyLines.keys()) > 1

    def covert_to_1_line(self, region_name: str) -> np.ndarray:
        # Fix region_name
        region_name = region_name.replace('-', '_')
        if self.is_multiline():
            # print('self.probablyLines')
            # print(self.probablyLines)
            cmd = self.load_region_plugin(region_name)
            return self.merge_lines(cmd)
        else:
            return self.imagePart

    @staticmethod
    def load_region_plugin(region_name: str) -> str:
        if region_name in multiline_plugins:
            mod = multiline_plugins[region_name]
        else:
            try:
                module = 'NpMultiline.%s' % region_name
                mod = importlib.import_module(module)
            except ModuleNotFoundError:
                print('Module {} is absent!'.format(region_name))
                mod = default
            multiline_plugins[region_name] = mod
        return mod

    def merge_lines(self, cmd: Any) -> np.ndarray:
        # Detect zones
        img_zones = get_cv_zonesRGBLite(self.imagePart, self.rects)
        img_zones = cmd.prepare_multiline_rects(self.rects, img_zones, self.probablyLines)

        target_zone_value = max([img_zone.shape[0] for img_zone in img_zones])

        print('target_zone_value {}'.format(target_zone_value))

        res_zone = []
        bg_fill = np.max([np.max(zone) for zone in img_zones])
        cnt = len(img_zones)
        for idx, zone in enumerate(img_zones):
            if target_zone_value != zone.shape[0]:
                zone = target_resize(zone, target_zone_value)
            res_zone.append(zone)
            if idx < cnt-1:
                res_zone.append(np.ones((target_zone_value, int(target_zone_value / 4), 3), dtype="uint8") * bg_fill)

        # Show result
        temp = np.hstack(res_zone)
        temp = normalize(temp)

        res_zone = [
            np.ones((target_zone_value, int(target_zone_value / 1.5)), dtype="uint8") * self.bg_fill,
            temp,
            np.ones((target_zone_value, int(target_zone_value / 6)), dtype="uint8") * self.bg_fill
        ]
        temp = np.hstack(res_zone)

        res_zone = [
            np.ones((int(target_zone_value / 8), temp.shape[1]), dtype="uint8") * self.bg_fill,
            temp,
            np.ones((int(target_zone_value / 8), temp.shape[1]), dtype="uint8") * self.bg_fill
        ]
        return np.vstack(res_zone)
