import os
import sys
import cv2
import numpy as np
import importlib

# load NomerooffNet packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from tools import distance
import NpMultiline

multiline_plugins = {}


def meanRectDistance(rect, startIdx):
    idxs = [(idx, idx - 4)[idx > 3] for idx in range(startIdx, len(rect) + startIdx)]
    return np.mean([distance(rect[idxs[0]], rect[idxs[1]]), distance(rect[idxs[2]], rect[idxs[3]])])


def findWidth(rect):
    return meanRectDistance(rect, 0)


def findHieght(rect):
    return meanRectDistance(rect, 1)


def to_pretty_point(points):
    sortedOnX = sorted([point for point in points], key=lambda x: x[0])
    res = [[0, 0], [0, 0], [0, 0], [0, 0]]
    if sortedOnX[0][1] < sortedOnX[1][1]:
        res[0], res[3] = sortedOnX[0], sortedOnX[1]
    else:
        res[0], res[3] = sortedOnX[1], sortedOnX[0]

    if sortedOnX[2][1] < sortedOnX[3][1]:
        res[1], res[2] = sortedOnX[2], sortedOnX[3]
    else:
        res[1], res[2] = sortedOnX[3], sortedOnX[2]
    return res


def rotate_to_pretty(points):
    points = to_pretty_point(points)
    w = findWidth(points)
    h = findHieght(points)
    return points, w, h


def get_cv_zonesRGBLite(img, rects):
    dsts = []
    for rect in rects:
        rect, w, h = rotate_to_pretty(rect)
        w = int(w)
        h = int(h)
        pts1 = np.float32(rect)
        pts2 = np.float32(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (w, h))
        dsts.append(dst)
    return dsts


def normalize(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img -= np.amin(img)
    img = img.astype(np.float32)
    img *= 255 / np.amax(img)
    img = img.astype("uint8")

    return img


def target_resize(zone, target_zone_value):
    height = target_zone_value
    k = target_zone_value / zone.shape[0]
    width = int(zone.shape[1] * k)
    dim = (width, height)
    # resize image
    return cv2.resize(zone, dim, interpolation=cv2.INTER_AREA)


class MultilineConverter:
    def __init__(self, imagePart, rects, line_dispersion=0.3, bg_fill=223):
        self.imagePart = imagePart
        self.rects = rects
        self.line_dispersion = line_dispersion
        self.bg_fill = bg_fill
        self.probablyLines = self.detect_line_count()

    def detect_line_count(self):
        lines = {}
        zones_cnt = len(self.rects)
        current_line = 1
        idx = 0
        lines[current_line] = [
            {'idx': idx, 'h': distance(self.rects[idx][3], self.rects[idx][0]), 'y': self.rects[idx][0][1]}
        ]
        if zones_cnt > 1:
            for idx in range(1, len(self.rects)):
                rect = self.rects[idx - 1]
                rect_next = self.rects[idx]
                h = distance(rect[3], rect[0])
                y = rect[0][1]
                dy = h * self.line_dispersion
                if y - dy < rect_next[0][1] < y + dy:
                    pass
                else:
                    current_line = current_line + 1
                if current_line not in lines.keys():
                    lines[current_line] = []
                lines[current_line].append({'idx': idx, 'h': h, 'y': y})
        return lines

    def is_multiline(self):
        return len(self.probablyLines.keys()) > 1

    def covert_to_1_line(self, region_name):
        if self.is_multiline():
            cmd = self.load_region_plugin(region_name)
            return self.merge_lines(cmd)
        else:
            return self.imagePart

    @staticmethod
    def load_region_plugin(region_name):
        if region_name in multiline_plugins:
            mod = multiline_plugins[region_name]
        else:
            try:
                module = 'NpMultiline.%s' % region_name
                mod = importlib.import_module(module)
            except ModuleNotFoundError:
                print('Module {} is absent!'.format(module))
                mod = NpMultiline.default
            multiline_plugins[region_name] = mod
        return mod

    def merge_lines(self, cmd):
        # Detect zones
        img_zones = get_cv_zonesRGBLite(self.imagePart, self.rects)
        img_zones = cmd.prepare_multiline_rects(self.rects, img_zones, self.probablyLines)

        target_zone_value = max([img_zone.shape[0] for img_zone in img_zones])
        res_zone = []
        for zone in img_zones:
            if target_zone_value != zone.shape[0]:
                zone = target_resize(zone, target_zone_value)
            res_zone.append(zone)
            res_zone.append(np.ones((target_zone_value, int(target_zone_value / 8), 3), dtype="uint8") * self.bg_fill)

        # Show result
        temp = np.hstack(res_zone)
        temp = normalize(temp)

        res_zone = [
            np.ones((target_zone_value, int(target_zone_value*2)), dtype="uint8") * self.bg_fill,
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
