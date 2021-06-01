import numpy as np
import math
from NomeroffNet.BBoxNpPoints import NpPointsCraft, minimum_bounding_rectangle, detectIntersection, \
    fixClockwise2, findMinXIdx
from tools import (fline,
                   distance,
                   linearLineMatrix,
                   getYByMatrix,
                   findDistances,
                   rotate,
                   buildPerspective,
                   getCvZoneRGB,
                   getMeanDistance,
                   reshapePoints,
                   getCvZonesRGB,
                   convertCvZonesRGBtoBGR,
                   getCvZonesBGR)
from BBoxNpMultiline import MultilineConverter


def normalizeMultilineRect(rect, mline_boxes):
    """
    TODO: describe function
    """
    rect = fixClockwise2(rect)
    minXIdx = findMinXIdx(rect)
    rect = reshapePoints(rect, minXIdx)

    w_max, w_max_idx, options = make_mline_boxes_options(mline_boxes)
    target_angle = options[w_max_idx]['angle']
    coef_ccw = fline(rect[0], rect[3])
    angle_ccw = round(coef_ccw[2], 2)

    if abs(target_angle-angle_ccw) > 45:
        rect = reshapePoints(rect, 3)
    return rect


def fix_box_angle(target_angle, mline_box, option):
    #print('target_angle {} mline_box: {} option {}'.format(target_angle, mline_box, option))
    print('target_angle {} option {}'.format(target_angle, option))
    line_matrix_left = linearLineMatrix(mline_box[0], mline_box[2])
    line_matrix_right = linearLineMatrix(mline_box[1], mline_box[3])
    origin = detectIntersection(line_matrix_left, line_matrix_right)
    angle = target_angle - option['angle']
    return [rotate(origin, point, angle) for point in mline_box]


def make_mline_boxes_options(mline_boxes):
    options = []
    w_max = 0
    w_max_idx = -1
    for mline_idx, mline_box in enumerate(mline_boxes):
        w = distance(mline_box[0], mline_box[1])
        if w_max < w:
            w_max = w
            w_max_idx = mline_idx
        angle = (fline(mline_box[0], mline_box[1]))[2]
        options.append({
            'w': w,
            'angle': angle
        })
    return w_max, w_max_idx, options


def fix_mline_boxes_angle(mline_boxes):
    max_delta_angle = 7
    w_max, w_max_idx, options = make_mline_boxes_options(mline_boxes)
    target_angle = options[w_max_idx]['angle']
    for mline_idx, mline_box in enumerate(mline_boxes):
        if abs(target_angle - options[mline_idx]['angle']) > max_delta_angle:
            mline_boxes[mline_idx] = fix_box_angle(target_angle, mline_box, options[mline_idx])
    return mline_boxes, target_angle


def check_line_side_faces(targetPoints, shape):
    """
    Неработающая заглушка
    которая должна фиксить угол наклона боковых граней.
    """
    h = shape[0]
    w = shape[1]
    out_of_bounds_points = []
    out_of_bounds_points_idx = {}
    for idx, point in enumerate(targetPoints):
        if point[0] < 0 or point[0] > w:
            out_of_bounds_points.append(idx)
            out_of_bounds_points_idx[idx] = True
        else:
            out_of_bounds_points_idx[idx] = False
    return out_of_bounds_points, out_of_bounds_points_idx


def get_center_point(p0, p1):
    """
    Ищем середину отрезка, заданного 2 точками
    :param p0:
    :param p1:
    :return:
    """
    return [(p0[0]+p1[0])/2, (p0[1]+p1[1])/2]


def calc_diff(p1, p2):
    return p1[0]-p2[0], p1[1]-p2[1]


def apply_new_point(point, dx, dy):
    return [point[0]+dx, point[1]+dy]


def apply_new_box_angle(box, dx, dy):
    left_reference_point = get_center_point(box[3], box[0])
    right_reference_point = get_center_point(box[1], box[2])
    top_matrix = linearLineMatrix(box[0], box[1])
    bottom_matrix = linearLineMatrix(box[3], box[2])

    left_matrix = linearLineMatrix(left_reference_point, apply_new_point(left_reference_point, dx, dy))
    right_matrix = linearLineMatrix(right_reference_point, apply_new_point(right_reference_point, dx, dy))

    return [
        detectIntersection(left_matrix, top_matrix),
        detectIntersection(top_matrix, right_matrix),
        detectIntersection(right_matrix, bottom_matrix),
        detectIntersection(bottom_matrix, left_matrix)
    ]


def build_new_points(left_reference_point, right_reference_point, dx, dy):
    return [
        apply_new_point(left_reference_point, dx, dy),
        apply_new_point(left_reference_point, -dx, -dy),
        apply_new_point(right_reference_point, -dx, -dy),
        apply_new_point(right_reference_point, dx, dy)
    ]

def fit_to_frame(targetPoints, mline_boxes,  shape):
    """
    Неработающая заглушка
    которая должна вписывать область с текстом в заданную рамку
    """
    h = shape[0]
    w = shape[1]
    out_of_bounds_points, out_of_bounds_points_idx = check_line_side_faces(targetPoints, shape)
    if len(out_of_bounds_points):
        left_reference_point = get_center_point(targetPoints[0], targetPoints[1])
        right_reference_point = get_center_point(targetPoints[2], targetPoints[3])
        reference_point_angle = (fline(left_reference_point, right_reference_point))[2]
        center_matrix = linearLineMatrix(left_reference_point, right_reference_point)
        top_matrix = linearLineMatrix(targetPoints[1], targetPoints[2])
        bottom_matrix = linearLineMatrix(targetPoints[0], targetPoints[3])
        if left_reference_point[0] < 0:
            left_reference_point = [0, getYByMatrix(center_matrix, 0)]
            #topLeftPoint = detectIntersection(line_matrix_left, top_matrix)
        if right_reference_point[0] > w:
            right_reference_point = [w, getYByMatrix(center_matrix, w)]
        if reference_point_angle > 0:
            p2_new = detectIntersection(top_matrix, linearLineMatrix([w, 0],[w, h]))
            p0_new = detectIntersection(bottom_matrix, linearLineMatrix([0, 0], [0, h]))
            p2_dx = w - right_reference_point[0]
            p0_dx = left_reference_point[0]
            if p2_dx > p0_dx:
                dx, dy = calc_diff(left_reference_point, p0_new)
            else:
                dx, dy = calc_diff(right_reference_point, p2_new)
                dx = -dx
                dy = -dy

            print('dx {}, dy {}'.format(dx, dy))
        else:
            p3_new = detectIntersection(bottom_matrix, linearLineMatrix([w, 0],[w, h]))
            p1_new = detectIntersection(top_matrix, linearLineMatrix([0, 0], [0, h]))
            p3_dx = w - right_reference_point[0]
            p1_dx = left_reference_point[0]
            if p3_dx > p1_dx:
                dx, dy = calc_diff(left_reference_point, p1_new)
            else:
                dx, dy = calc_diff(right_reference_point, p3_new)
                dx = -dx
                dy = -dy
        targetPoints = build_new_points(left_reference_point, right_reference_point, dx, dy)
        mline_boxes = [apply_new_box_angle(mline_box, dx, dy) for mline_box in mline_boxes]
    return targetPoints, mline_boxes


class CCraft(object):
    
    def __init__(self, **args):
        self.npPointsCraft = NpPointsCraft(
            low_text=0.38,
            link_threshold=0.7,  # 0.4
            text_threshold=0.6,
            canvas_size=1280,
            mag_ratio=1.5
        )
        self.npPointsCraft.load()


    def make1LineFromMany(self, image_part, region_name, debug = False):
        mline_boxes = self.npPointsCraft.detectProbablyMultilineZones(image_part)

        if len(mline_boxes) > 1:
            mline_boxes, target_angle = fix_mline_boxes_angle(mline_boxes)
            targetPoints = minimum_bounding_rectangle(np.concatenate(mline_boxes, axis=0))
            targetPoints = normalizeMultilineRect(targetPoints, mline_boxes)
            targetPoints, mline_boxes = fit_to_frame(targetPoints, mline_boxes, image_part.shape)
            multilineConverter = MultilineConverter(image_part, mline_boxes, targetPoints)
            one_line_img = multilineConverter.covert_to_1_line(region_name)
            # print('multilineConverter.probablyLines')
            # print(multilineConverter.probablyLines)
        else:
            one_line_img = image_part
            targetPoints = mline_boxes[0]

        #return one_line_img, targetPoints, mline_boxes
        return one_line_img