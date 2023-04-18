import math
import numpy as np
import cv2
from typing import List, Union
from scipy.spatial import ConvexHull


def fline(p0: List, p1: List, debug: bool = False) -> List:
    """
    Вычесление угла наклона прямой по 2 точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    if debug:
        print("Уравнение прямой, проходящей через эти точки:")
    if x1 - x2 == 0:
        k = math.inf
        b = y2
    else:
        k = (y1 - y2) / (x1 - x2)
        b = y2 - k*x2
    if debug:
        print(" y = %.4f*x + %.4f" % (k, b))
    r = math.atan(k)
    a = math.degrees(r)
    a180 = a
    if a < 0:
        a180 = 180 + a
    return [k, b, a, a180, r]


def distance(p0: List or np.ndarray, p1: List or np.ndarray) -> float:
    """
    distance between two points p0 and p1
    """
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def linear_line_matrix(p0: List, p1: List, verbode: bool = False) -> np.ndarray:
    """
    Вычесление коефициентов матрицы, описывающей линию по двум точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    matrix_a = y1 - y2
    matrix_b = x2 - x1
    matrix_c = x2*y1-x1*y2
    if verbode:
        print("Уравнение прямой, проходящей через эти точки:")
        print("%.4f*x + %.4fy = %.4f" % (matrix_a, matrix_b, matrix_c))
        print(matrix_a, matrix_b, matrix_c)
    return np.array([matrix_a, matrix_b, matrix_c])


def get_y_by_matrix(matrix: np.ndarray, x: float) -> np.ndarray:
    """
    TODO: describe function
    """
    matrix_a = matrix[0]
    matrix_b = matrix[1]
    matrix_c = matrix[2]
    if matrix_b != 0:
        return (matrix_c - matrix_a * x) / matrix_b


def find_distances(points: np.ndarray or List) -> List:
    """
    TODO: describe function
    """
    distanses = []
    cnt = len(points)

    for i in range(cnt):
        p0 = i
        if i < cnt - 1:
            p1 = i + 1
        else:
            p1 = 0
        distanses.append({"d": distance(points[p0], points[p1]), "p0": p0, "p1": p1,
                          "matrix": linear_line_matrix(points[p0], points[p1]),
                          "coef": fline(points[p0], points[p1])})
    return distanses


def rotate(origin, point, angle_degrees):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in degrees.
    """
    angle = math.radians(angle_degrees)

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def build_perspective(img: np.ndarray, rect: list, w: int, h: int) -> List:
    """
    image perspective transformation
    """
    img_h, img_w, img_c = img.shape
    if img_h < h:
        h = img_h
    if img_w < w:
        w = img_w
    pts1 = np.float32(rect)
    pts2 = np.float32(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
    moment = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, moment, (w, h))


def get_cv_zone_rgb(img: np.ndarray, rect: list, gw: float = 0, gh: float = 0,
                    coef: float = 4.6, auto_width_height: bool = True) -> List:
    """
    TODO: describe function
    """
    if gw == 0 or gh == 0:
        distanses = find_distances(rect)
        h = (distanses[0]['d'] + distanses[2]['d']) / 2
        if auto_width_height:
            w = int(h*coef)
        else:
            w = (distanses[1]['d'] + distanses[3]['d']) / 2
    else:
        w, h = gw, gh
    return build_perspective(img, rect, int(w), int(h))


def get_mean_distance(rect: List, start_idx: int, verbose: bool = False) -> np.ndarray:
    """
    TODO: describe function
    """
    end_idx = start_idx+1
    start2_idx = start_idx+2
    end2_idx = end_idx+2
    if end2_idx == 4:
        end2_idx = 0
    if verbose:
        print('startIdx: {}, endIdx: {}, start2Idx: {}, end2Idx: {}'.format(start_idx, end_idx, start2_idx, end2_idx))
    return np.mean([distance(rect[start_idx], rect[end_idx]), distance(rect[start2_idx], rect[end2_idx])])


def reshape_points(target_points: List or np.ndarray, start_idx: int) -> List:
    """
    TODO: describe function
    """
    if start_idx > 0:
        part1 = target_points[:start_idx]
        part2 = target_points[start_idx:]
        target_points = np.concatenate((part2, part1))
    return target_points


def get_cv_zones_rgb(img: np.ndarray, rects: list, gw: float = 0, gh: float = 0,
                     coef: float = 4.6, auto_width_height: bool = True) -> List:
    """
    TODO: describe function
    """
    dsts = []
    for rect in rects:
        h = get_mean_distance(rect, 0)
        w = get_mean_distance(rect, 1)
        if h > w and auto_width_height:
            h, w = w, h
        else:
            rect = reshape_points(rect, 3)
        if gw == 0 or gh == 0:
            w, h = int(h*coef), int(h)
        else:
            w, h = gw, gh
        dst = build_perspective(img, rect, int(w), int(h))
        dsts.append(dst)
    return dsts


def convert_cv_zones_rgb_to_bgr(dsts: List) -> List:
    """
    TODO: describe function
    """
    bgr_dsts = []
    for dst in dsts:
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        bgr_dsts.append(dst)
    return bgr_dsts


def get_cv_zones_bgr(img: np.ndarray, rects: list, gw: float = 0, gh: float = 0,
                     coef: float = 4.6, auto_width_height: bool = True) -> List:
    """
    TODO: describe function
    """
    dsts = get_cv_zones_rgb(img, rects, gw, gh, coef, auto_width_height=auto_width_height)
    return convert_cv_zones_rgb_to_bgr(dsts)


def normalize(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return normalize_color(img)


def normalize_color(img: np.ndarray) -> np.ndarray:
    img = cv2.normalize(img, None, alpha=0, beta=255,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img


def order_points_old(pts: np.ndarray):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    lp = np.argmin(s)

    # fix original code by Oleg Cherniy
    rp = lp + 2
    if rp > 3:
        rp = rp - 4
    rect[0] = pts[lp]
    rect[2] = pts[rp]
    pts_crop = [pts[idx] for idx in filter(lambda i: (i != lp) and (i != rp), range(len(pts)))]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    # Определяется так.
    # Предположим, у нас есть 3 точки: А(х1,у1), Б(х2,у2), С(х3,у3).
    # Через точки А и Б проведена прямая. И нам надо определить, как расположена точка С относительно прямой АБ.
    # Для этого вычисляем значение:
    # D = (х3 - х1) * (у2 - у1) - (у3 - у1) * (х2 - х1)
    # - Если D = 0 - значит, точка С лежит на прямой АБ.
    # - Если D < 0 - значит, точка С лежит слева от прямой.
    # - Если D > 0 - значит, точка С лежит справа от прямой.
    x1 = rect[0][0]
    y1 = rect[0][1]
    x2 = rect[2][0]
    y2 = rect[2][1]
    x3 = pts_crop[0][0]
    y3 = pts_crop[0][1]
    d = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)

    if d > 0:
        rect[1] = pts_crop[0]
        rect[3] = pts_crop[1]
    else:
        rect[1] = pts_crop[1]
        rect[3] = pts_crop[0]

    # return the ordered coordinates
    return rect


def fix_clockwise2(target_points: np.ndarray or List) -> np.ndarray:
    return order_points_old(np.array(target_points))


def minimum_bounding_rectangle(points: np.ndarray) -> np.ndarray:
    """
    Find the smallest bounding rectangle for a set of points.
    detail: https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def detect_intersection(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    www.math.by/geometry/eqline.html
    xn--80ahcjeib4ac4d.xn--p1ai/information/solving_systems_of_linear_equations_in_python/
    """
    x = np.array([matrix1[:2], matrix2[:2]])
    y = np.array([matrix1[2], matrix2[2]])
    return np.linalg.solve(x, y)


def find_min_x_idx(target_points: Union) -> int:
    """
    TODO: describe function
    """
    min_x_idx = 3
    for i in range(0, len(target_points)):
        if target_points[i][0] < target_points[min_x_idx][0]:
            min_x_idx = i
        if target_points[i][0] == target_points[min_x_idx][0] and target_points[i][1] < target_points[min_x_idx][1]:
            min_x_idx = i
    return min_x_idx


def grab_rotation_matrix(cx, cy, h, w, angle):
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    moment = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(moment[0, 0])
    sin = np.abs(moment[0, 1])

    # compute the new bounding dimensions of the image
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    moment[0, 2] += (nw / 2) - cx
    moment[1, 2] += (nh / 2) - cy
    return moment, nw, nh


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.float32
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cx, cy) = (w // 2, h // 2)
    moment, nw, nh = grab_rotation_matrix(cx, cy, h, w, angle)

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, moment, (nw, nh))

    return image


def get_corners(bboxes):
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(corners, angle, cx, cy, h, w):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.float32
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.float32
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    moment, nw, nh = grab_rotation_matrix(cx, cy, h, w, angle)

    # Prepare the vector to be transformed
    calculated = np.dot(moment, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners: np.ndarray):
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def rotate_image_and_bboxes(img, target_boxes, angle=0):
    if angle == 0:
        return img, target_boxes
    w, h = img.shape[1], img.shape[0]
    cx, cy = w // 2, h // 2
    rotated_img = rotate_im(img, angle)
    corners = get_corners(target_boxes)
    corners = np.hstack((corners, target_boxes[:, 4:]))
    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
    new_bbox = get_enclosing_box(corners)
    return rotated_img, new_bbox


def generate_image_rotation_variants(img, target_boxes, angles=None):
    if angles is None:
        angles = [90, 180, 270]
    target_boxes = np.array(target_boxes)[:, :4]
    variants_bboxes = [target_boxes]
    variant_images = [img]
    for angle in angles:
        rotated_img, new_bbox = rotate_image_and_bboxes(img, target_boxes, angle)

        variants_bboxes.append(new_bbox)
        variant_images.append(rotated_img)
    return variant_images, variants_bboxes


def normalize_img(img: np.ndarray,
                  height: int = 64,
                  width: int = 295,
                  to_gray: bool = False,
                  with_aug: bool = False) -> np.ndarray:
    if with_aug:
        from nomeroff_net.tools.augmentations import aug
        imgs = aug([img])
        img = imgs[0]
    if to_gray and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if not to_gray and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (width, height))
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if to_gray:
        img = np.reshape(img, [*img.shape, 1])
    return img


def crop_image(image, target_box):
    x = int(min(target_box[0], target_box[2]))
    w = int(abs(target_box[2] - target_box[0]))
    y = int(min(target_box[1], target_box[3]))
    h = int(abs(target_box[3] - target_box[1]))

    image_part = image[y:y + h, x:x + w]
    return image_part, (x, w, y, h)


def crop_number_plate_zones_from_images(images, images_points):
    zones = []
    image_ids = []
    for i, (image, image_points) in enumerate(zip(images, images_points)):
        image_zones = [get_cv_zone_rgb(image, reshape_points(rect, 1)) for rect in image_points]
        for zone in image_zones:
            zones.append(zone)
            image_ids.append(i)
    return zones, image_ids


def crop_number_plate_rect_zones_from_images(images, images_bboxs):
    zones = []
    image_ids = []
    for i, (img, bboxs) in enumerate(zip(images, images_bboxs)):
        for target_box in bboxs:
            x, w, y, h = (
                int(min(target_box[0], target_box[2])),
                int(abs(target_box[2] - target_box[0])),
                int(min(target_box[1], target_box[3])),
                int(abs(target_box[3] - target_box[1]))
            )
            image_part = img[y:y + h, x:x + w]
            zones.append(image_part)
            image_ids.append(i)
    return zones, image_ids


def group_by_image_ids(image_ids, props):
    images_props = [[[] for _ in range(max(image_ids or [0])+1)] for _ in props]
    for i, prop in enumerate(props):
        for image_id, val in zip(image_ids, prop):
            images_props[i][image_id].append(val)
    return images_props
