# Import all necessary libraries.
import os
import sys
import math
import pathlib
import collections

# clone and append to path craft
NOMEROFF_NET_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "../")
CRAFT_DIR = os.environ.get("CRAFT_DIR", os.path.join(NOMEROFF_NET_DIR, 'CRAFT-pytorch'))
CRAFT_URL = "https://github.com/clovaai/CRAFT-pytorch.git"

if not os.path.exists(CRAFT_DIR):
    from git import Repo

    Repo.clone_from(CRAFT_URL, CRAFT_DIR)
sys.path.append(CRAFT_DIR)

import time
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
import craft_utils
import imgproc
from scipy.spatial import ConvexHull

# load CRAFT packages
from craft import CRAFT
from refinenet import RefineNet

# load NomerooffNet packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Base')))

from typing import List, Dict, Tuple, Any, Union
from mcm.mcm import download_latest_model
from mcm.mcm import get_mode_torch
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


def copyStateDict(state_dict: Dict) -> OrderedDict:
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


def test_net(net: CRAFT, image: np.ndarray, text_threshold: float,
             link_threshold: float, low_text: float, cuda: bool,
             poly: bool, canvas_size: int, refine_net: RefineNet = None,
             mag_ratio: float = 1.5) -> Tuple[Any, Any, Any]:
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
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


def split_boxes(bboxes: List[Union[np.ndarray, np.ndarray]], dimensions: List[Dict],
                similarity_range: int = 0.7) -> Tuple[List[int], List[int]]:
    """
    TODO: describe function
    """
    np_bboxes_idx = []
    garbage_bboxes_idx = []
    maxDy = 0
    if len(bboxes):
        maxDy = max([dimension['dy'] for dimension in dimensions])
    for i, (bbox, dimension) in enumerate(zip(bboxes, dimensions)):
        if maxDy * similarity_range <= dimension['dy']:
            np_bboxes_idx.append(i)
        else:
            garbage_bboxes_idx.append(i)
    return np_bboxes_idx, garbage_bboxes_idx


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


def detectIntersection(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    www.math.by/geometry/eqline.html
    xn--80ahcjeib4ac4d.xn--p1ai/information/solving_systems_of_linear_equations_in_python/
    """
    x = np.array([matrix1[:2], matrix2[:2]])
    y = np.array([matrix1[2], matrix2[2]])
    return np.linalg.solve(x, y)


def detectIntersectionNormDD(matrix1: np.ndarray, matrix2: np.ndarray, d1: float, d2: float) -> np.ndarray:
    """
    TODO: describe function
    """
    X = np.array([matrix1[:2], matrix2[:2]])
    c0 = matrix1[2] - d1 * (matrix1[0] ** 2 + matrix1[1] ** 2) ** 0.5
    c1 = matrix2[2] - d2 * (matrix2[0] ** 2 + matrix2[1] ** 2) ** 0.5
    y = np.array([c0, c1])
    return np.linalg.solve(X, y)




def detectDistanceFromPointToLine(matrix: List[np.ndarray],
                                  point: Union) -> float:
    """
    Определение растояния от точки к линии
    https://ru.onlinemschool.com/math/library/analytic_geometry/p_line1/
    """
    A = matrix[0]
    B = matrix[1]
    C = matrix[2]
    x = point[0]
    y = point[1]
    return abs(A * x + B * y - C) / math.sqrt(A ** 2 + B ** 2)


def findMinXIdx(targetPoints: Union) -> int:
    """
    TODO: describe function
    """
    minXIdx = 3
    for i in range(0, len(targetPoints)):
        if targetPoints[i][0] < targetPoints[minXIdx][0]:
            minXIdx = i
        if targetPoints[i][0] == targetPoints[minXIdx][0] and targetPoints[i][1] < targetPoints[minXIdx][1]:
            minXIdx = i
    return minXIdx


def fixClockwise(targetPoints: List) -> List:
    """
    TODO: describe function
    """
    stat1 = fline(targetPoints[0], targetPoints[1])
    stat2 = fline(targetPoints[0], targetPoints[2])
    if targetPoints[0][0] == targetPoints[1][0] and (targetPoints[0][1] > targetPoints[1][1]):
        stat1[2] = -stat1[2]

    if stat2[2] < stat1[2]:
        targetPoints = np.array([targetPoints[0], targetPoints[3], targetPoints[2], targetPoints[1]])
    return targetPoints


def order_points_old(pts: np.ndarray) -> Union:
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
    diff = np.diff(pts_crop, axis=1)
    rect[1] = pts_crop[np.argmin(diff)]
    rect[3] = pts_crop[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def fixClockwise2(target_points: np.ndarray) -> np.ndarray:
    return order_points_old(np.array(target_points))


def addoptRectToBbox(targetPoints: List, Bbox: Tuple, distansesoffsetLeftMaxPercentage: float,
                     offsetTopMaxPercentage: float, offsetRightMaxPercentage: float,
                     offsetBottomMaxPercentage: float) -> np.ndarray:
    """
    TODO: describe function
    """
    distanses = findDistances(targetPoints)
    points = []

    distansesoffsetLeftPercentage = distansesoffsetLeftMaxPercentage
    offsetTopPercentage = offsetTopMaxPercentage
    offsetRightPercentage = offsetRightMaxPercentage
    offsetBottomPercentage = offsetBottomMaxPercentage

    k = Bbox[1] / Bbox[0]

    if k < 2:
        offsetTopPercentage = offsetTopPercentage / 2
        offsetBottomPercentage = offsetBottomPercentage / 2

    if k < 1:
        offsetTopPercentage = 0
        offsetBottomPercentage = 0

    offsets = [distansesoffsetLeftPercentage, offsetTopPercentage, offsetRightPercentage, offsetBottomPercentage]
    cnt = len(distanses)
    for i in range(cnt):
        iNext = i + 1
        if iNext == cnt:
            iNext = 0
        offsets[i] = distanses[iNext]['d'] * offsets[i] / 100
    for i in range(cnt):
        iPrev = i
        iNext = i + 1
        if iNext == cnt:
            iNext = 0
        offset1 = offsets[iPrev]
        offset2 = offsets[iNext]
        points.append(
            detectIntersectionNormDD(distanses[iPrev]['matrix'], distanses[iNext]['matrix'], offset1, offset2))

    # Step 2
    points = reshapePoints(points, 3)

    distanses = findDistances(points)

    if distanses[3]['coef'][2] == 90:
        return np.array(points)

    h = Bbox[0]
    w = Bbox[1]

    matrixLeft = linearLineMatrix([0, 0], [0, h])
    matrixRight = linearLineMatrix([w, 0], [w, h])

    pLeftTop = detectIntersection(matrixLeft, distanses[1]['matrix'])
    pLeftBottom = detectIntersection(matrixLeft, distanses[3]['matrix'])
    pRightTop = detectIntersection(matrixRight, distanses[1]['matrix'])
    pRightBottom = detectIntersection(matrixRight, distanses[3]['matrix'])

    offsetLeftBottom = distance(points[0], pLeftBottom)
    offsetLeftTop = distance(points[1], pLeftTop)
    offsetRightTop = distance(points[2], pRightTop)
    offsetRightBottom = distance(points[3], pRightBottom)

    overLeftTop = points[1][0] < 0
    overLeftBottom = points[0][0] < 0
    if not overLeftTop and not overLeftBottom:
        if offsetLeftTop > offsetLeftBottom:
            points[0] = pLeftBottom
            leftDistance = detectDistanceFromPointToLine(distanses[0]['matrix'], pLeftBottom)
            points[1] = detectIntersectionNormDD(distanses[0]['matrix'], distanses[1]['matrix'], leftDistance, 0)
        else:
            points[1] = pLeftTop
            leftDistance = detectDistanceFromPointToLine(distanses[0]['matrix'], pLeftTop)
            points[0] = detectIntersectionNormDD(distanses[3]['matrix'], distanses[0]['matrix'], 0, leftDistance)

    overRightTop = points[2][0] > w
    overRightBottom = points[3][0] > w
    if not overRightTop and not overRightBottom:
        if offsetRightTop > offsetRightBottom:
            points[3] = pRightBottom
            rightDistance = detectDistanceFromPointToLine(distanses[2]['matrix'], pRightBottom)
            points[2] = detectIntersectionNormDD(distanses[1]['matrix'], distanses[2]['matrix'], 0, rightDistance)
        else:
            points[2] = pRightTop
            rightDistance = detectDistanceFromPointToLine(distanses[2]['matrix'], pRightTop)
            points[3] = detectIntersectionNormDD(distanses[2]['matrix'], distanses[3]['matrix'], rightDistance, 0)

    return np.array(points)


def addCoordinatesOffset(points: List, x: float, y: float) -> List:
    """
    TODO: describe function
    """
    return [[point[0] + x, point[1] + y] for point in points]


def normalizeRect(rect: List) -> List:
    """
    TODO: describe function
    """
    rect = fixClockwise2(rect)
    minXIdx = findMinXIdx(rect)
    rect = reshapePoints(rect, minXIdx)
    coef_ccw = fline(rect[0], rect[3])
    angle_ccw = round(coef_ccw[2], 2)
    if angle_ccw < 0 or angle_ccw > 45:
        rect = reshapePoints(rect, 3)
    return rect


def prepareImageText(img: np.ndarray) -> np.ndarray:
    """
    сперва переведём изображение из RGB в чёрно серый
    значения пикселей будут от 0 до 255
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_min = np.amin(gray_image)
    gray_image -= img_min
    img_max = np.amax(img)
    k = 255 / img_max
    gray_image = gray_image.astype(np.float64)
    gray_image *= k
    gray_image = gray_image.astype(np.uint8)

    (thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage


def detectBestPerspective(bwImages: List[np.ndarray]) -> int:
    """
    TODO: describe function
    """
    res = []
    idx = 0
    diff = 1000000
    diffCnt = 0
    for i, img in enumerate(bwImages):
        s = np.sum(img, axis=0)
        imgStat = collections.Counter(s)
        imgStatDict = OrderedDict(imgStat.most_common())
        maxStat = max(imgStatDict, key=int)
        maxStatCount = imgStatDict[maxStat]
        minStat = min(imgStatDict, key=int)
        minStatCount = imgStatDict[minStat]
        res.append({'max': maxStat, 'min': minStat, 'maxCnt': maxStatCount, 'minCnt': minStatCount})

        if minStat < diff:
            idx = i
            diff = minStat
        if minStat == diff and maxStatCount + minStatCount > diffCnt:
            idx = i
            diffCnt = maxStatCount + minStatCount
    return idx


def addPointOffset(point: List, x: float, y: float) -> List:
    """
    TODO: describe function
    """
    return [point[0] + x, point[1] + y]


def addPointOffsets(points: List, dx: float, dy: float) -> List:
    """
    TODO: describe function
    """
    return [
        addPointOffset(points[0], -dx, -dy),
        addPointOffset(points[1], dx, dy),
        addPointOffset(points[2], dx, dy),
        addPointOffset(points[3], -dx, -dy),
    ]


def makeRectVariants(propably_points: List, quality_profile: List = None) -> List:
    """
    TODO: describe function
    """
    if quality_profile is None:
        quality_profile = [3, 1, 0]
    points_arr = []

    distanses = findDistances(propably_points)

    if distanses[0]['coef'][2] == 90:
        points_arr.append(propably_points)
        return points_arr

    point_centre_left = [propably_points[0][0] + (propably_points[1][0] - propably_points[0][0]) / 2,
                         propably_points[0][1] + (propably_points[1][1] - propably_points[0][1]) / 2]

    point_bottom_left = [point_centre_left[0], getYByMatrix(distanses[3]["matrix"], point_centre_left[0])]

    dx = propably_points[0][0] - point_bottom_left[0]
    dy = propably_points[0][1] - point_bottom_left[1]

    steps = quality_profile[0]
    steps_plus = quality_profile[1]
    steps_minus = quality_profile[2]

    dx_step = dx / steps
    dy_step = dy / steps

    points_arr = []
    for i in range(-steps_minus, steps + steps_plus + 1):
        points_arr.append(addPointOffsets(propably_points, i * dx_step, i * dy_step))
    return points_arr


def normalizePerspectiveImages(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    TODO: describe function
    """
    new_images = []
    for img in images:
        new_images.append(prepareImageText(img))
    return new_images


class NpPointsCraft(object):
    """
    NpPointsCraft Class
    git clone https://github.com/clovaai/CRAFT-pytorch.git
    """

    def __init__(self,
                 low_text=0.4,
                 link_threshold=0.7,  # 0.4
                 text_threshold=0.6,
                 canvas_size=1280,
                 mag_ratio=1.5
                 ):
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.text_threshold = text_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio

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
            model_info = download_latest_model(self.get_classname(), "mtl", ext="pth", mode=get_mode_torch())
            mtl_model_path = model_info["path"]
        if refiner_model_path == "latest":
            model_info = download_latest_model(self.get_classname(), "refiner", ext="pth", mode=get_mode_torch())
            refiner_model_path = model_info["path"]
        device = "cpu"
        if get_mode_torch() == "gpu":
            device = "cuda"
        self.loadModel(device, True, mtl_model_path, refiner_model_path)

    def loadModel(self,
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
            self.net.load_state_dict(copyStateDict(model))
        else:
            model = copyStateDict(torch.load(trained_model, map_location='cpu'))
            self.net.load_state_dict(model)

        if is_cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if is_refine:
            self.refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
            if is_cuda:
                self.refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
                self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))

            self.refine_net.eval()
            self.is_poly = True

    def detectByImagePath(self,
                          image_path: str,
                          target_boxes: List[Dict],
                          qualityProfile: List = None) -> Tuple[List[Dict], Any]:
        """
        TODO: describe method
        """
        if qualityProfile is None:
            qualityProfile = [1, 0, 0]
        image = imgproc.loadImage(image_path)
        for targetBox in target_boxes:
            x = min(targetBox['x1'], targetBox['x2'])
            w = abs(targetBox['x2'] - targetBox['x1'])
            y = min(targetBox['y1'], targetBox['y2'])
            h = abs(targetBox['y2'] - targetBox['y1'])

            image_part = image[y:y + h, x:x + w]
            points = self.detectInBbox(image_part)
            propablyPoints = addCoordinatesOffset(points, x, y)
            targetBox['points'] = []
            targetBox['imgParts'] = []
            if len(propablyPoints):
                targetPointsVariants = makeRectVariants(propablyPoints, qualityProfile)
                if len(targetPointsVariants) > 1:
                    imgParts = [getCvZoneRGB(image, reshapePoints(rect, 1)) for rect in targetPointsVariants]
                    normalized_perspective_img = normalizePerspectiveImages(imgParts)
                    idx = detectBestPerspective(normalized_perspective_img)
                    targetBox['points'] = targetPointsVariants[idx]
                    targetBox['imgParts'] = imgParts
                else:
                    targetBox['points'] = targetPointsVariants[0]
        return target_boxes, image

    def detect(self, image: np.ndarray, targetBoxes: List, qualityProfile: List = None) -> List:
        """
        TODO: describe method
        """
        if qualityProfile is None:
            qualityProfile = [1, 0, 0]
        all_points = []
        for targetBox in targetBoxes:
            x = int(min(targetBox[0], targetBox[2]))
            w = int(abs(targetBox[2] - targetBox[0]))
            y = int(min(targetBox[1], targetBox[3]))
            h = int(abs(targetBox[3] - targetBox[1]))

            image_part = image[y:y + h, x:x + w]
            propablyPoints = addCoordinatesOffset(self.detectInBbox(image_part), x, y)
            if len(propablyPoints):
                targetPointsVariants = makeRectVariants(propablyPoints, qualityProfile)
                if len(targetPointsVariants) > 1:
                    imgParts = [getCvZoneRGB(image, reshapePoints(rect, 1)) for rect in targetPointsVariants]
                    idx = detectBestPerspective(normalizePerspectiveImages(imgParts))
                    points = targetPointsVariants[idx]
                else:
                    points = targetPointsVariants[0]
                all_points.append(points)
            else:
                all_points.append([
                    [x, y + h],
                    [x, y],
                    [x + w, y],
                    [x + w, y + h]
                ])
        return all_points

    def detectInBbox(self, image: np.ndarray, craft_params={}, debug: bool = False):
        """
        TODO: describe method
        """
        low_text = craft_params.get('low_text', self.low_text)
        link_threshold = craft_params.get('link_threshold', self.link_threshold)
        text_threshold = craft_params.get('text_threshold', self.text_threshold)
        canvas_size = craft_params.get('canvas_size', self.canvas_size)
        mag_ratio = craft_params.get('mag_ratio', self.mag_ratio)

        t = time.time()
        bboxes, polys, score_text = test_net(self.net, image, text_threshold, link_threshold, low_text,
                                             self.is_cuda, self.is_poly, canvas_size, self.refine_net, mag_ratio)
        if debug:
            print("elapsed time : {}s".format(time.time() - t))
        dimensions = []
        for poly in bboxes:
            dimensions.append({'dx': distance(poly[0], poly[1]), 'dy': distance(poly[1], poly[2])})

        if debug:
            print(score_text.shape)
            # print(polys)
            print(dimensions)
            print(bboxes)

        np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions)

        targetPoints = []
        if debug:
            print('np_bboxes_idx')
            print(np_bboxes_idx)
            print('garbage_bboxes_idx')
            print(garbage_bboxes_idx)

        if len(np_bboxes_idx) == 1:
            targetPoints = bboxes[np_bboxes_idx[0]]

        if len(np_bboxes_idx) > 1:
            targetPoints = minimum_bounding_rectangle(np.concatenate([bboxes[i] for i in np_bboxes_idx], axis=0))

        if len(np_bboxes_idx) > 0:
            targetPoints = normalizeRect(targetPoints)
            if debug:
                print("[INFO] targetPoints", targetPoints)
                print('[INFO] image.shape', image.shape)
            targetPoints = addoptRectToBbox(targetPoints, image.shape, 7, 12, 0, 12)
        return targetPoints

    def detectProbablyMultilineZones(self, image, craft_params=None, debug=False):
        """
        TODO: describe method
        """
        if craft_params is None:
            craft_params = {}
        low_text = craft_params.get('low_text', self.low_text)
        link_threshold = craft_params.get('link_threshold', self.link_threshold)
        text_threshold = craft_params.get('text_threshold', self.text_threshold)
        canvas_size = craft_params.get('canvas_size', self.canvas_size)
        mag_ratio = craft_params.get('mag_ratio', self.mag_ratio)

        t = time.time()
        bboxes, polys, score_text = test_net(self.net, image, text_threshold, link_threshold, low_text,
                                             self.is_cuda, self.is_poly, canvas_size, self.refine_net, mag_ratio)
        if debug:
            print("elapsed time : {}s".format(time.time() - t))

        dimensions = []
        for poly in bboxes:
            dimensions.append({'dx': distance(poly[0], poly[1]), 'dy': distance(poly[1], poly[2])})

        np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions)

        return [bboxes[i] for i in np_bboxes_idx]