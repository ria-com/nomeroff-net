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

# load NomerooffNet packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Base')))

from mcm.mcm import download_latest_model
from mcm.mcm import get_mode_torch
from BBoxNpMultiline import MultilineConverter
from tools.image_processing import (distance,
                                    convertCvZonesRGBtoBGR,
                                    reshapePoints,
                                    fline,
                                    findDistances,
                                    linearLineMatrix,
                                    getCvZoneRGB,
                                    pi2)


def make_boxes(img, boxes , color=(0, 0, 255), thickness=2):
    for i, box in enumerate(boxes):
        polybox = np.array(box).astype(np.int32).reshape((-1))
        polybox = polybox.reshape(-1, 2)
        cv2.polylines(img, [polybox.reshape((-1, 1, 2))], True, color, thickness)
    return img


def remove_coordinates_offset(points, x, y):
    for point in points:
        point[0] = point[0] - x
        point[1] = point[1] - y
    return points


def copyStateDict(state_dict):
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


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, refine_net=None,
             mag_ratio=1.5):
    """
    TODO: describe function
    """
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size,
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
        if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


def split_boxes(bboxes, dimensions, similarity_range=0.7):
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


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    detail: https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
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


def detectIntersection(matrix1, matrix2):
    """
    http://www.math.by/geometry/eqline.html
    https://xn--80ahcjeib4ac4d.xn--p1ai/information/solving_systems_of_linear_equations_in_python/
    """
    X = np.array([matrix1[:2], matrix2[:2]])
    y = np.array([matrix1[2], matrix2[2]])
    return np.linalg.solve(X, y)


def detectIntersectionNormDD(matrix1, matrix2, d1, d2):
    """
    TODO: describe function
    """
    X = np.array([matrix1[:2], matrix2[:2]])
    c0 = matrix1[2] - d1 * (matrix1[0] ** 2 + matrix1[1] ** 2) ** 0.5
    c1 = matrix2[2] - d2 * (matrix2[0] ** 2 + matrix2[1] ** 2) ** 0.5
    y = np.array([c0, c1])
    return np.linalg.solve(X, y)


def getYByMatrix(matrix, x):
    """
    TODO: describe function
    """
    A = matrix[0]
    B = matrix[1]
    C = matrix[2]
    if B != 0:
        return (C - A * x) / B


def detectDistanceFromPointToLine(matrix, point):
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


def findMinXIdx(targetPoints):
    """
    TODO: describe function
    """
    minXIdx = 3
    for i in range(0, len(targetPoints)):
        if (targetPoints[i][0] < targetPoints[minXIdx][0]):
            minXIdx = i
        if (targetPoints[i][0] == targetPoints[minXIdx][0]) and (targetPoints[i][1] < targetPoints[minXIdx][1]):
            minXIdx = i
    return minXIdx


def fixClockwise(targetPoints):
    """
    TODO: describe function
    """
    stat1 = fline(targetPoints[0], targetPoints[1])
    stat2 = fline(targetPoints[0], targetPoints[2])
    if targetPoints[0][0] == targetPoints[1][0] and (targetPoints[0][1] > targetPoints[1][1]):
        stat1[2] = -stat1[2]

    if (stat2[2] < stat1[2]):
        targetPoints = np.array([targetPoints[0], targetPoints[3], targetPoints[2], targetPoints[1]])
    return targetPoints


def addOffsetManualPercentage(targetPoints, offsetLeftPercentage, offsetTopPercentage, offsetRightPercentage,
                              offsetBottomPercentage):
    """
    TODO: describe function
    """
    distanses = findDistances(targetPoints)
    points = []
    if distanses[0]['d'] > distanses[1]['d']:
        offsets = [offsetTopPercentage, offsetRightPercentage, offsetBottomPercentage, offsetLeftPercentage]
    else:
        offsets = [offsetLeftPercentage, offsetTopPercentage, offsetRightPercentage, offsetBottomPercentage]
    cnt = len(distanses)

    for i in range(cnt):
        iNext = i + 1
        if (iNext == cnt):
            iNext = 0
        offsets[i] = distanses[iNext]['d'] * offsets[i] / 100

    for i in range(cnt):
        iPrev = i
        iNext = i + 1
        if (iNext == cnt):
            iNext = 0
        offset1 = offsets[iPrev]
        offset2 = offsets[iNext]
        points.append(
            detectIntersectionNormDD(distanses[iPrev]['matrix'], distanses[iNext]['matrix'], offset1, offset2))
    return np.array(points)


def addoptRectToBbox(targetPoints, Bbox, distansesoffsetLeftMaxPercentage, offsetTopMaxPercentage,
                     offsetRightMaxPercentage,
                     offsetBottomMaxPercentage):
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

    if (k < 2):
        offsetTopPercentage = offsetTopPercentage / 2
        offsetBottomPercentage = offsetBottomPercentage / 2

    if (k < 1):
        offsetTopPercentage = 0
        offsetBottomPercentage = 0

    offsets = [distansesoffsetLeftPercentage, offsetTopPercentage, offsetRightPercentage, offsetBottomPercentage]
    cnt = len(distanses)
    for i in range(cnt):
        iNext = i + 1
        if (iNext == cnt):
            iNext = 0
        offsets[i] = distanses[iNext]['d'] * offsets[i] / 100
    for i in range(cnt):
        iPrev = i
        iNext = i + 1
        if (iNext == cnt):
            iNext = 0
        offset1 = offsets[iPrev]
        offset2 = offsets[iNext]
        points.append(
            detectIntersectionNormDD(distanses[iPrev]['matrix'], distanses[iNext]['matrix'], offset1, offset2))

    points = reshapePoints(points, 3)

    distanses = findDistances(points)

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
    if not (overLeftTop) and not (overLeftBottom):
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
    if not (overRightTop) and not (overRightBottom):
        if offsetRightTop > offsetRightBottom:
            points[3] = pRightBottom
            rightDistance = detectDistanceFromPointToLine(distanses[2]['matrix'], pRightBottom)
            points[2] = detectIntersectionNormDD(distanses[1]['matrix'], distanses[2]['matrix'], 0, rightDistance)
        else:
            points[2] = pRightTop
            rightDistance = detectDistanceFromPointToLine(distanses[2]['matrix'], pRightTop)
            points[3] = detectIntersectionNormDD(distanses[2]['matrix'], distanses[3]['matrix'], rightDistance, 0)

    return np.array(points)


def fixSideFacets(targetPoints, adoptToFrame=None):
    """
    TODO: describe function
    """
    distanses = findDistances(targetPoints)
    points = targetPoints.copy()

    cnt = len(distanses)
    if distanses[0]['d'] > distanses[1]['d']:
        targetSides = [1, 3]
    else:
        targetSides = [0, 2]

    for targetSideIdx in targetSides:
        iPrev = targetSideIdx - 1
        iNext = targetSideIdx + 1
        if (iNext == cnt):
            iNext = 0
        if (iPrev < 0):
            iPrev = 3

        pointCentre = [targetPoints[targetSideIdx][0] + (targetPoints[iNext][0] - targetPoints[targetSideIdx][0]) / 2,
                       targetPoints[targetSideIdx][1] + (targetPoints[iNext][1] - targetPoints[targetSideIdx][1]) / 2]

        if adoptToFrame != None:
            if pointCentre[0] < 0:
                pointCentre[0] = 0
            if pointCentre[0] >= adoptToFrame[1]:
                pointCentre[0] = adoptToFrame[1] - 1

        pointTo = [pointCentre[0], pointCentre[1] + 1]
        matrix = linearLineMatrix(pointCentre, pointTo)
        points[targetSideIdx] = detectIntersection(distanses[iPrev]["matrix"], matrix)
        points[iNext] = detectIntersection(matrix, distanses[iNext]["matrix"])
    return np.array(points)


def addCoordinatesOffset(points, x, y):
    """
    TODO: describe function
    """
    return [[point[0] + x, point[1] + y] for point in points]


def normalizeRect(rect):
    """
    TODO: describe function
    """
    rect = fixClockwise(rect)
    minXIdx = findMinXIdx(rect)
    rect = reshapePoints(rect, minXIdx)
    distanses = findDistances(rect)
    if distanses[0]['d'] > distanses[1]['d'] or distanses[0]['matrix'][0] == 0:
        rect = reshapePoints(rect, 3)
    return rect


def normalizeMultilineRect(rect, mline_rects):
    """
    TODO: describe function
    """
    rect = fixClockwise(rect)
    minXIdx = findMinXIdx(rect)
    rect = reshapePoints(rect, minXIdx)
    max_width = 0
    for mline_rect in mline_rects:
        distanses = findDistances(mline_rect)
        if distanses[0]['d'] > max_width:
            max_width = distanses[0]['d']
            max_p0 = mline_rect[0]
            max_p1 = mline_rect[1]
    if max_p0[0] < max_p1[0]:
        if max_p0[1] > max_p1[1]:
            rect = reshapePoints(rect, 3)
    return rect


def prepareImageText(img):
    """
    сперва переведём изображение из RGB в чёрно серый
    значения пикселей будут от 0 до 255
    """
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_min = np.amin(grayImage)
    grayImage -= img_min
    img_max = np.amax(img)
    k = 255 / img_max
    grayImage = grayImage.astype(np.float64)
    grayImage *= k
    grayImage = grayImage.astype(np.uint8)

    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage


def detectBestPerspective(bwImages):
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
        if (minStat == diff) and (maxStatCount + minStatCount > diffCnt):
            idx = i
            diffCnt = maxStatCount + minStatCount

    return idx


def addPointOffset(point, x, y):
    """
    TODO: describe function
    """
    return [point[0] + x, point[1] + y]


def addPointOffsets(points, dx, dy):
    """
    TODO: describe function
    """
    return [
        addPointOffset(points[0], -dx, -dy),
        addPointOffset(points[1], dx, dy),
        addPointOffset(points[2], dx, dy),
        addPointOffset(points[3], -dx, -dy),
    ]


def makeRectVariants2(propablyPoints, h, w, qualityProfile=[3, 1, 0]):
    """
    TODO: describe function
    """
    distanses = findDistances(propablyPoints)

    pointCentreLeft = [propablyPoints[0][0] + (propablyPoints[1][0] - propablyPoints[0][0]) / 2,
                       propablyPoints[0][1] + (propablyPoints[1][1] - propablyPoints[0][1]) / 2]

    pointBottomLeft = [pointCentreLeft[0], getYByMatrix(distanses[3]["matrix"], pointCentreLeft[0])]

    dx = propablyPoints[0][0] - pointBottomLeft[0]
    dy = propablyPoints[0][1] - pointBottomLeft[1]

    if dx == 0:
        return [propablyPoints]

    steps = qualityProfile[0]
    stepsPlus = qualityProfile[1]
    stepsMinus = qualityProfile[2]

    if len(qualityProfile) > 3:
        stepAdaptive = qualityProfile[3] > 0
    else:
        stepAdaptive = False
    step = 1

    dxStep = dx / steps
    dyStep = dy / steps

    if stepAdaptive and w/h > 2:
        # Need normalizing
        dd = math.sqrt(dx ** 2 + dy ** 2)
        #print('pointCentreLeft {}  propablyPoints[0] {}'.format(pointCentreLeft, propablyPoints[0]))
        d_max = distance(pointCentreLeft, propablyPoints[0])
        steps_all = int(d_max / dd)

        step = int((steps_all*2)/steps)
        stepsMinus = steps_all+stepsMinus*step
        stepsPlus = steps_all+stepsPlus*step
        #print('dd: {}  d_max: {}  steps_all {}  step {}'.format(dd, d_max, steps_all, step))


    pointsArr = []
    for i in range(-stepsMinus, steps + stepsPlus + 1):
        pointsArr.append(addPointOffsets(propablyPoints, i * dxStep, i * dyStep))
    return pointsArr


def makeRectVariants(propablyPoints, steps=5):
    """
    TODO: describe function
    """
    distanses = findDistances(propablyPoints)

    pointCentreLeft = [propablyPoints[0][0] + (propablyPoints[1][0] - propablyPoints[0][0]) / 2,
                       propablyPoints[0][1] + (propablyPoints[1][1] - propablyPoints[0][1]) / 2]

    matrixLeft = linearLineMatrix(pointCentreLeft, [pointCentreLeft[0], pointCentreLeft[1] - 1])
    pointBottomLeft = detectIntersection(distanses[3]["matrix"], matrixLeft)

    dx = propablyPoints[0][0] - pointBottomLeft[0]
    dy = propablyPoints[0][1] - pointBottomLeft[1]

    if dx == 0:
        return []

    if dy / dx > 1:
        steps = 5
        dxStep = dx / (steps - 2)
        dyStep = dy / (steps - 2)

        pointsArr = []
        for i in range(steps):
            pointsArr.append([
                addPointOffset(propablyPoints[0], -i * dxStep, -i * dyStep),
                addPointOffset(propablyPoints[1], i * dxStep, i * dyStep),
                addPointOffset(propablyPoints[2], i * dxStep, i * dyStep),
                addPointOffset(propablyPoints[3], -i * dxStep, -i * dyStep),
            ])
    else:
        steps = 5
        dxStep = dx / steps
        dyStep = dy / steps

        pointsArr = []
        for i in range(-2, steps + 3):
            pointsArr.append([
                addPointOffset(propablyPoints[0], -i * dxStep, -i * dyStep),
                addPointOffset(propablyPoints[1], i * dxStep, i * dyStep),
                addPointOffset(propablyPoints[2], i * dxStep, i * dyStep),
                addPointOffset(propablyPoints[3], -i * dxStep, -i * dyStep),
            ])

    return pointsArr


def normalizePerspectiveImages(images):
    """
    TODO: describe function
    """
    newImages = []
    for img in images:
        newImages.append(prepareImageText(img))
    return newImages


class NpPointsCraft(object):
    """
    NpPointsCraft Class
    git clone https://github.com/clovaai/CRAFT-pytorch.git
    """

    def __init__(self,
                 low_text=0.4,
                 link_threshold=0.7, # 0.4
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
    def get_classname(cls):
        return cls.__name__

    def load(self,
             mtl_model_path="latest",
             refiner_model_path="latest"
             ):
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
                  device="cuda",
                  is_refine=True,
                  trained_model=os.path.join(CRAFT_DIR, 'weights/craft_mlt_25k.pth'),
                  refiner_model=os.path.join(CRAFT_DIR, 'weights/craft_refiner_CTW1500.pth')
                  ):
        """
        TODO: describe method
        """
        is_cuda = device == "cuda"
        self.is_cuda = is_cuda

        # initialize CRAFT
        self.net = CRAFT()

        if is_cuda:
            self.net.load_state_dict(copyStateDict(torch.load(trained_model)))
        else:
            self.net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

        if is_cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if is_refine:
            from refinenet import RefineNet
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

    def detectByImagePath(self, image_path, targetBoxes, qualityProfile=[1, 0, 0]):
        """
        TODO: describe method
        """
        image = imgproc.loadImage(image_path)
        for targetBox in targetBoxes:
            x = min(targetBox['x1'], targetBox['x2'])
            w = abs(targetBox['x2'] - targetBox['x1'])
            y = min(targetBox['y1'], targetBox['y2'])
            h = abs(targetBox['y2'] - targetBox['y1'])
            image_part = image[y:y + h, x:x + w]
            points, probably_multiline = self.detectInBbox(image_part)
            propablyPoints = addCoordinatesOffset(points, x, y)
            targetBox['points'] = []
            targetBox['imgParts'] = []
            coef = 4.6
            if probably_multiline > 0:
                coef = 1.76
            targetBox['probably_multiline'] = {'coef': coef, 'proportion': probably_multiline}
            if len(propablyPoints):
                targetPointsVariants = makeRectVariants2(propablyPoints, h, w, qualityProfile)
                if len(targetPointsVariants) > 1:
                    imgParts = [getCvZoneRGB(image, reshapePoints(rect, 1), 0, 0, coef) for rect in targetPointsVariants]
                    idx = detectBestPerspective(normalizePerspectiveImages(imgParts))
                    targetBox['imgParts'] = imgParts
                else:
                    idx = 0
                targetBox['idx'] = idx
                targetBox['points'] = targetPointsVariants[idx]
        return targetBoxes, image

    def multiline_to_one_line(self, zones, all_points, all_properties, region_names, convert_all_by_region=None):
        for i, (points, properties, regions) in enumerate(
                zip(all_points, all_properties, region_names)):
            if convert_all_by_region:
                regions = convert_all_by_region
            if properties['proportion'] > 0:
                rects = self.detectProbablyMultilineZones(properties['parts'][properties['idx']])
                if len(rects) > 1:
                    mc = MultilineConverter(properties['parts'][properties['idx']], rects)
                    converted_part = mc.covert_to_1_line(regions)
                    zones[i] = np.reshape(converted_part, [*converted_part.shape, 1])
        return zones

    def detect(self, image, targetBoxes, qualityProfile=[1, 0, 0], return_properties=False):
        """
        TODO: describe method
        """
        all_points = []
        all_properties = []
        for targetBox in targetBoxes:
            x = int(min(targetBox[0], targetBox[2]))
            w = int(abs(targetBox[2] - targetBox[0]))
            y = int(min(targetBox[1], targetBox[3]))
            h = int(abs(targetBox[3] - targetBox[1]))

            image_part = image[y:y + h, x:x + w]
            points, probably_multiline = self.detectInBbox(image_part)
            propablyPoints = addCoordinatesOffset(points, x, y)
            coef = 4.6
            if probably_multiline > 0:
                coef = 1.76
            imgParts = None
            idx = 0
            if len(propablyPoints):
                targetPointsVariants = makeRectVariants2(propablyPoints, h, w, qualityProfile)
                if len(targetPointsVariants) > 1:
                    imgParts = [getCvZoneRGB(image, reshapePoints(rect, 1)) for rect in targetPointsVariants]

                    idx = detectBestPerspective(normalizePerspectiveImages(imgParts))
                    points = targetPointsVariants[idx]
                else:
                    points = targetPointsVariants[0]
                all_points.append(points)
                all_properties.append({'coef': coef,
                                       'idx': idx,
                                       'proportion': probably_multiline,
                                       "parts": imgParts})
            else:
                all_points.append([
                    [x, y + h],
                    [x, y],
                    [x + w, y],
                    [x + w, y + h]
                ])
                all_properties.append({'coef': coef,
                                       'idx': idx,
                                       'proportion': probably_multiline,
                                       "parts": imgParts})

        if return_properties:
            return all_points, all_properties
        return all_points

    def detectInBbox(self, image, debug=False):
        """
        TODO: describe method
        """
        t = time.time()
        bboxes, polys, score_text = test_net(self.net, image, self.text_threshold, self.link_threshold, self.low_text,
                                             self.is_cuda, self.is_poly, self.canvas_size, self.refine_net, self.mag_ratio)
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

        probably_multiline = 0
        if len(np_bboxes_idx) > 1:
            # probably multiline
            mline_boxes = [bboxes[i] for i in np_bboxes_idx]
            targetPoints = minimum_bounding_rectangle(np.concatenate(mline_boxes, axis=0))
            w = distance(targetPoints[0], targetPoints[1])
            h = distance(targetPoints[1], targetPoints[2])
            p = w/h

            # np sizes 3lines: 220х174, p=1.264 2lines: 300х150, p = 2 may be multiline
            if p < 2:
                probably_multiline = p

        if len(np_bboxes_idx) > 0:
            if probably_multiline:
                targetPoints = normalizeMultilineRect(targetPoints, mline_boxes)
            else:
                targetPoints = normalizeRect(targetPoints)

            if debug:
                print('###################################')
                print(targetPoints)

            if debug:
                print('image.shape')
                print(image.shape)
            # targetPoints = fixSideFacets(targetPoints, image.shape)
            targetPoints = addoptRectToBbox(targetPoints, image.shape, 7, 12, 0, 12)
        return targetPoints, probably_multiline

    def detectProbablyMultilineZones(self, image, craft_params={}, debug=False):
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

        np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions, 0.6)

        return [bboxes[i] for i in np_bboxes_idx]
