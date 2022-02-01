# Import all necessary libraries.
import os
import sys
import math
import collections

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from tools.mcm import (modelhub,
                       get_mode_torch)

info = modelhub.download_repo_for_model("craft_mlt")
CRAFT_DIR = info["repo_path"]

import time
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np

# load CRAFT packages
from craft_mlt import imgproc
from craft_mlt import craft_utils
from craft_mlt.craft import CRAFT
from craft_mlt.refinenet import RefineNet

from typing import List, Dict, Tuple, Any, Union
from tools import (fline,
                   distance,
                   linearLineMatrix,
                   getYByMatrix,
                   findDistances,
                   getCvZoneRGB,
                   convertCvZonesRGBtoBGR,
                   fixClockwise2,
                   findMinXIdx,
                   detectIntersection,
                   minimum_bounding_rectangle,
                   reshapePoints)


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


def get_det_boxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8),
                                                                         connectivity=4)

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
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
             poly: bool, canvas_size: int, refine_net: RefineNet = None,
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


def filter_boxes(bboxes: List[Union[np.ndarray, np.ndarray]], dimensions: List[Dict],
                 target_points: List[Union[np.ndarray, np.ndarray]],
                 np_bboxes_idx: List[int], filter_range: int = 0.7) -> Tuple[List[int], List[int], int]:
    """
    TODO: describe function
    """
    target_points = normalizeRect(target_points)
    probably_count_lines = 1
    #print('target_points')
    #print(target_points)
    dy = distance(target_points[0], target_points[1])
    dx = distance(target_points[1], target_points[2])
    #print('dx {}, dy {}'.format(dx, dy))
    new_np_bboxes_idx = []
    garbage_bboxes_idx = []
    max_dy = 0
    if len(bboxes):
        max_dy = max([dimension['dy'] for dimension in dimensions])
    for i, (bbox, dimension) in enumerate(zip(bboxes, dimensions)):
        if i in np_bboxes_idx:
            coef = dimension['dy']/max_dy
            #print('coef {}, max_dy {}, dimension["dy"] {}'.format(coef, max_dy, dimension['dy']))
            if coef > filter_range:
                new_np_bboxes_idx.append(i)
            else:
                boxify_factor = dimension['dx']/dimension['dy']
                #print('boxify_factor {}'.format(boxify_factor))
                #print('bbox[0] {}'.format(bbox[0]))
                dx_offset = round(dimension['dx']/2)
                #print('dx_offset {}'.format(dx_offset))
                if bbox[0][0]<=dx_offset and (boxify_factor > 0.7 and boxify_factor < 1.7):
                    #print('Box {} is filtered!'.format(i))
                    garbage_bboxes_idx.append(i)
                else:
                    new_np_bboxes_idx.append(i)
        else:
            garbage_bboxes_idx.append(i)

    probably_count_lines = round(dy/max_dy)
    probably_count_lines = 1 if probably_count_lines < 1 else probably_count_lines
    probably_count_lines = 3 if probably_count_lines > 3 else probably_count_lines
    return new_np_bboxes_idx, garbage_bboxes_idx, probably_count_lines


def detectIntersectionNormDD(matrix1: np.ndarray, matrix2: np.ndarray, d1: float, d2: float) -> np.ndarray:
    """
    TODO: describe function
    """
    x = np.array([matrix1[:2], matrix2[:2]])
    c0 = matrix1[2] - d1 * (matrix1[0] ** 2 + matrix1[1] ** 2) ** 0.5
    c1 = matrix2[2] - d2 * (matrix2[0] ** 2 + matrix2[1] ** 2) ** 0.5
    y = np.array([c0, c1])
    return np.linalg.solve(x, y)


def detectDistanceFromPointToLine(matrix: List[np.ndarray],
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


def fixClockwise(target_points: List) -> List:
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


def addCoordinatesOffset(points: List or np.ndarray, x: float, y: float) -> List:
    """
    TODO: describe function
    """
    return [[point[0] + x, point[1] + y] for point in points]


def normalizeRect(rect: List) -> np.ndarray or List:
    """
    TODO: describe function
    """
    rect = fixClockwise2(rect)
    minXIdx = findMinXIdx(rect)
    rect = reshapePoints(rect, minXIdx)
    coef_ccw = fline(rect[0], rect[3])
    angle_ccw = round(coef_ccw[2], 2)
    d_bottom = distance(rect[0], rect[3])
    d_left = distance(rect[0], rect[1])
    k = d_bottom / d_left
    if round(rect[0][0], 4) == round(rect[1][0], 4):
        pass
    else:
        if d_bottom < d_left:
            k = d_left / d_bottom
            if k > 1.5 or angle_ccw < 0 or angle_ccw > 45:
                rect = reshapePoints(rect, 3)
        else:
            if k < 1.5 and (angle_ccw < 0 or angle_ccw > 45):
                rect = reshapePoints(rect, 3)
    return rect


def prepareImageText(img: np.ndarray) -> np.ndarray:
    """
    сперва переведём изображение из RGB в чёрно серый
    значения пикселей будут от 0 до 255
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.normalize(gray_image, None, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

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
        quality_profile = [3, 1, 0, 0]

    steps = quality_profile[0]
    steps_plus = quality_profile[1]
    steps_minus = quality_profile[2]
    step = 1
    if len(quality_profile) > 3:
        step_adaptive = quality_profile[3] > 0
    else:
        step_adaptive = False

    distanses = findDistances(propably_points)

    point_centre_left = [propably_points[0][0] + (propably_points[1][0] - propably_points[0][0]) / 2,
                         propably_points[0][1] + (propably_points[1][1] - propably_points[0][1]) / 2]

    if distanses[3]["matrix"][1] == 0:
        return [propably_points]
    point_bottom_left = [point_centre_left[0], getYByMatrix(distanses[3]["matrix"], point_centre_left[0])]
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
        points_arr.append(addPointOffsets(propably_points, i * dx_step, i * dy_step))
    return points_arr


def normalizePerspectiveImages(images: List or np.ndarray) -> List[np.ndarray]:
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
                 canvas_size=300,
                 mag_ratio=1.0
                 ):
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.text_threshold = text_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
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
            qualityProfile = [1, 0, 0, 0]
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
                target_points_variants = makeRectVariants(propablyPoints, qualityProfile)
                if len(target_points_variants) > 1:
                    imgParts = [getCvZoneRGB(image, reshapePoints(rect, 1)) for rect in target_points_variants]
                    normalized_perspective_img = normalizePerspectiveImages(imgParts)
                    idx = detectBestPerspective(normalized_perspective_img)
                    targetBox['points'] = target_points_variants[idx]
                    targetBox['imgParts'] = imgParts
                else:
                    targetBox['points'] = target_points_variants[0]
        return target_boxes, image

    def detect(self, image: np.ndarray, targetBoxes: List, qualityProfile: List = None) -> List:
        """
        TODO: describe method
        """
        all_points, all_mline_boxes = self.detect_mline(image, targetBoxes, qualityProfile)
        return all_points

    def detect_mline(self, image: np.ndarray, targetBoxes: List, qualityProfile: List = None) -> Tuple:
        """
        TODO: describe method
        """
        all_points, all_mline_boxes, all_count_lines, all_image_parts = self.detect_mline_count_lines(image, targetBoxes, qualityProfile)
        return all_points, all_mline_boxes


    def detect_mline_count_lines(self, image: np.ndarray, targetBoxes: List, qualityProfile: List = None) -> Tuple:
        """
        TODO: describe method
        """
        if qualityProfile is None:
            qualityProfile = [1, 0, 0, 0]
        all_points = []
        all_mline_boxes = []
        all_image_parts = []
        all_count_lines = []
        for targetBox in targetBoxes:
            x = int(min(targetBox[0], targetBox[2]))
            w = int(abs(targetBox[2] - targetBox[0]))
            y = int(min(targetBox[1], targetBox[3]))
            h = int(abs(targetBox[3] - targetBox[1]))

            image_part = image[y:y + h, x:x + w]
            all_image_parts.append(image_part)
            if h / w > 3.5:
                image_part = cv2.rotate(image_part, cv2.ROTATE_90_CLOCKWISE)
            # image_part = normalize_color(image_part)
            local_propably_points, mline_boxes, probably_count_lines = self.detectInBboxCountLines(image_part)
            all_mline_boxes.append(mline_boxes)
            all_count_lines.append(probably_count_lines)
            propably_points = addCoordinatesOffset(local_propably_points, x, y)
            if len(propably_points):
                target_points_variants = makeRectVariants(propably_points, qualityProfile)
                if len(target_points_variants) > 1:
                    img_parts = [getCvZoneRGB(image, reshapePoints(rect, 1)) for rect in target_points_variants]
                    idx = detectBestPerspective(normalizePerspectiveImages(img_parts))
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
        return all_points, all_mline_boxes, all_count_lines, all_image_parts

    def detectInBbox(self,
                     image: np.ndarray,
                     craft_params: Dict = None,
                     debug: bool = False):
        target_points, mline_boxes, probably_count_lines = self.detectInBbox(image, craft_params, debug)
        return target_points, mline_boxes

    def detectInBboxCountLines(self,
                     image: np.ndarray,
                     craft_params: Dict = None,
                     debug: bool = False):
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

        bboxes, score_text = test_net(self.net, image, text_threshold, link_threshold, low_text,
                                      self.is_cuda, self.is_poly, canvas_size, self.refine_net, mag_ratio)
        #print('Len of boxes {}'.format(len(bboxes)))
        dimensions = []
        for poly in bboxes:
            dimensions.append({'dx': distance(poly[0], poly[1]), 'dy': distance(poly[1], poly[2])})

        np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions)

        target_points = []
        probably_count_lines = 1

        if len(np_bboxes_idx) == 1:
            target_points = bboxes[np_bboxes_idx[0]]

        if len(np_bboxes_idx) > 1:
            started_boxes = np.concatenate([bboxes[i] for i in np_bboxes_idx], axis=0)
            target_points = minimum_bounding_rectangle(started_boxes)
            np_bboxes_idx, garbage_bboxes_idx, probably_count_lines  = filter_boxes(bboxes, dimensions, target_points, np_bboxes_idx)
            filtred_boxes = np.concatenate([bboxes[i] for i in np_bboxes_idx], axis=0)
            if len(started_boxes) > len(filtred_boxes):
                target_points = minimum_bounding_rectangle(started_boxes)

        if len(np_bboxes_idx) > 0:
            target_points = normalizeRect(target_points)
            target_points = addoptRectToBbox(target_points, image.shape, 7, 12, 0, 12)
        #return target_points, [bboxes[i] for i in np_bboxes_idx], probably_count_lines
        return target_points, [bboxes[i] for i in np_bboxes_idx], probably_count_lines

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
        bboxes, score_text = test_net(self.net, image, text_threshold, link_threshold, low_text,
                                      self.is_cuda, self.is_poly, canvas_size, self.refine_net, mag_ratio)
        if debug:
            print("elapsed time : {}s".format(time.time() - t))

        dimensions = []
        for poly in bboxes:
            dimensions.append({'dx': distance(poly[0], poly[1]), 'dy': distance(poly[1], poly[2])})

        np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions)

        return [bboxes[i] for i in np_bboxes_idx]
