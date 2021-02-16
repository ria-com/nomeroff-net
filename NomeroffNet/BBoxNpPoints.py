# Import all necessary libraries.
import os
import sys
import math
import pathlib

# clone and append to path craft
NOMEROFF_NET_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "../")
CRAFT_DIR        = os.path.join(NOMEROFF_NET_DIR, 'CRAFT-pytorch')
CRAFT_URL        = "https://github.com/clovaai/CRAFT-pytorch.git"
if not os.path.exists(CRAFT_DIR):
    from git import Repo
    Repo.clone_from(CRAFT_URL, CRAFT_DIR)
sys.path.append(CRAFT_DIR)

# -*- coding: utf-8 -*-
import time
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# load CRAFT packages
from craft import CRAFT

# load NomerooffNet packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Base')))
from mcm.mcm import download_latest_model
from mcm.mcm import get_mode



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


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size,  refine_net=None, mag_ratio=1.5):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    #if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def distance(p0, p1):
    """
    distance between two points p0 and p1
    """
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def split_boxes(bboxes,dimensions,similarity_range = 0.7):
    np_bboxes_idx = []
    garbage_bboxes_idx =[]
    maxDy=0
    if len(bboxes):
        maxDy= max([dimension['dy'] for dimension in dimensions])
    #print('max dy: {}'.format(maxDy))
    for i, (bbox, dimension) in enumerate(zip(bboxes,dimensions)):
        #print('maxDy*similarity_range: {}'.format(maxDy*similarity_range))
        #print('dy: {}'.format(dimension['dy']))
        if (maxDy*similarity_range <=dimension['dy']):
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
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
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


def fline(p0,p1):
    """
    Вычесление угла наклона прямой по 2 точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    #print("Уравнение прямой, проходящей через эти точки:")
    if (x1 - x2 == 0):
        k = 1000000000
        b = y2
    else:
        k = (y1 - y2) / (x1 - x2)
        b = y2 - k*x2
    #print(" y = %.4f*x + %.4f" % (k, b))
    r = math.atan(k)
    a = math.degrees(r)
    a180 = a
    if (a < 0 ):
        a180 = 180 + a
    return [k, b, a, a180, r]


def detectIntersection(matrix1,matrix2):
    """
    http://www.math.by/geometry/eqline.html
    https://xn--80ahcjeib4ac4d.xn--p1ai/information/solving_systems_of_linear_equations_in_python/
    """
    X = np.array([matrix1[:2],matrix2[:2]])
    y = np.array([matrix1[2], matrix2[2]])
    return np.linalg.solve(X, y)


def detectIntersectionNormDD(matrix1,matrix2,d1,d2):
    X = np.array([matrix1[:2],matrix2[:2]])
    c0 = matrix1[2]-d1*(matrix1[0]**2 + matrix1[1]**2)**0.5
    c1 = matrix2[2]-d2*(matrix2[0]**2 + matrix2[1]**2)**0.5
    y = np.array([c0, c1])
    return np.linalg.solve(X, y)


def linearLineMatrix(p0,p1):
    """
    Вычесление коефициентов матрицы, описывающей линию по двум точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    #print("Уравнение прямой, проходящей через эти точки:")
    A = y1 - y2
    B = x2 - x1
    C = x2*y1-x1*y2
    #print("%.4f*x + %.4fy = %.4f" % (A, B, C))
    #print(A, B, C)
    return [A, B, C]


def findDistances(points):
    distanses = []
    cnt = len(points)

    for i in range(cnt):
        p0 = i
        if (i < cnt - 1):
            p1 = i + 1
        else:
            p1 = 0
        distanses.append({"d": distance(points[p0], points[p1]), "p0": p0, "p1": p1,
                          "matrix": linearLineMatrix(points[p0], points[p1]),
                          "coef": fline(points[p0], points[p1])})
    return distanses


def reshapePoints(targetPoints,startIdx):
    if [startIdx>0]:
        part1 = targetPoints[:(startIdx)]
        part2 = targetPoints[(startIdx):]
        targetPoints = np.concatenate((part2,part1))
    return targetPoints


def findMinXIdx(targetPoints):
    minXIdx = 3
    for i in range(0,len(targetPoints)):
        if (targetPoints[i][0] < targetPoints[minXIdx][0]):
            minXIdx = i
        if (targetPoints[i][0] == targetPoints[minXIdx][0]) and (targetPoints[i][1] < targetPoints[minXIdx][1]):
            minXIdx = i
    return minXIdx


def fixClockwise(targetPoints):
    stat1 = fline(targetPoints[0], targetPoints[1])
    stat2 = fline(targetPoints[0], targetPoints[2])
    if targetPoints[0][0] == targetPoints[1][0] and (targetPoints[0][1] > targetPoints[1][1]):
        stat1[2] = -stat1[2]

    if (stat2[2] < stat1[2]):
        targetPoints = np.array([targetPoints[0], targetPoints[3], targetPoints[2], targetPoints[1]])
    return targetPoints


def addOffsetManualPercentage(targetPoints, offsetLeftPercentage, offsetTopPercentage, offsetRightPercentage,
                              offsetBottomPercentage):
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


def fixSideFacets(targetPoints, adoptToFrame=None):
    distanses = findDistances(targetPoints)
    points = targetPoints.copy()
    #print('targetPoints: {}'.format(targetPoints))

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

        #print('targetSideIdx: {} iPrev: {} iNext: {}'.format(targetSideIdx, iPrev, iNext))
        pointCentre = [targetPoints[targetSideIdx][0] + (targetPoints[iNext][0] - targetPoints[targetSideIdx][0]) / 2,
                       targetPoints[targetSideIdx][1] + (targetPoints[iNext][1] - targetPoints[targetSideIdx][1]) / 2]

        if adoptToFrame != None:
            if pointCentre[0] < 0:
                pointCentre[0] = 0
            if pointCentre[0] >= adoptToFrame[1]:
                pointCentre[0] = adoptToFrame[1] - 1

        pointTo = [pointCentre[0], pointCentre[1] + 1]
        #print('pointCentre: {} pointTo: {}'.format(pointCentre, pointTo))
        matrix = linearLineMatrix(pointCentre, pointTo)
        #print('matrix: {}'.format(matrix))
        points[targetSideIdx] = detectIntersection(distanses[iPrev]["matrix"], matrix)
        #print('points[{}]: {}'.format(targetSideIdx, points[targetSideIdx]))
        points[iNext] = detectIntersection(matrix, distanses[iNext]["matrix"])
    # linearLineMatrix(points[p0],points[p1])
    return np.array(points)


def addCoordinatesOffset(points,x,y):
    for point in points:
        point[0] = point[0]+x
        point[1] = point[1]+y
    return points


class NpPointsCraft(object):
    """
    NpPointsCraft Class
    git clone https://github.com/clovaai/CRAFT-pytorch.git
    """
    def __init__(self, **args):
        pass
    
    @classmethod
    def get_classname(cls):
        return cls.__name__
    
    def load(self, 
             mtl_model_path="latest",
             refiner_model_path="latest"
            ):
        if mtl_model_path == "latest":
            model_info   = download_latest_model(self.get_classname(), "mtl", ext="pth")
            mtl_model_path   = model_info["path"]
        if refiner_model_path == "latest":
            model_info   = download_latest_model(self.get_classname(), "refiner", ext="pth")
            refiner_model_path   = model_info["path"]
        device = "cpu"
        if get_mode() == "gpu":
            device = "cuda"
        self.loadModel(device, True, mtl_model_path, refiner_model_path)
                  
    def loadModel(self, 
                  device="cuda",
                  is_refine=True,
                  trained_model=os.path.join(CRAFT_DIR, 'weights/craft_mlt_25k.pth'),
                  refiner_model=os.path.join(CRAFT_DIR, 'weights/craft_refiner_CTW1500.pth')
             ):
        is_cuda=torch.device(device)
        self.is_cuda = is_cuda
        # load net
        self.net = CRAFT()  # initialize

        print('Loading weights from checkpoint (' + trained_model + ')')
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

    def detect_by_path(self, image_path, targetBoxes, debug=False):
        image = imgproc.loadImage(image_path)
        for targetBox in targetBoxes:
            x = min(targetBox['x1'], targetBox['x2'])
            w = abs(targetBox['x2']-targetBox['x1'])
            y = min(targetBox['y1'], targetBox['y2'])
            h = abs(targetBox['y2']-targetBox['y1'])
            #print('x: {} w: {} y: {} h: {}'.format(x,w,y,h))
            image_part = image[y:y + h, x:x + w]
            targetBox['points'] = addCoordinatesOffset(self.detectInBbox(image_part),x,y)
        return targetBoxes

    def detect(self, image, targetBoxes, debug=False):
        all_points = []
        for targetBox in targetBoxes:
            x = int(min(targetBox[0], targetBox[2]))
            w = int(abs(targetBox[2]-targetBox[0]))
            y = int(min(targetBox[1], targetBox[3]))
            h = int(abs(targetBox[3]-targetBox[1]))
            
            image_part = image[y:y + h, x:x + w]
            points = addCoordinatesOffset(self.detectInBbox(image_part),x,y)
            all_points.append(points)
        return all_points

    def detectInBbox(self, image, debug=False):
        low_text = 0.4
        link_threshold = 0.7  # 0.4
        text_threshold = 0.6
        canvas_size = 1280
        mag_ratio = 1.5

        t = time.time()
        bboxes, polys, score_text = test_net(self.net, image, text_threshold, link_threshold, low_text,
                                                                   self.is_cuda, self.is_poly, canvas_size, self.refine_net, mag_ratio)
        if debug:
            print("elapsed time : {}s".format(time.time() - t))
        dimensions = []
        for poly in bboxes:
            dimensions.append({'dx': distance(poly[0], poly[1]), 'dy': distance(poly[1], poly[2])})

        if (debug):
            print(score_text.shape)
            # print(polys)
            print(dimensions)
            print(bboxes)

        np_bboxes_idx, garbage_bboxes_idx = split_boxes(bboxes, dimensions)

        targetPoints = []
        if (debug):
            print('np_bboxes_idx')
            print(np_bboxes_idx)
            print('garbage_bboxes_idx')
            print(garbage_bboxes_idx)
            print('raw_boxes')
            print(raw_boxes)
            print('raw_polys')
            print(raw_polys)

        if len(np_bboxes_idx) == 1:
            targetPoints = bboxes[np_bboxes_idx[0]]

        if len(np_bboxes_idx) > 1:
            targetPoints = minimum_bounding_rectangle(np.concatenate([bboxes[i] for i in np_bboxes_idx], axis=0))

        if len(np_bboxes_idx) > 0:
            minXIdx = findMinXIdx(targetPoints)
            targetPoints = reshapePoints(targetPoints, minXIdx)
            targetPoints = fixClockwise(targetPoints)
            if (debug):
                print('###################################')
                print(targetPoints)

            targetPoints = addOffsetManualPercentage(targetPoints, 11, 10, 3, 10)
            if (debug):
                print('image.shape')
                print(image.shape)
            targetPoints = fixSideFacets(targetPoints, image.shape)
        return targetPoints