import math
import numpy as np
import cv2


def fline(p0, p1, debug=False):
    """
    Вычесление угла наклона прямой по 2 точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    if debug:
        print("Уравнение прямой, проходящей через эти точки:")
    if (x1 - x2 == 0):
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
    if (a < 0):
        a180 = 180 + a
    return [k, b, a, a180, r]


def distance(p0, p1):
    """
    distance between two points p0 and p1
    """
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def linearLineMatrix(p0, p1):
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
    """
    TODO: describe function
    """
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


def buildPerspective(img, rect, w, h):
    """
    TODO: describe function
    """
    w = int(w)
    h = int(h)
    pts1 = np.float32(rect)
    pts2 = np.float32(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))


def getCvZoneRGB(img, rect, gw=0, gh=0, coef=4.6, auto_width_height=True):
    """
    TODO: describe function
    """
    if (gw == 0 or gh == 0):
        distanses = findDistances(rect)
        h = (distanses[0]['d'] + distanses[2]['d']) / 2
        if auto_width_height:
            w = int(h*coef)
        else:
            w = (distanses[1]['d'] + distanses[3]['d']) / 2
    else:
        w, h = gw, gh
    # print('h: {}, w: {}'.format(h, w))
    return buildPerspective(img, rect, w, h)


def getMeanDistance(rect, startIdx):
    """
    TODO: describe function
    """
    endIdx = startIdx+1
    start2Idx = startIdx+2
    end2Idx = endIdx+2
    if end2Idx == 4:
        end2Idx = 0
    #print('startIdx: {}, endIdx: {}, start2Idx: {}, end2Idx: {}'.format(startIdx, endIdx, start2Idx, end2Idx))
    return np.mean([distance(rect[startIdx], rect[endIdx]), distance(rect[start2Idx], rect[end2Idx])])


def reshapePoints(targetPoints,startIdx):
    """
    TODO: describe function
    """
    if [startIdx>0]:
        part1 = targetPoints[:(startIdx)]
        part2 = targetPoints[(startIdx):]
        targetPoints = np.concatenate((part2,part1))
    return targetPoints


def getCvZonesRGB(img, rects, gw=0, gh=0, coef=4.6, auto_width_height=True):
    """
    TODO: describe function
    """
    dsts = []
    for rect in rects:
        h = getMeanDistance(rect, 0)
        w = getMeanDistance(rect, 1)
        #print('h: {}, w: {}'.format(h,w))
        if h > w and auto_width_height:
            h, w = w, h
        else:
            rect = reshapePoints(rect, 3)
        if (gw == 0 or gh == 0):
            w, h = int(h*coef), int(h)
        else:
            w, h = gw, gh
        dst = buildPerspective(img, rect, w, h)
        dsts.append(dst)
    return dsts


def convertCvZonesRGBtoBGR(dsts):
    """
    TODO: describe function
    """
    bgrDsts = []
    for dst in dsts:
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        bgrDsts.append(dst)
    return bgrDsts


def getCvZonesBGR(img, rects, gw = 0, gh = 0, coef=4.6, auto_width_height = True):
    """
    TODO: describe function
    """
    dsts = getCvZonesRGB(img, rects, gw, gh, coef, auto_width_height = auto_width_height)
    return convertCvZonesRGBtoBGR(dsts)