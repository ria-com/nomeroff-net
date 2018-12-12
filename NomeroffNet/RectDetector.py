import os
import cv2
import numpy as np
import imutils
import math

class RectDetector(object):
    ''' Class for rectangle detection from the mask. '''

    def __init__(self, config):
        self.__dict__ = config

    def getApprox(self, c, peri):
        '''function get approx'''
        i = 0
        coef = self.COEF_APPROX
        min_c = 0
        max_c = coef*2
        while i < self.MAX_COUNT_STEP_APPROX:
            approx = cv2.approxPolyDP(c, coef * peri, True)
            cnt = len(approx)
            if cnt == self.TARGET_POINTS:
                break
            if cnt > self.TARGET_POINTS:
                min_c = coef
                max_c = max_c + coef
            if cnt < self.TARGET_POINTS:
                max_c = coef
            coef = (max_c + min_c)/2
            i += 1
        return approx

    def detectRect(self, image):
        '''function detect rect'''

        # load the image and resize it to a smaller factor so that the shapes can be approximated better
        resized = imutils.resize(image, width=1200)
        ratio = image.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly, and threshold it
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]

        # find contours in the thresholded image and initialize the shape detector
        cnts = cv2.findContours(thresh.copy(), 1, 2)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        res = []
        rows, cols = image.shape[:2]

        # loop over the contours
        for c in cnts:
            # compute the center of the contour, then detect the name of the shape using only the contour
            peri = cv2.arcLength(c, True)
            approx = self.getApprox(c,  peri)
            c = approx.astype("float")

            # multiply the contour (x, y)-coordinates by the resize ratio, then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            res.append(c)
        return res

    def linearLineMatrix(self, p0 ,p1):
        ''' The calculation of the coefficients of the matrix describing the line by two points. '''
        x1 = float(p0[0])
        y1 = float(p0[1])

        x2 = float(p1[0])
        y2 = float(p1[1])

        A = y1 - y2
        B = x2 - x1
        C = x2*y1-x1*y2
        return [A, B, C]

    def fline(self, p0, p1):
        ''' Calculation of the angle of inclination of a straight line by 2 points. '''
        x1 = float(p0[0])
        y1 = float(p0[1])

        x2 = float(p1[0])
        y2 = float(p1[1])

        if (x1 - x2 == 0):
            k = 100000000
            b = y2
        else:
            k = (y1 - y2) / (x1 - x2)
            b = y2 - k*x2
        r = math.atan(k)
        a = math.degrees(r)
        a180 = a
        if (a < 0 ):
            a180 = 180 + a
        return [k, b, a, a180, r]
    
    def distance(self, p0, p1):
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

    def findDistances(self, points):
        ''' Getting an array with line characteristics '''
        distanses = []
        cnt = len(points)
        for i in range(cnt):
            p0 = i
            if (i < cnt-1):
                p1 = i+1
            else:
                p1 = 0
            distanses.append({ "d": self.distance(points[p0][0], points[p1][0]), "p0": p0, "p1":p1,
                              "matrix": self.linearLineMatrix(points[p0][0],points[p1][0]),
                              "coef": self.fline(points[p0][0],points[p1][0])})
        return distanses

    def clacRectLines(self, distanses):
        ''' Sort lines to length '''
        return sorted(distanses, key=lambda x: x["d"])

    def filterInterestedLines(self, interestedLines, max, thresholdPercentage):
        threshold = interestedLines[len(interestedLines)-1]["d"]*thresholdPercentage
        return [x for x in interestedLines if x["d"] >= threshold]

    def gDiff(self, a, b):
        d1 = abs(a-b)
        d2 = 180-(d1)
        if (d1<d2):
            return d1
        else:
            return d2

    def cdist(self, X, centroids):
        lines = []
        for x in X:
            line = []
            for c in centroids:
                line.append(self.gDiff(x,c))
            lines.append(line)
        return np.array(lines)

    def cmean(self, X, centroid):
        lines = []
        for i in range(0,len(X)):
            x = X[i]
            if abs(x-centroid)>abs(x-180-centroid):
                X[i] = x-180
        return np.mean(X, axis=0)

    def gKMeans(self, X, maxDeep=0, currentLevel=0):
        ''' Simplified implementation of k-means clustering for 2 classes over a one-character array '''

        parts = np.int32(np.random.random(2)*currentLevel)
        centroids = np.array([0+parts[0],90+parts[1]])

        # Consider the distance from observations to centroid
        for i in range(0,10):
            distances = self.cdist(X, centroids)

            # look to which centroid of each point is closest
            labels = distances.argmin(axis=1)
            c0 = self.cmean(X[labels == 0], centroids[0])
            c1 = self.cmean(X[labels == 1], centroids[1])
            if not np.isnan(c0):
                centroids[0] = c0
            if not np.isnan(c1):
                centroids[1] = c1
        if (len(X[labels == 0])==0) or (len(X[labels == 1])==0):
            currentLevel = currentLevel+1
            if maxDeep>currentLevel:
                labels,distances,centroids = self.gKMeans(X,maxDeep,currentLevel)
        return [labels,distances,centroids]

    def gKMeansMajorLines(self, interestedLines,minElements=5):
        Xfull = [item["coef"][3] for item in interestedLines]
        Xfull = np.float32(Xfull)

        interestedLinesFiltered = self.filterInterestedLines(interestedLines,7,0.2)
        X = [item["coef"][3] for item in interestedLinesFiltered]
        X = np.float32(X)

        labels,distances,centroids = self.gKMeans(X,10)
        distances = self.cdist(Xfull, centroids)
        labels = distances.argmin(axis=1)
        if (len(Xfull[labels == 0])==0) or (len(Xfull[labels == 1])==0):
            if (minElements<len(interestedLines)):
                minElements = minElements+1
                labels = self.gKMeansMajorLines(interestedLines,minElements)
        return labels

    def detectIntersection(self, line1, line2):
        X = np.array([line1['matrix'][:2], line2['matrix'][:2]])
        y = np.array([line1['matrix'][2], line2['matrix'][2]])
        return np.linalg.solve(X, y)

    def isHaveCommonPoint(self, dist1, dist2):
        if (dist1["p0"] == dist2["p1"]) or (dist1["p1"] == dist2["p0"]):
            return True
        else:
            return False

    def isHaveCommonPointRecursive(self, C, i1, i2, deep=0):
        deep = deep + 1;
        if len(C[i1]["links"]) < 2 or deep > len(C):
            return False
        for iTarget in C[i1]["links"]:
            if iTarget == i2:
                return True
            else:
                srabotka = self.isHaveCommonPointRecursive(C, iTarget, i2, deep)
                if srabotka:
                    return True
        return False

    
    def makeCommonPointsLinks(self, C):
        for i in range(0, len(C)):
            p0 = C[i]["p0"]
            p1 = C[i]["p1"]
            C[i]["links"] = []
            for iTarget in range(0, len(C)):
                if i != iTarget:
                    if self.isHaveCommonPoint(C[i], C[iTarget]):
                        C[i]["links"].append(iTarget)

    
    def detectIndexesParalelLines(self, C):
        self.makeCommonPointsLinks(C)
        pointIdx1 = len(C) - 1
        idx = 2
        pointIdx2 = len(C) - idx

        while self.isHaveCommonPoint(C[pointIdx1], C[pointIdx2]):
            pointIdx1 = len(C) - idx
            idx = idx + 1
            pointIdx2 = len(C) - idx
        return [pointIdx1, pointIdx2]

    def makeTargetLines(self, A, B):
        pointsA = self.detectIndexesParalelLines(A)
        pointsB = self.detectIndexesParalelLines(B)
        return np.array([A[pointsA[0]], B[pointsB[0]], A[pointsA[1]], B[pointsB[1]]])

    def makeTargetPoints(self, targetLines):
        point1 = self.detectIntersection(targetLines[0], targetLines[1])
        point2 = self.detectIntersection(targetLines[1], targetLines[2])
        point3 = self.detectIntersection(targetLines[2], targetLines[3])
        point4 = self.detectIntersection(targetLines[3], targetLines[0])
        return np.array([point1, point2, point3, point4])

    def reshapePoints(self, targetPoints, startIdx):
        if [startIdx > 0]:
            part1 = targetPoints[:(startIdx)]
            part2 = targetPoints[(startIdx):]
            targetPoints = np.concatenate((part2, part1))
        return targetPoints

    def findMinXIdx(self, targetPoints):
        minXIdx = 0
        for i in range(1,len(targetPoints)):
            if (targetPoints[i][0] < targetPoints[minXIdx][0]):
                minXIdx = i
            if (targetPoints[i][0] == targetPoints[minXIdx][0]) and (targetPoints[i][1] < targetPoints[minXIdx][1]):
                minXIdx = i
        return minXIdx

    def detectIntersectionNormD(self, matrix1, matrix2, d):
        X = np.array([matrix1[:2], matrix2[:2]])
        c0 = matrix1[2] - d * (matrix1[0] ** 2 + matrix1[1] ** 2) ** 0.5
        c1 = matrix2[2] - d * (matrix2[0] ** 2 + matrix2[1] ** 2) ** 0.5
        y = np.array([c0, c1])
        return np.linalg.solve(X, y)

    def addOffset(self, targetPoints, offset):
        minXIdx = self.findMinXIdx(targetPoints)
        minXIdxPrev = minXIdx - 1
        if minXIdx == 0:
            minXIdxPrev = 3

        minXIdxNext = minXIdx + 1

        if minXIdxNext == 4:
            minXIdxNext = 0

        maxXIdx = minXIdx + 2
        if maxXIdx > 3:
            maxXIdx = maxXIdx - 4

        dy = targetPoints[minXIdxPrev][1] - targetPoints[minXIdx][1]
        sign = math.copysign(1, dy)

        matrixMin_Prev = self.linearLineMatrix(targetPoints[minXIdxPrev], targetPoints[minXIdx])
        matrixMin_Next = self.linearLineMatrix(targetPoints[minXIdx], targetPoints[minXIdxNext])
        matrixMax_Prev = self.linearLineMatrix(targetPoints[minXIdxNext], targetPoints[maxXIdx])
        matrixMax_Next = self.linearLineMatrix(targetPoints[maxXIdx], targetPoints[minXIdxPrev])

        point1 = self.detectIntersectionNormD(matrixMin_Prev, matrixMin_Next, offset * sign)
        point2 = self.detectIntersectionNormD(matrixMin_Next, matrixMax_Prev, offset * sign)
        point3 = self.detectIntersectionNormD(matrixMax_Prev, matrixMax_Next, offset * sign)
        point4 = self.detectIntersectionNormD(matrixMax_Next, matrixMin_Prev, offset * sign)
        return np.array([point1, point2, point3, point4])

    def fixClockwise(self, targetPoints):
        stat1=self.fline(targetPoints[0],targetPoints[1])
        stat2=self.fline(targetPoints[0],targetPoints[2])

        if targetPoints[0][0] == targetPoints[1][0] and (targetPoints[0][1] > targetPoints[1][1]):
            stat1[2] = -stat1[2]

        if (stat2[2] < stat1[2]):
            targetPoints = np.array([targetPoints[0],targetPoints[3],targetPoints[2],targetPoints[1]])

        return targetPoints

    def detect(self, image, outboundOffset=0):
        ''' Main method '''

        res = []
        arrPoints = self.detectRect(image)

        for points in arrPoints:
            distanses = self.findDistances(points)
            interestedLines = self.clacRectLines(distanses)

            # Get 4 lines that are part of a rectangle describing the license plate area.
            labels = self.gKMeansMajorLines(interestedLines)
            A = []
            B = []
            for i in range(len(interestedLines)):
                if (labels[i] == 0):
                    A.append(interestedLines[i])
                else:
                    B.append(interestedLines[i])
            targetLines = self.makeTargetLines(A,B)
            targetPoints = self.makeTargetPoints(targetLines)
            minXIdx = self.findMinXIdx(targetPoints)

            targetPoints=self.reshapePoints(targetPoints,minXIdx)

            targetPoints=self.fixClockwise(targetPoints)

            if outboundOffset:
                targetPoints=self.addOffset(targetPoints,outboundOffset)
            res.append(targetPoints)
        return res