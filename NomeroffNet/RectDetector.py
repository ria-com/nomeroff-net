import os
import cv2
import numpy as np
import imutils
import math
import asyncio

class RectDetector(object):
    ''' Class for rectangle detection from the mask. '''

    def __init__(self, coef_approx = 0.00001, max_count_step_approx = 300, target_points = 11):
        self.COEF_APPROX = coef_approx
        self.MAX_COUNT_STEP_APPROX = max_count_step_approx
        self.TARGET_POINTS = target_points

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
        if cv2.__version__[0] == "4":
            cnts, hierarchy = cv2.findContours(thresh.copy(), 1, 2)
        else:
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
        return math.sqrt((p0[0] - p1[0])*(p0[0] - p1[0]) + (p0[1] - p1[1])*(p0[1] - p1[1]))

    def rotate_points(self,rect):
        rect[0], rect[1], rect[2], rect[3] = rect[1], rect[2], rect[3], rect[0]
        return rect

    def findWidth(self, rect):
        line1 = self.distance(rect[0], rect[1])
        line3 = self.distance(rect[2], rect[3])
        h = np.mean([line1, line3])
        return h

    def findHieght(self, rect):
        line2 = self.distance(rect[1], rect[2])
        line4 = self.distance(rect[3], rect[0])
        w = np.mean([line2, line4])
        return w

    def to_pretty_point(self, points):
        sortedOnX = sorted([point for point in points], key=lambda x: x[0] )
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


    def rotate_to_pretty(self, points):
        new_arr = []
        points = self.to_pretty_point(points)
        new_arr.append(points)
        w = self.findWidth(points)
        h = self.findHieght(points)

        if h > w:
            h, w = w, h
            points = self.reshapePoints(points, 1)
        return points, w, h

    def detetct_pretty_w_h_to_zone(self, w, h, coef=4.6):
        return int(h*coef), int(h)

    async def get_cv_zoneRGB_async(self, img, rect, gw = 0, gh = 0, coef=4.6):
        rect, w, h = self.rotate_to_pretty(rect)
        if (gw == 0 or gh == 0):
            w, h = self.detetct_pretty_w_h_to_zone(w, h, coef)
        else:
            w, h = gw, gh
        pts1 = np.float32(rect)
        pts2 = np.float32(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img,M,(w,h))
        return dst;

    async def get_cv_zonesRGB_async(self, img, rects, gw = 0, gh = 0, coef=4.6):
        loop = asyncio.get_event_loop()
        promises = [loop.create_task(self.get_cv_zoneRGB_async(img, rect, gw = gw, gh = gh, coef=coef)) for rect in rects]
        if bool(promises):
            await asyncio.wait(promises)
        return [promise.result() for promise in promises]

    async def get_cv_zonesBGR_async(self, img, rects, gw = 0, gh = 0, coef=4.6):
        loop = asyncio.get_event_loop()
        promises = [loop.create_task(self.get_cv_zoneRGB_async(img, rect, gw = gw, gh = gh, coef=coef)) for rect in rects]
        if bool(promises):
            await asyncio.wait(promises)
        return [cv2.cvtColor(promise.result(), cv2.COLOR_RGB2BGR) for promise in promises]

    def get_cv_zonesRGB(self, img, rects, gw = 0, gh = 0, coef=4.6):
        dsts = []
        for rect in rects:
            rect, w, h = self.rotate_to_pretty(rect)
            if (gw == 0 or gh == 0):
                w, h = self.detetct_pretty_w_h_to_zone(w, h, coef)
            else:
                w, h = gw, gh
            pts1 = np.float32(rect)
            pts2 = np.float32(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img,M,(w,h))
            dsts.append(dst)
        return dsts;

    def get_cv_zonesBGR(self, img, rects, gw = 0, gh = 0, coef=4.6):
        dsts = self.get_cv_zonesRGB(img, rects, gw, gh, coef)
        bgrDsts = []
        for dst in dsts:
            dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
            bgrDsts.append(dst)
        return bgrDsts

    def makeUglyPoints(self, points):
        return [[p] for p in points]

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

    def filterInterestedLines(self, interestedLines,minElements,thresholdPercentage):
        if (len(interestedLines) > minElements):
            threshold = interestedLines[len(interestedLines)-1]["d"]*thresholdPercentage
            while True:
                interestedLinesFilterd = [x for x in interestedLines if x["d"] >= threshold]
                if len(interestedLinesFilterd) > minElements:
                    break
                threshold = threshold*.9
            # interestedLinesForFilter = interestedLines[:len(interestedLines)-8]
            # interestedLinesFiltered = [x for x in interestedLinesForFilter if x["d"] < threshold]
            return interestedLinesFilterd
        else:
            return interestedLines

        #threshold = interestedLines[len(interestedLines)-1]["d"]*thresholdPercentage
        #return [x for x in interestedLines if x["d"] >= threshold]

    def linearLineMatrixCrossPoint(self, p0,p1,point):
        arr = self.linearLineMatrix(p0,p1)
        x = point[0]
        y = point[1]
        arr[2] = arr[0]*x + arr[1]*y
        return arr

    def detectSimilarLines(self, C,targetLine, significantAlignPercentage=0.3):
        arr = []
        for i in range(len(C)):
            if ((targetLine['p0'] == C[i]['p1']) or (targetLine['p1'] == C[i]['p0'])):
                arr.append(C[i])
        if len(arr) > 1:
            dProp = arr[0]['d']/arr[1]['d'];
            if (dProp <= (1-significantAlignPercentage)):
                arr = [arr[1]]
            else:
                if (dProp >= (1-significantAlignPercentage)):
                    arr = [arr[0]]
                else:
                    arr = []
        return arr

    def detectAlignmentLines(self, targetLines,A,B):
        arrA0 = self.detectSimilarLines(A,targetLines[0])
        arrB0 = self.detectSimilarLines(B,targetLines[1])
        arrA1 = self.detectSimilarLines(A,targetLines[2])
        arrB1 = self.detectSimilarLines(B,targetLines[3])
        return np.array([arrA0,arrB0,arrA1,arrB1])

    def makeAligmentKeyPoints(self, line, arr):
        commonPoint = -1
        startPoint = -1
        endPoint = -1
        subline = arr[0]
        if subline['p0'] == line['p1']:
            commonPoint = line['p1']
            startPoint = line['p0']
            endPoint = subline['p1']
        else:
            commonPoint = line['p0']
            startPoint = line['p1']
            endPoint = subline['p0']
        return { "cPoint": commonPoint, "sPoint": startPoint, "ePoint": endPoint }

    def makeAligmentMatrix(self, line, arr, points):
        if (len(arr) == 0):
            return line["matrix"]
        else:
            pObj = self.makeAligmentKeyPoints(line, arr)
            return self.linearLineMatrixCrossPoint(points[pObj["sPoint"]][0],points[pObj["ePoint"]][0],points[pObj["cPoint"]][0])

    def makeTargetPoints2(self, targetLines,targetAlignmentLines,points):

        line0matrix = self.makeAligmentMatrix(targetLines[0],targetAlignmentLines[0],points)
        line1matrix = self.makeAligmentMatrix(targetLines[1],targetAlignmentLines[1],points)
        line2matrix = self.makeAligmentMatrix(targetLines[2],targetAlignmentLines[2],points)
        line3matrix = self.makeAligmentMatrix(targetLines[3],targetAlignmentLines[3],points)

        point1 = self.detectIntersection(line0matrix,line1matrix)
        point2 = self.detectIntersection(line1matrix,line2matrix)
        point3 = self.detectIntersection(line2matrix,line3matrix)
        point4 = self.detectIntersection(line3matrix,line0matrix)
        return np.array([point1,point2,point3,point4])

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

        interestedLinesFiltered = self.filterInterestedLines(interestedLines,4,0.2)
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

    def detectIntersection(self, matrix1,matrix2):
        X = np.array([matrix1[:2],matrix2[:2]])
        y = np.array([matrix1[2], matrix2[2]])
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def isHaveCommonPoint(self, dist1, dist2):
        if (dist1["p0"] == dist2["p1"]) or (dist1["p1"] == dist2["p0"]):
            return True
        else:
            return False

    def isHaveCommonPointRecursive(self, C, i1, i2, deep=0):
        deep = deep + 1;
        if len(C[i1]["links"]) < 1 or deep > len(C):
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

        while self.isHaveCommonPointRecursive(C,pointIdx1,pointIdx2):
            #pointIdx1 = len(C) - idx
            idx = idx + 1
            pointIdx2 = len(C) - idx
        return [pointIdx1, pointIdx2]

    def makeTargetLines(self, A, B):
        pointsA = self.detectIndexesParalelLines(A)
        pointsB = self.detectIndexesParalelLines(B)
        return np.array([A[pointsA[0]], B[pointsB[0]], A[pointsA[1]], B[pointsB[1]]])

    def makeTargetPoints(self, targetLines):
       point1 = self.detectIntersection(targetLines[0]["matrix"],targetLines[1]["matrix"])
       point2 = self.detectIntersection(targetLines[1]["matrix"],targetLines[2]["matrix"])
       point3 = self.detectIntersection(targetLines[2]["matrix"],targetLines[3]["matrix"])
       point4 = self.detectIntersection(targetLines[3]["matrix"],targetLines[0]["matrix"])
       return np.array([point1,point2,point3,point4])

    def reshapePoints(self, targetPoints, startIdx):
        if [startIdx > 0]:
            part1 = targetPoints[:(startIdx)]
            part2 = targetPoints[(startIdx):]
            targetPoints = np.concatenate((part2, part1))
        return targetPoints

    def findMinXIdx(self, targetPoints):
        minXIdx = 3
        for i in range(0,len(targetPoints)):
            if (targetPoints[i][0] < targetPoints[minXIdx][0]):
                minXIdx = i
            if (targetPoints[i][0] == targetPoints[minXIdx][0]) and (targetPoints[i][1] < targetPoints[minXIdx][1]):
                minXIdx = i
        return minXIdx

    def detectIntersectionNormDD(self, matrix1, matrix2, d1, d2):
        X = np.array([matrix1[:2], matrix2[:2]])
        c0 = matrix1[2] - d1 * (matrix1[0]*matrix1[0] + matrix1[1]*matrix1[1]) ** 0.5
        c1 = matrix2[2] - d2 * (matrix2[0]*matrix2[0] + matrix2[1]*matrix2[1]) ** 0.5
        y = np.array([c0, c1])
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def detectIntersectionNormD(self, matrix1, matrix2, d):
        X = np.array([matrix1[:2], matrix2[:2]])
        c0 = matrix1[2] - d * (matrix1[0]*matrix1[0] + matrix1[1]*matrix1[1]) ** 0.5
        c1 = matrix2[2] - d * (matrix2[0]*matrix2[0] + matrix2[1]*matrix2[1]) ** 0.5
        y = np.array([c0, c1])
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def addOffset(self, targetPoints, offsetHorisontal, offsetVertical):
       distanses = self.findDistances(self.makeUglyPoints(targetPoints))
       points=[]
       cnt = len(distanses)
       offsetFlag = distanses[0]['d']>distanses[1]['d']
       for i in range(cnt):
           iPrev = i
           iNext = i+1
           if (iNext == cnt):
               iNext=0
           if offsetFlag:
               offset1 = offsetVertical
               offset2 = offsetHorisontal
           else:
               offset2 = offsetVertical
               offset1 = offsetHorisontal
           offsetFlag = not offsetFlag
           points.append(self.detectIntersectionNormDD(distanses[iPrev]['matrix'],distanses[iNext]['matrix'],offset1,offset2))
       return np.array(points)

    def fixClockwise(self, targetPoints):
        stat1=self.fline(targetPoints[0],targetPoints[1])
        stat2=self.fline(targetPoints[0],targetPoints[2])

        if targetPoints[0][0] == targetPoints[1][0] and (targetPoints[0][1] > targetPoints[1][1]):
            stat1[2] = -stat1[2]

        if (stat2[2] < stat1[2]):
            targetPoints = np.array([targetPoints[0],targetPoints[3],targetPoints[2],targetPoints[1]])

        return targetPoints

    # fix rectangle points
    # http://www.math.by/geometry/eqline.html
    # https://xn--80ahcjeib4ac4d.xn--p1ai/information/solving_systems_of_linear_equations_in_python/
    def detectParalelMatrix(self, matrix,point):
        new_matrix = matrix
        new_matrix[2] = matrix[0]*point[0]+matrix[1]*point[1]
        return new_matrix

    def detectIntersectionParalelLine(self, line1,line2,point):
        line2matrix = self.detectParalelMatrix(line2['matrix'],point)
        return self.detectIntersection(line1['matrix'],line2matrix)

    def detectUnstableLines(self, distanses,points,d0,d1):
        angle0 = distanses[0]["coef"][3]-distanses[1]["coef"][3]
        if d0<d1:
            angle1 = distanses[0]["coef"][3]-distanses[3]["coef"][3]
            if abs(angle0)>abs(angle1):
                if (distanses[2]["d"] < distanses[0]["d"]):
                    points[3] = self.detectIntersectionParalelLine(distanses[2],distanses[1],points[distanses[0]["p0"]])
                else:
                    points[0] = self.detectIntersectionParalelLine(distanses[0],distanses[1],points[distanses[2]["p1"]])
            else:
                if (distanses[2]["d"] <distanses[0]["d"]):
                    points[2] = self.detectIntersectionParalelLine(distanses[2],distanses[3],points[distanses[0]["p1"]])
                else:
                    points[1] = self.detectIntersectionParalelLine(distanses[0],distanses[3],points[distanses[2]["p0"]])
        else:
            angle1 = distanses[1]["coef"][3]-distanses[2]["coef"][3]
            if abs(angle0)>abs(angle1):
                if (distanses[3]["d"] <distanses[1]["d"]):
                    points[3] = self.detectIntersectionParalelLine(distanses[3],distanses[0],points[distanses[1]["p1"]])
                else:
                    points[2] = self.detectIntersectionParalelLine( distanses[1],distanses[0],points[distanses[3]["p0"]])
            else:
                if (distanses[3]["d"] <distanses[1]["d"]):
                    points[0] = self.detectIntersectionParalelLine(distanses[3],distanses[2],points[distanses[1]["p0"]])
                else:
                    points[1] = self.detectIntersectionParalelLine(distanses[1],distanses[2],points[distanses[3]["p1"]])
        return points

    def fixRectangle(self, points, fixRectangleAngle=3):
        distanses = self.findDistances(self.makeUglyPoints(points))
        d0 = self.gDiff(distanses[0]["coef"][3],distanses[2]["coef"][3])
        d1 = self.gDiff(distanses[1]["coef"][3],distanses[3]["coef"][3])
        if (d0>fixRectangleAngle) or (d1>fixRectangleAngle):
            points = self.detectUnstableLines(distanses,points,d0,d1)
        if (d0>fixRectangleAngle) or (d1>fixRectangleAngle):
            points = self.fixRectangle(points, fixRectangleAngle)
        return points

    def findMaxs(self, points):
        maxX = max(points, key=lambda p: p[0])
        maxY = max(points, key=lambda p: p[1])
        minX = min(points, key=lambda p: p[0])
        minY = min(points, key=lambda p: p[1])
        return maxX, maxY, minX, minY

    def uniquePoints(self, arrPoints):
       arrPointsNew = []
       for i in range(0,len(arrPoints)):
           points = arrPoints[i]
           pointsNew = []
           pStart = [[-1,-1]]
           for j in range(0,len(points)):
               p = points[j]
               if (not (pStart==p).all()):
                   pointsNew.append(p)
               pStart = p
           arrPointsNew.append(pointsNew)
       return np.array(arrPointsNew)

    def checkIfIsSquare(self, points, coef = 2.):
        points = np.array(points)

        d1 = self.distance(points[0], points[1])
        d2 = self.distance(points[1], points[2])

        distanses = self.findDistances(self.makeUglyPoints(points))
        angle = distanses[0]["coef"][3]

        if d1 > d2:
            d1, d2 = d2, d1
        #print(d2/d1)
        #print(angle)
        if (d2/d1) <= coef:
            return True
        return False

    def sortBySize(self, mainArr):
        if len(mainArr) < 2:
            return mainArr
        return sorted(mainArr, key=lambda x: cv2.contourArea(np.array(x).astype(int)), reverse=True)

    async def detectOneAsync(self, image, outboundWidthOffset=3, outboundHeightOffset=0, fixRectangleAngle=3, fixGeometry=0):
        arrPoints = self.detectRect(image)
        arrPoints = self.uniquePoints(arrPoints)

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
            if fixGeometry:
                targetAlignmentLines = self.detectAlignmentLines(targetLines,A,B)
                targetPoints = self.makeTargetPoints2(targetLines,targetAlignmentLines,arrPoints[0])
            else:
                targetPoints = np.float32(self.makeTargetPoints(targetLines))

            minXIdx = self.findMinXIdx(targetPoints)

            targetPoints=self.reshapePoints(targetPoints,minXIdx)
            targetPoints=self.fixClockwise(targetPoints)
            targetPoints = self.fixRectangle(targetPoints, fixRectangleAngle)

            if outboundWidthOffset or outboundHeightOffset:
                targetPoints=self.addOffset(targetPoints,outboundWidthOffset,outboundHeightOffset)

            return targetPoints

    async def detectAsync(self, images, outboundWidthOffset=3, outboundHeightOffset=0, fixRectangleAngle=3, fixGeometry=0):
         ''' Main method '''
         loop = asyncio.get_event_loop()
         promises = [loop.create_task(self.detectOneAsync(image, outboundWidthOffset=outboundWidthOffset, outboundHeightOffset=outboundHeightOffset, fixRectangleAngle=fixRectangleAngle, fixGeometry=fixGeometry)) for image in images]
         if bool(promises):
            await asyncio.wait(promises)
         return np.array([promise.result() for promise in promises])

    def detect(self, images, outboundWidthOffset=3, outboundHeightOffset=0, fixRectangleAngle=3, fixGeometry=1):
        ''' Main method '''
        resPoints = []
        for image in images:
            res = []
            arrPoints = self.detectRect(image)
            arrPoints = self.uniquePoints(arrPoints)

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

                #print("targetLines")
                #print(targetLines)

                if fixGeometry:
                    targetAlignmentLines = self.detectAlignmentLines(targetLines,A,B)
                    targetPoints = self.makeTargetPoints2(targetLines,targetAlignmentLines,arrPoints[0])
                else:
                    targetPoints = np.float32(self.makeTargetPoints(targetLines))

                minXIdx = self.findMinXIdx(targetPoints)

                targetPoints=self.reshapePoints(targetPoints,minXIdx)
                targetPoints=self.fixClockwise(targetPoints)
                targetPoints = self.fixRectangle(targetPoints, fixRectangleAngle)

                if outboundWidthOffset or outboundHeightOffset:
                    targetPoints=self.addOffset(targetPoints,outboundWidthOffset,outboundHeightOffset)

                res.append(targetPoints)

            resPoints.append(res[0])

        return np.array(resPoints)