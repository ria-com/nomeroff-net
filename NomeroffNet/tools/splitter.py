import numpy
import cv2

def split(img, coef=0.1, line=2):
    n = int(img.shape[0]/line + (img.shape[0] * coef))
    firstPart = img[:n]
    for l in range(1, line - 1):
        middlePart = img[:l*n]
        firstPart  = numpy.concatenate((firstPart, middlePart), axis=1)
    lastPart  = img[-n:]
    return numpy.concatenate((firstPart, lastPart), axis=1)