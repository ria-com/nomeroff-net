import numpy
import cv2

def np_split(imgs, lines, coef=0.1):
    res_imgs = []
    for img, line in zip(imgs, lines):
        if line < 2:
            res_imgs.append(img)
        else:
            n = int(img.shape[0]/line + (img.shape[0] * coef))
            firstPart = img[:n]
            for l in range(1, line - 1):
                middlePart = img[:l*n]
                firstPart  = numpy.concatenate((firstPart, middlePart), axis=1)
            lastPart  = img[-n:]
            res_imgs.append(numpy.concatenate((firstPart, lastPart), axis=1))
    return imgs