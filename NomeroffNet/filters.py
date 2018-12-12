import skimage.io
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.morphology import convex_hull_image

def draw_box(image, boxs, color=(255,0,0), thickness=2):
  image = image.copy()

  # округление координат
  boxs = np.int0(boxs)

  cv2.drawContours(image,boxs,0,color,thickness)
  return image

def cv_img_mask(nns):
    res = []
    for nn in nns:
        masks = np.array(nn["masks"])
        for i in np.arange(masks.shape[2]):
            mask = np.array([[w[i] for w in h] for h in nn["masks"]])
            chull = convex_hull_image(mask)
            gray = skimage.color.gray2rgb(chull) * 255
            res.append(img_as_ubyte(gray))
    return res

def color_splash(image, nns, color=(0, 255, 0), white_balance=200):
    res = []
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * white_balance
    for nn in nns:
        if nn["masks"].shape[-1] > 0:
            mask = (np.sum(nn["masks"], -1, keepdims=True) >= 1)
            fulled = np.full(image.shape, color)
            splash = np.where(mask, fulled, gray).astype(np.uint8)
        else:
            splash = gray.astype(np.uint8)
        res.append(splash)
    return res