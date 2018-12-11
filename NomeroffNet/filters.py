import skimage.io
import numpy as np

def mask(nns):
    res = []
    for nn in nns:
        masks = np.array(nn["masks"])
        for i in np.arange(masks.shape[2]):
            mask = np.array([[w[i] for w in h] for h in nn["masks"]])
            gray = skimage.color.gray2rgb(mask) * 255
            res.append(gray)
    return res

def color_splash(image, nns):
    res = []
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 160
    for nn in nns:
        if nn["masks"].shape[-1] > 0:
            mask = (np.sum(nn["masks"], -1, keepdims=True) >= 1)
            fulled = np.full(image.shape, (127, 0, 127))
            splash = np.where(mask, fulled, gray).astype(np.uint8)
        else:
            splash = gray.astype(np.uint8)
        res.append(splash)
    return res