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