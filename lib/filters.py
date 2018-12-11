import skimage.io
import numpy as np

def mask(nns):
    masks = []
    for nn in nns:
        mask = np.array([[w[0] for w in h] for h in nn["masks"]])
        gray = skimage.color.gray2rgb(mask) * 255
        masks.append(gray)
    return masks