"""
TODO: remove in future release
"""
import numpy as np
from typing import List


def np_split(imgs: List[np.ndarray], lines: List, coef: float = 0.1) -> List[np.ndarray]:
    res_imgs = []
    for img, line in zip(imgs, lines):
        if line < 2:
            res_imgs.append(img)
        else:
            n = int(img.shape[0]/line + (img.shape[0] * coef))
            first_part = img[:n]
            for num_line in range(1, line - 1):
                middle_part = img[:num_line*n]
                first_part = np.concatenate((first_part, middle_part), axis=1)
            last_part = img[-n:]
            res_imgs.append(np.concatenate((first_part, last_part), axis=1))
    return res_imgs
