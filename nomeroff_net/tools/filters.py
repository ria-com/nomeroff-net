import cv2
import skimage.io
import numpy as np
import asyncio

from skimage import img_as_ubyte
from skimage.morphology import convex_hull_image
from skimage.color import gray2rgb
from typing import List, Dict, Tuple, Union


def draw_box(image: np.ndarray, boxs: List[np.ndarray],
             color: Tuple[int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
    for box in boxs:
        # округление координат
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color, thickness)
    return image


def gamma_lut(img: np.ndarray, gamma: float = 0.5) -> None:
    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    _ = cv2.LUT(img, look_up_table)


async def cv_one_img_mask_async(nn: Dict) -> List[np.ndarray]:
    return cv_img_mask([nn])


async def cv_img_mask_async(nns: List[Dict]) -> List:
    loop = asyncio.get_event_loop()
    promises = [loop.create_task(cv_one_img_mask_async(nn)) for nn in nns]
    if bool(promises):
        await asyncio.wait(promises)
    res = []
    for promise in promises:
        res += promise.result()
    return res


def cv_img_mask(nns: List[Dict]) -> List:
    res = []
    for nn in nns:
        masks = np.array(nn["masks"])
        for i in np.arange(masks.shape[2]):
            mask = np.array([[w[i] for w in h] for h in nn["masks"]], dtype=np.uint8)
            chull = np.array(convex_hull_image(mask), dtype=np.uint8)
            gray = gray2rgb(chull) * 255
            res.append(img_as_ubyte(gray))
    return res


def color_splash(image: np.ndarray, masks: List[np.ndarray],
                 color: Union = (0, 255, 0), white_balance: int = 200) -> List[np.ndarray]:
    res = []
    gray = gray2rgb(skimage.color.rgb2gray(image)) * white_balance
    for mask in masks:
        if mask.shape[-1] > 0:
            mask = (np.sum(mask, -1, keepdims=True) >= 1)
            fulled = np.full(image.shape, color)
            splash = np.where(mask, fulled, gray).astype(np.uint8)
        else:
            splash = gray.astype(np.uint8)
        res.append(splash)
    return res


def calc_normalize(hist: Union, reverse: bool = False, min_n: int = 5) -> int:
    level = 0
    iterable = hist
    if reverse:
        iterable = reversed(hist)
    for h in iterable:
        if min_n < h:
            break
        level += 1
    if reverse:
        level = len(hist)-level
    return level


def normalize(img: np.ndarray) -> None:
    cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(cv_img.ravel(), 256, [0, 255])
    alpha = calc_normalize(hist)
    beta = calc_normalize(hist, reverse=True)
    _ = cv2.normalize(cv_img, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
