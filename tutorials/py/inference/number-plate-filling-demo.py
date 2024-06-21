"""
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python3 number-plate-filling-demo.py
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
nomeroff_net_dir = os.path.join(current_dir, "../../../")
sys.path.append(nomeroff_net_dir)

import glob
import cv2
import argparse

from nomeroff_net import pipeline


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s",
                    "--show",
                    required=False,
                    default=0,
                    type=int,
                    help="Show filled number plate photos")

    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    kwargs = parse_args()
    is_show = kwargs["show"]

    number_plate_filling = pipeline("number_plate_filling", image_loader="opencv")

    root_dir = os.path.join(nomeroff_net_dir, './data/examples/oneline_images/example1.jpeg')
    images = number_plate_filling(glob.glob(root_dir))
    if is_show:
        for img in images:
            img = img[..., ::-1]  # RGB2BGR
            cv2.imshow("Display window", img)
            k = cv2.waitKey(0)
