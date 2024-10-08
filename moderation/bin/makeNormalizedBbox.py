#!/usr/bin/python3.12 -W ignore
"""
RUN EXAMPLE:

python3.12 -W ignore makeNormalizedBbox.py -src_image /mnt/sdd1/datasets/via_add_eu_mlines_all/p18159417.jpg \
           -dest_dir /mnt/sdd1/datasets/via_add_eu_mlines_all_boxes -x 72,60,85,95 -y 375,312,319,381
"""
import os
import sys
import argparse
import cv2
w=300
h=100

NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net.tools.via_boxes import VIABoxes

parser = argparse.ArgumentParser(description='Make boxes from via files')
parser.add_argument('-src_image', dest="src_image", required=True, help='Path to source image file', type=str)
parser.add_argument('-dest_dir', dest="dest_dir", required=True, help='Path to destination Bbox dir', type=str)
parser.add_argument('-x', dest="x", required=True, help='Array of x coordinates', type=str)
parser.add_argument('-y', dest="y", required=True, help='Array of y coordinates', type=str)
parser.add_argument('-debug', dest="debug", required=False, help='Debug mode', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

src_image = args.src_image
dest_dir = args.dest_dir
xArr = [int(i) for i in args.x.split(",")]
yArr = [int(i) for i in args.y.split(",")]
keypoints = [[x,y] for x,y in zip(xArr, yArr)]
basename = os.path.basename(src_image).split('.')[0]
debug = args.debug

min_x_box = min(xArr)
min_y_box = min(yArr)
max_x_box = max(xArr)
max_y_box = max(yArr)
bbox = [min_x_box, min_y_box, max_x_box, max_y_box]
bbox_filename = VIABoxes.get_bbox_filename(basename, bbox)
bbox_path = os.path.join(dest_dir, bbox_filename)
image = cv2.imread(src_image)
bbox_image = VIABoxes.get_aligned_image(image, keypoints, shift=0, w=w, h=h)
cv2.imwrite(bbox_path, bbox_image)
