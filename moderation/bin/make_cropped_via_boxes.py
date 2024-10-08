#!/usr/bin/python3.12 -W ignore
"""
RUN EXAMPLE:

python3.9 -W ignore visualize_via_boxes.py -dataset_json /path/to/via_region_data.json \
                      -target_dir /path/to/target_directory
"""

import os
import sys
import argparse

NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net.tools.via_boxes import VIABoxes

parser = argparse.ArgumentParser(description='Make boxes from via files')
parser.add_argument('-dataset_json', dest="dataset_json", required=True, help='Path to VIA json file')
parser.add_argument('-target_dir', dest="target_dir", required=True, help='Path to generate result boxes images')
parser.add_argument('-target_file', dest="target_file", required=False, help='Path to generate result boxes images', default=None)
parser.add_argument('-debug', dest="debug", required=False, help='Debug mode', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

dataset_json = args.dataset_json
target_dir = args.target_dir
target_file = args.target_file
debug = args.debug

via_boxes = VIABoxes(dataset_json, debug)
via_boxes.make_cropped_via_boxes(target_dir=target_dir,
                                 target_file=target_file,
                                 filtered_classes=[
                                     "numberplate",
                                     "brand_numberplate",
                                     "filled_numberplate",
                                     "empty_numberplate"
                                 ])
# via_boxes.make_transformed_boxes(target_dir=target_dir,
#                                  moderation_bbox_dir=moderation_bbox_dir,
#                                  moderation_image_dir=moderation_image_dir,
#                                  w=224, h=224, min_h=45, min_w=45,)

