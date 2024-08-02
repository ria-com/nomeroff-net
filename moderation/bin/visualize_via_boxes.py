#!/usr/bin/python3.9 -W ignore

import os
import sys
import argparse

NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net.tools.via_boxes import VIABoxes

parser = argparse.ArgumentParser(description='Image number plates data markup rebuilder')
parser.add_argument('-dataset_json', dest="dataset_json", required=True, help='Path to VIA json file')
parser.add_argument('-target_dir', dest="target_dir", required=True, help='Path to generate result boxes images')
parser.add_argument('-check_dir', dest="check_dir", required=False, help='Path by which the presence of the target zone is checked when shift is not zero', default=None)
parser.add_argument('-shift', dest="shift", required=False, help='Shift keypoints', default=0, type=int)
parser.add_argument('-debug', dest="debug", required=False, help='Debug mode', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

dataset_json = args.dataset_json
target_dir = args.target_dir
check_dir = args.check_dir
shift = args.shift
debug = args.debug

via_boxes = VIABoxes(dataset_json, debug)
via_boxes.make_transformed_boxes(target_dir, shift, check_dir)
