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
parser.add_argument('-checked_dir', dest="checked_dir", required=False, help='Path to checked regions files for update via json file')
parser.add_argument('-debug', dest="debug", required=False, help='Debug mode', default=False,  action=argparse.BooleanOptionalAction)
args = parser.parse_args()

dataset_json = args.dataset_json
checked_dir = args.checked_dir
debug = args.debug

via_boxes = VIABoxes(dataset_json, debug)
via_boxes.fix_checked_via_boxes(checked_dir=checked_dir)

