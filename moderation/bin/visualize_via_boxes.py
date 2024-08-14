#!/usr/bin/python3.9 -W ignore
"""
RUN EXAMPLE:

python3.9 -W ignore visualize_via_boxes.py -dataset_json /path/to/via_region_data.json \
                      -target_dir /path/to/target_directory \
                      -moderation_bbox_dir /path/to/moderation_bbox_directory \
                      -moderation_image_dir /path/to/moderation_image_directory \
                      -debug

python3.9 -W ignore visualize_via_boxes.py \
                      -dataset_json /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06/val/via_region_data.json \
                      -target_dir /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val \
                      -moderation_bbox_dir /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_bbox_moderate \
                      -moderation_image_dir /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_moderate


python3.9 -W ignore visualize_via_boxes.py \
                      -dataset_json /var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_blurred/via_region_data.json \
                      -target_dir /var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_blurred/val \
                      -moderation_bbox_dir /var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_blurred/val_bbox_moderate \
                      -moderation_image_dir /var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_blurred/val_moderate


python3.9 -W ignore visualize_via_boxes.py \
                      -dataset_json /var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_wrong_rotation/via_region_data.json \
                      -target_dir /var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_wrong_rotation/val \
                      -moderation_bbox_dir /var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_wrong_rotation/val_bbox_moderate \
                      -moderation_image_dir /var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_wrong_rotation/val_moderate


python3.9 -W ignore visualize_via_boxes.py \
                      -dataset_json /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06/train/via_region_data.json \
                      -target_dir /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/train \
                      -moderation_bbox_dir /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/train_bbox_moderate \
                      -moderation_image_dir /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/train_moderate

python3.9 -W ignore visualize_via_boxes.py \
                      -dataset_json /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06/added/via_region_data.json \
                      -target_dir /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/added_train \
                      -moderation_bbox_dir /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/added_train_bbox_moderate \
                      -moderation_image_dir /mnt/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/added_train_moderate

"""

import os
import sys
import argparse

NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net.tools.via_boxes import VIABoxes

parser = argparse.ArgumentParser(description='Image number plates data markup rebuilder')
parser.add_argument('-dataset_json', dest="dataset_json", required=True, help='Path to VIA json file')
parser.add_argument('-target_dir', dest="target_dir", required=True,
                    help='Path to generate result boxes images')
parser.add_argument('-moderation_bbox_dir', dest="moderation_bbox_dir", required=True,
                    help='Path to generate result boxes images')
parser.add_argument('-moderation_image_dir', dest="moderation_image_dir", required=True,
                    help='Path to generate result boxes images')
parser.add_argument('-debug', dest="debug", required=False, help='Debug mode', default=False,
                    action=argparse.BooleanOptionalAction)
args = parser.parse_args()

dataset_json = args.dataset_json
target_dir = args.target_dir
moderation_bbox_dir = args.moderation_bbox_dir
moderation_image_dir = args.moderation_image_dir
debug = args.debug

via_boxes = VIABoxes(dataset_json, debug)
via_boxes.make_transformed_boxes(target_dir=target_dir,
                                 moderation_bbox_dir=moderation_bbox_dir,
                                 moderation_image_dir=moderation_image_dir,)
# via_boxes.make_transformed_boxes(target_dir=target_dir,
#                                  moderation_bbox_dir=moderation_bbox_dir,
#                                  moderation_image_dir=moderation_image_dir,
#                                  w=224, h=224, min_h=45, min_w=45,)

