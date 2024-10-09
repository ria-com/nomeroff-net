"""
## Description:
This script roll four points annd make  another  via dataset

## Usage examples:

To roll four points run:
```
python roll_4_points.py -dataset_json=autoriaNumberplateDataset-2024-08-16/train/via_region_data.json -target_dir=./res
```

"""
import os
import shutil
import sys
import json
import numpy as np
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
NOMEROFF_NET_DIR = os.path.abspath(os.path.join(dir_path, "../../../"))
sys.path.append(NOMEROFF_NET_DIR)
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import (normalize_rect_new,
                                                                                      normalize_rect)


def moderate(dataset_json, target_dir):

    target_json = os.path.join(target_dir, "via_region_data.json")
    dataset_dir = os.path.dirname(dataset_json)
    via_data = {
        "_via_settings": {
            "ui": {
                "annotation_editor_height": 25,
                "annotation_editor_fontsize": 0.8,
                "leftsidebar_width": 18,
                "image_grid": {
                    "img_height": 80,
                    "rshape_fill": "none",
                    "rshape_fill_opacity": 0.3,
                    "rshape_stroke": "yellow",
                    "rshape_stroke_width": 2,
                    "show_region_shape": True,
                    "show_image_policy": "all"
                },
                "image": {
                    "region_label": "",
                    "region_label_font": "10px Sans",
                    "on_image_annotation_editor_placement": "NEAR_REGION"
                }
            },
            "core": {
                "buffer_size": "18",
                "filepath": {
                    "/mrcnn4/": 3
                },
                "default_filepath": "./data"
            },
            "project": {
                "name": "via_data_end"
            }
        },
        "_via_img_metadata": {},
        "_via_attributes": {
            "region": {
                "class": {
                    "type": "text"
                }
            },
            "file": {}
        }
    }
    with open(dataset_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    count_not_matched_normalized_points = 0
    count_not_matched_new_normalized_points = 0

    i = 0
    for key in data['_via_img_metadata']:
        filename = data['_via_img_metadata'][key]["filename"]
        img_file = os.path.join(dataset_dir, filename)
        target_file = os.path.join(target_dir, filename)
        for region in data['_via_img_metadata'][key]["regions"]:
            points = np.array(list(zip(region['shape_attributes']['all_points_x'],
                                       region['shape_attributes']['all_points_y'])))
            normalized_points = normalize_rect(points)
            new_normalized_points = normalize_rect_new(points)
            need_copy = 0
            i += 1
            if not np.array_equal(normalized_points, points):
                count_not_matched_normalized_points += 1
                need_copy = 1
            if not np.array_equal(new_normalized_points, points):
                count_not_matched_new_normalized_points += 1
                need_copy = 1
            if need_copy:
                via_data['_via_img_metadata'][key] = data['_via_img_metadata'][key]
                shutil.copyfile(img_file, target_file)
    print("Count examples", i)
    print("count not matched normalized points", count_not_matched_normalized_points)
    print("count not matched new normalized points", count_not_matched_new_normalized_points)

    with open(target_json, 'w', encoding='utf-8') as file:
        json.dump(via_data, file, indent=4, ensure_ascii=False)
    # print(f"Merged json contains {len(merged_data['_via_img_metadata'])} keys. Saved to file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='This script roll four points and make  another  via dataset ')
    parser.add_argument('-dataset_json', dest="dataset_json", required=True, help='Path to VIA json file')
    parser.add_argument('-target_dir', dest="target_dir", required=True, help='Path to output dir')

    args = parser.parse_args()

    moderate(args.dataset_json, args.target_dir)


if __name__ == "__main__":
    main()
