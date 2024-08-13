import os
import cv2
import shutil
import json
import numpy as np
from typing import List


def split_numberplate(aligned_img: np.ndarray, parts_count: int = 2, overlap_percentage: float = 0.03):
    parts = []
    aligned_h, aligned_w = aligned_img.shape[0:2]
    line_h = round(aligned_h/parts_count)
    overlap = round(aligned_h*overlap_percentage)
    for part in range(parts_count):
        start_h = part*line_h-overlap
        end_h = (part+1)*line_h+overlap
        if start_h < 0:
            start_h = 0
        if start_h > aligned_h:
            start_h = aligned_h
        image_part = aligned_img[start_h:end_h, 0:aligned_w]
        parts.append(image_part)
    return parts


def add_np(fname, zone, region_id, count_line, desc, predicted_text,
           img_dir, ann_dir, replace_template=None):
    if replace_template is None:
        replace_template = {}
    height, width = zone.shape[:2]
    cv2.imwrite(os.path.join(img_dir, f'{fname}.png'), zone)
    data = {
        "description": desc,
        "name": fname,
        "region_id": region_id,
        "count_lines": count_line,
        "size": {
            "width": width,
            "height": height
        },
    }
    data.update(replace_template)
    if "moderation" not in data:
        data["moderation"] = {}
    data["moderation"]["predicted"] = predicted_text
    with open(os.path.join(ann_dir, f'{fname}.json'), "w", encoding='utf8') as jsonWF:
        json.dump(data, jsonWF, ensure_ascii=False)


class NumberplateDatasetItem:

    # default constructor
    def __init__(self,
                 dataset_path: str,
                 orig_filename: str,
                 bbox: List or np.ndarray,
                 keypoints: List,
                 lines: List,
                 region_id: int,
                 zone_bbox: np.ndarray,
                 zone_norm: np.ndarray,
                 ann_subdir: str = 'ann',
                 img_subdir: str = 'img',
                 src_subdir: str = 'src',
                 anb_subdir: str = 'anb',
                 box_subdir: str = 'box',
                 ):
        self.version = 2
        self.dataset_path = dataset_path
        self.orig_filename = orig_filename
        self.bbox = bbox
        self.keypoints = keypoints
        self.lines = lines
        self.region_id = region_id
        self.zone_bbox = zone_bbox
        self.zone_norm = zone_norm

        self.basename, self.orig_ext = os.path.basename(orig_filename).split(".")

        self.ann_subdir = ann_subdir
        self.img_subdir = img_subdir

        self.ann_dir = os.path.join(dataset_path, ann_subdir)
        self.img_dir = os.path.join(dataset_path, img_subdir)
        self.src_dir = os.path.join(dataset_path, src_subdir)
        self.anb_dir = os.path.join(dataset_path, anb_subdir)
        self.box_dir = os.path.join(dataset_path, box_subdir)
        self.img_ext = 'png'
        self.json_ext = 'json'

        self.check_dir(self.ann_dir)
        self.check_dir(self.img_dir)
        self.check_dir(self.src_dir)
        self.check_dir(self.anb_dir)
        self.check_dir(self.box_dir)

    @staticmethod
    def check_dir(path):
        if not os.path.exists(path):
            os.mkdir(path, mode=0o755)

    def get_bbox_basename(self):
        bbox = self.bbox
        basename = self.basename
        return f'{basename}-{int(bbox[0])}x{int(bbox[1])}-{int(bbox[2])}x{int(bbox[3])}'

    def get_src_filename(self):
        return f'{self.basename}.{self.orig_ext}'

    def get_json_filename(self):
        return f'{self.basename}.{self.json_ext}'

    def get_bbox_img(self):
        # bbox = self.bbox
        # x_box = int(min(bbox[0], bbox[2]))
        # w_box = int(abs(bbox[2] - bbox[0]))
        # y_box = int(min(bbox[1], bbox[3]))
        # h_box = int(abs(bbox[3] - bbox[1]))
        # bbox_img = img[y_box:y_box + h_box, x_box:x_box + w_box]
        return self.zone_bbox

    def copy_src(self):
        src_filename = os.path.join(self.src_dir, self.get_src_filename())
        if not os.path.isfile(src_filename):
            shutil.copyfile(self.orig_filename, src_filename)

    def get_bbox_description(self):
        return {
            "bbox": self.bbox.tolist(),
            "keypoints": self.keypoints.tolist(),
            "lines": self.lines,
            "region_id": self.region_id
        }

    def write_bbox_description(self):
        src_json_name = os.path.join(self.anb_dir, self.get_json_filename())
        if os.path.isfile(src_json_name):
            with open(src_json_name, 'r') as f:
                data = json.load(f)
        else:
            data = {
                "src": self.get_src_filename(),
                "version": self.version,
                "regions": {}
            }
        data["regions"][self.get_bbox_basename()] = self.get_bbox_description()
        with open(src_json_name, "w", encoding='utf8') as jsonWF:
            json.dump(data, jsonWF, ensure_ascii=False)

            # a method for printing data members

    def write_orig_dataset(self):
        self.copy_src()
        bbox = self.bbox
        basename = self.get_bbox_basename()
        bbox_filename = f'{basename}.{self.img_ext}'
        bbox_path = os.path.join(self.box_dir, bbox_filename)
        bbox_img = self.get_bbox_img()
        cv2.imwrite(bbox_path, bbox_img)
        self.write_bbox_description()

    # a method for printing data members
    def write_normalize_dataset(self, replace_template=None):
        basename = self.get_bbox_basename()
        parts = split_numberplate(self.zone_norm, len(self.lines))
        idx = 0
        for i, line in self.lines.items():
            if idx == i:
                norm_basename = f'{basename}-line-{i}'
                add_np(norm_basename, parts[i], self.region_id, 1, line, "",
                       self.img_dir, self.ann_dir, replace_template)
            idx += 1
