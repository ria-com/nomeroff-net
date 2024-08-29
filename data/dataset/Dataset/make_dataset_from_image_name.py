#!/usr/bin/env python
# coding: utf-8
"""
python3.9 make_dataset_from_image_name.py
"""

import os
import re
import sys
import cv2
import math
import glob
import warnings
import shutil
import json
import numpy as np
from typing import List
from ultralytics import YOLO
from matplotlib import pyplot as plt
import easyocr

# add noneroff_net path
dir_path = os.path.dirname(os.path.realpath(__file__))
NOMEROFF_NET_DIR = os.path.abspath(os.path.join(dir_path, "../../../"))
sys.path.append(NOMEROFF_NET_DIR)


from nomeroff_net.tools.mcm import get_device_torch
from nomeroff_net.pipes.number_plate_classificators.options_detector import OptionsDetector
from nomeroff_net.tools.image_processing import distance
from nomeroff_net.pipes.number_plate_text_readers.text_postprocessing import translit_cyrillic_to_latin
from upscaler import HAT
device_torch = get_device_torch()


classifiactor = OptionsDetector()
_ = classifiactor.load("latest")

from nomeroff_net.tools import unzip
from nomeroff_net.tools.mcm import modelhub
from nomeroff_net.pipelines.number_plate_text_reading import NumberPlateTextReading
model_info = modelhub.download_model_by_name('yolov8x')

# Load last model
model = YOLO(model_info['path'])  # load a custom model


plt.rcParams["figure.figsize"] = (10, 5)


def addCoordinatesOffset(points: List or np.ndarray, x: float, y: float) -> List:
    """
    TODO: describe function
    """
    return [[point[0] + x, point[1] + y] for point in points]


def applyCoefficient(points: List or np.ndarray, coef_w: float, coef_h: float) -> List:
    """
    TODO: resize points coordinates
    """
    return [[point[0] * coef_w, point[1] * coef_h] for point in points]


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


def remove_bad_text_zones(easyocr_arr, exclude_zones_list = ['UA']):
    result = []
    for item in easyocr_arr:
        if item[1].upper() not in exclude_zones_list:
            result.append(item)
    return result


def remove_small_zones(easyocr_arr, delete_threshold = 0.4):
    result = []
    if not len(easyocr_arr):
        return result
    dy_arr = [{'dy': distance(item[0][1], item[0][2]), 'idx': idx } for idx, item in enumerate(easyocr_arr)]
    max_dy = max(item["dy"] for item in dy_arr)
    dy_arr = filter(lambda x: x["dy"]/max_dy>=delete_threshold, dy_arr)
    dy_idx = [item['idx'] for item in dy_arr]
    for idx, item in enumerate(easyocr_arr):
        if idx in dy_idx:
            result.append(item)
    return result


def append_text_to_line(easyocr_arr, img, count_lines):
    dimensions = {}
    lines = {}
    lines_text = {}
    h,w = img.shape[:2]
    part_y = h/count_lines
    for idx, item in enumerate(easyocr_arr):
        min_x = min(point[0] for point in item[0])
        min_y = min(point[1] for point in item[0])
        max_y = max(point[1] for point in item[0])
        center_y = round(min_y + (max_y-min_y)/2)
        dimensions[idx] = {'dx': distance(item[0][0], item[0][1]), 'dy': distance(item[0][1], item[0][2]), 'center_y': center_y, 'min_x': min_x, 'idx': idx }
        line = math.floor(center_y/part_y)
        if line not in lines:
            lines[line] = []
        lines[line].append(dimensions[idx])
    for line in lines:
        sorted_arr = sorted(lines[line], key=lambda x: x['min_x'])
        lines_text[line] = ''.join([easyocr_arr[item['idx']][1] for item in sorted_arr])
    return lines_text


def get_easyocr_lines(easyocr_arr, img, count_lines, exclude_zones_list=None):
    if exclude_zones_list is None:
        exclude_zones_list = ['UA']
    if len(easyocr_arr) > 0:
        cleared_arr = remove_bad_text_zones(easyocr_arr, exclude_zones_list)
        if len(cleared_arr) > 0:
            cleared_arr = remove_small_zones(cleared_arr)
            lines_text = append_text_to_line(cleared_arr, img, count_lines)
        else:
            lines_text = {}
    else:
        lines_text = {}
    return lines_text


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
    if translit_cyrillic_to_latin(desc) == translit_cyrillic_to_latin(predicted_text):
        data["moderation"]["isModerated"] = 1
        data["moderation"]["moderatedBy"] = "auto"
    data["moderation"]["predicted"] = predicted_text
    with open(os.path.join(ann_dir, f'{fname}.json'), "w", encoding='utf8') as jsonWF:
        json.dump(data, jsonWF, ensure_ascii=False)


class NumberplateDatasetItem:

    # default constructor
    def __init__(self, 
                 numberplate_lines: List,
                 photo_id: str, 
                 numberplate: str,
                 dataset_path: str,
                 orig_filename: str,
                 bbox:List or np.ndarray,
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
        self.numberplate_lines = numberplate_lines
        self.photo_id = photo_id
        self.numberplate = numberplate
        self.dataset_path = dataset_path
        self.orig_filename = orig_filename
        self.bbox = bbox
        self.keypoints = keypoints
        self.lines = lines
        self.region_id = region_id
        self.zone_bbox = zone_bbox
        self.zone_norm = zone_norm
        
        basename_splits = os.path.basename(orig_filename).split(".")
        self.basename = ".".join(basename_splits[:-1])
        self.orig_ext = basename_splits[-1]
        if photo_id is not None:
            self.basename = photo_id
        
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
            os.makedirs(path, exist_ok=True, mode = 0o755)

    def get_bbox_basename(self):
        bbox = self.bbox
        basename = self.basename
        return f'{basename}-{int(bbox[0])}x{int(bbox[1])}-{int(bbox[2])}x{int(bbox[3])}'

    def get_src_filename(self):
        return f'{self.basename}.{self.orig_ext}'

    def get_json_filename(self):
        return f'{self.basename}.{self.json_ext}'

    def get_bbox_img(self):
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
        else :
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
        for (i, line), npline in zip(self.lines.items(), self.numberplate_lines):
            norm_basename = f'{basename}-line-{i}'
            add_np(norm_basename, parts[i], self.region_id, 1, line, npline, 
                   self.img_dir, self.ann_dir, replace_template)


# In[15]:


def fix_text_line(str):
    return str.replace(" ", "").replace("-", "").replace("|", "I").replace("0", "O").replace("/", "I")


def fix_number_line(str):
    return str.replace(" ", "").replace("-", "").replace("O", "0").replace("I", "1")


def fix_lines(orig_lines, lines, region_id):
    lines = lines.values()
    if len(orig_lines) != len(lines):
        return {i: l for i, l in enumerate(lines)}
    new_lines = []
    for ol, l in zip(orig_lines, lines):
        ol = (ol.replace(" ", "").replace("-", "").replace(".", "").replace(",", "").
              replace("'", "").replace('"', "").replace("`", "").replace("*", "").replace("[", "Г").upper())
        l = (l.replace(" ", "").replace("-", "").replace(".", "").replace(",", "").
             replace("'", "").replace('"', "").replace("`", "").replace("*", "").replace("[", "Г").upper())
        if len(ol) != len(l):
            new_lines.append(l)
            continue
        new_line = ""
        for letter_ol, letter_l in zip(ol, l):
            if letter_ol == letter_l:
                new_line += letter_l
            else:
                if letter_ol == "1" and letter_l in ("I", "І", "|", "/", "\\"):
                    new_line += letter_ol
                elif letter_ol in ("I", 'І') and letter_l in ("1", "|", "/", "\\"):
                    new_line += letter_ol
                elif letter_ol in ("O", "О") and letter_l == "0":
                    new_line += letter_ol
                elif letter_ol == "0" and letter_l in ("O", "О"):
                    new_line += "0"
                else:
                    new_line += letter_l
        new_lines.append(new_line)
    return {i: l for i, l in enumerate(new_lines)}


def normalize_easyocr_output(result):
    new_result = []
    for item in result:
        new_item = (
            item[0],
            item[1].upper().replace('-', '').replace(' ', '').replace('.', '').replace(',', ''),
            item[2]
        )
        new_result.append(new_item)
    return new_result


def format_moldovan_plate(plate):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = plate.replace(" ", "").upper()
    
    # Знаходимо всі літери та цифри
    letters = re.findall(r'[A-Z]', plate)
    digits = re.findall(r'\d', plate)
    
    # Якщо літери в кінці, переставляємо їх на початок
    return ''.join(digits+letters), [''.join(digits), ''.join(letters)]


# In[18]:


def format_default_plate(plate):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = plate.upper()
    _plate_lines = plate.split(" ")
    if len(_plate_lines) != 2:
        warnings.warn(f"!!![WRONG COUNT LINES]!!! {plate} = {_plate_lines}")
    return plate.replace(" ", ""), _plate_lines


# In[29]:


def format_kz_plate(plate):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = plate.upper()
    _plate_lines = plate.split(" ")
    if len(_plate_lines) == 2:
        return plate.replace(" ", ""), _plate_lines
    elif len(_plate_lines) == 3:
        plate = _plate_lines[0] + _plate_lines[2] + _plate_lines[1]
        _plate_lines = [_plate_lines[0], _plate_lines[2] + _plate_lines[1]]
        return plate.replace(" ", ""), _plate_lines
    else:
        warnings.warn(f"!!![WRONG COUNT LINES]!!! {plate} = {_plate_lines}")
        return plate.replace(" ", ""), _plate_lines
    


# In[30]:


def format_ro_plate(plate):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = plate.upper()
    _plate_lines = plate.split(" ")
    if len(_plate_lines) == 2:
        return plate.replace(" ", ""), _plate_lines
    elif len(_plate_lines) == 3:
        _plate_lines = [_plate_lines[0] + _plate_lines[1], _plate_lines[2]]
        return plate.replace(" ", ""), _plate_lines
    else:
        warnings.warn(f"!!![WRONG COUNT LINES]!!! {plate} = {_plate_lines}")
        return plate.replace(" ", ""), _plate_lines
    

class EasyOCRReader:
    def __init__(self, easyocr_readers=None, exclude_zones_list=None):
        if exclude_zones_list is None:
            exclude_zones_list = []
        if easyocr_readers is None:
            easyocr_readers = ['en']
        self.reader = easyocr.Reader(easyocr_readers)
        self.exclude_zones_list = exclude_zones_list

    def predict(self, img, count_lines, flag_show=False):
        # Display the aligned and cropped image
        result = self.reader.readtext(img)
        result = normalize_easyocr_output(result)

        easyocr_lines = get_easyocr_lines(result, img, count_lines[0],
                                          exclude_zones_list=self.exclude_zones_list)
        if count_lines[0] > len(easyocr_lines):
            count_lines[0] = len(easyocr_lines)
            easyocr_lines = get_easyocr_lines(result, img, count_lines[0],

                                              exclude_zones_list=self.exclude_zones_list)

        if count_lines[0] > 1:
            parts = split_numberplate(img, parts_count=count_lines[0])
            for a_img_part in parts:
                if flag_show:
                    plt.imshow(a_img_part)
                    plt.show()
        return easyocr_lines


class NomeroffNetReader:
    def __init__(self, presets=None):
        if presets is None:
            presets = {
                "eu_2lines_efficientnet_b2": {
                    "for_regions": ["eu_ua_2015", "eu_ua_2004", "eu_ua_1995", "eu_ua_custom", "xx_transit",
                                    "eu", "xx_unknown", "ru", "eu_ua_ordlo_lpr", "eu_ua_ordlo_dpr", "kz",
                                    "kg", "ge", "su", "am", "by"],
                    "for_count_lines": [2, 3],
                    "model_path": "latest"
                },
            }
        self.number_plate_text_reading = NumberPlateTextReading(
            "number_plate_text_reading",
            image_loader=None,
            presets=presets,
            default_label="eu",
            default_lines_count=2,
            multiline_splitter=" ",
        )

    def predict(self, img, count_lines, flag_show=False):
        number_plate_text_reading_res = unzip(
            self.number_plate_text_reading(unzip([[img],
                                                  ["eu"],
                                                  count_lines, [img]])))
        if len(number_plate_text_reading_res):
            texts, _ = number_plate_text_reading_res
            return {i: _np for i, _np in enumerate(texts[0].strip().split(" "))}
        return {}

# In[38]:


def format_al_plate(plate):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = plate.replace(" ", "").upper()

    # Визначаємо шаблони для різних форматів
    patterns = [
        r'^([A-Z]{2})(\d{3})$',  # @@###
        r'^([A-Z]{2})(\d{3}[A-Z]{2})$',  # @@###@@
        r'^([A-Z]{2})(\d{2}[A-Z]{2})$',  # @@##@@
        r'^([A-Z]{2})(\d{4}[A-Z])$'  # @@####@
    ]

    for pattern in patterns:
        match = re.match(pattern, plate)
        if match:
            return plate, list(match.groups())

    # Якщо номер не відповідає жодному з форматів
    warnings.warn(f"!!![НЕПРАВИЛЬНИЙ ФОРМАТ]!!! {plate}")
    return plate, [plate]


def format_at_plate(plate):
    # Видаляємо всі пробіли та переводимо у верхній регістр
    plate = re.sub(r'\s+', '', plate.upper())

    # Визначаємо паттерни для різних форматів
    patterns = [
        (r'^([A-Z]{2})(\d{3}[A-Z]{2})$', lambda m: (m.group(1), m.group(2))),  # @@###@@
        (r'^([A-Z]{2})(\d{2}[A-Z]{2})$', lambda m: (m.group(1), m.group(2))),  # @@##@@
        (r'^([A-Z]{2})(\d{5})$', lambda m: (m.group(1), m.group(2))),  # @@#####
        (r'^([A-Z]{2})([A-Z]{2}\d{2})$', lambda m: (m.group(1), m.group(2))),  # @@@@##
        (r'^([A-Z])(\d[A-Z]{3})$', lambda m: (m.group(1), m.group(2))),  # @#@@@
        (r'^([A-Z]{2})(\d[A-Z]{3})$', lambda m: (m.group(1), m.group(2))),  # @@#@@@
        (r'^([A-Z]\d)(\d{3}[A-Z])$', lambda m: (m.group(1), m.group(2))),  # @####@
        (r'^([A-Z]\d)(\d{4}[A-Z])$', lambda m: (m.group(1), m.group(2))),  # @#####@
        (r'^(\d{3})([A-Z]{3})$', lambda m: (m.group(1), m.group(2))),  # ###@@@
        (r'^([A-Z]\d{2})([A-Z]{3})$', lambda m: (m.group(1), m.group(2))),  # @##@@@
        (r'^([A-Z]{2})([A-Z]{2}\d)$', lambda m: (m.group(1), m.group(2))),  # @@@@#
        (r'^(\d{4})(\d{3})$', lambda m: (m.group(1), m.group(2))),  # #######
        (r'^([A-Z]\d{3})(\d{3})$', lambda m: (m.group(1), m.group(2))),  # @######
        (r'^([A-Z]\d{2})(\d{3})$', lambda m: (m.group(1), m.group(2))),  # @#####
        (r'^(\d{2})(\d{3}[A-Z])$', lambda m: (m.group(1), m.group(2))),  # #####@
        (r'^([A-Z])(\d{3}[A-Z])$', lambda m: (m.group(1), m.group(2))),  # @###@

    ]

    for pattern, formatter in patterns:
        match = re.match(pattern, plate)
        if match:
            formatted = formatter(match)
            return plate, list(formatted)

    warnings.warn(f"!!![INVALID FORMAT]!!! {plate}")
    return plate, [plate]


fromats_parse = {
    "md": format_moldovan_plate,
    "kz": format_kz_plate,
    'ro': format_ro_plate,
    "default": format_default_plate,
    "fi": format_default_plate,
    "al": format_al_plate,
    "at": format_at_plate,
}


# In[44]:


def create_dataset(img_dir="/mnt/datasets/nomeroff-net/2lines_np_parsed/md/*/*",
                   target_dataset="/mnt/datasets/nomeroff-net/2lines_np_parsed/mlines_md_dataset",
                   parse_fromat="md", flag_show=False,
                   reader=None, need_upscale_image=False
                   ):
    if need_upscale_image:
        up = HAT(tile_size=320, num_gpu=int(device_torch == "cuda"))
    if reader is None:
        reader = EasyOCRReader()

    for img_path in glob.glob(img_dir):
        print("====>IMAGE:", img_path)
        if parse_fromat == "fi":
            try:
                photo_id, _, _, numberplate_part1, numberplate_part2, *_ = os.path.basename(img_path).split("-")
                numberplate = f"{numberplate_part1} {numberplate_part2}"
            except Exception as e:
                warnings.warn(f"NO numberplate in filename {img_path}")
                photo_id, *_ = os.path.basename(img_path).split("-")
                numberplate = ""
        else:
            photo_id, _, _, numberplate, *_ = os.path.basename(img_path).split("-")
        photo_id = "p"+photo_id
        numberplate, numberplate_lines = fromats_parse[parse_fromat](numberplate)
        
        # Predict with the model
        results = model(img_path)  # predict on an image
        
        # Load the image using OpenCV
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]       
    
        # Loop over the results
        for result in results:
            if not len(result.boxes):
                warnings.warn("result.boxes is empty")
                continue
            # Extract keypoints and bounding boxes
            array_of_keypoints = result.keypoints.cpu().xy
            array_of_boxes = result.boxes.xyxy.cpu()
            for keypoints, bbox in zip(array_of_keypoints, array_of_boxes):
                if not ((bbox[0] == 0) or (bbox[2] >= img_w-1)):
                    x_box = int(min(bbox[0], bbox[2]))
                    w_box = int(abs(bbox[2] - bbox[0]))
                    y_box = int(min(bbox[1], bbox[3]))
                    h_box = int(abs(bbox[3] - bbox[1]))

                    image_part = img[y_box:y_box + h_box, x_box:x_box + w_box]

                    try:
                        if need_upscale_image:
                            image_part_upscale = up.run(cv2.cvtColor(image_part, cv2.COLOR_BGR2RGB))
                        else:
                            image_part_upscale = cv2.cvtColor(image_part, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        warnings.warn(f"FAILED UPSCALER {e}")
                        image_part_upscale = cv2.cvtColor(image_part, cv2.COLOR_BGR2RGB)

                    if flag_show:
                        plt.imshow(image_part_upscale)
                        plt.show()

                    # Calculation of the scaling factor coefficient
                    image_part_h, image_part_w, _ = image_part_upscale.shape
                    coef_h = h_box/image_part_h
                    coef_w = w_box/image_part_w

                    localKeypoints = addCoordinatesOffset(keypoints, -x_box, -y_box)
                    localKeypoints_upscale = applyCoefficient(localKeypoints, 1/coef_w, 1/coef_h)

                    h = 100
                    w = 400
                    target_points = np.float32(np.array([[0, h], [0, 0], [w, 0], [w, h]]))

                    # Convert keypoints to numpy array
                    src_points = np.array(localKeypoints_upscale, dtype="float32")

                    # Compute the perspective transform matrix
                    M = cv2.getPerspectiveTransform(src_points, target_points)

                    # Apply the perspective transformation to the image
                    aligned_img = cv2.warpPerspective(image_part_upscale, M, (w, h))
                    region_ids, count_lines, confidences, predicted = classifiactor.predict_with_confidence([aligned_img])

                    # Тут далі можна шось робити
                    if count_lines[0] == 2:
                        # w = 200
                        h = 300
                        target_points = np.float32(np.array([[0, h], [0, 0], [w, 0], [w, h]]))
                        # Compute the perspective transform matrix
                        M = cv2.getPerspectiveTransform(src_points, target_points)

                        # Apply the perspective transformation to the image
                        aligned_img = cv2.warpPerspective(image_part_upscale, M, (w, h))
                    if count_lines[0] == 3:
                        h = 300
                        target_points = np.float32(np.array([[0, h], [0, 0], [w, 0], [w, h]]))
                        # Compute the perspective transform matrix
                        M = cv2.getPerspectiveTransform(src_points, target_points)

                        # Apply the perspective transformation to the image
                        aligned_img = cv2.warpPerspective(image_part_upscale, M, (w, h))

                    predicted_lines = reader.predict(aligned_img, count_lines)

                    if count_lines[0] > 1:
                        # Make dataset
                        numberplate_dataset_item = NumberplateDatasetItem(numberplate_lines, photo_id, numberplate,
                                                                          target_dataset, img_path, bbox, keypoints,
                                                                          fix_lines(numberplate_lines,
                                                                                    predicted_lines, region_ids[0]),
                                                                          region_ids[0], image_part,
                                                                          cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR))
                        numberplate_dataset_item.write_orig_dataset()
                        numberplate_dataset_item.write_normalize_dataset()
                    else:
                        warnings.warn("count_lines <= 1")

                    if flag_show:
                        plt.imshow(aligned_img)
                        plt.show()


                    for i, point in enumerate(keypoints):
                        x, y = point
                        # Малюємо точку
                        cv2.circle(img, (int(x), int(y)), int(img.shape[0]/100), (255, 0, 0), -1)
                        # Виводимо номер точки
                        cv2.putText(img, str(i+1), (int(x)+10, int(y)+10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)


            if flag_show:
                # Draw bounding box
                for bbox in array_of_boxes:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        if flag_show:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()


if __name__ == "__main__":
    # create_dataset(img_dir="/mnt/datasets/nomeroff-net/2lines_np_parsed/fi/*/*",
    #                target_dataset="/mnt/datasets/nomeroff-net/2lines_np_parsed/mlines_fi_dataset",
    #                parse_fromat="fi",
    #                reader=EasyOCRReader(easyocr_readers=["en"], exclude_zones_list=["FIN"]))
    #
    # create_dataset(img_dir="/mnt/datasets/nomeroff-net/2lines_np_parsed/md/*/*",
    #                target_dataset="/mnt/datasets/nomeroff-net/2lines_np_parsed/mlines_md_dataset",
    #                parse_fromat="md",
    #                reader=EasyOCRReader(easyocr_readers=["en"], exclude_zones_list=["MD"]))
    #
    # create_dataset(img_dir="/mnt/datasets/nomeroff-net/2lines_np_parsed/pl/*/*",
    #                target_dataset="/mnt/datasets/nomeroff-net/2lines_np_parsed/mlines_pl_dataset",
    #                parse_fromat="default",
    #                reader=EasyOCRReader(easyocr_readers=["en"], exclude_zones_list=["PL"]))
    #
    # create_dataset(img_dir="/mnt/datasets/nomeroff-net/2lines_np_parsed/by/*/*",
    #                target_dataset="/mnt/datasets/nomeroff-net/2lines_np_parsed/mlines_by_dataset",
    #                parse_fromat="default",
    #                reader=EasyOCRReader(easyocr_readers=["en"], exclude_zones_list=[]))
    #
    # create_dataset(img_dir="/mnt/datasets/nomeroff-net/2lines_np_parsed/kz/*/*",
    #                target_dataset="/mnt/datasets/nomeroff-net/2lines_np_parsed/mlines_kz_dataset",
    #                parse_fromat="kz",
    #                reader=EasyOCRReader(easyocr_readers=["en"], exclude_zones_list=["KZ"]))
    #
    # create_dataset(img_dir="/mnt/datasets/nomeroff-net/2lines_np_parsed/ro/*/*",
    #                target_dataset="/mnt/datasets/nomeroff-net/2lines_np_parsed/mlines_ro_dataset",
    #                parse_fromat="ro",
    #                reader=EasyOCRReader(easyocr_readers=["en"], exclude_zones_list=["RO"]))
    #
    # create_dataset(img_dir="/mnt/datasets/nomeroff-net/2lines_np_parsed/lv/*/*",
    #                target_dataset="/mnt/datasets/nomeroff-net/2lines_np_parsed/mlines_lv_dataset",
    #                parse_fromat="default",
    #                reader=EasyOCRReader(easyocr_readers=["en"], exclude_zones_list=["LV"]))
    #
    # create_dataset(img_dir="/mnt/datasets/nomeroff-net/2lines_np_parsed/lt/*/*",
    #                target_dataset="/mnt/datasets/nomeroff-net/2lines_np_parsed/mlines_lt_dataset",
    #                parse_fromat="default",
    #                reader=EasyOCRReader(easyocr_readers=["en"], exclude_zones_list=["LT"]))

    # create_dataset(img_dir="/var/www/projects_computer_vision/nomeroff-net/data/dataset/Dataset/src_test_platesmania/*",
    #                target_dataset="/var/www/projects_computer_vision/nomeroff-net/data/dataset/Dataset"
    #                               "/src_test_platesmania_dataset",
    #                parse_fromat="default",
    #                reader=EasyOCRReader(easyocr_readers=["ru", "uk"], exclude_zones_list=[]))



    # create_dataset(img_dir="/var/www/projects_computer_vision/nomeroff-net/data/dataset/Dataset/test/al/*/*",
    #                target_dataset="/var/www/projects_computer_vision/nomeroff-net/data/dataset/Dataset/test_dataset/al",
    #                parse_fromat="al",
    #                reader=NomeroffNetReader(),
    #                )

    create_dataset(img_dir="/var/www/projects_computer_vision/nomeroff-net/data/dataset/Dataset/test/at/*/*",
                   target_dataset="/var/www/projects_computer_vision/nomeroff-net/data/dataset/Dataset/test_dataset/at",
                   parse_fromat="at",
                   reader=NomeroffNetReader(),
                   )

