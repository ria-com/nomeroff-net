#!/usr/bin/env python
# coding: utf-8

# In[18]:


import sys
import os
import json
import ujson
import yaml
import shutil
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../../../../')
sys.path.append(NOMEROFF_NET_DIR)


# In[19]:


# Формат для однієї анотації:
# class_id x_center y_center width height keypoint1_x keypoint1_y keypoint2_x keypoint2_y keypoint3_x keypoint3_y keypoint4_x keypoint4_y


# In[20]:


# auto download latest dataset
from nomeroff_net.tools import modelhub
from nomeroff_net.tools.image_processing import normalize_img, convert_cv_zones_rgb_to_bgr
from nomeroff_net.tools.image_processing import (fline,
                                                 distance,
                                                 linear_line_matrix,
                                                 get_y_by_matrix,
                                                 find_distances,
                                                 fix_clockwise2,
                                                 find_min_x_idx,
                                                 detect_intersection,
                                                 reshape_points)

# auto download latest dataset
info = modelhub.download_dataset_for_model("yolov8")
PATH_TO_DATASET = info["dataset_path"]

# local path dataset
#PATH_TO_DATASET = os.path.join(NOMEROFF_NET_DIR, "./data/dataset/Detector/autoria_numberplate_dataset_example")


# In[21]:


def normalize_rect(rect):
    """
    TODO: describe function
    """
    rect = fix_clockwise2(rect)
    min_x_idx = find_min_x_idx(rect)
    rect = reshape_points(rect, min_x_idx)
    # print("Start rect")
    # print(rect)
    coef_ccw = fline(rect[0], rect[3])
    angle_ccw = round(coef_ccw[2], 2)
    d_bottom = distance(rect[0], rect[3])
    d_left = distance(rect[0], rect[1])
    k = d_bottom / d_left
    if not round(rect[0][0], 4) == round(rect[1][0], 4):
        if d_bottom < d_left:
            k = d_left / d_bottom
            #print("d_bottom < d_left")
            #print("k", k, angle_ccw)
            if k > 1.5 or angle_ccw > 45:
                rect = reshape_points(rect, 3)
        else:
            # print("d_bottom >= d_left")
            # print("k", k, angle_ccw)
            primary_diag = distance(rect[0], rect[2])
            secondary_diag = distance(rect[1], rect[3])
            # print("primary_diag",round(primary_diag,2))
            # print("secondary_diag", round(secondary_diag,2))
            if k < 1.5 and (angle_ccw > 45) and (primary_diag>secondary_diag):
                rect = reshape_points(rect, 3)
    return rect


# In[22]:


PATH_TO_DATASET


# In[23]:


def rotate_image_by_exif(image):
    """
    Rotate photo

    Parameters
    ----------
    image
    """
    try:
        orientation = 274  # key of orientation ExifTags
        if image._getexif() is not None:
            exif = dict(image._getexif().items())
            if orientation in exif.keys():
                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                    image = ImageOps.mirror(image)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                    image = ImageOps.mirror(image)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
                    image = ImageOps.mirror(image)
    except AttributeError:
        pass
    return image


# In[24]:


import copy
import os
import json
import cv2
import torch
import numpy as np
import random
from glob import glob
from tqdm import tqdm
from typing import List, Tuple, Generator
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from nomeroff_net.tools.image_processing import normalize_img


# In[29]:


PATH_TO_DATASET = "/var/www/nomeroff-net/data/dataset/Detector/yolov8/autoriaNumberplateDataset-2023-03-06"


# # Val

# In[30]:


DEBUG = 0


res_dataset = "/mnt/store/var/www/nomeroff-net/nomeroff_net/tools/../../data/./dataset/Detector/yolov8/keypoints_yolo/labels/val"
os.makedirs(res_dataset, exist_ok=True)
path_to_images = os.path.join(PATH_TO_DATASET, "val")

ann_filepath = os.path.join(path_to_images, 'via_region_data.json')
with open(ann_filepath) as ann:
    ann_data = json.load(ann)
image_list = ann_data

img_id = 0
corupted_images = []
for _id, annotation in tqdm(image_list["_via_img_metadata"].items()):
    regions = annotation['regions']
    img_filename = annotation['filename']
    img_base, _ = os.path.splitext(img_filename)
    yolo_annotations = []

    image_id = image_list["_via_img_metadata"][_id]["filename"]
    filename = f'{path_to_images}/{image_id}'
    pil_image = Image.open(filename)
    if image_id == "369353060-28729225.jpeg":
        print("img", image_id, pil_image)
    
    
    pil_image = rotate_image_by_exif(pil_image)
    image = np.array(pil_image)
    h, w, c = image.shape
    target_boxes = []
    labels = []
    for region in image_list["_via_img_metadata"][_id]["regions"]:
        shape_attrs = region['shape_attributes']
        all_points_x = shape_attrs['all_points_x']
        all_points_y = shape_attrs['all_points_y']
        if region["shape_attributes"].get("all_points_x", None) is None or len(region["shape_attributes"]["all_points_x"]) != 4:
            corupted_images.append(_id)
            continue
        if c == 3:
            bbox = [
                int(min(region["shape_attributes"]["all_points_x"])),
                int(min(region["shape_attributes"]["all_points_y"])),
                int(max(region["shape_attributes"]["all_points_x"])),
                int(max(region["shape_attributes"]["all_points_y"])),
            ]
            x_min = min(all_points_x)
            x_max = max(all_points_x)
            y_min = min(all_points_y)
            y_max = max(all_points_y)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            x_center = x_min + bbox_width / 2
            y_center = y_min + bbox_height / 2
            
            roi_img = image[bbox[1]:bbox[3], bbox[0]: bbox[2]]
            xs = np.array([_item-bbox[0] for _item in region["shape_attributes"]["all_points_x"]])
            ys = np.array([_item-bbox[1] for _item in region["shape_attributes"]["all_points_y"]])
            
            rect = normalize_rect(list(zip(xs, ys)))
            xs = [item[0]+x_min for item in rect]
            ys = [item[1]+y_min for item in rect]
            
            if roi_img.shape[0] and roi_img.shape[1]:
                if DEBUG:
                    # Відображення зображення
                    plt.imshow(roi_img)
                    # Нанесення ключових точок
                    plt.scatter(xs, ys, c='red')
                    # Підписання ключових точок
                    for i in range(4):
                        plt.text(xs[i], ys[i], str(i+1), 
                                 fontsize=12, color='blue')
                    # Показати результат
                    plt.show()
                keypoints = []
                for x, y in zip(xs, ys):
                    keypoints.extend([x/w, y/h])
                # Формат анотацій YOLO
                class_id = 0  # Припустимо, що клас номерного знаку має індекс 0
                yolo_annotation = [class_id, x_center/w, y_center/h, bbox_width/w, bbox_height/h] + keypoints
                yolo_annotations.append(yolo_annotation)
            else:
                corupted_images.append(_id)
        else:
            corupted_images.append(_id)
    # Записати анотації в файл
    yolo_annotation_str = "\n".join([" ".join(map(str, anno)) for anno in yolo_annotations])
    output_path = os.path.join(os.path.join(res_dataset), f"{img_base}.txt")
    with open(output_path, "w") as f:
        f.write(yolo_annotation_str)


# In[31]:


corupted_images


# In[32]:


res_dataset = "/mnt/store/var/www/nomeroff-net/nomeroff_net/tools/../../data/./dataset/Detector/yolov8/keypoints_yolo/labels/val"
os.makedirs(res_dataset, exist_ok=True)
path_to_images = os.path.join(PATH_TO_DATASET, "val")

ann_filepath = os.path.join(path_to_images, 'via_region_data.json')
with open(ann_filepath) as ann:
    ann_data = json.load(ann)
image_list = ann_data

img_id = 0
tmp = list(image_list["_via_img_metadata"].items())
for _id, annotation in tqdm(tmp):
    if _id not in corupted_images:
        del image_list["_via_img_metadata"][_id]


# In[33]:


len(image_list["_via_img_metadata"])


# In[34]:


with open("corrupted_via_region_data_val.json", "w") as ann:
    json.dump(image_list, ann)


# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:


DEBUG = 0


res_dataset = "/mnt/store/var/www/nomeroff-net/nomeroff_net/tools/../../data/./dataset/Detector/yolov8/keypoints_yolo/labels/train"
os.makedirs(res_dataset, exist_ok=True)
path_to_images = os.path.join(PATH_TO_DATASET, "train")

ann_filepath = os.path.join(path_to_images, 'via_region_data.json')
with open(ann_filepath) as ann:
    ann_data = json.load(ann)
image_list = ann_data

img_id = 0
corupted_train_images = []
for _id, annotation in tqdm(image_list["_via_img_metadata"].items()):
    regions = annotation['regions']
    img_filename = annotation['filename']
    img_base, _ = os.path.splitext(img_filename)
    yolo_annotations = []

    image_id = image_list["_via_img_metadata"][_id]["filename"]
    filename = f'{path_to_images}/{image_id}'
    
    pil_image = Image.open(filename)
    pil_image = rotate_image_by_exif(pil_image)
    image = np.array(pil_image)
    h, w, c = image.shape
    target_boxes = []
    labels = []
    for region in image_list["_via_img_metadata"][_id]["regions"]:
        shape_attrs = region['shape_attributes']
        if region["shape_attributes"].get("all_points_x", None) is None or len(region["shape_attributes"]["all_points_x"]) != 4:
            corupted_train_images.append(_id)
            continue
        all_points_x = shape_attrs['all_points_x']
        all_points_y = shape_attrs['all_points_y']
        if c == 3:
            bbox = [
                int(min(region["shape_attributes"]["all_points_x"])),
                int(min(region["shape_attributes"]["all_points_y"])),
                int(max(region["shape_attributes"]["all_points_x"])),
                int(max(region["shape_attributes"]["all_points_y"])),
            ]
            x_min = min(all_points_x)
            x_max = max(all_points_x)
            y_min = min(all_points_y)
            y_max = max(all_points_y)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            x_center = x_min + bbox_width / 2
            y_center = y_min + bbox_height / 2
            
            roi_img = image[bbox[1]:bbox[3], bbox[0]: bbox[2]]
            xs = np.array([_item-bbox[0] for _item in region["shape_attributes"]["all_points_x"]])
            ys = np.array([_item-bbox[1] for _item in region["shape_attributes"]["all_points_y"]])
            
            rect = normalize_rect(list(zip(xs, ys)))
            xs = [item[0]+x_min for item in rect]
            ys = [item[1]+y_min for item in rect]
            
            if roi_img.shape[0] and roi_img.shape[1]:
                if DEBUG:
                    # Відображення зображення
                    plt.imshow(roi_img)
                    # Нанесення ключових точок
                    plt.scatter(xs, ys, c='red')
                    # Підписання ключових точок
                    for i in range(4):
                        plt.text(xs[i], ys[i], str(i+1), 
                                 fontsize=12, color='blue')
                    # Показати результат
                    plt.show()
                keypoints = []
                for x, y in zip(xs, ys):
                    keypoints.extend([x/w, y/h])
                # Формат анотацій YOLO
                class_id = 0  # Припустимо, що клас номерного знаку має індекс 0
                yolo_annotation = [class_id, x_center/w, y_center/h, bbox_width/w, bbox_height/h] + keypoints
                yolo_annotations.append(yolo_annotation)
            else:
                corupted_train_images.append(_id)
        else:
            corupted_train_images.append(_id)
    # Записати анотації в файл
    yolo_annotation_str = "\n".join([" ".join(map(str, anno)) for anno in yolo_annotations])
    output_path = os.path.join(os.path.join(res_dataset), f"{img_base}.txt")
    with open(output_path, "w") as f:
        f.write(yolo_annotation_str)


# In[36]:


corupted_train_images


# In[37]:


res_dataset = "/mnt/store/var/www/nomeroff-net/nomeroff_net/tools/../../data/./dataset/Detector/yolov8/keypoints_yolo/labels/train"
os.makedirs(res_dataset, exist_ok=True)
path_to_images = os.path.join(PATH_TO_DATASET, "train")

ann_filepath = os.path.join(path_to_images, 'via_region_data.json')
with open(ann_filepath) as ann:
    ann_data = json.load(ann)
image_list = ann_data

img_id = 0
tmp = list(image_list["_via_img_metadata"].items())
for _id, annotation in tqdm(tmp):
    if _id not in corupted_train_images:
        del image_list["_via_img_metadata"][_id]


# In[38]:


len(image_list["_via_img_metadata"])


# In[39]:


with open("corrupted_via_region_data_train.json", "w") as ann:
    json.dump(image_list, ann)


# In[ ]:




