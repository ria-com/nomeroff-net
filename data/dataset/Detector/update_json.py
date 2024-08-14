#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../../../../')
sys.path.append(NOMEROFF_NET_DIR)


# # update train

# In[2]:


PATH_TO_DATASET = '/mnt/store/var/www/nomeroff-net/nomeroff_net/tools/../../data/./dataset/Detector/yolov8/autoriaNumberplateDataset-2023-03-06'


# In[3]:


fixed_ann_filepath = '/mnt/store/var/www/nomeroff-net/data/dataset/Detector/yolov8/via_data_end-6(861)1.train (1).json'
#fixed_ann_filepath = '/mnt/store/var/www/nomeroff-net/data/dataset/Detector/yolov8/via_data_end-6(861)_1Val.json'


# In[4]:


with open(fixed_ann_filepath) as ann:
    fixed_ann_data = json.load(ann)


# In[5]:


path_to_images = os.path.join(PATH_TO_DATASET, "train")
#path_to_images = os.path.join(PATH_TO_DATASET, "val")

ann_filepath = os.path.join(path_to_images, 'via_region_data.json')
with open(ann_filepath) as ann:
    ann_data = json.load(ann)


# In[6]:


ann_data.keys()


# In[7]:


nees_moderation = {
    '_via_attributes': ann_data['_via_attributes'],
    '_via_settings': ann_data['_via_settings'],
    '_via_img_metadata': {}
}
for _id in tqdm(ann_data["_via_img_metadata"]):
    if _id in fixed_ann_data["_via_img_metadata"]:
        ann_data["_via_img_metadata"][_id] = fixed_ann_data["_via_img_metadata"][_id]
        ann_data["_via_img_metadata"][_id]["moderated"] = True
        have_4_points = all([len(item['shape_attributes']["all_points_x"]) == 4 and len(item['shape_attributes']["all_points_y"]) == 4 for item in fixed_ann_data["_via_img_metadata"][_id]["regions"]])
        if not have_4_points:
            print("\n\n\n")
            print(_id)
            print(ann_data["_via_img_metadata"][_id])
            print(fixed_ann_data["_via_img_metadata"][_id])
    else:
        nees_moderation["_via_img_metadata"][_id] = ann_data["_via_img_metadata"][_id]


# In[8]:


# res_ann_filepath = '/mnt/store/var/www/nomeroff-net/data/dataset/Detector/yolov8/train_via_region_data.json'
# with open(res_ann_filepath, "w") as ann:
#     json.dump(ann_data, ann)


# In[9]:


len(nees_moderation["_via_img_metadata"])


# In[10]:


# res_ann_filepath = '/mnt/store/var/www/nomeroff-net/data/dataset/Detector/yolov8/need_moder_train_via_region_data.json'
# with open(res_ann_filepath, "w") as ann:
#     json.dump(nees_moderation, ann)


# In[11]:


from copy import deepcopy

batch_size = 1000
_from = 0
for i in range(len(nees_moderation["_via_img_metadata"])//batch_size+1):
    _to = _from + batch_size
    tmp_nees_moderation = deepcopy(nees_moderation)
    tmp_nees_moderation["_via_img_metadata"] = dict(list(tmp_nees_moderation["_via_img_metadata"].items())[_from:_to])

    _from = _from+batch_size
    print(len(tmp_nees_moderation["_via_img_metadata"]))
    res_ann_filepath = f'/mnt/store/var/www/nomeroff-net/data/dataset/Detector/yolov8/need_moder_train_via_region_data_{i}.json'
    #res_ann_filepath = f'/mnt/store/var/www/nomeroff-net/data/dataset/Detector/yolov8/need_moder_val_via_region_data_{i}.json'
    with open(res_ann_filepath, "w") as ann:
        json.dump(tmp_nees_moderation, ann)


# # make fixed json train

# In[4]:


PATH_TO_DATASET = '/mnt/store/var/www/nomeroff-net/nomeroff_net/tools/../../data/./dataset/Detector/yolov8/autoriaNumberplateDataset-2023-03-06'


# In[5]:


fixed_ann_filepath = '/mnt/store/var/www/nomeroff-net/data/dataset/Detector/yolov8/via_data_end-6(861)1.train (1).json'


# In[6]:


with open(fixed_ann_filepath) as ann:
    fixed_ann_data = json.load(ann)


# In[7]:


path_to_images = os.path.join(PATH_TO_DATASET, "train")
#path_to_images = os.path.join(PATH_TO_DATASET, "val")

ann_filepath = os.path.join(path_to_images, 'via_region_data.json')
with open(ann_filepath) as ann:
    ann_data = json.load(ann)


# In[8]:


ann_data["_via_img_metadata"].update(fixed_ann_data["_via_img_metadata"])


# In[10]:


with open("fixed_via_region_data_train.json", "w") as ann:
    json.dump(ann_data, ann)


# # make fixed json val
# 

# In[11]:


fixed_ann_filepath = '/mnt/store/var/www/nomeroff-net/data/dataset/Detector/yolov8/via_data_end-6(861)_1Val.json'


# In[12]:


with open(fixed_ann_filepath) as ann:
    fixed_ann_data = json.load(ann)


# In[13]:


path_to_images = os.path.join(PATH_TO_DATASET, "val")

ann_filepath = os.path.join(path_to_images, 'via_region_data.json')
with open(ann_filepath) as ann:
    ann_data = json.load(ann)


# In[14]:


ann_data["_via_img_metadata"].update(fixed_ann_data["_via_img_metadata"])


# In[17]:


with open("fixed_via_region_data_val.json", "w") as ann:
    json.dump(ann_data, ann)


# In[ ]:




