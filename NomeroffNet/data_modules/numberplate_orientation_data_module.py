import os
import sys
import json
import torch
import tqdm
import cv2
import numpy as np
import pytorch_lightning as pl
from .data_loaders import normalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../base')))


def prepare_data(json_path,
                 img_path,
                 w=300,
                 h=300):
    features = []
    targets = []
    print("Loading dataset...")
    with open(json_path) as json_file:
        data = json.load(json_file)
        for p in tqdm.tqdm(data['_via_img_metadata']):
            item = data['_via_img_metadata'][p]
            filename = item["filename"]
            image_path = os.path.join(img_path, filename)
            img = cv2.imread(image_path)
            for region in item['regions']:
                if len(region['shape_attributes']['all_points_x']) != 4:
                    continue
                if len(region['shape_attributes']['all_points_y']) != 4:
                    continue
                xs = np.array(region['shape_attributes']['all_points_x'])
                ys = np.array(region['shape_attributes']['all_points_y'])
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                img_part = img[min_y:max_y, min_x:max_x]
                img_part = normalize(img_part, width=w, height=h)
                features.append(img_part)
                targets.append(0)
    print("Prepared", len(features), "images")
    return features, targets


class OrientationDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dir="../datasets/mask/val",
                 train_json_path="../datasets/mask/val/via_region_data_sorted.json",
                 validation_dir="../datasets/mask/train",
                 validation_json_path="../datasets/mask/train/via_region_data_new7.json",
                 width=300,
                 height=300,
                 batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height

        self.val_img_path = validation_dir
        self.val_json_path = validation_json_path

        self.train_img_path = train_dir
        self.train_json_path = train_json_path

        self.val_features = None
        self.val_targets = None
        self.train_features = None
        self.train_targets = None
        self.train = None
        self.val = None

    def prepare_data(self):
        val_features, val_targets = prepare_data(self.val_json_path,
                                                 self.val_img_path,
                                                 w=self.width,
                                                 h=self.height)
        self.val_features = torch.Tensor(val_features)
        self.val_targets = torch.Tensor(val_targets)

        train_features, train_targets = prepare_data(self.train_json_path,
                                                     self.train_img_path,
                                                     w=self.width,
                                                     h=self.height)
        self.train_features = torch.Tensor(train_features)
        self.train_targets = torch.Tensor(train_targets)

    def setup(self, stage=None):
        self.train = torch.utils.data.TensorDataset(self.val_features, self.val_targets)
        self.val = torch.utils.data.TensorDataset(self.train_features, self.train_targets)

    # return the dataloader for each split
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size)
