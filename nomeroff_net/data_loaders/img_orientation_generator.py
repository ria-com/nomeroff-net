import os
import json
import cv2
import torch
import numpy as np
import random
import tqdm
from typing import List, Tuple, Generator
from torch.utils.data import Dataset

from nomeroff_net.tools.image_processing import rotate_image_and_bboxes, normalize_img


class ImgOrientationGenerator(Dataset):

    def __init__(self,
                 json_path: str,
                 img_path: str,
                 img_w: int = 295,
                 img_h: int = 64,
                 batch_size: int = 32,
                 angles: List = None,
                 with_aug: bool = False) -> None:
        if angles is None:
            angles = [0, 90, 180, 270]
        self.cur_index = 0
        self.paths = []
        self.discs = []

        self.img_h = img_h
        self.img_w = img_w
        self.orientations_count = len(angles)

        with open(json_path) as json_file:
            data = json.load(json_file)

        self.samples = []
        for p in tqdm.tqdm(data['_via_img_metadata']):
            item = data['_via_img_metadata'][p]
            file_name = item["file_name"]
            image_path = os.path.join(img_path, file_name)
            target_boxes = [
                [
                    min(np.array(region['shape_attributes']['all_points_x'])),
                    min(np.array(region['shape_attributes']['all_points_y'])),
                    max(np.array(region['shape_attributes']['all_points_x'])),
                    max(np.array(region['shape_attributes']['all_points_y'])),
                ] for region in item['regions']
                if len(region['shape_attributes']['all_points_x']) == 4
                and len(region['shape_attributes']['all_points_y']) == 4
            ]
            if not len(target_boxes):
                continue

            for label, angle in enumerate(angles):
                for bbox in target_boxes:
                    self.samples.append([
                        [image_path, angle, bbox],
                        np.array([1 if i == label else 0 for i in range(self.orientations_count)])
                    ])

        self.n = len(self.samples)
        self.batch_size = batch_size
        self.indexes = list(range(self.n))
        self.batch_count = int(self. n / batch_size)
        self.with_aug = with_aug
        self.rezero()

    def build_data(self):
        return

    def rezero(self) -> None:
        self.cur_index = 0
        random.shuffle(self.indexes)

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return self.n

    def get_x_from_path(self,
                        path: str,
                        bbox: List,
                        angle: int) -> np.ndarray:
        img = cv2.imread(path)
        rotated_img, rotated_bbox = rotate_image_and_bboxes(img, np.array([bbox]), int(angle))
        rotated_bbox = rotated_bbox[0]
        rotated_bbox[0] = rotated_bbox[0] if rotated_bbox[0] > 0 else 0
        rotated_bbox[1] = rotated_bbox[1] if rotated_bbox[1] > 0 else 0
        min_x = rotated_bbox[0]
        max_x = rotated_bbox[2]
        min_y = rotated_bbox[1]
        max_y = rotated_bbox[3]

        img_part = rotated_img[min_y:max_y, min_x:max_x]
        img_part = normalize_img(img_part, width=self.img_w, height=self.img_h)
        img_part = np.moveaxis(np.array(img_part), 2, 0)
        return img_part

    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        img_data, y = self.samples[index]
        img_path, angle, bbox = img_data
        x = self.get_x_from_path(img_path, bbox, angle)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y

    def next_sample(self) -> Tuple:
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
        img_data, y = self.samples[self.cur_index]
        img_path, angle, bbox = img_data
        x = self.get_x_from_path(img_path, bbox, angle)
        return img_path, x, y

    def run_iteration(self):
        ys = []
        xs = []
        paths = []
        for _ in np.arange(self.batch_size):
            img_path, x, y = self.next_sample()
            paths.append(img_path)
            xs.append(x)
            ys.append(y)
        ys = np.array(ys, dtype=np.float32)
        xs = np.array(xs, dtype=np.float32)
        return paths, xs, ys

    def path_generator(self) -> Generator:
        for _ in np.arange(self.batch_count):
            paths, xs, ys = self.run_iteration()
            yield paths, xs, ys
