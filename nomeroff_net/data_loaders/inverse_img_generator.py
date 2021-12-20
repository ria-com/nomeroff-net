import copy
import os
import json
import cv2
import torch
import numpy as np
from typing import List, Tuple

from nomeroff_net.tools.image_processing import normalize_img
from .img_generator import ImgGenerator


class InverseImgGenerator(ImgGenerator):
    def __init__(self,
                 dirpath: str,
                 img_w: int = 295,
                 img_h: int = 64,
                 batch_size: int = 32,
                 labels_counts: List or Tuple = tuple([2]),
                 with_aug: bool = False) -> None:
        super(ImgGenerator, self).__init__(dirpath, img_w, img_h, batch_size, labels_counts, with_aug)

    def load_dataset(self):
        img_dirpath = os.path.join(self.dirpath, 'img')
        ann_dirpath = os.path.join(self.dirpath, 'ann')
        self.samples = []
        for file_name in os.listdir(img_dirpath):
            name, ext = os.path.splitext(file_name)
            if ext == '.png':
                img_filepath = os.path.join(img_dirpath, file_name)
                json_filepath = os.path.join(ann_dirpath, name + '.json')
                if os.path.exists(json_filepath):
                    description = json.load(open(json_filepath, 'r'))
                    self.samples.append([img_filepath, int(description.get("orientation", -1))])
        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.batch_count = int(self.n / self.batch_size)

    def __getitem__(self, index):
        """
        Generates one sample of data
        """

        x, y = copy.deepcopy(self.paths[self.indexes[index]]), copy.deepcopy(self.discs[self.indexes[index]])
        x = self.get_x_from_path(x)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y

    def build_data(self) -> None:
        self.paths = []
        self.discs = []
        for i, (img_filepath, disc) in enumerate(self.samples):
            self.paths.append(img_filepath)
            self.discs.append(
                np.eye(self.labels_counts[0])[disc],
            )

    def run_iteration(self, with_aug=False):
        ys = []
        xs = []
        paths = []
        for _ in np.arange(self.batch_size):
            x, y = self.next_sample()
            paths.append(x)
            img = cv2.imread(x)
            x = normalize_img(img, with_aug=with_aug, width=self.img_w, height=self.img_h)
            xs.append(x)
            ys.append(y)
        xs = np.moveaxis(np.array(xs), 3, 1)
        return paths, xs, ys
