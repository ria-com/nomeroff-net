import copy
import os
import json
import cv2
import torch
import numpy as np
import random
from tqdm import tqdm
from typing import List, Tuple, Generator
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from nomeroff_net.tools.image_processing import normalize_img


class ImgGenerator(Dataset):
    def __init__(self,
                 dirpath: str,
                 img_w: int = 256,
                 img_h: int = 256,
                 batch_size: int = 32,
                 with_aug: bool = False) -> None:
        self.with_aug = with_aug
        self.cur_index = 0
        self.paths = []
        self.discs = []

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size

        self.dirpath = dirpath
        self.samples = []
        self.images_path = []
        self.list_transforms = None

        self.prepare_transformers()
        self.load_dataset(with_aug, dirpath)

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.batch_count = int(self.n/batch_size)
        self.rezero()

    def generate_numberplate_fraud_and_true(self, image_filename, corners):
        fraud_paths, true_paths = [], []
        x_filepath = self.generate_cache_x_in_path(image_filename, corners)

        return fraud_paths, true_paths

    def load_dataset(self, with_aug: bool, dir_path: str, json_path: str):
        with open(json_path) as jsonFile:
            json_data = json.load(jsonFile)

        self.samples = []
        self.images_path = []
        for key in json_data["_via_img_metadata"]:
            metadata = json_data["_via_img_metadata"][key]

            # define image_id
            image_filename = metadata["filename"]
            print("image_filename", image_filename)

            for region in metadata["regions"]:
                segmentation = [[]]
                for x, y in zip(region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"]):
                    segmentation[0].append(x)
                    segmentation[0].append(y)

                # define area
                corners = [(x, y) for x, y in
                           zip(region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"])]
                fraud_paths, true_paths = self.generate_numberplate_fraud_and_true(image_filename, corners)

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.batch_count = int(self.n / self.batch_size)

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return self.n

    def get_x_from_path(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        x = normalize_img(img,
                          with_aug=self.with_aug,
                          width=self.img_w,
                          height=self.img_h)
        x = np.moveaxis(np.array(x), 2, 0)
        return x

    def generate_cache_x_in_path(self,
                                 img_path: str,
                                 cache_dirpath: str,
                                 newsize: Tuple = None) -> str:
        x_path = self.generate_x_path(img_path, cache_dirpath)

        if os.path.exists(x_path):
            return x_path

        if newsize is None:
            newsize = (self.img_w, self.img_h)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(newsize)
        if self.with_aug:
            from nomeroff_net.tools.augmentations import aug
            img = np.array(img)
            imgs = aug([img])
            img = Image.fromarray(imgs[0])
        x = self.transform(img)
        torch.save(x, x_path)
        return x_path

    @staticmethod
    def generate_x_path(img_path: str, cache_dirpath: str):
        filename, file_extension = os.path.splitext(img_path)
        filename = os.path.basename(filename)
        x_path = os.path.join(cache_dirpath, f'{filename}.pt')
        return x_path

    def __getitem__(self, index):
        """
        Generates one sample of data
        """

        x = copy.deepcopy(self.paths[self.indexes[index]])
        y = copy.deepcopy(self.discs[self.indexes[index]])
        x = self.get_x_from_path(x)
        x = torch.from_numpy(x)
        y[0] = torch.from_numpy(y[0])
        y[1] = torch.from_numpy(y[1])
        return x, y

    def prepare_transformers(self):
        self.list_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def transform(self, img) -> torch.Tensor:
        x = self.list_transforms(img)
        return x

    def rezero(self) -> None:
        self.cur_index = 0
        random.shuffle(self.indexes)

    def build_data(self) -> None:
        self.paths = []
        self.discs = []
        for i, (img_filepath, disc) in enumerate(self.samples):
            self.paths.append(img_filepath)
            self.discs.append(
                [
                    np.eye(self.labels_counts[0])[disc[0]],
                    np.eye(self.labels_counts[1])[disc[1]]
                ]
            )

    def next_sample(self) -> Tuple:
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
        return self.images_path[self.indexes[self.cur_index]], self.discs[self.indexes[self.cur_index]]

    def run_iteration(self, with_aug=False):
        ys = [[], []]
        xs = []
        paths = []
        for _ in np.arange(self.batch_size):
            x, y = self.next_sample()
            paths.append(x)
            img = cv2.imread(x)
            x = normalize_img(img, with_aug=with_aug, width=self.img_w, height=self.img_h)
            xs.append(x)
            ys[0].append(y[0])
            ys[1].append(y[1])
        ys[0] = np.array(ys[0], dtype=np.float32)
        ys[1] = np.array(ys[1], dtype=np.float32)
        xs = np.moveaxis(np.array(xs), 3, 1)
        return paths, xs, ys

    def generator(self, with_aug: bool = False) -> Generator:
        for _ in np.arange(self.batch_count):
            _, xs, ys = self.run_iteration(with_aug)
            yield xs, ys

    def torch_generator(self, with_aug: bool = False) -> Generator:
        for _ in np.arange(self.batch_count):
            _, xs, ys = self.run_iteration(with_aug)
            xs = torch.from_numpy(ys)
            ys = torch.from_numpy(ys)
            yield xs, ys

    def path_generator(self, with_aug: bool = False) -> Generator:
        for _ in np.arange(self.batch_count):
            paths, xs, ys = self.run_iteration(with_aug)
            yield paths, xs, ys
