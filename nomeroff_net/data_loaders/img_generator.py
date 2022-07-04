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
                 img_w: int = 295,
                 img_h: int = 64,
                 batch_size: int = 32,
                 labels_counts: List = (14, 4, 2),
                 with_aug: bool = False) -> None:
        self.with_aug = with_aug
        self.cur_index = 0
        self.paths = []
        self.discs = []

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size

        self.labels_counts = labels_counts
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

    def load_dataset(self, with_aug: bool, dirpath: str, cache_postfix: str = "cache_options"):
        img_dirpath = os.path.join(self.dirpath, 'img')
        ann_dirpath = os.path.join(self.dirpath, 'ann')

        if with_aug:
            cache_postfix = f"{cache_postfix}_aug"
        cache_dirpath = os.path.join(dirpath, cache_postfix)
        os.makedirs(cache_dirpath, exist_ok=True)
        self.samples = []
        self.images_path = []
        for file_name in tqdm(os.listdir(img_dirpath)):
            name, ext = os.path.splitext(file_name)
            if ext == '.png':
                img_filepath = os.path.join(img_dirpath, file_name)
                self.images_path.append(img_filepath)
                json_filepath = os.path.join(ann_dirpath, name + '.json')
                x_filepath = self.generate_cache_x_in_path(img_filepath, cache_dirpath)
                if os.path.exists(json_filepath):
                    description = json.load(open(json_filepath, 'r'))
                    self.samples.append([x_filepath, [
                        int(description.get("region_id", -1)),
                        int(description.get("count_lines", -1)),
                        int(description.get("orientation", -1))]])

        self.added_samples_to_round_batch()
        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.batch_count = int(self.n / self.batch_size)

    def added_samples_to_round_batch(self):
        while len(self.samples) % self.batch_size != 0 and len(self.samples):
            for sample, images_path in zip(self.samples, self.images_path):
                self.samples.append(sample)
                self.images_path.append(images_path)
                if len(self.samples) % self.batch_size == 0:
                    break

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return self.n

    @staticmethod
    def get_x_from_path(x_path: str) -> torch.Tensor:
        return torch.load(x_path)

    def generate_cache_x_in_path(self, img_path: str, cache_dirpath: str, newsize: Tuple = None) -> str:
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
            img = img[:, :, ::-1]
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
