import copy
import os
import json
import cv2
import torch
import numpy as np
import random
from typing import List, Tuple, Generator
from NomeroffNet.tools.ocr_tools import is_valid_str, text_to_labels


def normalize(img: np.ndarray,
              height: int = 64,
              width: int = 295,
              to_gray: bool = False,
              with_aug: bool = False) -> np.ndarray:
    if with_aug:
        from .augmentations import aug
        imgs = aug([img])
        img = imgs[0]
    if to_gray and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)

    # advanced normalisation
    img_min = np.amin(img)
    img -= img_min
    img_max = np.amax(img)
    img /= (img_max or 1)
    img[img == 0] = .0001

    if to_gray:
        img = np.reshape(img, [*img.shape, 1])
    return img


class ImgGenerator(torch.utils.data.Dataset):

    def __init__(self,
                 dirpath: str,
                 img_w: int = 295,
                 img_h: int = 64,
                 batch_size: int = 32,
                 labels_counts: List = (14, 4, 2),
                 with_aug: bool = False) -> None:

        self.cur_index = 0
        self.paths = []
        self.discs = []

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size

        self.labels_counts = labels_counts

        img_dirpath = os.path.join(dirpath, 'img')
        ann_dirpath = os.path.join(dirpath, 'ann')
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext == '.png':
                img_filepath = os.path.join(img_dirpath, filename)
                json_filepath = os.path.join(ann_dirpath, name + '.json')
                if os.path.exists(json_filepath):
                    description = json.load(open(json_filepath, 'r'))
                    self.samples.append([img_filepath, [
                        int(description["region_id"]),
                        int(description["count_lines"]),
                        int(description.get("orientation", -1))]])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.batch_count = int(self.n/batch_size)
        self.with_aug = with_aug
        self.rezero()

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return self.n

    def get_x_from_path(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        x = normalize(img,
                      with_aug=self.with_aug,
                      width=self.img_w,
                      height=self.img_h)
        x = np.moveaxis(np.array(x), 2, 0)
        return x

    def __getitem__(self, index):
        """
        Generates one sample of data
        """

        x, y = copy.deepcopy(self.paths[self.indexes[index]]), copy.deepcopy(self.discs[self.indexes[index]])
        x = self.get_x_from_path(x)
        x = torch.from_numpy(x)
        y[0] = torch.from_numpy(y[0])
        y[1] = torch.from_numpy(y[1])
        return x, y

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
        return self.paths[self.indexes[self.cur_index]], self.discs[self.indexes[self.cur_index]]

    def run_iteration(self, with_aug=False):
        ys = [[], []]
        xs = []
        paths = []
        for _ in np.arange(self.batch_size):
            x, y = self.next_sample()
            paths.append(x)
            img = cv2.imread(x)
            x = normalize(img, with_aug=with_aug, width=self.img_w, height=self.img_h)
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


class OrientationImgGenerator(ImgGenerator):
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
            if disc[2] >= 0:
                self.paths.append(img_filepath)
                self.discs.append(
                    np.eye(self.labels_counts[2])[disc[2]]
                )

    def run_iteration(self, with_aug=False):
        ys = []
        xs = []
        paths = []
        for _ in np.arange(self.batch_size):
            x, y = self.next_sample()
            paths.append(x)
            img = cv2.imread(x)
            x = normalize(img, with_aug=with_aug, width=self.img_w, height=self.img_h)
            xs.append(x)
            ys.append(y)
        ys = np.array(ys, dtype=np.float32)
        xs = np.moveaxis(np.array(xs), 3, 1)
        return paths, xs, ys


class TextImageGenerator(object):
    def __init__(self,
                 dirpath: str,
                 letters: List,
                 max_text_len: int,
                 img_w: int = 128,
                 img_h: int = 64,
                 batch_size: int = 1,
                 with_aug: bool = False) -> None:

        self.dirpath = dirpath
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.letters = letters

        img_dirpath = os.path.join(dirpath, 'img')
        ann_dirpath = os.path.join(dirpath, 'ann')
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext == '.png':
                img_filepath = os.path.join(img_dirpath, filename)
                json_filepath = os.path.join(ann_dirpath, name + '.json')
                description = json.load(open(json_filepath, 'r'))['description']
                if is_valid_str(description, self.letters):
                    self.samples.append([img_filepath, description])
                else:
                    raise Warning(f"Image {img_filepath} does not have a valid description!")
            else:
                raise Warning(f"Image {filename} is not png!")

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.with_aug = with_aug
        self.count_ep = 0
        self.letters_max = len(letters) + 1
        self.imgs = None
        self.texts = None

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return self.n

    def get_x_from_path(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        x = normalize(img,
                      with_aug=self.with_aug,
                      width=self.img_w,
                      height=self.img_h,
                      to_gray=True)
        x = np.moveaxis(np.array(x), 2, 0)
        return x

    def __getitem__(self, index):
        """
        Generates one sample of data
        """

        img_path, text = copy.deepcopy(self.samples[index])
        x = self.get_x_from_path(img_path)
        y = np.array(text_to_labels(text, self.letters, self.max_text_len))
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        input_lengths = torch.full(size=(1,),
                                   fill_value=self.img_w // (2 * 2) - 2,
                                   dtype=torch.long)
        target_lengths = np.array([len(text)])
        target_lengths = torch.from_numpy(target_lengths)
        return (x, input_lengths, target_lengths), y
