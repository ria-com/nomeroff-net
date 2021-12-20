import copy
import os
import json
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Generator, Any
from torchvision import transforms

from nomeroff_net.tools.ocr_tools import is_valid_str


class TextImageGenerator(object):
    def __init__(self,
                 dirpath: str,
                 letters: List,
                 max_text_len: int,
                 label_converter: Any = None,
                 img_w: int = 128,
                 img_h: int = 64,
                 batch_size: int = 1,
                 max_plate_length: int = 8,
                 with_aug: bool = False) -> None:

        self.dirpath = dirpath
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.max_plate_length = max_plate_length
        self.letters = letters

        self.label_converter = label_converter
        self.list_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        img_dirpath = os.path.join(dirpath, 'img')
        ann_dirpath = os.path.join(dirpath, 'ann')
        self.pathes = [os.path.join(img_dirpath, file_name) for file_name in os.listdir(img_dirpath)]
        self.samples = []
        for file_name in os.listdir(img_dirpath):
            name, ext = os.path.splitext(file_name)
            if ext == '.png':
                img_filepath = os.path.join(img_dirpath, file_name)
                json_filepath = os.path.join(ann_dirpath, name + '.json')
                description = json.load(open(json_filepath, 'r'))['description']
                if is_valid_str(description, self.letters):
                    self.samples.append([img_filepath, description])
                else:
                    raise Warning(f"Image {img_filepath} does not have a valid description!")
            else:
                raise Warning(f"Image {file_name} is not png!")

        self.n = len(self.samples)
        self.batch_count = int(self.n / batch_size)
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

    def get_x_from_path(self, img_path: str, newsize: Tuple = None) -> torch.Tensor:
        if newsize is None:
            newsize = (200, 50)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(newsize)
        if self.with_aug:
            from nomeroff_net.tools.augmentations import aug
            img = np.array(img)
            imgs = aug([img])
            img = Image.fromarray(imgs[0])
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        """
        Generates one sample of data
        """

        img_path, text = copy.deepcopy(self.samples[index])
        text = text.lower()
        img = self.get_x_from_path(img_path)
        return img, text

    def transform(self, img) -> torch.Tensor:
        return self.list_transforms(img)

    def next_sample(self) -> Tuple:
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
        img_path, text = copy.deepcopy(self.samples[self.cur_index])
        text = text.lower()
        x = self.get_x_from_path(img_path)
        return img_path, x, text

    def run_iteration(self):
        ys = []
        xs = []
        paths = []
        for _ in np.arange(self.batch_size):
            img_path, x, y = self.next_sample()
            paths.append(img_path)
            x = x.reshape([1, *x.shape])
            xs.append(x)
            ys.append(y)
        xs = torch.cat(xs, dim=0)
        return paths, xs, ys

    def path_generator(self) -> Generator:
        for _ in np.arange(self.batch_count):
            paths, xs, ys = self.run_iteration()
            yield paths, xs, ys
