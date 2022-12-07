import copy
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List, Tuple, Generator, Any
from torchvision import transforms

from nomeroff_net.tools.mcm import get_device_torch
from nomeroff_net.tools.ocr_tools import is_valid_str

device_torch = get_device_torch()


class TextImageGenerator(object):
    def __init__(self,
                 dirpath: str,
                 letters: List,
                 max_text_len: int,
                 label_converter: Any = None,
                 img_w: int = 128,
                 img_h: int = 64,
                 batch_size: int = 1,
                 seed: int = 42,
                 with_aug: bool = False) -> None:

        self.dirpath = dirpath
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.letters = letters
        self.with_aug = with_aug
        self.label_converter = label_converter
        self.prepare_transformers()

        img_dirpath = os.path.join(dirpath, 'img')
        ann_dirpath = os.path.join(dirpath, 'ann')
        cache_postfix = "cache_ocr"
        if with_aug:
            cache_postfix = f"{cache_postfix}_aug_{seed}"
        cache_dirpath = os.path.join(dirpath, cache_postfix)
        os.makedirs(cache_dirpath, exist_ok=True)
        self.paths = []
        self.samples = []
        for file_name in tqdm(os.listdir(img_dirpath)):
            name, ext = os.path.splitext(file_name)
            if ext == '.png':
                img_filepath = os.path.join(img_dirpath, file_name)
                json_filepath = os.path.join(ann_dirpath, name + '.json')
                if not os.path.exists(json_filepath):
                    continue
                self.paths.append(os.path.join(img_dirpath, file_name))
                x_filepath = self.generate_cache_x_in_path(img_filepath, cache_dirpath)
                description = json.load(open(json_filepath, 'r'))['description']
                if is_valid_str(description, self.letters):
                    self.samples.append([x_filepath, description])
                else:
                    raise Warning(f"Image {img_filepath} does not have a valid description!")
            else:
                raise Warning(f"Image {file_name} is not png!")

        self.n = len(self.samples)
        self.batch_count = int(self.n / batch_size)
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.count_ep = 0
        self.letters_max = len(letters) + 1
        self.imgs = None
        self.texts = None

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return self.n

    def generate_x_path(self, img_path: str, cache_dirpath: str):
        filename, file_extension = os.path.splitext(img_path)
        filename = os.path.basename(filename)
        x_path = os.path.join(cache_dirpath, f'{filename}.pt')
        return x_path

    def generate_cache_x_in_path(self, img_path: str, cache_dirpath: str, newsize: Tuple = None) -> torch.Tensor:
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
    def get_x_from_path(x_path: str) -> torch.Tensor:
        return torch.load(x_path)

    def __getitem__(self, index):
        """
        Generates one sample of data
        """

        img_path, text = copy.deepcopy(self.samples[index])
        text = text.lower()
        img = self.get_x_from_path(img_path)
        return img, text

    def prepare_transformers(self):
        self.list_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def transform(self, img) -> torch.Tensor:
        x = self.list_transforms(img)
        return x

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
