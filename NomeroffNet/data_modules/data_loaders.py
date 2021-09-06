import copy
import os
import json
import cv2
import torch
import numpy as np
import random
import tqdm
from PIL import Image
from typing import List, Tuple, Generator, Any
from NomeroffNet.tools.ocr_tools import is_valid_str
from NomeroffNet.tools.image_processing import rotate_image_and_bboxes
from torchvision import transforms


def aug_seed(num: int = None) -> None:
    import imgaug as ia

    if num is None:
        ia.seed()
    else:
        ia.seed(num)


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
    if not to_gray and  len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
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


class ImgOrientationGenerator(torch.utils.data.Dataset):

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
            filename = item["filename"]
            image_path = os.path.join(img_path, filename)
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
        self.batch_count = int(self.n/batch_size)
        self.with_aug = with_aug
        self.rezero()
    
    def build_data(self):
        pass
        
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
        img_part = normalize(img_part, width=self.img_w, height=self.img_h)
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
        self.pathes = [os.path.join(img_dirpath, filename) for filename in os.listdir(img_dirpath)]
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
        self.batch_count = int(self.n/batch_size)
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
            from .augmentations import aug
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
