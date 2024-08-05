"""
Numberplate Orientation Image Generator
python3 -m nomeroff_net.data_loaders.img_orientation_generator -f nomeroff_net/data_loaders/img_orientation_generator_from_orig.py
"""
import os
import json
import cv2
import torch
import numpy as np
import random
import tqdm
from typing import List, Tuple, Generator
from torch.utils.data import Dataset

from nomeroff_net.tools.image_processing import normalize_img, build_perspective
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import normalize_rect


def rotate_rect(rect, angle):
    """
    Повертає rect на заданий кут (90, 180 або 270 градусів)
    """
    if angle not in [0, 90, 180, 270]:
        raise ValueError("Кут повинен бути 0, 90, 180 або 270 градусів")
    rect = normalize_rect(rect)

    # Конвертуємо rect у numpy масив для зручності
    rect = np.array(rect)

    if angle == 0:
        return [rect[1], rect[2], rect[3], rect[0]]
    elif angle == 90:
        # Зсув для повороту на 90 градусів за годинниковою стрілкою
        return [rect[2], rect[3], rect[0], rect[1]]
    elif angle == 180:
        # Зсув для повороту на 180 градусів
        return [rect[3], rect[0], rect[1], rect[2]]
    elif angle == 270:
        # Зсув для повороту на 270 градусів за годинниковою стрілкою (90 проти годинникової)
        return rect



class ImgOrientationGenerator(Dataset):

    def __init__(self,
                 json_path: str,
                 img_path: str,
                 img_w: int = 300,
                 img_h: int = 100,
                 batch_size: int = 32) -> None:
        angles = {
            0: 0,
            90: 1,
            180: 2,
            270: 1
        }

        # generate label_abgle_map
        label_abgle_map = {}
        for angle, label in angles.items():
            if angle not in label_abgle_map:
                label_abgle_map[label] = []
            label_abgle_map[label].append(angle)

        self.cur_index = 0
        self.paths = []
        self.discs = []

        self.img_h = img_h
        self.img_w = img_w

        with open(json_path) as json_file:
            data = json.load(json_file)

        self.samples = []
        for p in tqdm.tqdm(data['_via_img_metadata']):
            item = data['_via_img_metadata'][p]
            file_name = item["filename"]
            image_path = os.path.join(img_path, file_name)
            target_points = [
                list(zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']))
                for region in item['regions']
                if len(region['shape_attributes']['all_points_x']) == 4
                and len(region['shape_attributes']['all_points_y']) == 4
            ]
            if not len(target_points):
                continue

            unique_labels_count = len(set(angles.values()))
            for label, angles_to_choice in label_abgle_map.items():
                # choise random angle for label
                angle = angles_to_choice[0]
                if len(angles_to_choice) > 1:
                    angle = random.choice(angles_to_choice)
                for bbox in target_points:
                    self.samples.append([
                        [image_path, angle, bbox],
                        np.array([1 if i == label else 0 for i in range(unique_labels_count)])
                    ])

        self.n = len(self.samples)
        self.batch_size = batch_size
        self.indexes = list(range(self.n))
        self.batch_count = int(self. n / batch_size)
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
                        points: List,
                        angle: int) -> np.ndarray:
        points = rotate_rect(points, angle)
        points = np.array(points)

        img = cv2.imread(path)
        rotated_img = build_perspective(img, points, self.img_w, self.img_h)

        normalized_img = normalize_img(rotated_img, width=self.img_w, height=self.img_h)
        img_part = np.moveaxis(np.array(normalized_img), 2, 0)
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


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "../../data/dataset/Detector/autoria_numberplate_dataset_example/train/")
    json_path = os.path.join(img_path, "via_region_data.json")
    show = False

    img_orientation_generator = ImgOrientationGenerator(json_path, img_path, img_w=300, img_h=100, batch_size=32)
    for i in range(len(img_orientation_generator.samples)):
        img_data, y = img_orientation_generator.samples[i]
        img_path, angle, points = img_data
        points = rotate_rect(points, angle)
        points = np.array(list(points))
        img = cv2.imread(img_path)
        rotated_img = build_perspective(img, points, img_orientation_generator.img_w, img_orientation_generator.img_h)
        if show:
            cv2.imshow(f"img{angle}", rotated_img)
            cv2.waitKey(0)
