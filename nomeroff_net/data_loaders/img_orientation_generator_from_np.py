"""
Numberplate Orientation Image Generator
python3 -m nomeroff_net.data_loaders.img_orientation_generator_from_np -f nomeroff_net/data_loaders/img_orientation_generator_from_np.py
"""
import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple


class ImgOrientationGenerator(Dataset):

    def __init__(self, root_dir: str, img_w: int = 300, img_h: int = 300, split: str = 'train',
                 batch_size: int = 32, classes=None) -> None:
        self.root_dir = root_dir
        self.img_w = img_w
        self.img_h = img_h
        self.split = split
        self.samples = self._load_samples()
        self.indexes = list(range(len(self.samples)))
        self.cur_index = 0
        if classes is None:
            classes = {'0': 0, '90': 1, '180': 2}
        self.classes = classes

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Завантаження зразків з відповідних директорій."""
        samples = []
        for class_name, class_label in self.classes.items():
            class_dir = os.path.join(self.root_dir, self.split, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                samples.append((img_path, class_label))
        return samples

    def __len__(self) -> int:
        """Повертає загальну кількість зразків."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Генерує один зразок даних."""
        img_path, class_label = self.samples[self.indexes[index]]
        img = cv2.imread(img_path)
        img = self._preprocess_image(img)
        return img, class_label

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Попередня обробка зображення (нормалізація)."""
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img / 255.0
        img = img.transpose((2, 0, 1))  # Переставлення каналів
        img = torch.tensor(img, dtype=torch.float32)
        return img

    def rezero(self) -> None:
        """Перемішує індекси зразків і встановлює поточний індекс на початок."""
        self.cur_index = 0
        random.shuffle(self.indexes)

    def build_data(self):
        return


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "../../data/dataset/OrientationDetector/numberplate_orientation_example/")
    dataset = ImgOrientationGenerator(root_dir, img_w=300, img_h=100, split='train')
    for i in range(len(dataset)):
        img, label = dataset[i]
        print(f"Зображення {i}: мітка {label}, розмір {img.size()}")
