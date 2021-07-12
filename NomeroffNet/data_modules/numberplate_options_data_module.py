import os
import sys
import torch
import pytorch_lightning as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../base')))
from .data_loaders import ImgGenerator


class OptionsNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dir,
                 val_dir,
                 test_dir,
                 class_region=None,
                 class_count_line=None,
                 orientations=None,
                 data_loader=ImgGenerator,
                 width=295,
                 height=64,
                 batch_size=32):
        super().__init__()
        self.batch_size = batch_size

        if orientations is None:
            orientations = [0, 180]
        if class_region is None:
            class_region = []
        if class_count_line is None:
            class_count_line = []

        # init train generator
        self.train = None
        self.train_image_generator = data_loader(
            train_dir,
            width,
            height,
            batch_size,
            [len(class_region), len(class_count_line), len(orientations)])

        # init validation generator
        self.val = None
        self.val_image_generator = data_loader(
            val_dir,
            width,
            height,
            batch_size,
            [len(class_region), len(class_count_line), len(orientations)])

        # init test generator
        self.test = None
        self.test_image_generator = data_loader(
            test_dir,
            width,
            height,
            batch_size,
            [len(class_region), len(class_count_line), len(orientations)])

    def prepare_data(self):
        self.train_image_generator.build_data()
        self.val_image_generator.build_data()
        self.test_image_generator.build_data()

    def setup(self, stage=None):
        self.train_image_generator.rezero()
        self.train = self.train_image_generator

        self.val_image_generator.rezero()
        self.val = self.val_image_generator

        self.test_image_generator.rezero()
        self.test = self.test_image_generator

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size)
