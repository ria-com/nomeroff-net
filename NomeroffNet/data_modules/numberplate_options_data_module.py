import os
import sys
import torch
import pytorch_lightning as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../base')))
from .option_img_generator import ImgGenerator


class OptionsNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dir,
                 val_dir,
                 test_dir,
                 class_region,
                 class_count_line,
                 width=295,
                 height=64,
                 batch_size=32):
        super().__init__()
        self.batch_size = batch_size

        # init train generator
        self.train = None
        self.train_image_generator = ImgGenerator(
            train_dir,
            width,
            height,
            batch_size,
            [len(class_region), len(class_count_line)])

        # init validation generator
        self.val = None
        self.val_image_generator = ImgGenerator(
            val_dir,
            width,
            height,
            batch_size,
            [len(class_region), len(class_count_line)])

        # init test generator
        self.test = None
        self.test_image_generator = ImgGenerator(
            test_dir,
            width,
            height,
            batch_size,
            [len(class_region), len(class_count_line)])

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
