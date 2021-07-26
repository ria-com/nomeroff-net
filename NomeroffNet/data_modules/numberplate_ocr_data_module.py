import os
import sys
import torch
import pytorch_lightning as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../base')))
from .data_loaders import TextImageGenerator


class OcrNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dir,
                 val_dir,
                 test_dir,
                 letters,
                 max_text_len,
                 width=128,
                 height=64,
                 batch_size=32,
                 max_plate_length=8,
                 num_workers=0,
                 with_aug=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # init train generator
        self.train = None
        self.train_image_generator = TextImageGenerator(
            train_dir,
            letters,
            max_text_len,
            width,
            height,
            batch_size,
            max_plate_length,
            with_aug)

        # init validation generator
        self.val = None
        self.val_image_generator = TextImageGenerator(
            val_dir,
            letters,
            max_text_len,
            width,
            height,
            batch_size,
            max_plate_length,
            with_aug)

        # init test generator
        self.test = None
        self.test_image_generator = TextImageGenerator(
            test_dir,
            letters,
            max_text_len,
            width,
            height,
            batch_size,
            max_plate_length,
            with_aug)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train = self.train_image_generator
        self.val = self.val_image_generator
        self.test = self.test_image_generator

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train,
                                           batch_size=self.batch_size,
                                           #drop_last=True,
                                           num_workers=self.num_workers,
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val,
                                           batch_size=self.batch_size,
                                           #drop_last=True,
                                           num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test,
                                           batch_size=self.batch_size,
                                           #drop_last=True,
                                           num_workers=self.num_workers)
