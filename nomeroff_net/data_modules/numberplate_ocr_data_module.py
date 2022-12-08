import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from typing import Optional

from nomeroff_net.data_loaders import TextImageGenerator


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
                 num_workers=0,
                 seed=42,
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
            img_w=width,
            img_h=height,
            batch_size=batch_size,
            seed=seed,
            with_aug=with_aug)

        # init validation generator
        self.val = None
        self.val_image_generator = TextImageGenerator(
            val_dir,
            letters,
            max_text_len,
            img_w=width,
            img_h=height,
            batch_size=batch_size)

        # init test generator
        self.test = None
        self.test_image_generator = TextImageGenerator(
            test_dir,
            letters,
            max_text_len,
            img_w=width,
            img_h=height,
            batch_size=batch_size)

    def prepare_data(self):
        return

    def setup(self, stage=None):
        self.train = self.train_image_generator
        self.val = self.val_image_generator
        self.test = self.test_image_generator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None):
        # clean up after fit or test
        # called on every process in DDP
        ...
