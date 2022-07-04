import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from typing import Optional
from nomeroff_net.data_loaders.fraud_numberplate_generator import FraudImgGenerator as ImgGenerator


class FraudNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dir=None,
                 val_dir=None,
                 test_dir=None,
                 data_loader=ImgGenerator,
                 width=256,
                 height=256,
                 batch_size=32,
                 num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # init train generator
        self.train = None
        self.train_image_generator = None
        if train_dir is not None:
            self.train_image_generator = data_loader(
                train_dir,
                width,
                height,
                batch_size)

        # init validation generator
        self.val = None
        self.val_image_generator = None
        if val_dir is not None:
            self.val_image_generator = data_loader(
                val_dir,
                width,
                height,
                batch_size)

        # init test generator
        self.test = None
        self.test_image_generator = None
        if test_dir is not None:
            self.test_image_generator = data_loader(
                test_dir,
                width,
                height,
                batch_size)

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
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

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
