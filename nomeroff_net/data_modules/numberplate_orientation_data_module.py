"""
Numberplate Orientation Image Generator
python3 -m nomeroff_net.data_modules.numberplate_orientation_data_module -f nomeroff_net/data_modules/numberplate_orientation_data_module.py
"""
from nomeroff_net.data_loaders import ImgOrientationGenerator
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from .numberplate_options_data_module import OptionsNetDataModule
from torch.utils.data import DataLoader
from typing import Optional


class OrientationDataModule(OptionsNetDataModule):
    def __init__(self,
                 dataset_dir="../../data/dataset/OrientationDetector/numberplate_orientation_example/",
                 width=300,
                 height=300,
                 batch_size=32,
                 num_workers=0,
                 classes=None,):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.width = width
        self.height = height

        self.train_image_generator = ImgOrientationGenerator(
            dataset_dir,
            split="train",
            img_w=width,
            img_h=height,
            classes=classes,
        )

        self.val_image_generator = ImgOrientationGenerator(
            dataset_dir,
            split="val",
            img_w=width,
            img_h=height,
            classes=classes,
        )

        self.test_image_generator = ImgOrientationGenerator(
            dataset_dir,
            split="test",
            img_w=width,
            img_h=height,
            classes=classes,
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_image_generator.rezero()
            self.val_image_generator.rezero()
        if stage == 'test' or stage is None:
            self.test_image_generator.rezero()

    def train_dataloader(self):
        return DataLoader(self.train_image_generator,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_image_generator,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_image_generator is None or len(self.test_image_generator) == 0:
            raise ValueError("Test dataset is empty or not properly initialized.")
        return DataLoader(self.test_image_generator,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None):
        pass


if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "../../data/dataset/OrientationDetector/numberplate_orientation_example/")

    dm = OrientationDataModule(root_dir)
    dm.setup(stage='test')
    test_loader = dm.test_dataloader()
    for batch in test_loader:
        print(batch)  # Перевірка вмісту батчу
