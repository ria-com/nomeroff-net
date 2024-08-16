"""
Orientation Detector

python3 -m nomeroff_net.pipes.number_plate_classificators.orientation_detector -f nomeroff_net/pipes/number_plate_classificators/orientation_detector.py
"""
import os
from typing import List, Dict, Tuple
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from nomeroff_net.tools.mcm import (modelhub, get_device_torch)
from nomeroff_net.data_modules.numberplate_orientation_data_module import OrientationDataModule
from nomeroff_net.nnmodels.numberplate_orientation_model import NPOrientationNet
from nomeroff_net.tools.image_processing import normalize_img

device_torch = get_device_torch()


def prettify_orientation(photo_orientation):
    pretty_orientation = 0
    if photo_orientation == 0:
        pretty_orientation = 0
    if photo_orientation == 1:
        pretty_orientation = 90
    if photo_orientation == 2:
        pretty_orientation = 180
    if photo_orientation == 3:
        pretty_orientation = 270
    return pretty_orientation


class OrientationDetector(object):
    """
    TODO: describe class
    """

    def __init__(self,
                 classes=None,
                 ) -> None:
        """
        TODO: describe __init__
        """
        super().__init__()

        # input
        self.height = 224
        self.width = 224
        self.color_channels = 3
        self.backbone = "efficientnet_v2_l"

        # output
        if classes is None:
            classes = {'0': 0, '90': 1}
        self.classes = classes
        self.output_size = len(self.classes)

        # model
        self.model = None
        self.trainer = None
        self.checkpoint_callback = None

        # data module
        self.dm = None

        # train hyperparameters
        self.batch_size = 64
        self.epochs = 100
        self.gpus = 0

    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def create_model(self) -> NPOrientationNet:
        """
        TODO: describe method
        """
        self.model = NPOrientationNet(height=self.height,
                                      width=self.width,
                                      output_size=self.output_size,
                                      batch_size=self.batch_size)
        self.model = self.model.to(device_torch)
        return self.model

    def prepare(self,
                base_dir: str,
                num_workers=0,
                verbose=False) -> None:
        """
        TODO: describe method
        """

        # compile generators
        self.dm = OrientationDataModule(
            base_dir,
            width=self.width,
            height=self.height,
            classes=self.classes,
            num_workers=num_workers,
            batch_size=self.batch_size)

    def load_model(self, path_to_model):
        self.model = NPOrientationNet.load_from_checkpoint(
            path_to_model,
            map_location=torch.device('cpu'),
            output_size=self.output_size,
            backbone=self.backbone,
        )
        self.model = self.model.to(device_torch)
        self.model.eval()
        return self.model

    def test(self) -> List:
        """
        TODO: describe method
        """
        return self.trainer.test()

    def save(self, path: str, verbose: bool = True) -> None:
        """
        TODO: describe method
        """
        if self.model is not None:
            if bool(verbose):
                print("model save to {}".format(path))
            if self.trainer is None:
                torch.save({"state_dict": self.model.state_dict()}, path)
            else:
                self.trainer.save_checkpoint(path, weights_only=True)

    def load(self, path_to_model: str = "latest") -> NPOrientationNet:
        """
        Load model
        path_to_model - http, path or latest
        """
        self.create_model()
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("numberplate_orientation")
            path_to_model = model_info["path"]
            if model_info.get("classes", None) is not None:
                self.classes = model_info["classes"]
                self.output_size = len(self.classes)
                self.backbone = model_info.get("backbone", "vit_l_16")
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model, self.get_classname(), "numberplate_orientation")
            path_to_model = model_info["path"]
        elif path_to_model.startswith("modelhub://"):
            path_to_model = path_to_model.split("modelhub://")[1]
            model_info = modelhub.download_model_by_name(path_to_model)
            path_to_model = model_info["path"]
            if model_info.get("classes", None) is not None:
                self.classes = model_info["classes"]
                self.output_size = len(self.classes)
                self.backbone = model_info.get("backbone", "vit_l_16")

        return self.load_model(path_to_model)

    def define_callbacks(self, log_dir):
        self.checkpoint_callback = ModelCheckpoint(dirpath=log_dir, monitor='val_loss')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return [self.checkpoint_callback, lr_monitor]

    def train(self,
              log_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '../data/logs/options'))
              ) -> NPOrientationNet:
        """
        TODO: describe method
        TODO: add ReduceLROnPlateau callback
        """
        self.create_model()
        self.dm.setup('fit')
        if self.gpus:
            self.trainer = pl.Trainer(max_epochs=self.epochs, accelerator='gpu', devices=self.gpus,
                                      callbacks=self.define_callbacks(log_dir))
        else:
            self.trainer = pl.Trainer(max_epochs=self.epochs, accelerator='cpu',
                                      callbacks=self.define_callbacks(log_dir))
        self.trainer.fit(self.model, datamodule=self.dm)
        print("[INFO] best model path", self.checkpoint_callback.best_model_path)
        self.trainer.test(datamodule=self.dm)
        return self.model

    def tune(self) -> Dict:
        """
        TODO: describe method
        TODO: add ReduceLROnPlateau callback
        """
        model = self.create_model()
        trainer = pl.Trainer(auto_lr_find=True,
                             max_epochs=self.epochs,
                             # gpus=self.gpus,
                             accelerator='gpu', devices=self.gpus,
                             )
        return trainer.tune(model, self.dm)

    def predict(self, imgs: List[np.ndarray], return_acc=False) -> Tuple:
        """
        TODO: describe method
        """
        orientations, confidences, predicted = self.predict_with_confidence(imgs)
        if return_acc:
            return orientations, predicted
        return orientations

    def _predict(self, xs):
        x = torch.tensor(np.moveaxis(np.array(xs), 3, 1))
        x = x.to(device_torch)
        predicted = [p.cpu().numpy() for p in self.model(x)]
        return predicted

    @staticmethod
    def unzip_predicted(predicted):
        confidences, orientations = [], []
        for orientation in predicted:
            orientation_index = int(np.argmax(orientation))
            orientations.append(orientation_index)
            confidences.append(orientation[orientation_index].tolist())
        return orientations, confidences

    @torch.no_grad()
    def predict_with_confidence(self, imgs: List[np.ndarray]) -> Tuple:
        """
        TODO: describe method
        """
        xs = [normalize_img(img, height=self.height, width=self.width) for img in imgs]
        if not bool(xs):
            return [], [], []
        predicted = self._predict(xs)
        orientations, confidences = self.unzip_predicted(predicted)
        return orientations, confidences, predicted

    def get_orientations(self, index: int) -> int:
        """
        TODO: describe method
        """
        return self.orientations[index]


if __name__ == "__main__":
    import os
    import cv2
    import glob

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "../../../data/dataset/OrientationDetector/numberplate_orientation_example/")

    orientation_detector = OrientationDetector()
    orientation_detector.load()
    images_path = list(glob.glob(os.path.join(root_dir, "test", "180", "*")))
    images = [cv2.imread(image_path) for image_path in images_path]
    res = orientation_detector.predict(images)
    print(res)
