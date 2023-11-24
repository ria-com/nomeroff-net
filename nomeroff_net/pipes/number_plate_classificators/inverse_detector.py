import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl

from typing import List, Dict, Tuple
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from nomeroff_net.tools.mcm import (modelhub, get_device_torch)
from nomeroff_net.data_modules.numberplate_inverse_data_module import InverseNetDataModule
from nomeroff_net.nnmodels.numberplate_inverse_model import NPInverseNet
from nomeroff_net.tools.image_processing import normalize_img

device_torch = get_device_torch()

ORIENTATION_ALL = [
    "0",
    "180"
]


def imshow(img: np.ndarray) -> None:
    """
    # functions to show an image
    """
    import matplotlib.pyplot as plt
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class InverseDetector(object):
    """
    TODO: describe class
    """

    def __init__(self, options: Dict = None) -> None:
        """
        TODO: describe __init__
        """
        if options is None:
            options = dict()

        # input
        self.height = 64
        self.width = 295
        self.color_channels = 3

        # outputs
        self.orientations = options.get("orientations", ORIENTATION_ALL)

        # model
        self.model = None
        self.trainer = None

        # data module
        self.dm = None

        # train hyperparameters
        self.batch_size = 64
        self.epochs = 100
        self.gpus = 1

    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    @staticmethod
    def get_class_inverse_all() -> List:
        return ORIENTATION_ALL

    def create_model(self) -> NPInverseNet:
        """
        TODO: describe method
        """
        if self.model is None:
            self.model = NPInverseNet(len(self.orientations),
                                      self.height,
                                      self.width,
                                      self.batch_size)
            self.model = self.model.to(device_torch)
        return self.model

    def prepare(self,
                base_dir: str,
                num_workers: int = 0,
                verbose: bool = True) -> None:
        """
        TODO: describe method
        """
        if verbose:
            print("START PREPARING")
        # you mast split your data on 3 directory
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'val')
        test_dir = os.path.join(base_dir, 'test')

        # compile generators
        self.dm = InverseNetDataModule(
            train_dir,
            validation_dir,
            test_dir,
            self.orientations,
            width=self.width,
            height=self.height,
            batch_size=self.batch_size,
            num_workers=num_workers)

        if verbose:
            print("DATA PREPARED")

    @staticmethod
    def define_callbacks(log_dir):
        checkpoint_callback = ModelCheckpoint(dirpath=log_dir, monitor='val_loss')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return [checkpoint_callback, lr_monitor]

    def train(self,
              log_dir=sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/logs/options')))
              ) -> NPInverseNet:
        """
        TODO: describe method
        """
        self.create_model()
        self.trainer = pl.Trainer(max_epochs=self.epochs,
                                  #gpus=self.gpus,
                                  accelerator='gpu', devices=self.gpus,
                                  callbacks=self.define_callbacks(log_dir))
        self.trainer.fit(self.model, self.dm)
        self.trainer.test()
        return self.model

    def tune(self) -> Dict:
        """
        TODO: describe method
        TODO: add ReduceLROnPlateau callback
        """
        model = self.create_model()
        trainer = pl.Trainer(auto_lr_find=True,
                             max_epochs=self.epochs,
                             #gpus=self.gpus,
                             accelerator='gpu', devices=self.gpus,
                             )
        return trainer.tune(model, self.dm)

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
            self.trainer.save_checkpoint(path, weights_only=True)

    def is_loaded(self) -> bool:
        """
        TODO: describe method
        """
        if self.model is None:
            return False
        return True

    def load_model(self, path_to_model):
        self.model = NPInverseNet.load_from_checkpoint(path_to_model,
                                                       map_location=torch.device('cpu'),
                                                       orientation_output_size=len(self.orientations))
        self.model = self.model.to(device_torch)
        self.model.eval()
        return self.model

    def get_inverse_label(self, index: int) -> str:
        """
        TODO: describe method
        """
        return self.orientations[index]

    def get_inverse_labels(self, indexes: List[int]) -> List[str]:
        """
        TODO: describe method
        """
        return [self.orientations[index] for index in indexes]

    def load(self, path_to_model: str = "latest", options: Dict = None) -> NPInverseNet:
        """
        TODO: describe method
        """
        if options is None:
            options = dict()
        self.__dict__.update(options)

        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("numberplate_orientations")
            path_to_model = model_info["path"]
            self.orientations = model_info["orientations"]
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model, self.get_classname(), "numberplate_orientations")
            path_to_model = model_info["path"]
        self.create_model()
        return self.load_model(path_to_model)

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
        return confidences, orientations

    @torch.no_grad()
    def predict_with_confidence(self, imgs: List[np.ndarray]) -> Tuple:
        """
        TODO: describe method
        """
        xs = [normalize_img(img) for img in imgs]
        if not bool(xs):
            return [], []
        predicted = self._predict(xs)
        confidences, orientations = self.unzip_predicted(predicted)
        return orientations, confidences, predicted
