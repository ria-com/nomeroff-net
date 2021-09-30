import os
import sys
from typing import List, Dict, Tuple
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from tools import (modelhub,
                   get_mode_torch)
from data_modules.numberplate_inverse_data_module import InverseNetDataModule
from nnmodels.numberplate_inverse_model import NPInverseNet
from data_modules.data_loaders import normalize

mode_torch = get_mode_torch()

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
            if mode_torch == "gpu":
                self.model = self.model.cuda()
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

    def train(self,
              log_dir=sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/logs/inverse')))
              ) -> NPInverseNet:
        """
        TODO: describe method
        TODO: add ReduceLROnPlateau callback
        """
        self.create_model()
        checkpoint_callback = ModelCheckpoint(dirpath=log_dir, monitor='val_loss')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.trainer = pl.Trainer(max_epochs=self.epochs,
                                  gpus=self.gpus,
                                  callbacks=[checkpoint_callback, lr_monitor])
        self.trainer.fit(self.model, self.dm)
        print("[INFO] best model path", checkpoint_callback.best_model_path)
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
                             gpus=self.gpus)
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
            self.trainer.save_checkpoint(path)

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
        if mode_torch == "gpu":
            self.model = self.model.cuda()
        self.model.eval()
        return self.model

    def getInverseLabel(self, index: int) -> int:
        """
        TODO: describe method
        """
        return self.orientations[index]

    def getInverseLabels(self, indexes: List[int]) -> List[int]:
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
            self.orientation = model_info["orientation"]
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

    @torch.no_grad()
    def predict_with_confidence(self, imgs: List[np.ndarray]) -> Tuple:
        """
        TODO: describe method
        """
        xs = []
        for img in imgs:
            xs.append(normalize(img))

        predicted = [[], []]
        if bool(xs):
            x = torch.tensor(np.moveaxis(np.array(xs), 3, 1))
            if mode_torch == "gpu":
                x = x.cuda()
            predicted = self.model(x)
            predicted = [p.cpu().numpy() for p in predicted]

        confidences = []
        orientations = []
        for orientation in predicted:
            orientation_index = int(np.argmax(orientation))
            orientations.append(orientation_index)
            orientation_confidence = orientation[orientation_index].tolist()
            confidences.append(orientation_confidence)
        return orientations, confidences, predicted
