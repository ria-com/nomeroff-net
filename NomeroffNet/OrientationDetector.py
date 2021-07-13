import os
import sys
from typing import List, Dict, Tuple
import numpy as np

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'base')))
from .tools import (modelhub,
                    get_mode_torch)
from data_modules.numberplate_options_data_module import OptionsNetDataModule
from nnmodels.numberplate_orientation_model import NPOrientationNet
from data_modules.data_loaders import OrientationImgGenerator
from data_modules.data_loaders import normalize
from .OptionsDetector import OptionsDetector

mode_torch = get_mode_torch()


class OrientationDetector(OptionsDetector):
    """
    TODO: describe class
    """
    def __init__(self, orientations: List = None) -> None:
        """
        TODO: describe __init__
        """
        super().__init__()

        if orientations is None:
            orientations = [0, 180]

        # input
        self.height = 64
        self.width = 295
        self.color_channels = 3

        self.orientations = orientations

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

    def create_model(self) -> OptionsNetDataModule:
        """
        TODO: describe method
        """
        self.model = NPOrientationNet(len(self.orientations),
                                      self.height,
                                      self.width,
                                      self.batch_size)
        if mode_torch == "gpu":
            self.model = self.model.cuda()
        return self.model

    def prepare(self, base_dir: str, verbose=False) -> None:
        """
        TODO: describe method
        """
        # you mast split your data on 3 directory
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'val')
        test_dir = os.path.join(base_dir, 'test')

        # compile generators
        self.dm = OptionsNetDataModule(
            train_dir,
            validation_dir,
            test_dir,
            orientations=self.orientations,
            data_loader=OrientationImgGenerator,
            width=self.width,
            height=self.height,
            batch_size=self.batch_size)

    def load_model(self, path_to_model):
        if mode_torch == "gpu":
            self.model = NPOrientationNet.load_from_checkpoint(path_to_model,
                                                               orientation_output_size=len(self.orientations))
        else:
            self.model = NPOrientationNet.load_from_checkpoint(path_to_model,
                                                               map_location=torch.device('cpu'),
                                                               orientation_output_size=len(self.orientations))
        self.model.eval()
        return self.model

    def load(self, path_to_model: str = "latest", options: Dict = None) -> NPOrientationNet:
        """
        TODO: describe method
        """
        if options is None:
            options = dict()
        self.create_model()
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("numberplate_orientation")
            path_to_model = model_info["path"]
            options["orientations"] = model_info["orientations"]
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model, self.get_classname(), "numberplate_orientation")
            path_to_model = model_info["path"]

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
        for orientation in zip(predicted):
            print(orientation)
            orientation_confidence = orientation[int(np.argmax(orientation))]
            orientations.append(int(np.argmax(orientation)))
            confidences.append(orientation_confidence)
        return orientations, confidences, predicted

    def get_orientations(self, index: int) -> int:
        """
        TODO: describe method
        """
        return self.orientations[index]
