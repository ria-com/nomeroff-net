import os
import sys
from typing import List, Dict, Tuple
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'base')))
from .tools import (modelhub,
                    get_mode_torch)
from data_modules.numberplate_options_data_module import OptionsNetDataModule
from nnmodels.numberplate_options_model import NPOptionsNet
from data_modules.option_img_generator import normalize

mode_torch = get_mode_torch()

CLASS_REGION_ALL = [
            "xx-unknown",
            "eu-ua-2015",
            "eu-ua-2004",
            "eu-ua-1995",
            "eu",
            "xx-transit",
            "ru",
            "kz",
            "eu-ua-ordlo-dpr",
            "eu-ua-ordlo-lpr",
            "ge",
            "by",
            "su",
            "kg",
            "am"
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


class OptionsDetector(object):
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

        # outputs 1
        self.class_region = options.get("class_region", CLASS_REGION_ALL)

        # outputs 2
        self.count_lines = options.get("count_lines", [
            0,
            1,
            2,
            3
        ])

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
    def get_class_region_all() -> List:
        return CLASS_REGION_ALL

    def create_model(self) -> NPOptionsNet:
        """
        TODO: describe method
        """
        self.model = NPOptionsNet(len(self.class_region),
                                  len(self.count_lines),
                                  self.height,
                                  self.width,
                                  self.batch_size)
        if mode_torch == "gpu":
            self.model = self.model.cuda()
        return self.model

    def prepare(self, base_dir: str, verbose: bool = True) -> None:
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
        self.dm = OptionsNetDataModule(
            train_dir,
            validation_dir,
            test_dir,
            self.class_region,
            self.count_lines,
            width=self.width,
            height=self.height,
            batch_size=self.batch_size)

        if verbose:
            print("DATA PREPARED")

    def train(self,
              log_dir=sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/logs/options')))
              ) -> NPOptionsNet:
        """
        TODO: describe method
        TODO: add ReduceLROnPlateau callback
        """
        self.create_model()
        checkpoint_callback = ModelCheckpoint(dirpath=log_dir, monitor='val_loss')
        self.trainer = pl.Trainer(max_epochs=self.epochs,
                                  gpus=self.gpus,
                                  callbacks=[checkpoint_callback])
        self.trainer.fit(self.model, self.dm)
        print("[INFO] best model path", checkpoint_callback.best_model_path)
        self.trainer.test()
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
            self.trainer.save_checkpoint(path)

    def is_loaded(self) -> bool:
        """
        TODO: describe method
        """
        if self.model is None:
            return False
        return True

    def load(self, path_to_model: str = "latest", options: Dict = None) -> NPOptionsNet:
        """
        TODO: describe method
        """
        if options is None:
            options = dict()
        self.create_model()
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("numberplate_options")
            path_to_model = model_info["path"]
            options["class_region"] = model_info["class_region"]
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model, self.get_classname(), "numberplate_options")
            path_to_model = model_info["path"]

        self.class_region = options.get("class_region", CLASS_REGION_ALL)

        if mode_torch == "gpu":
            self.model.load_from_checkpoint(path_to_model,
                                            region_output_size=len(self.class_region),
                                            count_line_output_size=len(self.count_lines))
        else:
            self.model.load_from_checkpoint(path_to_model,
                                            map_location=torch.device('cpu'),
                                            region_output_size=len(self.class_region),
                                            count_line_output_size=len(self.count_lines))
        self.model.eval()
        return self.model

    def predict(self, imgs: List[np.ndarray], return_acc=False) -> Tuple:
        """
        TODO: describe method
        """
        region_ids, count_lines, confidences, predicted = self.predict_with_confidence(imgs)
        if return_acc:
            return region_ids, count_lines, predicted
        return region_ids, count_lines

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
        region_ids = []
        count_lines = []
        for region, count_line in zip(predicted[0], predicted[1]):
            region_ids.append(int(np.argmax(region)))
            count_lines.append(int(np.argmax(count_line)))
            region = region.tolist()
            count_line = count_line.tolist()
            region_confidence = region[int(np.argmax(region))]
            count_lines_confidence = count_line[int(np.argmax(count_line))]
            confidences.append([region_confidence, count_lines_confidence])
        return region_ids, count_lines, confidences, predicted

    def get_region_label(self, index: int) -> str:
        """
        TODO: describe method
        """
        return self.class_region[index].replace("-", "_")

    def get_region_labels(self, indexes: List[int]) -> List[str]:
        """
        TODO: describe method
        """
        return [self.class_region[index].replace("-", "_") for index in indexes]
