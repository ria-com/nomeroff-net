import os
import io
import sys
from typing import List, Dict, Tuple
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from nomeroff_net.tools.mcm import (modelhub, get_device_torch)
from nomeroff_net.data_modules.numberplate_options_data_module import OptionsNetDataModule
from nomeroff_net.nnmodels.numberplate_options_model import NPOptionsNet
from nomeroff_net.tools.image_processing import normalize_img, convert_cv_zones_rgb_to_bgr

device_torch = get_device_torch()

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
    "am",
    "ua-military",
    "ru-military",
    "md",
    "eu-ua-custom",
    "ae",
]

CLASS_LINES_ALL = [
    "0",  # garbage
    "1",  # one line
    "2",  # two line
    "3",  # three line
]

CLASS_STATE_ALL = [
    "garbage",  # garbage
    "filled",  # manual filled number
    "not filled",  # two line
    "empty"  # deprecated
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
        self.height = 50
        self.width = 200
        self.color_channels = 3

        # outputs 1
        self.class_region = options.get("class_region", CLASS_REGION_ALL)

        # outputs 2
        self.count_lines = options.get("count_lines", CLASS_LINES_ALL)

        # model
        self.model = None
        self.trainer = None

        # data module
        self.dm = None

        # train hyperparameters
        self.batch_size = 64
        self.epochs = 100
        self.gpus = 0
        self.train_regions = True
        self.train_count_lines = True

        self.class_region_indexes = None
        self.class_region_indexes_global = None

        self.class_lines_indexes = None
        self.class_lines_indexes_global = None

    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    @staticmethod
    def get_class_region_all() -> List:
        return CLASS_REGION_ALL

    @staticmethod
    def get_class_count_lines_all() -> List:
        return CLASS_LINES_ALL

    @staticmethod
    def get_class_state_all() -> List:
        return CLASS_STATE_ALL

    def get_class_region_for_report(self) -> List:
        """
        TODO: Get class_region list for classification_report routine
        """
        class_regions = []
        for region in self.class_region:
            region_item = region
            if type(region) == list:
                region_item = ','.join(region_item)
            class_regions.append(region_item)
        return class_regions

    def create_model(self) -> NPOptionsNet:
        """
        TODO: describe method
        """
        if self.model is None:
            self.model = NPOptionsNet(len(self.class_region),
                                      len(self.count_lines),
                                      batch_size=self.batch_size,
                                      train_regions=self.train_regions,
                                      train_count_lines=self.train_count_lines, )
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
        self.dm = OptionsNetDataModule(
            train_dir,
            validation_dir,
            test_dir,
            self.class_region,
            self.count_lines,
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
              ) -> NPOptionsNet:
        """
        TODO: describe method
        """
        self.create_model()
        if self.gpus:
            self.trainer = pl.Trainer(max_epochs=self.epochs,
                                      accelerator='gpu', devices=self.gpus,
                                      callbacks=self.define_callbacks(log_dir))
        else:
            self.trainer = pl.Trainer(max_epochs=self.epochs,
                                      accelerator='cpu',
                                      callbacks=self.define_callbacks(log_dir))
        self.trainer.fit(self.model, self.dm)
        return self.model

    def tune(self, percentage=0.1) -> Dict:
        """
        TODO: describe method
        TODO: add ReduceLROnPlateau callback
        """
        model = self.create_model()
        if self.gpus:
            trainer = pl.Trainer(
                max_epochs=self.epochs,
                accelerator='gpu', devices=self.gpus
            )
        else:
            trainer = pl.Trainer(
                max_epochs=self.epochs,
                accelerator='cpu'
            )
        num_training = int(len(self.dm.train_image_generator) * percentage) or 1
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model,
                                  self.dm,
                                  num_training=num_training,
                                  early_stop_threshold=None)
        lr = lr_finder.suggestion()
        print(f"Found lr: {lr}")
        model.hparams["learning_rate"] = lr

        return lr_finder

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

    def is_loaded(self) -> bool:
        """
        TODO: describe method
        """
        if self.model is None:
            return False
        return True

    def load_model(self, path_to_model):
        # Load the checkpoint
        checkpoint = torch.load(path_to_model, map_location=torch.device('cpu'))

        # Add a fake pytorch-lightning_version key to the checkpoint if it doesn't exist
        if 'pytorch-lightning_version' not in checkpoint:
            checkpoint['pytorch-lightning_version'] = pl.__version__

        # Save the modified checkpoint to an in-memory buffer
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)

        self.model = NPOptionsNet.load_from_checkpoint(buffer,
                                                       map_location=torch.device('cpu'),
                                                       region_output_size=len(self.class_region),
                                                       count_line_output_size=len(self.count_lines),
                                                       img_h=self.height,
                                                       img_w=self.width,
                                                       batch_size=self.batch_size,
                                                       train_regions=self.train_regions,
                                                       train_count_lines=self.train_count_lines, )
        self.model = self.model.to(device_torch)
        self.model.eval()
        return self.model

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

    def custom_regions_id_to_all_regions(self, indexes: List[int]) -> List[int]:
        """
        TODO: describe method
        """
        return [CLASS_REGION_ALL.index(str(self.class_region[index].replace("_", "-"))) for index in indexes]

    @staticmethod
    def get_regions_label_global(indexes: List[int]) -> List[str]:
        """
        TODO: describe method
        """
        return [CLASS_REGION_ALL[index].replace("-", "_") for index in indexes]

    def get_count_lines_label(self, index: int) -> int:
        """
        TODO: describe method
        """
        return int(self.count_lines[index])

    def custom_regions_id_to_all_regions_with_confidences(self,
                                                          indexes: List[int],
                                                          confidences: List) -> Tuple[List[int],
                                                                                      List]:
        """
        TODO: describe method
        """
        global_indexes = self.custom_regions_id_to_all_regions(indexes)
        self.class_region_indexes = [i for i, _ in enumerate(self.class_region)]
        self.class_region_indexes_global = self.custom_regions_id_to_all_regions(
            self.class_region_indexes)
        global_confidences = [[confidence[self.class_region_indexes.index(self.class_region_indexes_global.index(i))]
                               if i in self.class_region_indexes_global
                               else 0
                               for i, _
                               in enumerate(CLASS_REGION_ALL)]
                              for confidence in confidences]
        return global_indexes, global_confidences

    def custom_count_lines_id_to_all_count_lines(self, indexes: List[int]) -> List[int]:
        """
        TODO: describe method
        """
        return [CLASS_LINES_ALL.index(str(self.count_lines[index])) for index in indexes]

    def custom_count_lines_id_to_all_count_lines_with_confidences(self,
                                                                  global_indexes: List[int],
                                                                  confidences: List) -> Tuple[List[int],
                                                                                              List]:
        """
        TODO: describe method
        """
        self.class_lines_indexes = [i for i, _ in enumerate(self.count_lines)]
        self.class_lines_indexes_global = self.custom_count_lines_id_to_all_count_lines(
            self.class_lines_indexes)
        global_confidences = [[confidence[self.class_lines_indexes.index(self.class_lines_indexes_global.index(i))]
                               if i in self.class_lines_indexes_global
                               else 0
                               for i, _
                               in enumerate(CLASS_LINES_ALL)]
                              for confidence in confidences]
        return global_indexes, global_confidences

    @staticmethod
    def get_count_lines_labels_global(indexes: List[int]) -> List[int]:
        """
        TODO: describe method
        """
        return [int(CLASS_LINES_ALL[index]) for index in indexes]

    def get_count_lines_labels(self, indexes: List[int]) -> List[int]:
        """
        TODO: describe method
        """
        return [int(self.count_lines[index]) for index in indexes]

    def load_meta(self, path_to_model: str = "latest", options: Dict = None) -> NPOptionsNet:
        if options is None:
            options = dict()
        self.__dict__.update(options)

        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("numberplate_options")
            path_to_model = model_info["path"]
            self.class_region = model_info["class_region"]
            self.count_lines = model_info["count_lines"]
            self.height = model_info.get("height", self.height)
            self.width = model_info.get("width", self.width)
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model, self.get_classname(), "numberplate_options")
            path_to_model = model_info["path"]
        elif path_to_model.startswith("modelhub://"):
            path_to_model = path_to_model.split("modelhub://")[1]
            model_info = modelhub.download_model_by_name(path_to_model)
            path_to_model = model_info["path"]
            self.class_region = model_info["class_region"]
            self.count_lines = model_info["count_lines"]
            self.height = model_info.get("height", self.height)
            self.width = model_info.get("width", self.width)
        return path_to_model

    def load(self, path_to_model: str = "latest", options: Dict = None) -> NPOptionsNet:
        """
        TODO: describe method
        """
        path_to_model = self.load_meta(path_to_model, options)
        self.create_model()
        return self.load_model(path_to_model)

    def predict(self, imgs: List[np.ndarray], return_acc: bool = False) -> Tuple:
        """
        Predict options(region, count lines) by numberplate images
        """
        region_ids, count_lines, confidences, predicted = self.predict_with_confidence(imgs)
        if return_acc:
            return region_ids, count_lines, predicted
        return region_ids, count_lines

    def _predict(self, xs):
        x = torch.tensor(np.moveaxis(np.array(xs), 3, 1))
        x = x.to(device_torch)
        predicted = [p.cpu().numpy() for p in self.model(x)]
        return predicted

    @staticmethod
    def unzip_predicted(predicted):
        confidences, region_ids, count_lines = [], [], []
        for region, count_line in zip(predicted[0], predicted[1]):
            region_ids.append(int(np.argmax(region)))
            count_lines.append(int(np.argmax(count_line)))
            region = region.tolist()
            count_line = count_line.tolist()
            region_confidence = region[int(np.argmax(region))]
            count_lines_confidence = count_line[int(np.argmax(count_line))]
            confidences.append([region_confidence, count_lines_confidence])
        return confidences, region_ids, count_lines

    def preprocess(self, images):
        x = [normalize_img(img, height=self.height, width=self.width) for img in images]
        x = np.moveaxis(np.array(x), 3, 1)
        return x

    def forward(self, inputs):
        x = torch.tensor(inputs)
        x = x.to(device_torch)
        model_output = self.model(x)
        return model_output

    @torch.no_grad()
    def predict_with_confidence(self, imgs: List[np.ndarray or List]) -> Tuple:
        """
        Predict options(region, count lines) with confidence by numberplate images
        """
        xs = [normalize_img(img, height=self.height, width=self.width) for img in imgs]
        if not bool(xs):
            return [], [], [], []
        predicted = self._predict(xs)

        confidences, region_ids, count_lines = self.unzip_predicted(predicted)
        count_lines = self.custom_count_lines_id_to_all_count_lines(count_lines)
        return region_ids, count_lines, confidences, predicted
