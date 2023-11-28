"""
python3 -m nomeroff_net.text_detectors.base.ocr -f nomeroff_net/text_detectors/base/ocr.py
"""
import os
import cv2
import json
import numpy as np
import torch
import pytorch_lightning as pl

from collections import Counter
from torch.nn import functional
from pytorch_lightning.tuner.tuning import Tuner
from typing import List, Tuple, Any, Dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import models
from nomeroff_net.data_modules.numberplate_ocr_data_module import OcrNetDataModule
from nomeroff_net.nnmodels.ocr_model import NPOcrNet, weights_init

from nomeroff_net.tools.image_processing import normalize_img
from nomeroff_net.tools.errors import OCRError
from nomeroff_net.tools.mcm import modelhub, get_device_torch
from nomeroff_net.tools.augmentations import aug_seed
from nomeroff_net.tools.ocr_tools import (StrLabelConverter,
                                          decode_prediction,
                                          decode_batch)

device_torch = get_device_torch()


class OCR(object):

    def __init__(self, model_name: str = None, letters: List = None, linear_size: int = 512,
                 max_text_len: int = 0, height: int = 50, width: int = 200, color_channels: int = 3,
                 hidden_size: int = 32, backbone: str = "resnet18",
                 off_number_plate_classification=True, **_) -> None:
        self.model_name = model_name
        self.letters = []
        if letters is not None:
            self.letters = letters

        # model
        self.dm = None
        self.model = None
        self.trainer = None

        # Input parameters
        self.linear_size = linear_size
        self.max_text_len = max_text_len
        self.height = height
        self.width = width
        self.color_channels = color_channels

        # Train hyperparameters
        self.hidden_size = hidden_size
        self.backbone = getattr(models, backbone)
        self.batch_size = 32
        self.epochs = 1
        self.gpus = 1

        self.label_converter = None
        self.path_to_model = None

    def init_label_converter(self):
        self.label_converter = StrLabelConverter("".join(self.letters), self.max_text_len)

    @staticmethod
    def get_counter(dirpath: str, verbose: bool = True) -> Tuple[Counter, int]:
        dir_name = os.path.basename(dirpath)
        ann_dirpath = os.path.join(dirpath, 'ann')
        letters = ''
        lens = []
        for file_name in os.listdir(ann_dirpath):
            json_filepath = os.path.join(ann_dirpath, file_name)
            description = json.load(open(json_filepath, 'r'))['description']
            lens.append(len(description))
            letters += description
        max_text_len = max(Counter(lens).keys())
        if verbose:
            print('Max plate length in "%s":' % dir_name, max_text_len)
        return Counter(letters), max_text_len

    def get_alphabet(self, train_path: str, test_path: str, val_path: str, verbose: bool = True) -> Tuple[List, int]:
        c_val, max_text_len_val = self.get_counter(val_path)
        c_train, max_text_len_train = self.get_counter(train_path)
        c_test, max_text_len_test = self.get_counter(test_path)

        letters_train = set(c_train.keys())
        letters_val = set(c_val.keys())
        letters_test = set(c_test.keys())
        if verbose:
            print("Letters train ", letters_train)
            print("Letters val ", letters_val)
            print("Letters test ", letters_test)

        if max_text_len_val == max_text_len_train:
            if verbose:
                print('Max plate length in train, test and val do match')
        else:
            raise OCRError('Max plate length in train, test and val do not match')

        if letters_train == letters_val:
            if verbose:
                print('Letters in train, val and test do match')
        else:
            raise OCRError('Letters in train, val and test do not match')

        self.letters = sorted(list(letters_train))
        self.max_text_len = max_text_len_train
        if verbose:
            print('Letters:', ' '.join(self.letters))
        return self.letters, self.max_text_len

    def prepare(self,
                path_to_dataset: str,
                use_aug: bool = False,
                seed: int = 42,
                verbose: bool = True,
                num_workers: int = 0) -> None:
        train_dir = os.path.join(path_to_dataset, "train")
        test_dir = os.path.join(path_to_dataset, "test")
        val_dir = os.path.join(path_to_dataset, "val")

        if verbose:
            print("GET ALPHABET")
        self.letters, self.max_text_len = self.get_alphabet(
            train_dir,
            test_dir,
            val_dir,
            verbose=verbose)
        self.init_label_converter()

        if verbose:
            print("START BUILD DATA")
        # compile generators
        self.dm = OcrNetDataModule(
            train_dir,
            val_dir,
            test_dir,
            self.letters,
            self.max_text_len,
            width=self.width,
            height=self.height,
            batch_size=self.batch_size,
            num_workers=num_workers,
            seed=seed,
            with_aug=use_aug)
        if verbose:
            print("DATA PREPARED")

    def create_model(self):
        """
        TODO: describe method
        """
        self.model = NPOcrNet(self.letters,
                              linear_size=self.linear_size,
                              hidden_size=self.hidden_size,
                              backbone=self.backbone,
                              letters_max=len(self.letters) + 1,
                              label_converter=self.label_converter,
                              height=self.height,
                              width=self.width,
                              color_channels=self.color_channels,
                              max_text_len=self.max_text_len)
        if 'resnet' in str(self.backbone):
            self.model.apply(weights_init)
        self.model = self.model.to(device_torch)
        return self.model

    def train(self,
              log_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../data/logs/ocr')),
              seed: int = None,
              ckpt_path: str = None
              ) -> NPOcrNet:
        """
        TODO: describe method
        """
        if seed is not None:
            aug_seed(seed)
            pl.seed_everything(seed)
        if self.model is None:
            self.create_model()
        checkpoint_callback = ModelCheckpoint(dirpath=log_dir, monitor='val_loss')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        if self.gpus:
            self.trainer = pl.Trainer(max_epochs=self.epochs,
                                      accelerator='gpu', devices=self.gpus,
                                      callbacks=[checkpoint_callback, lr_monitor])
        else:
            self.trainer = pl.Trainer(max_epochs=self.epochs,
                                      accelerator='cpu',
                                      callbacks=[checkpoint_callback, lr_monitor])
        self.trainer.fit(self.model, self.dm, ckpt_path=ckpt_path)
        print("[INFO] best model path", checkpoint_callback.best_model_path)
        return self.model

    def validation(self, val_losses, device):
        with torch.no_grad():
            self.model.eval()
            for batch_img, batch_text in self.dm.val_dataloader():
                logits = self.model(batch_img.to(device))
                val_loss = self.model.calculate_loss(logits, batch_text)
                val_losses.append(val_loss.item())
        return val_losses

    def tune(self, percentage=0.05) -> Dict:
        """
        TODO: describe method
        """
        if self.model is None:
            self.create_model()

        if self.gpus:
            trainer = pl.Trainer(max_epochs=self.epochs,
                                 accelerator='gpu', devices=self.gpus,
                                 )
        else:
            trainer = pl.Trainer(max_epochs=self.epochs,
                                 accelerator='cpu'
                                 )


        num_training = int(len(self.dm.train_image_generator)*percentage) or 1
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(self.model,
                                  self.dm,
                                  num_training=num_training,
                                  early_stop_threshold=None)
        lr = lr_finder.suggestion()
        print(f"Found lr: {lr}")
        self.model.hparams["learning_rate"] = lr

        return lr_finder

    def preprocess(self, imgs, need_preprocess=True):
        xs = []
        if need_preprocess:
            for img in imgs:
                x = normalize_img(img,
                                  width=self.width,
                                  height=self.height)
                xs.append(x)
            xs = np.moveaxis(np.array(xs), 3, 1)
        else:
            xs = np.array(imgs)
        xs = torch.tensor(xs)
        xs = xs.to(device_torch)
        return xs

    def forward(self, xs):
        return self.model(xs)

    def postprocess(self, net_out_value):
        net_out_value = [p.cpu().numpy() for p in net_out_value]
        pred_texts = decode_batch(torch.Tensor(net_out_value), self.label_converter)
        pred_texts = [pred_text.upper() for pred_text in pred_texts]
        return pred_texts

    @torch.no_grad()
    def predict(self, xs: List or torch.Tensor, return_acc: bool = False) -> Any:
        net_out_value = self.model(xs)
        net_out_value = [p.cpu().numpy() for p in net_out_value]
        pred_texts = decode_batch(torch.Tensor(net_out_value), self.label_converter)
        pred_texts = [pred_text.upper() for pred_text in pred_texts]
        if return_acc:
            if len(net_out_value):
                net_out_value = np.array(net_out_value)
                net_out_value = net_out_value.reshape((net_out_value.shape[1],
                                                       net_out_value.shape[0],
                                                       net_out_value.shape[2]))
            return pred_texts, net_out_value
        return pred_texts

    def save(self, path: str, verbose: bool = True, weights_only=True) -> None:
        """
        TODO: describe method
        """
        if bool(verbose):
            print("model save to {}".format(path))
        if self.model is None:
            raise ValueError("self.model is not defined")
        if self.trainer is None:
            torch.save({"state_dict": self.model.state_dict()}, path)
        else:
            self.trainer.save_checkpoint(path, weights_only=weights_only)

    def is_loaded(self) -> bool:
        """
        TODO: describe method
        """
        if self.model is None:
            return False
        return True

    def load_model(self, path_to_model, nn_class=NPOcrNet):
        self.path_to_model = path_to_model
        self.model = nn_class.load_from_checkpoint(path_to_model,
                                                   map_location=torch.device('cpu'),
                                                   letters=self.letters,
                                                   linear_size=self.linear_size,
                                                   hidden_size=self.hidden_size,
                                                   backbone=self.backbone,
                                                   letters_max=len(self.letters) + 1,
                                                   label_converter=self.label_converter,
                                                   height=self.height,
                                                   width=self.width,
                                                   color_channels=self.color_channels,
                                                   max_text_len=self.max_text_len,
                                                   **{'pytorch_lightning_version': '0.0.0'})
        self.model = self.model.to(device_torch)
        self.model.eval()
        return self.model

    def load_meta(self, path_to_model: str = "latest") -> str:
        model_info = {}
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name(self.model_name)
            path_to_model = model_info["path"]
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model,
                                                        self.model_name,
                                                        self.model_name)
            path_to_model = model_info["path"]
        elif path_to_model.startswith("modelhub://"):
            path_to_model = path_to_model.split("modelhub://")[1]
            model_info = modelhub.download_model_by_name(path_to_model)
            path_to_model = model_info["path"]
        self.hidden_size = model_info.get("hidden_size", self.hidden_size)
        self.backbone = model_info.get("backbone", self.backbone)
        if type(self.backbone) == str:
            self.backbone = getattr(models, self.backbone)
        self.letters = model_info.get("letters", self.letters)
        self.max_text_len = model_info.get("max_text_len", self.max_text_len)
        self.height = model_info.get("height", self.height)
        self.width = model_info.get("width", self.width)
        self.color_channels = model_info.get("color_channels", self.color_channels)
        self.linear_size = model_info.get("linear_size", self.linear_size)
        return path_to_model

    def load(self, path_to_model: str = "latest", nn_class=NPOcrNet) -> NPOcrNet:
        """
        TODO: describe method
        """
        path_to_model = self.load_meta(path_to_model)
        self.create_model()
        return self.load_model(path_to_model, nn_class=nn_class)

    @torch.no_grad()
    def get_acc(self, predicted: List, decode: List) -> torch.Tensor:
        decode = [pred_text.lower() for pred_text in decode]
        self.init_label_converter()

        logits = torch.tensor(predicted)
        logits = logits.reshape(logits.shape[1],
                                logits.shape[0],
                                logits.shape[2])
        input_len, batch_size, vocab_size = logits.size()
        device = logits.device

        logits = logits.log_softmax(2)

        encoded_texts, text_lens = self.label_converter.encode(decode)
        text_lens = torch.tensor([self.max_text_len for _ in range(batch_size)])
        logits_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)

        acc = functional.ctc_loss(
            logits,
            encoded_texts,
            logits_lens.to(device),
            text_lens
        )
        return 1 - acc / len(self.letters)
    
    @torch.no_grad()
    def acc_calc(self, dataset, verbose: bool = False, save_test_result = False) -> float:
        acc = 0
        self.model = self.model.to(device_torch)
        self.model.eval()
        for idx in range(len(dataset)):
            img, text = dataset[idx]
            img = img.unsqueeze(0).to(device_torch)
            logits = self.model(img)
            pred_text = decode_prediction(logits.cpu(), self.label_converter)

            if save_test_result:
                img_path = dataset.paths[idx]
                ann_path = img_path.replace("/img/", "/ann/").replace(".png", ".json")
                ann_data = json.load(open(ann_path, 'r'))
                if "moderation" not in ann_data:
                    ann_data["moderation"] = {}

            if pred_text == text:
                acc += 1
                if save_test_result:
                    ann_data["moderation"]["isModerated"]=1
            elif verbose:
                print(f'\n[INFO] {dataset.paths[idx]}\nPredicted: {pred_text.upper()} \t\t\t True: {text.upper()}')
                if save_test_result:
                    ann_data["moderation"]["isModerated"]=0
                    ann_data["moderation"]["predicted"]=pred_text.upper()
            if save_test_result:
                with open(ann_path, "w") as outfile:
                    outfile.write(json.dumps(ann_data, indent=4))
        return acc / len(dataset)

    def val_acc(self, verbose=False) -> float:
        acc = self.acc_calc(self.dm.val_image_generator, verbose=verbose)
        print('Validaton Accuracy: ', acc, "in", len(self.dm.val_image_generator))
        return acc

    def test_acc(self, verbose=True, save_test_result=False) -> float:
        acc = self.acc_calc(self.dm.test_image_generator, verbose=verbose, save_test_result=save_test_result)
        print('Testing Accuracy: ', acc, "in", len(self.dm.test_image_generator))
        return acc

    def train_acc(self, verbose=False) -> float:
        acc = self.acc_calc(self.dm.train_image_generator, verbose=verbose)
        print('Training Accuracy: ', acc, "in", len(self.dm.train_image_generator))
        return acc


if __name__ == "__main__":
    det = OCR()
    det.get_classname = lambda: "Eu"
    det.letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I",
                   "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    det.max_text_len = 9
    det.letters_max = len(det.letters)+1
    det.init_label_converter()
    det.load()

    image_path = os.path.join(os.getcwd(), "./data/examples/numberplate_zone_images/JJF509.png")
    img = cv2.imread(image_path)
    xs = det.preprocess([img])
    y = det.predict(xs)
    print("y", y)

    image_path = os.path.join(os.getcwd(), "./data/examples/numberplate_zone_images/RP70012.png")
    img = cv2.imread(image_path)
    xs = det.preprocess([img])
    y = det.predict(xs)
    print("y", y)
