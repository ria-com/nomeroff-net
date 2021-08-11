# import modules
from typing import List, Tuple, Any, Dict

import os
import json
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from collections import Counter

from NomeroffNet.tools import (modelhub,
                               get_mode_torch)
from NomeroffNet.data_modules.numberplate_ocr_data_module import OcrNetDataModule
from NomeroffNet.nnmodels.ocr_model import NPOcrNet, weights_init
from NomeroffNet.data_modules.data_loaders import normalize
from NomeroffNet.tools.ocr_tools import (strLabelConverter,
                                         decode_prediction,
                                         decode_batch)


mode_torch = get_mode_torch()


class OCR(object):
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self) -> None:
        # model
        self.dm = None
        self.model = None
        self.trainer = None
        self.letters = []
        self.max_text_len = 0

        # Input parameters
        self.max_plate_length = 0
        self.height = 64
        self.width = 128
        self.color_channels = 1
        self.label_length = 13

        # Train hyperparameters
        self.batch_size = 32
        self.epochs = 1
        self.gpus = 1
        
        self.label_converter = None
    
    def init_label_converter(self):
        self.label_converter = strLabelConverter("".join(self.letters), self.max_text_len)
        
    @staticmethod
    def get_counter(dirpath: str, verbose: bool = True) -> Tuple[Counter, int]:
        dirname = os.path.basename(dirpath)
        ann_dirpath = os.path.join(dirpath, 'ann')
        letters = ''
        lens = []
        for filename in os.listdir(ann_dirpath):
            json_filepath = os.path.join(ann_dirpath, filename)
            description = json.load(open(json_filepath, 'r'))['description']
            lens.append(len(description))
            letters += description
        max_plate_length = max(Counter(lens).keys())
        if verbose:
            print('Max plate length in "%s":' % dirname, max_plate_length)
        return Counter(letters), max_plate_length

    def get_alphabet(self, train_path: str, test_path: str, val_path: str, verbose: bool = True) -> Tuple[List, int]:
        c_val, max_plate_length_val = self.get_counter(val_path)
        c_train, max_plate_length_train = self.get_counter(train_path)
        c_test, max_plate_length_test = self.get_counter(test_path)

        letters_train = set(c_train.keys())
        letters_val = set(c_val.keys())
        letters_test = set(c_test.keys())
        if verbose:
            print("Letters train ", letters_train)
            print("Letters val ", letters_val)
            print("Letters test ", letters_test)

        if max_plate_length_val == max_plate_length_train:
            if verbose:
                print('Max plate length in train, test and val do match')
        else:
            raise Exception('Max plate length in train, test and val do not match')

        if letters_train == letters_val:
            if verbose:
                print('Letters in train, val and test do match')
        else:
            raise Exception('Letters in train, val and test do not match')

        self.letters = sorted(list(letters_train))
        self.max_text_len = max_plate_length_train
        if verbose:
            print('Letters:', ' '.join(self.letters))
        return self.letters, self.max_text_len

    def explain_text_generator(self,
                               train_dir: str,
                               letters: list,
                               max_plate_length: int) -> None:
        pass

    def prepare(self,
                path_to_dataset: str,
                use_aug: bool = False,
                verbose: bool = True,
                num_workers: int = 0) -> None:
        train_dir = os.path.join(path_to_dataset, "train")
        test_dir = os.path.join(path_to_dataset, "test")
        val_dir = os.path.join(path_to_dataset, "val")

        if verbose:
            print("GET ALPHABET")
        self.letters, self.max_plate_length = self.get_alphabet(
            train_dir,
            test_dir,
            val_dir,
            verbose=verbose)
        self.init_label_converter()

        if verbose:
            print("\nEXPLAIN DATA TRANSFORMATIONS")
            self.explain_text_generator(train_dir,
                                        self.letters,
                                        self.max_plate_length)

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
            max_plate_length=self.max_plate_length,
            num_workers=num_workers,
            with_aug=use_aug)
        if verbose:
            print("DATA PREPARED")

    def create_model(self) -> NPOcrNet:
        """
        TODO: describe method
        """
        self.model = NPOcrNet(self.letters,
                              letters_max=len(self.letters) + 1,
                              img_h=self.height,
                              img_w=self.width,
                              label_converter=self.label_converter,
                              max_plate_length=self.max_plate_length)
        self.model.apply(weights_init)
        if mode_torch == "gpu":
            self.model = self.model.cuda()
        return self.model

    def train(self,
              log_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/logs/ocr'))
              ) -> NPOcrNet:
        """
        TODO: describe method
        TODO: add ReduceLROnPlateau callback
        """
        self.create_model()
        checkpoint_callback = ModelCheckpoint(dirpath=log_dir, monitor='val_loss')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.trainer = pl.Trainer(max_epochs=self.epochs,
                                  gpus=self.gpus,
                                  callbacks=[checkpoint_callback, lr_monitor],
                                  weights_summary=None)
        self.trainer.fit(self.model, self.dm)
        print("[INFO] best model path", checkpoint_callback.best_model_path)
        self.trainer.test()
        return self.model
    
    def validation(self, val_losses, device):
        with torch.no_grad():
            self.model.eval()
            for batch_img, batch_text in self.dm.val_dataloader():
                logits = self.model(batch_img.to(device))
                val_loss = self.model.calculate_loss(logits, batch_text)
                val_losses.append(val_loss.item())
        return val_losses
    
    def tune(self) -> Dict:
        """
        TODO: describe method
        TODO: add ReduceLROnPlateau callback
        """
        trainer = pl.Trainer(auto_lr_find=True,
                             max_epochs=self.epochs,
                             gpus=self.gpus)
        
        model = self.create_model()
        lr_finder = trainer.tuner.lr_find(model, self.dm, early_stop_threshold=None, min_lr=1e-30)
        lr = lr_finder.suggestion()
        print(f"Found lr: {lr}")
        model.hparams["learning_rate"] = lr
        
        return lr_finder
    
    @torch.no_grad()
    def predict(self, imgs: List, return_acc: bool = False) -> Any:
        xs = []
        for img in imgs:
            x = normalize(img,
                          width=self.width,
                          height=self.height,
                          to_gray=True)
            xs.append(x)
        pred_texts = []
        net_out_value = []
        if bool(xs):
            xs = torch.tensor(np.moveaxis(np.array(xs), 3, 1))
            if mode_torch == "gpu":
                xs = xs.cuda()
                self.model = self.model.cuda()
            net_out_value = self.model(xs)
            # net_out_value = [p.cpu().numpy() for p in net_out_value]
            pred_texts = decode_batch(net_out_value, self.label_converter)
        if return_acc:
            return pred_texts, net_out_value
        return pred_texts
        
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
        if mode_torch == "gpu":
            self.model = NPOcrNet.load_from_checkpoint(path_to_model,
                                                       map_location=torch.device('cuda'),
                                                       letters=self.letters,
                                                       letters_max=len(self.letters) + 1,
                                                       img_h=self.height,
                                                       img_w=self.width,
                                                       label_converter=self.label_converter,
                                                       max_plate_length=self.max_plate_length)
        else:
            self.model = NPOcrNet.load_from_checkpoint(path_to_model,
                                                       map_location=torch.device('cpu'),
                                                       letters=self.letters,
                                                       letters_max=len(self.letters) + 1,
                                                       img_h=self.height,
                                                       img_w=self.width,
                                                       label_converter=self.label_converter,
                                                       max_plate_length=self.max_plate_length)
        self.model.eval()
        return self.model

    def load(self, path_to_model: str = "latest") -> NPOcrNet:
        """
        TODO: describe method
        """
        self.create_model()
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name(self.get_classname())
            path_to_model = model_info["path"]
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model, self.get_classname(), self.get_classname())
            path_to_model = model_info["path"]

        return self.load_model(path_to_model)
    
    def acc_calc(self, dataset, verbose: bool = False) -> float:
        acc = 0
        with torch.no_grad():
            self.model.eval()
            for idx in range(len(dataset)):
                img, text = dataset[idx]
                logits = self.model(img.unsqueeze(0))
                pred_text = decode_prediction(logits.cpu(), self.label_converter)

                if pred_text == text:
                    acc += 1
                elif verbose:
                    print(f'\n[INFO] {dataset.pathes[idx]}\nPredicted: {pred_text} \t\t\t True: {text}')
        return acc / len(dataset)
    
    def val_acc(self, verbose=False) -> float:
        acc = self.acc_calc(self.dm.val_image_generator, verbose=verbose)
        print('Validaton Accuracy: ', acc)
        return acc
        
    def test_acc(self, verbose=True) -> float:
        acc = self.acc_calc(self.dm.test_image_generator, verbose=verbose)
        print('Testing Accuracy: ', acc)
        return acc
        
    def train_acc(self, verbose=False) -> float:
        acc = self.acc_calc(self.dm.train_image_generator, verbose=verbose)
        print('Training Accuracy: ', acc)
        return acc
