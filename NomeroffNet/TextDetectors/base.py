# import modules
from typing import List, Tuple, Any

import os
import sys
import time
import json
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import Counter

from NomeroffNet.tools import (modelhub,
                               get_mode_torch)
from NomeroffNet.data_modules.numberplate_ocr_data_module import OcrNetDataModule
from NomeroffNet.nnmodels.ocr_model import NPOcrNet

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
        self.letters = None
        self.max_text_len = 0

        # Input parameters
        self.height = 64
        self.width = 128
        self.color_channels = 1

        # Train hyperparameters
        self.batch_size = 32
        self.epochs = 1
        self.gpus = 1

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
        self.letters, max_plate_length = self.get_alphabet(train_dir,
                                                           test_dir,
                                                           val_dir,
                                                           verbose=verbose)

        if verbose:
            print("\nEXPLAIN DATA TRANSFORMATIONS")
            self.explain_text_generator(train_dir,
                                        self.letters,
                                        max_plate_length)

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
            with_aug=use_aug)
        if verbose:
            print("DATA PREPARED")

    def create_model(self) -> NPOcrNet:
        """
        TODO: describe method
        """
        self.model = NPOcrNet(self.max_text_len,
                              self.height,
                              self.width)
        if mode_torch == "gpu":
            self.model = self.model.cuda()
        return self.model

    def train(self,
              log_dir=sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/logs/ocr')))
              ) -> NPOcrNet:
        """
        TODO: describe method
        TODO: add ReduceLROnPlateau callback
        """
        self.create_model()
        checkpoint_callback = ModelCheckpoint(dirpath=log_dir, monitor='val_loss')
        self.trainer = pl.Trainer(max_epochs=self.epochs,
                                  gpus=self.gpus,
                                  callbacks=[checkpoint_callback],
                                  weights_summary=None)
        self.trainer.fit(self.model, self.dm)
        print("[INFO] best model path", checkpoint_callback.best_model_path)
        self.trainer.test()
        return self.model

    def test(self) -> List:
        """
        TODO: describe method
        """
        return self.trainer.test()
