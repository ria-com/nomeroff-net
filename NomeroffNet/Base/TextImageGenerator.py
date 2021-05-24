from os.path import join, basename
import cv2
import os
import json
import numpy as np
from tensorflow.keras import backend
import random
from typing import List, Tuple, Generator
from .aug import aug, aug_seed
from .ocr_base import BaseOCR


class TextImageGenerator(BaseOCR):
    def __init__(self,
                 dirpath: str,
                 img_w: int,
                 img_h: int,
                 batch_size: int,
                 downsample_factor: float,
                 letters: List,
                 max_text_len: int,
                 cname="") -> None:
        BaseOCR.__init__(self)

        self.CNAME = cname
        self.dirpath = dirpath
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.letters = letters

        img_dirpath = join(dirpath, 'img')
        ann_dirpath = join(dirpath, 'ann')
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext == '.png':
                img_filepath = join(img_dirpath, filename)
                json_filepath = join(ann_dirpath, name + '.json')
                description = json.load(open(json_filepath, 'r'))['description']
                if TextImageGenerator.is_valid_str(self, description):
                    self.samples.append([img_filepath, description])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.count_ep = 0
        self.letters_max = len(letters)+1
        self.imgs = None
        self.texts = None

    def is_valid_str(self, s: str) -> bool:
        for ch in s:
            if ch not in self.letters:
                return False
        return True

    def build_data(self, use_aug: bool = False, aug_debug: bool = False,
                   aug_suffix: str = 'aug', aug_seed_num: int = None) -> None:
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []

        img_dirpath_aug = ""
        if use_aug:
            aug_seed(aug_seed_num)
            img_dirpath_aug = join(self.dirpath, 'img_{}'.format(aug_suffix))
            if not os.path.exists(img_dirpath_aug) and aug_debug:
                print('Creating path "{}" for aug images'.format(img_dirpath_aug))
                os.mkdir(img_dirpath_aug)

        for i, (img_filepath, text) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            
            if use_aug:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs = aug([img])
                img = imgs[0]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if aug_debug:
                    cv2.imwrite(join(img_dirpath_aug, basename(img_filepath)), img)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img -= np.amin(img)
            img /= np.amax(img)
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.texts.append(text)
            
        self.n = len(self.imgs)
        self.indexes = list(range(self.n))

    def get_output_size(self) -> int:
        return len(self.letters) + 1

    def next_sample(self, is_random: int = True) -> Tuple:
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.count_ep += 1
            self.cur_index = 0
            if is_random:
                random.shuffle(self.indexes)
        img = self.imgs[self.indexes[self.cur_index]]
        labels = self.texts[self.indexes[self.cur_index]]
        img = img.T
        if backend.image_data_format() == 'channels_first':
            img = np.expand_dims(img, 0)
        else:
            img = np.expand_dims(img, -1)
        return img, labels

    def init_xs_ys(self):
        if backend.image_data_format() == 'channels_first':
            x_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
        else:
            x_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
        y_data = np.ones([self.batch_size, self.max_text_len])
        return x_data, y_data

    def next_batch(self, is_random: int = 1, input_name: str = None, output_name: str = "ctc") -> Tuple:
        if not input_name:
            input_name = 'the_input_{}'.format(self.CNAME)
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            x_data, y_data = self.init_xs_ys()
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                img, text = self.next_sample(is_random)
                x_data[i] = img
                y_data[i] = np.array(self.text_to_labels(text))
                label_length[i] = len(text)

            inputs = {
                '{}'.format(input_name): x_data,
                'the_labels_{}'.format(self.CNAME): y_data,
                'input_length_{}'.format(self.CNAME): input_length,
                'label_length_{}'.format(self.CNAME): label_length
            }
            outputs = {'{}'.format(output_name): np.zeros([self.batch_size])}
            yield inputs, outputs

    def next_batch_pb(self, is_random: int = 1) -> Generator:
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            x_data, y_data = self.init_xs_ys()

            for i in range(self.batch_size):
                img, text = self.next_sample(is_random)
                x_data[i] = img
                y_data[i] = np.array(self.text_to_labels(text))

            inputs = x_data
            outputs = y_data
            yield inputs, outputs
