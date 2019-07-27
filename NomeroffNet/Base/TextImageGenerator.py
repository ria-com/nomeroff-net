from os.path import join
import cv2
import os
import json
import numpy as np
from tensorflow.keras import backend as K
import random
import itertools
from .aug import aug

class TextImageGenerator:
    def __init__(self,
                 dirpath,
                 img_w, img_h,
                 batch_size,
                 downsample_factor,
                 letters,
                 max_text_len,
                 cname=""):

        self.CNAME = cname
        #print(self.CNAME)
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

    def labels_to_text(self, labels):
        data = ''.join(list(map(lambda x: "" if x==self.letters_max else self.letters[int(x)], labels)))
        return data

    def text_to_labels(self, text):
        data = list(map(lambda x: self.letters.index(x), text))
        while len(data) < self.max_text_len:
            data.append(self.letters_max)
        return data

    def is_valid_str(self, s):
        for ch in s:
            if not ch in self.letters:
                return False
        return True

    def decode_batch(self, out):
        # Most change
        # For a real OCR application, this should be beam search with a dictionary
        # and language model.  For this example, best path is sufficient.
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if c < len(self.letters):
                    outstr += self.letters[c]
            ret.append(outstr)
        return ret

    def build_data(self, aug_count=0):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)

            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img -= np.amin(img)
            img /= np.amax(img)
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.texts.append(text)
        while aug_count:
            for i, (img_filepath, text) in enumerate(self.samples):
                img = cv2.imread(img_filepath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs = aug([img])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = imgs[0]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img = clahe.apply(img)

                img = cv2.resize(img, (self.img_w, self.img_h))
                img = img.astype(np.float32)
                img -= np.amin(img)
                img /= np.amax(img)
                # width and height are backwards from typical Keras convention
                # because width is the time dimension when it gets fed into the RNN
                self.imgs[i, :, :] = img
                self.texts.append(text)
            aug_count -= 1


    def get_output_size(self):
        return len(self.letters) + 1

    def normalize(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (self.IMG_W, self.IMG_H))

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)

        img = img.astype(np.float32)
        img -= np.amin(img)
        img /= np.amax(img)
        img = [[[h] for h in w] for w in img.T]

        x = np.zeros((self.IMG_W, self.IMG_H, 1))
        x[:, :, :] = img
        return x

    def next_sample(self, is_random=1):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.count_ep += 1
            self.cur_index = 0
            if is_random:
                random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self, is_random=1):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])

            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                img, text = self.next_sample(is_random)
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = self.text_to_labels(text)
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input_{}'.format(self.CNAME): X_data,
                'the_labels_{}'.format(self.CNAME): Y_data,
                'input_length_{}'.format(self.CNAME): input_length,
                'label_length_{}'.format(self.CNAME): label_length,
                #'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)