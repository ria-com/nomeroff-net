import os
import json
import cv2
import numpy as np
import random
from keras.utils import to_categorical
from .aug import aug

class ImgGenerator:

    def __init__(self,
                 dirpath,
                 img_w, img_h,
                 batch_size,
                 labels_counts):

        self.HEIGHT = img_h
        self.WEIGHT = img_w
        self.batch_size = batch_size

        self.labels_counts = labels_counts

        img_dirpath = os.path.join(dirpath, 'img')
        ann_dirpath = os.path.join(dirpath, 'ann')
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext == '.png':
                img_filepath = os.path.join(img_dirpath, filename)
                json_filepath = os.path.join(ann_dirpath, name + '.json')
                if os.path.exists(json_filepath):
                    description = json.load(open(json_filepath, 'r'))
                    self.samples.append([img_filepath, [description["state_id"], description["region_id"], description["count_lines"]]])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.count_ep = 0
        self.count_ep_need_to_aug = 1

    def build_data(self):
        self.paths = []
        self.discs = []
        for i, (img_filepath, disc) in enumerate(self.samples):
            self.paths.append(img_filepath)
            self.discs.append(
                [
                    to_categorical(disc[0], self.labels_counts[0]),
                    to_categorical(disc[1], self.labels_counts[1]),
                    to_categorical(disc[2], self.labels_counts[2])
                ]
            )

    def normalize(self, img, with_aug=False):
        if with_aug:
            imgs = aug([img])
            img = imgs[0]
        img = cv2.resize(img, (self.WEIGHT, self.HEIGHT))
        img = img.astype(np.float32)

        # advanced normalisation
        img_min = np.amin(img)
        img -= img_min
        img_max = np.amax(img)
        img /= (img_max or 1)
        img[img == 0] = .0001
        return img

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.count_ep += 1
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.paths[self.indexes[self.cur_index]], self.discs[self.indexes[self.cur_index]]

    def generator(self):
        while True:
            Ys = [[], [], []]
            Xs = []
            for i in np.arange(self.batch_size):
                x, y = self.next_sample()
                img = cv2.imread(x)
                if self.count_ep >= self.count_ep_need_to_aug:
                    x = self.normalize(img, with_aug=1)
                else:
                    x = self.normalize(img)
                Xs.append(x)
                Ys[1].append(y[0])
                Ys[0].append(y[1])
                Ys[2].append(y[2])
            yield np.array(Xs), Ys
