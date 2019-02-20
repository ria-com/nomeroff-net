import os
import json
import cv2
import numpy as np
import random
from keras.utils import to_categorical

class ImgGenerator:

    def __init__(self,
                 dirpath,
                 img_w, img_h,
                 batch_size,
                 labels_counts):

        self.img_h = img_h
        self.img_w = img_w
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
                    self.samples.append([img_filepath, [description["state_id"], description["region_id"]]])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0

    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w, 3))
        self.discs = []
        for i, (img_filepath, disc) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)

            # advanced normalisation
            #img /= 255
            img_min = np.amin(img)
            img -= img_min
            img_max = np.amax(img)
            img /= img_max
            self.imgs[i, :, :] = img
            self.discs.append(
                [
                    to_categorical(disc[0], self.labels_counts[0]),
                    to_categorical(disc[1], self.labels_counts[1])
                ]
            )

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.discs[self.indexes[self.cur_index]]

    def generator(self):
        while True:
            Ys = [[], []]
            Xs = []
            for i in np.arange(self.batch_size):
                x, y = self.next_sample()
                Xs.append(x)
                Ys[1].append(y[0])
                Ys[0].append(y[1])
            #print("Send batch")
            yield np.array(Xs), Ys
