# import labaris
import keras
import tensorflow as tf

import os
from os.path import join
import json
import random
import itertools
import re
import datetime
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
from collections import Counter

from .TextImageGenerator import TextImageGenerator

class OCR(TextImageGenerator):
    def __init__(self):
        # Input parameters
        self.IMG_H = 64
        self.IMG_W = 128
        self.IMG_C = 1

        # Train parameters
        self.BATCH_SIZE = 8
        self.EPOCHS = 1

        # Network parameters
        self.CONV_FILTERS = 16
        self.KERNEL_SIZE = (3, 3)
        self.POOL_SIZE = 2
        self.TIME_DENSE_SIZE = 32
        self.RNN_SIZE = 512
        self.ACTIVATION = 'relu'
        self.DOWNSAMPLE_FACROT = self.POOL_SIZE ** 2

    def get_counter(self, dirpath, verbose=1):
        dirname = os.path.basename(dirpath)
        ann_dirpath = join(dirpath, 'ann')
        letters = ''
        lens = []
        for filename in os.listdir(ann_dirpath):
            json_filepath = join(ann_dirpath, filename)
            description = json.load(open(json_filepath, 'r'))['description']
            lens.append(len(description))
            letters += description
        max_plate_length = max(Counter(lens).keys())
        if verbose:
            print('Max plate length in "%s":' % dirname, max_plate_length)
        return Counter(letters), max_plate_length

    def get_alphabet(self, train_path, test_path, val_path, verbose=1):
        c_val, max_plate_length_val     = self.get_counter(val_path)
        c_train, max_plate_length_train = self.get_counter(train_path)
        c_test, max_plate_length_test   = self.get_counter(test_path)

        letters_train = set(c_train.keys())
        letters_val = set(c_val.keys())
        letters_test = set(c_test.keys())

        if max_plate_length_val == max_plate_length_train and max_plate_length_train == max_plate_length_test:
            if verbose:
                print('Max plate length in train, test and val do match')
        else:
            raise Exception('Max plate length in train, test and val do not match')

        if letters_train == letters_val and letters_test == letters_val:
            if verbose:
                print('Letters in train, val and test do match')
        else:
            raise Exception('Letters in train, val and test do not match')

        self.letters = sorted(list(letters_train))
        self.max_text_len = max_plate_length_train
        if verbose:
            print('Letters:', ' '.join(self.letters))
        return self.letters, self.max_text_len

    def explainTextGenerator(self, train_dir, letters, max_plate_length, verbose=1):
        tiger = TextImageGenerator(train_dir, self.IMG_W, self.IMG_H, 1, self.POOL_SIZE ** 2, letters, max_plate_length)
        tiger.build_data()

        for inp, out in tiger.next_batch():
            print('Text generator output (data which will be fed into the neutral network):')
            print('1) the_input (image)')
            if K.image_data_format() == 'channels_first':
                img = inp['the_input'][0, 0, :, :]
            else:
                img = inp['the_input'][0, :, :, 0]

            plt.imshow(img.T, cmap='gray')
            plt.show()
            print('2) the_labels (plate number): %s is encoded as %s' %
                  (tiger.labels_to_text(inp['the_labels'][0]), list(map(int, inp['the_labels'][0]))))
            print('3) input_length (width of image that is fed to the loss function): %d == %d / 4 - 2' %
                  (inp['input_length'][0], tiger.img_w))
            print('4) label_length (length of plate number): %d' % inp['label_length'][0])
            break

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def _train(self, train_path, test_path, val_path, letters, max_plate_length,  model_path="./", load=False, verbose=1):
        if K.image_data_format() == 'channels_first':
            input_shape = (1, self.IMG_W, self.IMG_H)
        else:
            input_shape = (self.IMG_W, self.IMG_H, 1)

        tiger_train = TextImageGenerator(train_path, self.IMG_W, self.IMG_H, self.BATCH_SIZE, self.DOWNSAMPLE_FACROT, letters, max_plate_length)
        tiger_train.build_data()
        tiger_val = TextImageGenerator(val_path,  self.IMG_W, self.IMG_H, self.BATCH_SIZE, self.DOWNSAMPLE_FACROT, letters, max_plate_length)
        tiger_val.build_data()

        input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        inner = Conv2D(self.CONV_FILTERS, self.KERNEL_SIZE, padding='same',
                       activation=self.ACTIVATION, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(self.POOL_SIZE , self.POOL_SIZE ), name='max1')(inner)
        inner = Conv2D(self.CONV_FILTERS, self.KERNEL_SIZE, padding='same',
                       activation=self.ACTIVATION, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(self.POOL_SIZE , self.POOL_SIZE ), name='max2')(inner)

        conv_to_rnn_dims = (self.IMG_W // (self.POOL_SIZE  ** 2), (self.IMG_H // (self.POOL_SIZE ** 2)) * self.CONV_FILTERS)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(self.TIME_DENSE_SIZE, activation=self.ACTIVATION, name='dense1')(inner)

        # Two layers of bidirecitonal GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(self.RNN_SIZE, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(self.RNN_SIZE, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(self.RNN_SIZE, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(self.RNN_SIZE, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

        # transforms RNN output to character activations:
        inner = Dense(tiger_train.get_output_size(), kernel_initializer='he_normal',
                      name='dense2')(concatenate([gru_2, gru_2b]))
        y_pred = Activation('softmax', name='softmax')(inner)
        Model(inputs=input_data, outputs=y_pred).summary()

        labels = Input(name='the_labels', shape=[tiger_train.max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        # clipnorm seems to speeds up convergence
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        if load:
            model = load_model(model_path, compile=False)
        else:
            model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

        if not load:
            # captures output of softmax so we can decode the output during visualization
            test_func = K.function([input_data], [y_pred])

            model.fit_generator(generator=tiger_train.next_batch(),
                                steps_per_epoch=tiger_train.n,
                                epochs=self.EPOCHS,
                                validation_data=tiger_val.next_batch(),
                                validation_steps=tiger_val.n)

        return model

    def save(self, path):
        if self.MODEL:
            self.MODEL.save(path)

    def test(self, test_path, letters, max_plate_length, verbose=1):
        tiger_test = TextImageGenerator(test_path, self.IMG_W, self.IMG_H, 1, self.DOWNSAMPLE_FACROT, letters, max_plate_length)
        tiger_test.build_data()

        net_inp = self.MODEL.get_layer(name='the_input').input
        net_out = self.MODEL.get_layer(name='softmax').output

        err_arr = []
        succ_arr = []
        for inp_value, _ in tiger_test.next_batch():
            bs = inp_value['the_input'].shape[0]
            X_data = inp_value['the_input']
            print(X_data.shape)
            print(X_data)
            net_out_value = self.SESS.run(net_out, feed_dict={net_inp:X_data})
            pred_texts = tiger_test.decode_batch(net_out_value)
            labels = inp_value['the_labels']
            print(labels)
            texts = []
            for label in labels:
                text = ''.join(list(map(lambda x: letters[int(x)], label)))
                texts.append(text)

            for i in range(bs):
                if (pred_texts[i] != texts[i]):
                    if verbose:
                        print('\nPredicted: \t\t %s\nTrue: \t\t\t %s' % (pred_texts[i], texts[i]))
                    err_arr.append({'pred':pred_texts[i],'true':texts[i]})
                else:
                    succ_arr.append({'pred':pred_texts[i],'false':texts[i]})
            break
        print(f"loss: {len(err_arr)/(len(err_arr)+len(succ_arr))}")
        print(f"acc: {len(succ_arr)/(len(err_arr)+len(succ_arr))}")

    def predict(self, img):
        net_inp = self.MODEL.get_layer(name='the_input').input
        net_out = self.MODEL.get_layer(name='softmax').output

        X = self.normalize(img)

        model = Model(input=net_inp, output=net_out)
        net_out_value = model.predict(X)
        pred_texts = self.decode_batch(net_out_value)
        return pred_texts[0]

    def load(self, path_to_model, verbose = 0):
        self.MODEL = load_model(path_to_model, compile=False)

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        if verbose:
            self.MODEL.summary()
        return self.MODEL

    def train(self, path_to_dataset, model_path="./model.h5", load=False, verbose=1):
        self.SESS = tf.Session()
        K.set_session(self.SESS)

        train_path = os.path.join(path_to_dataset, "train")
        test_path  = os.path.join(path_to_dataset, "test")
        val_path   = os.path.join(path_to_dataset, "val")

        if verbose:
            print("GET ALPHABET")
        letters, max_plate_length = self.get_alphabet(train_path, test_path, val_path, verbose=verbose)

        if verbose:
            print("\nEXPLAIN DATA TRANSFORMATIONS")
            self.explainTextGenerator(train_path, letters, max_plate_length)

        if verbose:
            print("\nSTART TRAINING")
        self.MODEL = self._train(train_path, test_path, val_path, letters, max_plate_length, model_path=model_path, load=load, verbose=verbose)
        self.MODEL.save(model_path)

        if verbose:
            print("\nRUN TEST")
        self.test(test_path, letters, max_plate_length, verbose=verbose)

        return self.MODEL