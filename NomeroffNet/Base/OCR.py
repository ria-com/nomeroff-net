# import labaris
import tensorflow.keras
import tensorflow as tf

import os
from os.path import join
import json
import random
import numpy as np

from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
from collections import Counter

from keras.layers import CuDNNGRU as GRUgpu
from keras.layers.recurrent import GRU as GRUcpu

from .TextImageGenerator import TextImageGenerator
from NomeroffNet.mcm.mcm import download_latest_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class OCR(TextImageGenerator):
    @classmethod
    def get_classname(cls):
        return cls.__name__

    def __init__(self):
        # Input parameters
        self.IMG_H = 64
        self.IMG_W = 128
        self.IMG_C = 1

        # Train parameters
        self.BATCH_SIZE = 32
        self.EPOCHS = 1

        # Network parameters
        self.CONV_FILTERS = 16
        self.KERNEL_SIZE = (3, 3)
        self.POOL_SIZE = 2
        self.TIME_DENSE_SIZE = 32
        self.RNN_SIZE = 512
        self.ACTIVATION = 'relu'
        self.DOWNSAMPLE_FACROT = self.POOL_SIZE * self.POOL_SIZE

        self.GRU = GRUcpu

        self.INPUT_NODE = "the_input_{}:0".format(type(self).__name__)
        self.OUTPUT_NODE = "softmax_{}/truediv:0".format(type(self).__name__)

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

    def explainTextGenerator(self, train_dir, letters, max_plate_length, verbose=1):
        tiger = TextImageGenerator(train_dir, self.IMG_W, self.IMG_H, 1, self.POOL_SIZE * self.POOL_SIZE, letters, max_plate_length, cname=type(self).__name__)
        tiger.build_data()

        for inp, out in tiger.next_batch():
            print('Text generator output (data which will be fed into the neutral network):')
            print('1) the_input (image)')
            if K.image_data_format() == 'channels_first':
                img = inp['the_input_{}'.format(type(self).__name__)][0, 0, :, :]
            else:
                img = inp['the_input_{}'.format(type(self).__name__)][0, :, :, 0]

            #import matplotlib.pyplot as plt
            #plt.imshow(img.T, cmap='gray')
            #plt.show()
            print('2) the_labels (plate number): %s is encoded as %s' %
                  (tiger.labels_to_text(inp['the_labels_{}'.format(type(self).__name__)][0]), list(map(int, inp['the_labels_{}'.format(type(self).__name__)][0]))))
            print('3) input_length (width of image that is fed to the loss function): %d == %d / 4 - 2' %
                  (inp['input_length_{}'.format(type(self).__name__)][0], tiger.img_w))
            print('4) label_length (length of plate number): %d' % inp['label_length_{}'.format(type(self).__name__)][0])
            break

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def save(self, path, verbose=1):
        if self.MODEL:
            self.MODEL.save(path)
            if verbose:
                print("SAVED TO {}".format(path))

    def test(self, verbose=1):
        if verbose:
            print("\nRUN TEST")
        net_inp = self.MODEL.get_layer(name='the_input_{}'.format(type(self).__name__)).input
        net_out = self.MODEL.get_layer(name='softmax_{}'.format(type(self).__name__)).output

        err_c = 0
        succ_c = 0
        for inp_value, _ in self.tiger_test.next_batch():
            bs = inp_value['the_input_{}'.format(type(self).__name__)].shape[0]
            X_data = inp_value['the_input_{}'.format(type(self).__name__)]
            net_out_value = self.SESS.run(net_out, feed_dict={net_inp:X_data})
            pred_texts = self.tiger_test.decode_batch(net_out_value)
            labels = inp_value['the_labels_{}'.format(type(self).__name__)]
            texts = []
            for label in labels:
                text = self.tiger_test.labels_to_text(label)
                texts.append(text)

            for i in range(bs):
                if (pred_texts[i] != texts[i]):
                    if verbose:
                        print('\nPredicted: \t\t %s\nTrue: \t\t\t %s' % (pred_texts[i], texts[i]))
                    err_c += 1
                else:
                    succ_c += 1
            break
        print("acc: {}".format(succ_c/(err_c+succ_c)))

    def predict(self, imgs, *argv):
        Xs = []
        for img in imgs:
            x = self.normalize(img)
            Xs.append(x)
        pred_texts = []
        if bool(Xs):
            net_out_value = self.MODEL.predict(np.array(Xs))
            #print(net_out_value)
            pred_texts = self.decode_batch(net_out_value)
        return pred_texts

    def load(self, path_to_model, mode="cpu", verbose = 0):
        if path_to_model =="latest":
            model_info = download_latest_model("TextDetector", self.get_classname(), mode=mode)
            path_to_model = model_info["path"]


        self.MODEL = load_model(path_to_model, compile=False)

        net_inp = self.MODEL.get_layer(name='the_input_{}'.format(type(self).__name__)).input
        net_out = self.MODEL.get_layer(name='softmax_{}'.format(type(self).__name__)).output

        self.MODEL = Model(input=net_inp, output=net_out)

        if verbose:
            self.MODEL.summary()

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        #sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        #self.MODEL.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

        return self.MODEL

    def prepare(self, path_to_dataset, aug_count=0, verbose=1):
        self.SESS = tf.Session()
        K.set_session(self.SESS)

        train_path = os.path.join(path_to_dataset, "train")
        test_path  = os.path.join(path_to_dataset, "test")
        val_path   = os.path.join(path_to_dataset, "val")

        if verbose:
            print("GET ALPHABET")
        self.letters, max_plate_length = self.get_alphabet(train_path, test_path, val_path, verbose=verbose)

        if verbose:
            print("\nEXPLAIN DATA TRANSFORMATIONS")
            self.explainTextGenerator(train_path, self.letters, max_plate_length)

        if verbose:
            print("START BUILD DATA")
        self.tiger_train = TextImageGenerator(train_path, self.IMG_W, self.IMG_H, self.BATCH_SIZE, self.DOWNSAMPLE_FACROT, self.letters, max_plate_length, cname=type(self).__name__)
        self.tiger_train.build_data(aug_count=aug_count)
        self.tiger_val = TextImageGenerator(val_path,  self.IMG_W, self.IMG_H, self.BATCH_SIZE, self.DOWNSAMPLE_FACROT, self.letters, max_plate_length, cname=type(self).__name__)
        self.tiger_val.build_data()

        self.tiger_test = TextImageGenerator(test_path, self.IMG_W, self.IMG_H, len(os.listdir(os.path.join(test_path, "img"))), self.DOWNSAMPLE_FACROT, self.letters, max_plate_length, cname=type(self).__name__)
        self.tiger_test.build_data()
        if verbose:
            print("DATA PREPARED")

    def train(self, mode="cpu", is_random=1, model_path="./model.h5", load=False, verbose=1):
        if mode == "gpu":
            self.GRU = GRUgpu
        if mode == "cpu":
            self.GRU = GRUcpu

        if verbose:
            print("\nSTART TRAINING")
        if K.image_data_format() == 'channels_first':
            input_shape = (1, self.IMG_W, self.IMG_H)
        else:
            input_shape = (self.IMG_W, self.IMG_H, 1)

        input_data = Input(name='the_input_{}'.format(type(self).__name__), shape=input_shape, dtype='float32')
        inner = Conv2D(self.CONV_FILTERS, self.KERNEL_SIZE, padding='same',
                       activation=self.ACTIVATION, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(self.POOL_SIZE , self.POOL_SIZE ), name='max1')(inner)
        inner = Conv2D(self.CONV_FILTERS, self.KERNEL_SIZE, padding='same',
                       activation=self.ACTIVATION, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(self.POOL_SIZE , self.POOL_SIZE ), name='max2')(inner)

        conv_to_rnn_dims = (self.IMG_W // (self.POOL_SIZE  * self.POOL_SIZE), (self.IMG_H // (self.POOL_SIZE * self.POOL_SIZE)) * self.CONV_FILTERS)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(self.TIME_DENSE_SIZE, activation=self.ACTIVATION, name='dense1')(inner)

        # Two layers of bidirecitonal GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = self.GRU(self.RNN_SIZE, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = self.GRU(self.RNN_SIZE, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = self.GRU(self.RNN_SIZE, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = self.GRU(self.RNN_SIZE, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

        # transforms RNN output to character activations:
        inner = Dense(self.tiger_train.get_output_size(), kernel_initializer='he_normal',
                      name='dense2')(concatenate([gru_2, gru_2b]))
        y_pred = Activation('softmax', name='softmax_{}'.format(type(self).__name__))(inner)
        Model(inputs=input_data, outputs=y_pred).summary()

        labels = Input(name='the_labels_{}'.format(type(self).__name__), shape=[self.tiger_train.max_text_len], dtype='float32')
        input_length = Input(name='input_length_{}'.format(type(self).__name__), shape=[1], dtype='int64')
        label_length = Input(name='label_length_{}'.format(type(self).__name__), shape=[1], dtype='int64')
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

            model.fit_generator(generator=self.tiger_train.next_batch(is_random),
                                steps_per_epoch=self.tiger_train.n,
                                epochs=self.EPOCHS,
                                validation_data=self.tiger_val.next_batch(is_random),
                                validation_steps=self.tiger_val.n)

        net_inp = model.get_layer(name='the_input_{}'.format(type(self).__name__)).input
        net_out = model.get_layer(name='softmax_{}'.format(type(self).__name__)).output
        self.MODEL = Model(input=net_inp, output=net_out)
        return self.MODEL

    def load_frozen(self, FROZEN_MODEL_PATH, mode="cpu"):
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(FROZEN_MODEL_PATH, "rb") as f:
            graph_def.ParseFromString(f.read())

        #print([x.name for x in graph_def.node])
        graph = tf.Graph()
        with graph.as_default():
            self.net_inp, self.net_out = tf.import_graph_def(
                graph_def, return_elements = [self.INPUT_NODE, self.OUTPUT_NODE]
            )

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=sess_config)

    def frozen_predict(self, imgs):
        Xs = []
        for img in imgs:
            x = self.normalize(img)
            Xs.append(x)
        pred_texts = []
        if bool(Xs):
            net_out_value = self.sess.run([self.net_out], feed_dict={self.net_inp: np.array(Xs)})
            #print(net_out_value)
            pred_texts = self.decode_batch(net_out_value[0])
        return pred_texts