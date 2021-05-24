# import modules
import tensorflow as tf
from typing import List, Tuple, Any

import os
from os.path import join
import json
import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Reshape, Lambda
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks

from collections import Counter

from tensorflow.keras.layers import GRU

from .ocr_base import BaseOCR
from .TextImageGenerator import TextImageGenerator
from .mcm.mcm import download_latest_model

import time

from tensorflow.keras import backend


class OCR(BaseOCR):
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self) -> None:
        BaseOCR.__init__(self)

        # init model params
        self.MODEL = None
        self.PB_MODEL = None
        self.tiger_train = None
        self.tiger_val = None
        self.tiger_test = None
        self.CALLBACKS_LIST = []

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

        self.INPUT_NODE = "the_input_{}:0".format(type(self).__name__)
        self.OUTPUT_NODE = "softmax_{}".format(type(self).__name__)
        
        # callbacks hyperparameters
        self.REDUCE_LRO_N_PLATEAU_PATIENCE = 3
        self.REDUCE_LRO_N_PLATEAU_FACTOR = 0.1

    @staticmethod
    def get_counter(dirpath: str, verbose: bool = True) -> Tuple[Counter, int]:
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

    def explain_text_generator(self, train_dir: str, letters: list,
                               max_plate_length: int) -> None:
        tiger = TextImageGenerator(train_dir, self.IMG_W, self.IMG_H, 1, self.POOL_SIZE * self.POOL_SIZE,
                                   letters, max_plate_length, cname=type(self).__name__)
        tiger.build_data()

        for inp, out in tiger.next_batch():
            print('Text generator output (data which will be fed into the neutral network):')
            print('1) the_input (image)')
            if backend.image_data_format() == 'channels_first':
                img = inp['the_input_{}'.format(type(self).__name__)][0, 0, :, :]
            else:
                img = inp['the_input_{}'.format(type(self).__name__)][0, :, :, 0]

            import matplotlib.pyplot as plt
            plt.imshow(img.T, cmap='gray')
            plt.show()

            print('2) the_labels (plate number): %s is encoded as %s' %
                  (tiger.labels_to_text(inp['the_labels_{}'.format(type(self).__name__)][0]),
                   list(map(int, inp['the_labels_{}'.format(type(self).__name__)][0]))))
            print('3) input_length (width of image that is fed to the loss function): %d == %d / 4 - 2' %
                  (inp['input_length_{}'.format(type(self).__name__)][0], tiger.img_w))
            print('4) label_length (length of plate number): %d' % inp['label_length_{}'
                  .format(type(self).__name__)][0])
            break

    @staticmethod
    def ctc_lambda_func(args: list) -> np.ndarray:
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        y_pred = y_pred[:, 2:, :]
        return backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def save(self, path: str, verbose: bool = True) -> None:
        if self.MODEL:
            self.MODEL.save(path)
            if verbose:
                print("SAVED TO {}".format(path))

    def test(self, verbose: bool = True, random_state: int = 0) -> None:
        start_time = time.time()
        if verbose:
            print("\nRUN TEST")
        net_inp = self.MODEL.get_layer(name='{}'.format(self.MODEL.layers[0].name)).input
        net_out = self.MODEL.get_layer(name='{}'.format(self.MODEL.layers[-1].name)).output
        if verbose:
            print("[INFO] net_inp", net_inp)
            print("[INFO] net_out", net_out)

        err_c = 0
        succ_c = 0
        for inp_value, _ in self.tiger_test.next_batch(random_state,
                                                       input_name=self.MODEL.layers[0].name,
                                                       output_name=self.MODEL.layers[-1].name):
            bs = inp_value['the_input_{}'.format(type(self).__name__)].shape[0]
            x_data = inp_value['the_input_{}'.format(type(self).__name__)]
            
            net_out_value = self.MODEL.predict(np.array(x_data))
            pred_texts = self.decode_batch(net_out_value)
            
            labels = inp_value['the_labels_{}'.format(type(self).__name__)]
            texts = []
            for label in labels:
                text = self.tiger_test.labels_to_text(label)
                texts.append(text)

            for i in range(bs):
                if pred_texts[i] != texts[i]:
                    if verbose:
                        print('\nPredicted: \t\t %s\nTrue: \t\t\t %s' % (pred_texts[i], texts[i]))
                    err_c += 1
                else:
                    succ_c += 1
            break
        if verbose:
            print("Test processing time: {} seconds".format(time.time() - start_time))
        print("acc: {}".format(succ_c/(err_c+succ_c)))
   
    def test_pb(self, verbose: bool = True, random_state: int = 0):
        start_time = time.time()
        if verbose:
            print("\nRUN TEST")

        err_c = 0
        succ_c = 0
        for X_data, labels in self.tiger_test.next_batch_pb(random_state):
            tensor_x = tf.convert_to_tensor(np.array(X_data).astype(np.float32))
            net_out_value = self.PB_MODEL(tensor_x)[self.OUTPUT_NODE]
            pred_texts = self.decode_batch(net_out_value)
            
            texts = []
            for label in labels:
                text = self.tiger_test.labels_to_text(label)
                texts.append(text)
            
            bs = len(labels)
            for i in range(bs):
                if pred_texts[i] != texts[i]:
                    if verbose:
                        print('\nPredicted: \t\t %s\nTrue: \t\t\t %s' % (pred_texts[i], texts[i]))
                    err_c += 1
                else:
                    succ_c += 1
            break
        if verbose:
            print("Test processing time: {} seconds".format(time.time() - start_time))
        print("acc: {}".format(succ_c/(err_c+succ_c)))

    def predict(self, imgs: List, return_acc: bool = False) -> Any:
        xs = []
        for img in imgs:
            x = self.normalize(img)
            xs.append(x)
        pred_texts = []
        net_out_value = []
        if bool(xs):
            if len(xs) == 1:
                net_out_value = self.MODEL.predict_on_batch(np.array(xs))
            else:
                net_out_value = self.MODEL(np.array(xs), training=False)
            pred_texts = self.decode_batch(net_out_value)
        if return_acc:
            return pred_texts, net_out_value
        return pred_texts
    
    def predict_pb(self, imgs: List, return_acc: bool = False) -> Any:
        xs = []
        for img in imgs:
            x = self.normalize_pb(img)
            xs.append(x)
        pred_texts = []
        net_out_value = []
        if bool(xs):
            tensor_x = tf.convert_to_tensor(np.array(xs).astype(np.float32))
            net_out_value = self.PB_MODEL(tensor_x)[self.OUTPUT_NODE]
            pred_texts = self.decode_batch(net_out_value)
        if return_acc:
            return pred_texts, net_out_value
        return pred_texts

    def load(self, path_to_model: str, mode: str = "cpu", verbose: bool = False) -> Model:
        if path_to_model == "latest":
            model_info = download_latest_model("TextDetector", self.get_classname(), mode=mode)
            path_to_model = model_info["path"]

        self.MODEL = load_model(path_to_model, compile=False)

        net_inp = self.MODEL.get_layer(name='{}'.format(self.MODEL.layers[0].name)).input
        net_out = self.MODEL.get_layer(name='{}'.format(self.MODEL.layers[-1].name)).output

        self.MODEL = Model(inputs=net_inp, outputs=net_out)

        if verbose:
            self.MODEL.summary()

        return self.MODEL
    
    def load_pb(self, model_dir: str) -> tf.Tensor:
        pb_model = tf.saved_model.load(model_dir)
        self.PB_MODEL = pb_model.signatures["serving_default"]
        return self.PB_MODEL

    def prepare(self, path_to_dataset: str, use_aug: bool = False, verbose: bool = True,
                aug_debug: bool = False, aug_suffix: str = 'aug', aug_seed_num: int = 42) -> None:
        train_path = os.path.join(path_to_dataset, "train")
        test_path = os.path.join(path_to_dataset, "test")
        val_path = os.path.join(path_to_dataset, "val")

        if verbose:
            print("GET ALPHABET")
        self.letters, max_plate_length = self.get_alphabet(train_path, test_path, val_path, verbose=verbose)

        if verbose:
            print("\nEXPLAIN DATA TRANSFORMATIONS")
            self.explain_text_generator(train_path, self.letters, max_plate_length)

        if verbose:
            print("START BUILD DATA")
        self.tiger_train = TextImageGenerator(train_path,
                                              self.IMG_W,
                                              self.IMG_H,
                                              self.BATCH_SIZE,
                                              self.DOWNSAMPLE_FACROT,
                                              self.letters,
                                              max_plate_length,
                                              cname=type(self).__name__)
        self.tiger_train.build_data(use_aug=use_aug,
                                    aug_debug=aug_debug,
                                    aug_suffix=aug_suffix,
                                    aug_seed_num=aug_seed_num)
        self.tiger_val = TextImageGenerator(val_path,
                                            self.IMG_W,
                                            self.IMG_H,
                                            self.BATCH_SIZE,
                                            self.DOWNSAMPLE_FACROT,
                                            self.letters, max_plate_length,
                                            cname=type(self).__name__)
        self.tiger_val.build_data()

        self.tiger_test = TextImageGenerator(test_path,
                                             self.IMG_W,
                                             self.IMG_H,
                                             len(os.listdir(os.path.join(test_path, "img"))),
                                             self.DOWNSAMPLE_FACROT,
                                             self.letters,
                                             max_plate_length,
                                             cname=type(self).__name__)
        self.tiger_test.build_data()
        if verbose:
            print("DATA PREPARED")

    def train(self, is_random: bool = True, load_trained_model_path: str = None, load_last_weights: bool = False,
              verbose: bool = True, log_dir: str = "./") -> Model:
        if verbose:
            print("\nSTART TRAINING")
        if backend.image_data_format() == 'channels_first':
            input_shape = (1, self.IMG_W, self.IMG_H)
        else:
            input_shape = (self.IMG_W, self.IMG_H, 1)

        input_data = Input(name='the_input_{}'.format(type(self).__name__), shape=input_shape, dtype='float32')
        inner = Conv2D(self.CONV_FILTERS, self.KERNEL_SIZE, padding='same',
                       activation=self.ACTIVATION, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(self.POOL_SIZE, self.POOL_SIZE), name='max1')(inner)
        inner = Conv2D(self.CONV_FILTERS, self.KERNEL_SIZE, padding='same',
                       activation=self.ACTIVATION, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(self.POOL_SIZE, self.POOL_SIZE), name='max2')(inner)

        conv_to_rnn_dims = (self.IMG_W // (self.POOL_SIZE * self.POOL_SIZE),
                            (self.IMG_H // (self.POOL_SIZE * self.POOL_SIZE)) * self.CONV_FILTERS)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(self.TIME_DENSE_SIZE, activation=self.ACTIVATION, name='dense1')(inner)

        # Two layers of bidirecitonal GRUs
        gru_1 = GRU(self.RNN_SIZE,
                    return_sequences=True,
                    kernel_initializer='he_normal',
                    name='gru1')(inner)
        gru_1b = GRU(self.RNN_SIZE,
                     return_sequences=True, 
                     go_backwards=True, 
                     kernel_initializer='he_normal', 
                     name='gru1_b')(inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(self.RNN_SIZE,
                    return_sequences=True,
                    kernel_initializer='he_normal',
                    name='gru2')(gru1_merged)
        gru_2b = GRU(self.RNN_SIZE,
                     return_sequences=True, 
                     go_backwards=True, 
                     kernel_initializer='he_normal', 
                     name='gru2_b')(gru1_merged)

        # transforms RNN output to character activations:
        inner = Dense(self.tiger_train.get_output_size(), kernel_initializer='he_normal',
                      name='dense2')(concatenate([gru_2, gru_2b]))
        y_pred = Activation('softmax', name='softmax_{}'.format(type(self).__name__))(inner)
        Model(inputs=input_data, outputs=y_pred).summary()

        labels = Input(name='the_labels_{}'.format(type(self).__name__),
                       shape=[self.tiger_train.max_text_len],
                       dtype='float32')
        input_length = Input(name='input_length_{}'.format(type(self).__name__),
                             shape=[1],
                             dtype='int64')
        label_length = Input(name='label_length_{}'.format(type(self).__name__),
                             shape=[1],
                             dtype='int64')
        
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(self.ctc_lambda_func,
                          output_shape=(1,),
                          name='ctc')([y_pred, labels, input_length, label_length])
        
        # clipnorm seems to speeds up convergence
        adam = tf.keras.optimizers.Adam(lr=0.0001)

        if load_trained_model_path is not None:
            model = load_model(load_trained_model_path, compile=False)
        else:
            model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss={'{}'.format(model.layers[-1].name): lambda y_true, y_predict: y_predict}, optimizer=adam)

        # captures output of softmax so we can decode the output during visualization
        _ = backend.function([input_data], [y_pred])
        
        # traine callbacks
        self.CALLBACKS_LIST = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'buff_weights.h5'),
                monitor='val_loss',
                save_best_only=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.REDUCE_LRO_N_PLATEAU_FACTOR,
                patience=self.REDUCE_LRO_N_PLATEAU_PATIENCE,
            )
        ]
        
        if load_last_weights:
            model.load_weights(os.path.join(log_dir, 'buff_weights.h5'))
        
        model.fit_generator(generator=self.tiger_train.next_batch(is_random,
                                                                  input_name=model.layers[0].name,
                                                                  output_name=model.layers[-1].name),
                            steps_per_epoch=self.tiger_train.n,
                            epochs=self.EPOCHS,
                            callbacks=self.CALLBACKS_LIST,
                            validation_data=self.tiger_val.next_batch(is_random,
                                                                      input_name=model.layers[0].name,
                                                                      output_name=model.layers[-1].name),
                            validation_steps=self.tiger_val.n)
        # load best model
        model.load_weights(os.path.join(log_dir, 'buff_weights.h5'))
        
        net_inp = model.get_layer(name='{}'.format(model.layers[0].name)).input
        net_out = model.get_layer(name='{}'.format(model.layers[-5].name)).output
        if load_trained_model_path is not None:
            net_inp = model.get_layer(name='{}'.format(model.layers[0].name)).input
            net_out = model.get_layer(name='{}'.format(model.layers[-1].name)).output

        self.MODEL = Model(inputs=net_inp, outputs=net_out)
        return self.MODEL

    def get_acc(self, predicted: List, decode: List) -> float:
        labels = []
        for text in decode:
            labels.append(self.text_to_labels(text))
        loss = tf.keras.backend.ctc_batch_cost(
            np.array(labels),
            np.array(predicted)[:, 2:, :],
            np.array([[self.label_length] for _ in labels]),
            np.array([[self.max_text_len] for _ in labels])
        )
        return 1 - tf.keras.backend.eval(loss)
