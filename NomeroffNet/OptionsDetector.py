import cv2
import keras
import os
import numpy as np
from keras.layers import merge
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras import models
from keras import layers
from keras import backend as K
from keras.models import Model, Input
from keras.preprocessing import image
from keras.applications import VGG16
from keras import callbacks
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from .mcm.mcm import download_latest_model
from .Base.ImgGenerator import ImgGenerator

class OptionsDetector(ImgGenerator):
    def __init__(self, options = {}):
        # input
        self.DEPTH          = 1
        self.HEIGHT         = 64
        self.WEIGHT         = 295
        self.COLOR_CHANNELS = 3

        # outputs 1
        self.CLASS_REGION = options.get("class_region", ["xx_unknown", "eu_ua_2015", "eu_ua_2004", "eu_ua_1995", "eu", "xx_transit", "ru", "kz"])

        # outputs 2
        self.CLASS_STATE = options.get("class_state", ["BACKGROUND", "FILLED", "NOT_FILLED"])

        # outputs 3
        self.CLASS_COUNT_LINE = options.get("class_count_line", ["0", "1", "2"])

        # model
        self.MODEL = None

        # model hyperparameters
        self.OUT_DENSE_INIT                = 'uniform'
        self.OUT_DENSE_ACTIVATION          = 'softmax'
        self.DENSE_ACTIVATION              = 'softmax'
        self.DROPOUT_1                     = 0.2
        self.DROPOUT_2                     = 0.5
        self.DENSE_LAYERS                  = 512
        self.ENSEMBLES                     = 1
        self.BATCH_NORMALIZATION_AXIS      = -1
        self.L2_LAMBDA                     = 0.001
        self.W_REGULARIZER                 = l2(self.L2_LAMBDA)

        # callbacks hyperparameters
        self.REDUCE_LRO_N_PLATEAU_PATIENCE = 10
        self.REDUCE_LRO_N_PLATEAU_FACTOR   = 0.1

        # train hyperparameters
        self.BATCH_SIZE       = 32
        self.STEPS_PER_EPOCH  = 0 # defain auto
        self.VALIDATION_STEPS = 0 # defain auto
        self.EPOCHS           = 150

        # compile model hyperparameters
        self.LOSSES = {
            "REGION":     "categorical_crossentropy",  # tf.losses.softmax_cross_entropy
            "STATE":      "categorical_crossentropy",  # tf.losses.softmax_cross_entropy
            "COUNT_LINE": "categorical_crossentropy",  # tf.losses.softmax_cross_entropy
        }
        self.LOSS_WEIGHTS = {"REGION": 1.0, "STATE": 1.0, "COUNT_LINE": 1.0}
        self.OPT = "adamax" #  tf.keras.optimizers.Adamax
        self.METRICS = ["accuracy"]

        # for frozen graph
        self.INPUT_NODE = "input_2:0"
        self.OUTPUT_NODES = ("REGION/Softmax:0", "STATE/Softmax:0")

    def change_dimension(self, w, h):
        if w != self.WEIGHT and h != self.HEIGHT:
            self.HEIGHT         = h
            self.WEIGHT         = w
            if self.MODEL != None:
                self.MODEL.layers.pop(0)
                newInput = Input(shape=(self.HEIGHT, self.WEIGHT, self.COLOR_CHANNELS))
                newOutputs = self.MODEL(newInput)
                self.MODEL = Model(newInput, newOutputs)

    def create_model(self, input_model, conv_base, dropout_1, dropout_2, dense_layers, output_labels1, \
                     output_labels2, output_labels3, out_dense_init, W_regularizer, \
                     out_dense_activation, dense_activation, BatchNormalization_axis):
        # cnn
        x = conv_base

        # classificator 1
        x1 = layers.Flatten()(x)
        x1 = layers.Dropout(dropout_2)(x1)
        x1 = layers.Dense(dense_layers, activation=dense_activation)(x1)
        x1 = layers.BatchNormalization(axis=BatchNormalization_axis)(x1)
        x1 = layers.Dense(output_labels1, init=out_dense_init, W_regularizer=W_regularizer)(x1)
        x1 = layers.Activation(out_dense_activation, name="REGION")(x1)

        # classificator 2
        x2 = layers.Flatten()(x)
        x2 = layers.Dropout(dropout_2)(x2)
        x2 = layers.Dense(dense_layers, activation=dense_activation)(x2)
        x2 = layers.BatchNormalization(axis=BatchNormalization_axis)(x2)
        x2 = layers.Dense(output_labels2, init=out_dense_init, W_regularizer=W_regularizer)(x2)
        x2 = layers.Activation(out_dense_activation, name="STATE")(x2)

        # classificator 3
        x3 = layers.Flatten()(x)
        x3 = layers.Dropout(dropout_2)(x3)
        x3 = layers.Dense(dense_layers, activation=dense_activation)(x3)
        x3 = layers.BatchNormalization(axis=BatchNormalization_axis)(x3)
        x3 = layers.Dense(output_labels3, init=out_dense_init, W_regularizer=W_regularizer)(x3)
        x3 = layers.Activation(out_dense_activation, name="COUNT_LINE")(x3)

        #x = keras.layers.concatenate([x1, x2], axis=1)
        model = Model(input=input_model, outputs=[x1, x2, x3])

        # compile model
        model.compile(
            optimizer=self.OPT,
            loss=self.LOSSES,
            loss_weights=self.LOSS_WEIGHTS,
            metrics=self.METRICS
        )
        return model

    def ensemble(self, models, model_input):
        # collect outputs of models in a list
        outputs = [model(model_input) for model in models]

        # averaging outputs
        y = layers.Average()(outputs)

        # build model from same input and avg output
        model = Model(inputs=model_input, outputs=y, name='ensemble')

        return model

    def prepare(self, base_dir, verbose=1):
        if verbose:
            print("START PREPARING")
        # you mast split your data on 3 directory
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'val')
        test_dir = os.path.join(base_dir, 'test')

        # compile generators
        self.train_generator = self.compile_train_generator(train_dir, (self.HEIGHT, self.WEIGHT), self.BATCH_SIZE)
        self.validation_generator = self.compile_test_generator(validation_dir, (self.HEIGHT, self.WEIGHT), self.BATCH_SIZE)

        self.test_generator = self.compile_test_generator(test_dir, (self.HEIGHT, self.WEIGHT), self.BATCH_SIZE)
        if verbose:
            print("DATA PREPARED")

    def create_conv(self, inp):
         # trainable cnn model
        conv_base = VGG16(weights='imagenet',
            include_top=False)
        # block trainable cnn parameters
        conv_base.trainable = False
        return conv_base(inp)

    def create_simple_conv(self, inp):
        conv_base = layers.Conv2D(32, (3, 3), activation='relu')(inp)
        conv_base = layers.MaxPooling2D((2, 2))(conv_base)

        conv_base = layers.Conv2D(64, (3, 3), activation='relu')(conv_base)
        conv_base = layers.MaxPooling2D((2, 2))(conv_base)

        conv_base = layers.Conv2D(128, (3, 3), activation='relu')(conv_base)
        conv_base = layers.MaxPooling2D((2, 2))(conv_base)

        conv_base = layers.Conv2D(128, (3, 3), activation='relu')(conv_base)
        conv_base = layers.MaxPooling2D((2, 2))(conv_base)

        return conv_base

    def train(self, log_dir="./", verbose=1, cnn="simple"):
        # init count outputs
        self.OTPUT_LABELS_1 = len(self.CLASS_REGION)
        self.OTPUT_LABELS_2 = len(self.CLASS_STATE)
        self.OTPUT_LABELS_3 = len(self.CLASS_COUNT_LINE)

        # create input
        input_model = Input(shape=(self.HEIGHT, self.WEIGHT, self.COLOR_CHANNELS))

        if (cnn == "simple"):
            conv_base = self.create_simple_conv(input_model)
        else:
            conv_base = self.create_conv(input_model)

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

        # train all
        modelsArr = []
        for i in np.arange(self.ENSEMBLES):
            # create model
            model = self.create_model(input_model = input_model, conv_base = conv_base, \
                                 dropout_1 = self.DROPOUT_1 , dropout_2 = self.DROPOUT_2, dense_layers = self.DENSE_LAYERS, \
                                 output_labels1 = self.OTPUT_LABELS_1, output_labels2 = self.OTPUT_LABELS_2, \
                                 output_labels3 = self.OTPUT_LABELS_3,
                                 out_dense_init = self.OUT_DENSE_INIT, W_regularizer = self.W_REGULARIZER, \
                                 out_dense_activation = self.OUT_DENSE_ACTIVATION , dense_activation = self.DENSE_ACTIVATION, \
                                 BatchNormalization_axis = self.BATCH_NORMALIZATION_AXIS)

            # train
            history = model.fit_generator(
                self.train_generator,
                steps_per_epoch=self.STEPS_PER_EPOCH,
                epochs=self.EPOCHS,
                callbacks=self.CALLBACKS_LIST,
                validation_data=self.validation_generator,
                validation_steps=self.VALIDATION_STEPS,
                verbose=verbose
            )

            # load best model
            model.load_weights(os.path.join(log_dir, 'buff_weights.h5'))

            # append to models
            modelsArr.append(model)

        # merge ensembles
        if len(modelsArr) > 1:
            self.MODEL = self.ensemble(modelsArr, input_model)
        elif len(modelsArr) == 1:
            self.MODEL = modelsArr[0]

        return  self.MODEL

    def test(self):
        test_loss, test_loss1, test_loss2, test_loss3, test_acc1, test_acc2, test_acc3 = self.MODEL.evaluate_generator(self.test_generator, steps=self.VALIDATION_STEPS)
        print("test loss: {}".format(test_loss))
        print("test loss: {}    test loss: {}       test loss: {}".format(test_loss1, test_loss2, test_loss3))
        print("test acc: {}    test acc {}      test acc {}".format(test_acc1, test_acc2, test_acc3))
        return test_loss, test_loss1, test_loss2, test_loss3, test_acc1, test_acc2, test_acc3

    def save(self, path, verbose=1):
        if self.MODEL != None:
            if bool(verbose):
                print("model save to {}".format(path))
            self.MODEL.save(path)

    def isLoaded(self):
        if self.MODEL == None:
            return False
        return True

    @classmethod
    def get_classname(cls):
        return cls.__name__

    def load(self, path_to_model, options={}, verbose = 0):
        if path_to_model == "latest":
            model_info   = download_latest_model(self.get_classname(), "simple")
            path_to_model   = model_info["path"]
            options["class_region"] = model_info["class_region"]

        self.CLASS_REGION = options.get("class_region", ["xx_unknown", "eu_ua_2015", "eu_ua_2004", "eu_ua_1995", "eu", "xx_transit", "ru", "kz"])
        if path_to_model.split(".")[-1] != "pb":
            self.MODEL = load_model(path_to_model)
            if verbose:
                self.MODEL.summary()
        else:
            self.load_frozen(path_to_model)
            self.predict = self.frozen_predict

    def getRegionLabel(self, index):
        return self.CLASS_REGION[index]

    def getStateLabel(self, index):
        return self.CLASS_STATE[index]

    def predict(self, imgs):
        Xs = []
        for img in imgs:
            Xs.append(self.normalize(img))

        predicted = [[], [], []]
        if bool(Xs):
            predicted = self.MODEL.predict(np.array(Xs))
	    
        regionIds = []
        for region in predicted[0]:
            regionIds.append(int(np.argmax(region)))

        stateIds = []
        for state in predicted[1]:
            stateIds.append(int(np.argmax(state)))

        countLines = []
        for countL in predicted[2]:
            countLines.append(int(np.argmax(countL)))

        return regionIds, stateIds, countLines

    def getRegionLabels(self, indexes):
        return [self.CLASS_REGION[index] for index in indexes]

    def compile_train_generator(self, train_dir, target_size, batch_size=32):
        # with data augumentation
        imageGenerator = ImgGenerator(train_dir, self.WEIGHT, self.HEIGHT, self.BATCH_SIZE, [len(self.CLASS_STATE), len(self.CLASS_REGION), len(self.CLASS_COUNT_LINE)])
        print("start train build")
        imageGenerator.build_data()
        self.STEPS_PER_EPOCH = self.STEPS_PER_EPOCH or imageGenerator.n / imageGenerator.batch_size or imageGenerator.n / imageGenerator.batch_size + 1
        print("end train build")
        return  imageGenerator.generator()

    def compile_test_generator(self, test_dir, target_size, batch_size=32):
        imageGenerator = ImgGenerator(test_dir, self.WEIGHT, self.HEIGHT, self.BATCH_SIZE, [len(self.CLASS_STATE), len(self.CLASS_REGION), len(self.CLASS_COUNT_LINE)])
        print("start test build")
        imageGenerator.build_data()
        self.VALIDATION_STEPS = self.VALIDATION_STEPS or imageGenerator.n / imageGenerator.batch_size or imageGenerator.n / imageGenerator.batch_size + 1
        print("end test build")
        return  imageGenerator.generator()

    def load_frozen(self, FROZEN_MODEL_PATH):
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(FROZEN_MODEL_PATH, "rb") as f:
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            self.net_inp, self.net_out1, self.net_out2 = tf.import_graph_def(
                graph_def, return_elements = [self.INPUT_NODE, self.OUTPUT_NODES[0], self.OUTPUT_NODES[1]]
            )
            #print([x.name for x in graph_def.node])

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=sess_config)

    def frozen_predict(self, imgs):
        Xs = []
        for img in imgs:
            img = self.normalize(img)
            Xs.append(img)

        predicted = [[], []]
        if bool(Xs):
            predicted = self.sess.run([self.net_out1, self.net_out2], feed_dict={self.net_inp:Xs})

        regionIds = []
        for region in predicted[0]:
            regionIds.append(int(np.argmax(region)))

        stateIds = []
        for state in predicted[1]:
            stateIds.append(int(np.argmax(state)))

        return regionIds, stateIds