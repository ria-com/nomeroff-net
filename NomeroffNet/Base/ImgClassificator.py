import cv2
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from keras.layers import merge
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras import models
from keras import layers
from keras.models import Model, Input
from sklearn.model_selection import GridSearchCV
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import ImageFile
from keras.wrappers.scikit_learn import KerasClassifier
from keras.applications import VGG16
from keras import callbacks
from keras.models import load_model

class ImgClassificator():
    def __init__(self):
        # input
        self.DEPTH          = 1
        self.HEIGHT         = 128
        self.WEIGHT         = 128
        self.COLOR_CHANNELS = 3

        # outputs
        self.CLASS_LABELS = ["BACKGROUND"]

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

        # compile model hyperparameters
        self.OPTIMIZER = 'Adamax'
        self.LOSS      = 'categorical_crossentropy'
        self.METRICS   = ['accuracy']

        # callbacks hyperparameters
        self.REDUCE_LRO_N_PLATEAU_PATIENCE = 10
        self.REDUCE_LRO_N_PLATEAU_FACTOR   = 0.1

        # train hyperparameters
        self.BATCH_SIZE       = 32
        self.STEPS_PER_EPOCH  = 100
        self.VALIDATION_STEPS = 50
        self.EPOCHS           = 150

    def change_dimension(self, w, h):
        if w != self.WEIGHT and h != self.HEIGHT:
            self.HEIGHT         = h
            self.WEIGHT         = w
            if self.MODEL != None:
                self.MODEL.layers.pop(0)
                newInput = Input(shape=(self.HEIGHT, self.WEIGHT, self.COLOR_CHANNELS))
                newOutputs = self.MODEL(newInput)
                self.MODEL = Model(newInput, newOutputs)

    def create_model(self, input_model, conv_base, dropout_1, dropout_2, dense_layers, output_labels, \
                     optimizer, loss, metrics, out_dense_init, W_regularizer, out_dense_activation, \
                     dense_activation, BatchNormalization_axis):
        # cnn
        conv_base = conv_base(input_model)
        x = layers.Dropout(dropout_1)(conv_base)

        # classificator
        x = layers.Flatten()(x)
        x = layers.Dropout(dropout_2)(x)
        x = layers.Dense(dense_layers, activation=dense_activation)(x)
        x = layers.BatchNormalization(axis=BatchNormalization_axis)(x)
        x = layers.Dense(output_labels, init=out_dense_init, W_regularizer=W_regularizer, \
                         activation=out_dense_activation)(x)
        model = Model(input=input_model, output=x)

        # compile model
        model.compile(optimizer=optimizer,
            loss=loss,
            metrics=metrics
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

    def train(self, base_dir, results_dir, model_name="model", verbose=1):
        # init count outputs
        self.OTPUT_LABELS = len(self.CLASS_LABELS)

        # you mast split your data on 3 directory
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')
        test_dir = os.path.join(base_dir, 'test')

        # trainable cnn model
        conv_base = VGG16(weights='imagenet',
            include_top=False)
        # block trainable cnn parameters
        conv_base.trainable = False

        # create input
        input_model = Input(shape=(self.HEIGHT, self.WEIGHT, self.COLOR_CHANNELS))

        # compile generators
        train_generator = self.compile_train_generator(train_dir, (self.HEIGHT, self.WEIGHT), self.BATCH_SIZE)
        validation_generator = self.compile_test_generator(validation_dir, (self.HEIGHT, self.WEIGHT), self.BATCH_SIZE)

        # traine callbacks
        self.CALLBACKS_LIST = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(results_dir, 'buff_weights.h5'),
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
                                 output_labels = self.OTPUT_LABELS, optimizer = self.OPTIMIZER, loss = self.LOSS, \
                                 metrics = self.METRICS, out_dense_init = self.OUT_DENSE_INIT, W_regularizer = self.W_REGULARIZER, \
                                 out_dense_activation = self.OUT_DENSE_ACTIVATION , dense_activation = self.DENSE_ACTIVATION, \
                                 BatchNormalization_axis = self.BATCH_NORMALIZATION_AXIS)

            # train
            history = model.fit_generator(
                train_generator,
                steps_per_epoch=self.STEPS_PER_EPOCH,
                epochs=self.EPOCHS,
                callbacks=self.CALLBACKS_LIST,
                validation_data=validation_generator,
                validation_steps=self.VALIDATION_STEPS,
                verbose=verbose
            )

            # load best model
            model.load_weights(os.path.join(results_dir, 'buff_weights.h5'))

            # append to models
            modelsArr.append(model)

        # merge ensembles
        if len(modelsArr) > 1:
            self.MODEL = self.ensemble(modelsArr, input_model)
        elif len(modelsArr) == 1:
            self.MODEL = modelsArr[0]

        # test
        test_loss, test_acc = self.test(test_dir)
        print(f"test acc: {test_acc} test loss: {test_loss}")

        # save model
        self.save_model(results_dir, model_name, verbose)

    def test(self, test_dir):
        # compile generator
        test_generator = self.compile_test_generator(test_dir, (self.HEIGHT, self.WEIGHT), self.BATCH_SIZE)

        # test
        return self.MODEL.evaluate_generator(test_generator, steps=self.VALIDATION_STEPS)

    def save_model(self, dir, model_name="model", verbose=1):
        now = datetime.now()
        path = os.path.join(dir, f"{model_name}_{now.year}_{now.month}_{now.day}.h5")
        if self.MODEL != None:
            if bool(verbose):
                print(f"model save to {path}")
            self.MODEL.save(path)

    def isLoaded(self):
        if self.MODEL == None:
            return False
        return True

    def load(self, path_to_model, verbose = 0):
        self.MODEL = load_model(path_to_model)
        if verbose:
            self.MODEL.summary()

    def getLabels(self, index):
        return self.CLASS_LABELS[index]

    def normalize(self, img):
        img = img / 255.
        img = cv2.resize(img, (self.WEIGHT, self.HEIGHT))
        return img

    def predict(self, img):
        img = self.normalize(img)
        predicted = self.MODEL.predict(np.array([img]))
        return int(np.argmax(predicted))

    def compile_train_generator(self, train_dir, target_size, batch_size=32):
        # with data augumentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=5,
            shear_range=0.05,
            height_shift_range=0.1,
            zoom_range=[1.0, 1.1],
            brightness_range=(0.5, 1.5),
            data_format='channels_last',
            channel_shift_range=0.1,
            fill_mode='nearest'
        )
        return train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

    def compile_test_generator(self, test_dir, target_size, batch_size=32):
        test_datagen = ImageDataGenerator(rescale=1./255)
        return test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical')