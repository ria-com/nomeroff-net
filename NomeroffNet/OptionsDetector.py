import os
import cv2
import numpy as np
import tensorflow as tf

class OptionsDetector():
    def __init__(self):
        # input
        self.DEPTH          = 1
        self.HEIGHT         = 64
        self.WEIGHT         = 295
        self.COLOR_CHANNELS = 3

        # outputs
        self.CLASS_REGION = ["xx_unknown", "eu_ua_2015", "eu_ua_2004", "eu_ua_1995", "eu", "xx_transit"]
        self.CLASS_STATE = ["BACKGROUND", "FILLED", "NOT_FILLED"]

        # frozen graph inp and out
        self.INPUT_NODE = "input_2:0"
        self.OUTPUT_NODES = ("REGION/Softmax:0", "STATE/Softmax:0")

    def getRegionLabels(self, indexes):
        return [self.CLASS_REGION[index] for index in indexes]

    def normalize(self, img, with_aug=False):
        img = cv2.resize(img, (self.WEIGHT, self.HEIGHT))
        img = img.astype(np.float32)

        # advanced normalisation
        #img /= 255
        img_min = np.amin(img)
        img -= img_min
        img_max = np.amax(img)
        img /= img_max
        return img

    def getStateLabels(self, indexes):
        return [self.CLASS_STATE[index] for index in indexes]

    def load(self, model_path):
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        graph = tf.Graph()
        with graph.as_default():
            self.net_inp, self.net_out1, self.net_out2 = tf.import_graph_def(
                graph_def, return_elements = [self.INPUT_NODE, self.OUTPUT_NODES[0], self.OUTPUT_NODES[1]]
            )
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=sess_config)

    def predict(self, imgs):
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