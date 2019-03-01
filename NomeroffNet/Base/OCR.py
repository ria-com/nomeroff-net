import cv2
import numpy as np
import itertools
import tensorflow as tf

class OCR():
    def __init__(self):
        self.IMG_H = 64
        self.IMG_W = 128
        self.IMG_C = 1

    def decode_batch(self, out):
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

    def load(self, model_path):
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            self.net_inp, self.net_out = tf.import_graph_def(
                graph_def, return_elements = [self.INPUT_NODE, self.OUTPUT_NODE]
            )

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=sess_config)

    def predict(self, imgs):
        Xs = []
        for img in imgs:
            x = self.normalize(img)
            Xs.append(x)
        pred_texts = []
        if bool(Xs):
            net_out_value = self.sess.run([self.net_out], feed_dict={self.net_inp: np.array(Xs)})
            pred_texts = self.decode_batch(net_out_value[0])
        return pred_texts

    def normalize(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (self.IMG_W, self.IMG_H))
        img = img.astype(np.float32)
        img -= np.amin(img)
        img /= np.amax(img)
        img = [[[h] for h in w] for w in img.T]

        x = np.zeros((self.IMG_W, self.IMG_H, 1))
        x[:, :, :] = img
        return x