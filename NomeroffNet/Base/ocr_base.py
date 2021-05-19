# import modules
import itertools
from typing import List
import numpy as np
import cv2


class BaseOCR(object):
    def __init__(self) -> None:
        self.IMG_H = 64
        self.IMG_W = 128
        self.letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "E", "H", "I", "K", "M",
                        "O", "P", "T", "X", "Z"]
        self.max_text_len = 8
        self.letters_max = len(self.letters)+1
        self.label_length = 32 - 2

    def labels_to_text(self, labels: List) -> str:
        data = ''.join(list(map(lambda x: "" if x == self.letters_max else self.letters[int(x)], labels)))
        return data

    def text_to_labels(self, text: str) -> List:
        data = list(map(lambda x: self.letters.index(x), text))
        while len(data) < self.max_text_len:
            data.append(self.letters_max)
        return data

    def decode_batch(self, out: np.ndarray) -> List:
        """
        this should be beam search with a dictionary and language model.
        For this example, best path is sufficient.
        """
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

    def normalize(self, img: np.ndarray) -> np.ndarray:
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (self.IMG_W, self.IMG_H))

        img = img.astype(np.float32)
        img -= np.amin(img)
        img /= np.amax(img)
        img = [[[h] for h in w] for w in img.T]

        x = np.zeros((self.IMG_W, self.IMG_H, 1))
        x[:, :, :] = img
        return x

    def normalize_pb(self, img: np.ndarray) -> np.ndarray:
        return self.normalize(img)
