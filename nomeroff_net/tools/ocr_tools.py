import itertools
import torch
import numpy as np
from numpy import mean
from PIL import Image, ImageDraw
from typing import List

import collections

try:
    collections_abc = collections.abc
except AttributeError:
    collections_abc = collections


class StrLabelConverter(object):
    """Convert between str and label.
        Insert `blank` to the alphabet for CTC.
    Args:
        letters (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, letters: str,
                 max_text_len: int,
                 ignore_case: bool = True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            letters = letters.lower()
        self.letters = letters
        self.letters_max = len(self.letters) + 1
        self.max_text_len = max_text_len

    def labels_to_text(self, labels: List) -> str:
        out_best = [k for k, g in itertools.groupby(labels)]
        outstr = ''
        for c in out_best:
            if c != 0:
                outstr += self.letters[c - 1]
        return outstr

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        length = []
        if isinstance(text, str):
            text = list(map(lambda x: self.letters.index(x) + 1, text))
            while len(text) < self.max_text_len:
                text.append(0)
            length = [len(text)]
        elif isinstance(text, collections_abc.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            out_best = list(np.argmax(t[0, :], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if c != 0:
                    outstr += self.letters[c - 1]
            return outstr
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                texts.append(
                    self.decode(
                        t[index:index + length[i]], torch.IntTensor([length[i]])))
                index += length[i]
        return texts


def decode_prediction(logits: torch.Tensor,
                      label_converter: StrLabelConverter) -> str:
    tokens = logits.softmax(2).argmax(2)
    tokens = tokens.squeeze(1).numpy()

    text = label_converter.labels_to_text(tokens)
    return text


def decode_batch(net_out_value: torch.Tensor,
                 label_converter: StrLabelConverter) -> str or List:
    texts = []
    for i in range(net_out_value.shape[1]):
        logits = net_out_value[:, i:i+1, :]
        pred_texts = decode_prediction(logits, label_converter)
        texts.append(pred_texts)
    return texts


def is_valid_str(s: str, letters: List) -> bool:
    for ch in s:
        if ch not in letters:
            return False
    return True


def plot_loss(epoch: int,
              train_losses: list,
              val_losses: list,
              n_steps: int = 100):
    """
    Plots train and validation losses
    """
    import matplotlib.pyplot as plt

    # making titles
    train_title = f'Epoch:{epoch} | Train Loss:{mean(train_losses[-n_steps:]):.6f}'
    val_title = f'Epoch:{epoch} | Val Loss:{mean(val_losses[-n_steps:]):.6f}'

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(train_losses)
    ax[1].plot(val_losses)

    ax[0].set_title(train_title)
    ax[1].set_title(val_title)

    plt.show()


def print_prediction(model,
                     dataset,
                     device,
                     label_converter,
                     w=200,
                     h=50,
                     count_zones=16):
    import matplotlib.pyplot as plt

    idx = np.random.randint(len(dataset))
    path = dataset.pathes[idx]

    with torch.no_grad():
        model.eval()
        img, target_text = dataset[idx]
        img = img.unsqueeze(0)
        logits = model(img.to(device))

    pred_text = decode_prediction(logits.cpu(), label_converter)
    img = Image.open(path).convert('L')
    img = img.resize((w, h))
    draw = ImageDraw.Draw(img)
    for i in np.arange(0, w, w / count_zones):
        if 1 > i or i > w:
            continue
        draw.line((i, 0, i, img.size[0]), fill=256)
    img = np.asarray(img)
    title = f'Truth: {target_text} | Pred: {pred_text}'
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()
