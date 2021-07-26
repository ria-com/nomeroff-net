import collections
import torch
import torch.nn as nn
import numpy as np
from numpy import mean
from PIL import Image
from typing import List


class strLabelConverter(object):
    """Convert between str and label.
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet: str, ignore_case: bool = True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.char2idx = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.char2idx[char] = i + 1
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.char2idx[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
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
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
        return texts


def decode_prediction(logits: torch.Tensor, 
                      label_converter: strLabelConverter) -> str:
    tokens = logits.softmax(2).argmax(2)
    tokens = tokens.squeeze(1).numpy()
    
    # convert tor stings tokens
    tokens = ''.join([label_converter.idx2char[token] 
                      if token != 0  else '-' 
                      for token in tokens])
    tokens = tokens.split('-')
    
    # remove duplicates
    text = [char 
            for batch_token in tokens 
            for idx, char in enumerate(batch_token)
            if char != batch_token[idx-1] or len(batch_token) == 1]
    text = ''.join(text)
    return text


def decode_batch(net_out_value: torch.Tensor, 
                 label_converter: strLabelConverter) -> str:
    position_size, batch_size, char_size = net_out_value.shape
    net_out_value = net_out_value.reshape([batch_size, position_size, char_size])

    texts = []
    for logits in net_out_value:
        logits = logits.reshape([position_size, 1, char_size])
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
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    
    # clear previous graph
    clear_output(True)
    # making titles
    train_title = f'Epoch:{epoch} | Train Loss:{mean(train_losses[-n_steps:]):.6f}'
    val_title = f'Epoch:{epoch} | Val Loss:{mean(val_losses[-n_steps:]):.6f}'

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(train_losses)
    ax[1].plot(val_losses)

    ax[0].set_title(train_title)
    ax[1].set_title(val_title)

    plt.show()

def print_prediction(model, dataset, device, label_converter):
    import matplotlib.pyplot as plt
    
    idx = np.random.randint(len(dataset))
    path = dataset.pathes[idx]
    
    with torch.no_grad():
        model.eval()
        img, target_text = dataset[idx]
        img = img.unsqueeze(0)
        logits = model(img.to(device))
        
    pred_text = decode_prediction(logits.cpu(), label_converter)

    img = np.asarray(Image.open(path).convert('L'))
    title = f'Truth: {target_text} | Pred: {pred_text}'
    plt.imshow(img)
    plt.title(title)
    plt.axis('off');