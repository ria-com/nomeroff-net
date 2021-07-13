import itertools
from typing import List
import numpy as np


def labels_to_text(labels: List, letters: List) -> str:
    letters_max = len(letters) + 1
    data = ''.join(list(map(lambda x: "" if x == letters_max else letters[int(x)], labels)))
    return data


def text_to_labels(text: str, letters: List, max_text_len: int) -> List:
    letters_max = len(letters) + 1
    data = list(map(lambda x: letters.index(x), text))
    while len(data) < max_text_len:
        data.append(letters_max)
    return data


def decode_batch(out: np.ndarray, letters: List) -> List:
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
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret


def is_valid_str(s: str, letters: List) -> bool:
    for ch in s:
        if ch not in letters:
            return False
    return True
