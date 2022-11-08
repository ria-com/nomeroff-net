"""
python3 -m nomeroff_net.text_detectors.ge -f nomeroff_net/text_detectors/ge.py
"""
import torch
from .base.ocr import OCR
from nomeroff_net.tools.mcm import get_device_torch


class Ge(OCR):
    def __init__(self) -> None:
        OCR.__init__(self)
        self.letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I",
                        "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        self.max_text_len = 8
        self.height = 50
        self.width = 200
        self.color_channels = 3
        self.label_length = 13
        self.letters_max = len(self.letters)+1
        self.init_label_converter()


ge = Ge

if __name__ == "__main__":
    ocr = Ge()
    ocr.load()
    device = get_device_torch()
    xs = torch.rand((1, 3, 50, 200)).to(device)
    y = ocr.predict(xs)
    print(y)
