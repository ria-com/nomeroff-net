from .base.ocr import OCR


class RuMilitary(OCR):
    def __init__(self) -> None:
        OCR.__init__(self)
        self.letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "E", "H", "K", "M", "O",
                        "P", "T", "X", "Y"]
        self.max_text_len = 8
        self.max_plate_length = 8
        self.letters_max = len(self.letters)+1
        self.label_length = 32 - 2
        self.init_label_converter()
ru_military = RuMilitary