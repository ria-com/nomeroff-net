from Base import OCR


class Su(OCR):
    def __init__(self) -> None:
        OCR.__init__(self)
        self.letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        'І', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И',
                        'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У',
                        'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я']
        self.max_text_len = 7
        self.letters_max = len(self.letters)+1
        self.label_length = 32 - 2


su = Su()
