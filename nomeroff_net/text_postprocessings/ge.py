"""
python3 -m nomeroff_net.text_postprocessings.ge -f nomeroff_net/text_postprocessings/ge.py
"""
from .xx_xx import XxXx
import string


class Ge(XxXx):
    def __init__(self) -> None:
        super().__init__()
        self.ALLOWED_LITERS = [x for x in string.ascii_letters]
        self.ALLOWED_LITERS.append("0")

        self.STANDARTS = ["@@@###", "@@###@@"]
        self.STANDART = ""

    def find(self, text: str, strong: bool = False) -> str:
        for standart in self.STANDARTS:
            self.STANDART = standart
            match = self.find_fully(text)
            if match:
                text = match.group(0)
                newtext = ""
                for i, standart_letter in enumerate(standart):
                    if standart_letter[0] == "@" and text[i] == "0":
                        newtext += "O"
                    else:
                        newtext += text[i]
                return newtext
        return text


ge = Ge()

if __name__ == "__main__":
    postprocessor = Ge()
    assert postprocessor.find("A0C123") == "AOC123"
    assert postprocessor.find("A0123C0") == "AO123CO"
