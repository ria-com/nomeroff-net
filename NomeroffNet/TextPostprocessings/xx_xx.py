import re
import numpy as np
import string
from typing import List, Optional


class XxXx(object):
    def __init__(self, standart: str = "", allowed_liters: str = string.ascii_letters,
                 black_list: List[str] = None) -> None:
        if black_list is None:
            black_list = ["\s", '\*', '\,', '\.', '\-', "\'", '\"', "\’", "_", "\+"]
        self.STANDART = self.check_pattern_standart(standart)
        self.ALLOWED_LITERS = allowed_liters
        self.BLACK_LIST = black_list
        self.ALLOWED_NUMBERS = [str(item) for item in np.arange(10)]
        self.REPLACEMENT = {
            "#": {
                "I": "1",
                "Z": "2",  # 7
                "O": "0",
                "Q": "0",
                "B": "8",
                "D": "0",
                "S": "5",  # 8
                "T": "7"
            },
            "@": {
                "/": "I",
                "|": "I",
                "¥": "X",
                "€": "C"
            }
        }

    def delete_all_black_list_characters(self, text: str) -> str:
        reg = "[{}]".format("".join(self.BLACK_LIST))
        return re.sub(re.compile(reg), "", text)\
                 .replace("\\", "/")\
                 .replace("\[", "|").replace("\]", "|")

    @staticmethod
    def check_pattern_standart(standart: str) -> str:
        if not re.match(r"^[#@]*$", standart):
            raise Exception("Standart {} not correct".format(standart))
        return standart

    @staticmethod
    def check_is_str(text: str) -> str:
        if type(text) is not str:
            raise ValueError("{} is not str".format(text))
        return text

    def find_fully(self, text: str) -> Optional:
        reg = ""
        for item in self.STANDART:
            if item == "@":
                reg = "{}[{}]".format(reg, "".join(self.ALLOWED_LITERS))
            elif item == "#":
                reg = "{}[{}]".format(reg, "".join(self.ALLOWED_NUMBERS))
        reg_all = re.compile(reg)
        return re.search(reg_all, text)

    def replace(self, text: str) -> str:
        res = ""
        for i in np.arange(len(self.STANDART)):
            l_dict = self.ALLOWED_LITERS
            if self.STANDART[i] == "#":
                l_dict = self.ALLOWED_NUMBERS

            if text[i] in l_dict:
                res = "{}{}".format(res, text[i])
            else:
                replace_l = self.REPLACEMENT[self.STANDART[i]][text[i]]
                res = "{}{}".format(res, replace_l)
        return res

    def find_similary(self, text: str) -> str:
        vcount = len(text) - len(self.STANDART) + 1
        reg = ""
        for item in self.STANDART:
            main = ""
            dop = ""
            if item == "@":
                dop = list(self.REPLACEMENT["@"].keys())
                main = self.ALLOWED_LITERS
            elif item == "#":
                dop = list(self.REPLACEMENT["#"].keys())
                main = self.ALLOWED_NUMBERS
            buf_reg = "".join(main + dop)
            reg = "{}[{}]".format(reg, buf_reg)
        reg_sim = re.compile(reg)
        for i in np.arange(vcount):
            buff_text = text[int(i):int(len(self.STANDART)+i)]
            match = re.search(reg_sim, buff_text)
            if match:
                return self.replace(match.group(0))
        return text

    def find(self, text: str, strong: bool = True) -> str:
        text = self.check_is_str(text)
        text = self.delete_all_black_list_characters(text)
        text = text.upper()

        if len(text) < len(self.STANDART):
            return text

        if len(self.STANDART):
            match = self.find_fully(text)
            if match:
                return match.group(0)

        if not strong:
            return self.find_similary(text)
        return text


xx_xx = XxXx()
