import re
import numpy as np
import string

class NPPatterns():
    def __init__(self, standart = "", allowed_liters = string.ascii_letters, black_list=["\s"]):
        self.STANDART = self.check_pattern_standart(standart)
        self.ALLOWED_LITERS = allowed_liters
        self.BLACK_LIST = black_list

    def delete_all_black_list_characters(self, text):
        reg = "[{}]".format("".join(self.BLACK_LIST))
        return re.sub(re.compile(reg), "", text)

    def check_pattern_standart(self, standart):
        if not re.match(r"^[#@]*$", standart):
            raise Exception("Standart {} not correct".format(standart))
        return standart

    def check_is_str(self, text):
        if type(text) is not str:
            raise ValueError("{} is not str".format(text))
        return text

    def findFully(self, text):
        reg = ""
        for item in self.STANDART:
            if item == "@":
                reg = "{}[{}]".format(reg,"".join(self.ALLOWED_LITERS))
            elif item == "#":
                reg = "{}[{}]".format(reg,"".join([str(item) for item in np.arange(10)]))
        reg_all = re.compile(reg)
        return re.search(reg_all, text)

    def findSimilary(self, text, min_count_litters = 4):
        res_reg_all = []
        for i in np.arange(len(self.STANDART)):
            item = self.STANDART[i]
            if item == "@":
                reg = "[{}]{}".format("".join(self.ALLOWED_LITERS), "."*(len(self.STANDART)-i-1))
            elif item == "#":
                reg = "[{}]{}".format("".join([str(item) for item in np.arange(10)]), "."*(len(self.STANDART)-i-1))
            reg = re.compile(reg)
            res_reg_all.append(re.findall(reg, text))
        return res_reg_all

    def find(self, text, min_count_litters = 4):
        text = self.check_is_str(text)
        text = self.delete_all_black_list_characters(text)
        text = text.upper()

        match = self.findFully(text)
        if match:
            return match.group(0)
        return text