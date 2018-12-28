from .NPPatterns import NPPatterns

class ukr2015(NPPatterns):
    def __init__(self,
                 standart = "@@####@@",
                 allowed_liters = ["A", "B", "E", "I", "K", "M", "H", "O", "P", "C", "T", "X"],
                 black_list = ["\s"]):
        super().__init__(standart, allowed_liters, black_list)