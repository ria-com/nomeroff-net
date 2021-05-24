from .xx_xx import XxXx


class EuUa1995(XxXx):
    def __init__(self) -> None:
        super().__init__()
        self.STANDART = "#####@@"
        self.ALLOWED_LITERS = ["A", "B", "E", "I", "K", "M", "H", "O", "P", "C", "T", "X"]
        self.REPLACEMENT = {
            "#": {
                "O": "0",
                "Q": "0",
                "D": "0",
                "I": "1",
                "Z": "2",  # 7
                "S": "5",  # 8
                "T": "7",
                "B": "8"
            },
            "@": {
                "/": "I",
                "|": "I",
                "L": "I",
                "1": "I",
                "5": "B",
                "8": "B",
                "R": "B",
                "0": "O",
                "Q": "O",
                "¥": "X",
                "Y": "X",
                "€": "C",
                "F": "E"
            }
        }

    def find(self, text: str, strong: bool = False) -> str:
        text = super().find(text, strong)
        return text


eu_ua_1995 = EuUa1995()
