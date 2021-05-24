from .eu_ua_2004 import EuUa2004


class EuUa2015(EuUa2004):
    def __init__(self) -> None:
        super().__init__()


eu_ua_2015 = EuUa2015()
