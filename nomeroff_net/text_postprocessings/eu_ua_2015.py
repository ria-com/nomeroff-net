"""
python3 -m nomeroff_net.text_postprocessings.eu_ua_2015 -f nomeroff_net/text_postprocessings/eu_ua_2015.py
"""
from .eu_ua_2004 import EuUa2004


class EuUa2015(EuUa2004):
    def __init__(self) -> None:
        super().__init__()


eu_ua_2015 = EuUa2015()

if __name__ == "__main__":
    postprocessor = EuUa2015()
    assert postprocessor.find("A€12Q4CH") == "AC1204CH"
