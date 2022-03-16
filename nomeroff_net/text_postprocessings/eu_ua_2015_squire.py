"""
python3 -m nomeroff_net.text_postprocessings.eu_ua_2015_squire -f nomeroff_net/text_postprocessings/eu_ua_2015_squire.py
"""
from .eu_ua_2004_squire import EuUa2004Squire


class EuUa2015Squire(EuUa2004Squire):
    def __init__(self) -> None:
        super().__init__()


eu_ua_2015_squire = EuUa2015Squire()

if __name__ == "__main__":
    postprocessor = EuUa2015Squire()
    assert postprocessor.find("ABC1234") == "ABC1234"
