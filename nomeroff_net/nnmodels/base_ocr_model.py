"""
Numberplate OCR Model
python3 -m nomeroff_net.nnmodels.ocr_model -f nomeroff_net/nnmodels/base_ocr_model.py
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


class BaseOcrNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=False)
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)

    def forward(self, xs: torch.float64):
        return self.resnet(xs)


if __name__ == "__main__":
    net = BaseOcrNet()
    xs = torch.rand((1, 3, 64, 295))
    ys = net(xs)
    print(ys)
