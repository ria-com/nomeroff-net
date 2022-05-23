"""
Numberplate Orientation Model
python3 -m nomeroff_net.nnmodels.numberplate_orientation_model -f nomeroff_net/nnmodels/numberplate_orientation_model.py
"""
import torch
import torch.nn as nn
from .numberplate_inverse_model import NPInverseNet
from nomeroff_net.tools.mcm import get_device_torch


class NPOrientationNet(NPInverseNet):
    def __init__(self,
                 orientation_output_size: int,
                 img_h: int = 300,
                 img_w: int = 300,
                 batch_size: int = 1,
                 learning_rate=0.005754399373371567):
        super(NPInverseNet, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.inp_conv = nn.Conv2d(3, 32, (3, 3),
                                  stride=(1, 1),
                                  padding=(0, 0))
        self.conv1 = nn.Conv2d(32, 64, (3, 3),
                               stride=(1, 1),
                               padding=(0, 0))
        self.conv2 = nn.Conv2d(64, 128, (3, 3),
                               stride=(1, 1),
                               padding=(0, 0))
        self.conv3 = nn.Conv2d(128, 128, (3, 3),
                               stride=(1, 1),
                               padding=(0, 0))
        self.pool = nn.MaxPool2d(2, 2)

        img_w = int(img_w / 2 / 2 / 2 / 2 - 2)
        img_h = int(img_h / 2 / 2 / 2 / 2 - 2)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * img_w * img_h, 512)
        self.fc2 = nn.Linear(512, 256)
        self.batch_norm = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(256, orientation_output_size)


if __name__ == "__main__":
    net = NPOrientationNet(2)
    device = get_device_torch()
    net = net.to(device)
    xs = torch.rand((1, 3, 300, 300)).to(device)
    y = net(xs)
    print(y)
