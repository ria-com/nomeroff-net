"""
Numberplate Inverse Model
TEST: python3 -m nomeroff_net.nnmodels.numberplate_inverse_model -f nomeroff_net/nnmodels/numberplate_inverse_model.py
"""
import torch
import torch.nn as nn
from torch.nn import functional
from .numberplate_classification_model import ClassificationNet
from nomeroff_net.tools.mcm import get_device_torch


class NPInverseNet(ClassificationNet):
    """
    Numberplate Inverse Model

    Examples:
        net = NPInverseNet(2)
        device = get_device_torch()
        net = net.to(device)
        xs = torch.rand((1, 3, 64, 295)).to(device)
        y = net(xs)
        print(y)
    """
    def __init__(self,
                 orientation_output_size: int,
                 img_h: int = 64,
                 img_w: int = 295,
                 batch_size: int = 1,
                 learning_rate: float = 0.005):
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

    def forward(self, x):
        x = self.pool(functional.relu(self.inp_conv(x)))
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = self.pool(functional.relu(self.conv3(x)))

        x1 = x.reshape(x.size(0), -1)
        x1 = self.dropout(x1)
        x1 = functional.relu(self.fc1(x1))
        if self.batch_size > 1:
            x1 = self.batch_norm(x1)
        x1 = functional.relu(self.fc2(x1))
        y = functional.softmax(self.fc3(x1))

        return y

    def step(self, batch):
        x, label = batch
        output = self.forward(x)

        loss = functional.cross_entropy(output, torch.max(label, 1)[1])
        acc = (torch.max(output, 1)[1] == torch.max(label, 1)[1]).float().sum() / self.batch_size

        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adamax(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    net = NPInverseNet(2)
    device = get_device_torch()
    net = net.to(device)
    xs = torch.rand((1, 3, 64, 295)).to(device)
    y = net(xs)
    print(y)
