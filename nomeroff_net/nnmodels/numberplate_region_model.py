"""
Numberplate Classification Model
python3 -m nomeroff_net.nnmodels.numberplate_region_model -f nomeroff_net/nnmodels/numberplate_region_model.py
"""
import torch
import torch.nn as nn
from torch.nn import functional
from .numberplate_classification_model import ClassificationNet
from torchvision.models import efficientnet_v2_s
from nomeroff_net.tools.mcm import get_device_torch


class NPRegionNet(ClassificationNet):
    def __init__(self, region_output_size: int, batch_size: int = 1, learning_rate: float = 0.001, backbone=None):
        super(NPRegionNet, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if backbone is None:
            backbone = efficientnet_v2_s
        self.model = backbone()

        if 'efficientnet' in str(backbone):
            in_features = self.model.classifier[1].in_features
        else:
            raise NotImplementedError(backbone)

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=in_features, out_features=region_output_size, bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        return functional.softmax(x, dim=1)

    def step(self, batch):
        x, ys = batch
        y = ys[0]
        y_hat = self(x)
        loss = functional.cross_entropy(y_hat, torch.max(y, 1)[1])
        acc = (torch.max(y_hat, 1)[1] == torch.max(y, 1)[1]).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)


if __name__ == "__main__":
    device = get_device_torch()

    np_region_net = NPRegionNet(13).to(device)

    xs = torch.rand((1, 3, 100, 300)).to(device)
    y = np_region_net(xs)
    print(y)
