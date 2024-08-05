"""
Numberplate Orientation Model
python3 -m nomeroff_net.nnmodels.numberplate_orientation_model -f nomeroff_net/nnmodels/numberplate_orientation_model.py
"""
import torch
from torch.nn import functional
import torch.nn as nn
from torchvision.models import vit_l_16
from .numberplate_classification_model import ClassificationNet
from nomeroff_net.tools.mcm import get_device_torch


class NPOrientationNet(ClassificationNet):
    def __init__(self,
                 height: int = 224,
                 width: int = 224,
                 output_size: int = 3,
                 batch_size: int = 1,
                 learning_rate: float = 0.001,
                 backbone=None):
        super(NPOrientationNet, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.learning_rate = learning_rate

        if backbone is None:
            backbone = vit_l_16(pretrained=True)
        self.model = backbone

        # Замінюємо класифікаційний шар
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, output_size)

    def forward(self, x):
        y = self.model(x)
        y = functional.softmax(y)
        return y

    def step(self, batch):
        x, y = batch

        # Прогнозування
        outputs = self.forward(x)

        # Обчислення втрат
        loss = functional.cross_entropy(outputs, y)

        # Обчислення точності
        pred = outputs.argmax(dim=1)
        correct = (pred == y).float()
        acc = correct.sum() / self.batch_size

        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'train_loss': loss,
            'acc': acc,
        }
        return {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def validation_step(self, batch, batch_idx):
        loss, acc, = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        tqdm_dict = {
            'val_loss': loss,
            'acc': acc,
        }
        return {
            'val_loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'test_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'test_loss': loss,
            'acc': acc,
        }
        return {
            'test_loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }


if __name__ == "__main__":
    net = NPOrientationNet()
    device = get_device_torch()
    net = net.to(device)
    xs = torch.rand((1, 3, 224, 224)).to(device)
    y = net(xs)
