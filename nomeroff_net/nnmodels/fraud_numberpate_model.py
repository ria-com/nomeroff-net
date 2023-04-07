"""
Numberplate Classification Model
python3 -m nomeroff_net.nnmodels.fraud_numberpate_options -f nomeroff_net/nnmodels/fraud_numberpate_options.py
"""
import torch
import torch.nn as nn
from torch.nn import functional
from .numberplate_classification_model import ClassificationNet
from torchvision.models import resnet18
from nomeroff_net.tools.mcm import get_device_torch


class FraudNPNet(ClassificationNet):
    """

    """
    def __init__(self,
                 batch_size: int = 1,
                 learning_rate: float = 0.005):
        super(FraudNPNet, self).__init__()  # activation='relu'
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()

        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)

    def training_step(self, batch, _):
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

    def validation_step(self, batch, _):
        loss, acc = self.step(batch)
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

    def test_step(self, batch, _):
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

    def forward(self, x):
        y = self.resnet(x)
        y = y.reshape(y.size(0), -1)
        y = self.relu(self.fc1(y))
        if self.batch_size > 1:
            y = self.batchnorm1(y)
        y = self.relu(self.fc2(y))
        if self.batch_size > 1:
            y = self.batchnorm2(y)
        y = self.dropout(y)
        y = functional.sigmoid(self.fc3(y))

        return y

    def step(self, batch):
        xs, true_ys = batch

        ys = self.forward(xs)

        loss = self.criterion(ys, true_ys.unsqueeze(1))

        correct_results_sum = (torch.round(ys) == true_ys.unsqueeze(1)).sum().float()
        acc = correct_results_sum / true_ys.shape[0]
  
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.ASGD(self.parameters(),
                                     lr=self.learning_rate,
                                     lambd=0.0001,
                                     alpha=0.75,
                                     t0=1000000.0,
                                     weight_decay=0)
        return optimizer


if __name__ == "__main__":
    net = FraudNPNet()
    device = get_device_torch()
    net = net.to(device)
    xs = torch.rand((1, 3, 256, 256)).to(device)
    predicted = net(xs)
    print(predicted)
