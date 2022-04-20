"""
Numberplate Classification Model
python3 -m nomeroff_net.nnmodels.numberplate_options_model -f nomeroff_net/nnmodels/numberplate_options_model.py
"""
import torch
import torch.nn as nn
from torch.nn import functional
from .numberplate_classification_model import ClassificationNet
from torchvision.models import resnet18
from nomeroff_net.tools.errors import NPOptionsNetError
import contextlib
from nomeroff_net.tools.mcm import get_device_torch


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


class NPOptionsNet(ClassificationNet):
    def __init__(self,
                 region_output_size: int,
                 count_line_output_size: int,
                 batch_size: int = 1,
                 learning_rate: float = 0.005,
                 train_regions=True,
                 train_count_lines=True):
        super(NPOptionsNet, self).__init__() 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.train_regions = train_regions
        self.train_count_lines = train_count_lines

        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)

        self.dropout_reg = nn.Dropout(0.2)
        self.fc1_reg = nn.Linear(256 * 4 * 19, 512)
        self.fc2_reg = nn.Linear(512, 256)
        self.batch_norm_reg = nn.BatchNorm1d(512)
        self.fc3_reg = nn.Linear(256, region_output_size)

        self.dropout_line = nn.Dropout(0.2)
        self.fc1_line = nn.Linear(256 * 4 * 19, 512)
        self.fc2_line = nn.Linear(512, 256)
        self.batch_norm_line = nn.BatchNorm1d(512)
        self.fc3_line = nn.Linear(256, count_line_output_size)

    def training_step(self, batch, batch_idx):
        loss, acc, acc_reg, acc_line = self.step(batch)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_reg', acc_reg, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_acc_line', acc_line, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'train_loss': loss,
            'acc': acc,
            'acc_reg': acc_reg,
            'acc_line': acc_line,
        }
        return {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def validation_step(self, batch, batch_idx):
        loss, acc, acc_reg, acc_line = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_reg', acc_reg, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'val_acc_line', acc_line, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'val_loss': loss,
            'acc': acc,
            'acc_reg': acc_reg,
            'acc_line': acc_line,
        }
        return {
            'val_loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def test_step(self, batch, batch_idx):
        loss, acc, acc_reg, acc_line = self.step(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'test_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_reg', acc_reg, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'test_acc_line', acc_line, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'test_loss': loss,
            'acc': acc,
            'acc_reg': acc_reg,
            'acc_line': acc_line,
        }
        return {
            'test_loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def forward(self, x):
        x = self.resnet(x)
        
        with dummy_context_mgr() if self.train_regions else torch.no_grad():
            x1 = x.reshape(x.size(0), -1)
            x1 = self.dropout_reg(x1)
            x1 = functional.relu(self.fc1_reg(x1))
            if self.batch_size > 1:
                x1 = self.batch_norm_reg(x1)
            x1 = functional.relu(self.fc2_reg(x1))
            x1 = functional.softmax(self.fc3_reg(x1))
        
        with dummy_context_mgr() if self.train_count_lines else torch.no_grad():
            x2 = x.reshape(x.size(0), -1)
            x2 = self.dropout_line(x2)
            x2 = functional.relu(self.fc1_line(x2))
            if self.batch_size > 1:
                x2 = self.batch_norm_line(x2)
            x2 = functional.relu(self.fc2_line(x2))
            x2 = functional.softmax(self.fc3_line(x2))

        return x1, x2

    def step(self, batch):
        x, ys = batch

        outputs = self.forward(x)
        label_reg = ys[0]
        label_cnt = ys[1]
        
        loss_reg = functional.cross_entropy(outputs[0], torch.max(label_reg, 1)[1])
        loss_line = functional.cross_entropy(outputs[1], torch.max(label_cnt, 1)[1])
        if self.train_regions and self.train_count_lines:
            loss = (loss_reg + loss_line) / 2
        elif self.train_regions:
            loss = loss_reg
        elif self.train_count_lines:
            loss = loss_line
        else:
            raise NPOptionsNetError("train_regions and train_count_lines can not to be False both!")

        acc_reg = (torch.max(outputs[0], 1)[1] == torch.max(label_reg, 1)[1]).float().sum() / self.batch_size
        acc_line = (torch.max(outputs[1], 1)[1] == torch.max(label_cnt, 1)[1]).float().sum() / self.batch_size
        acc = (acc_reg + acc_line) / 2

        return loss, acc, acc_reg, acc_line

    def configure_optimizers(self):
        optimizer = torch.optim.ASGD(self.parameters(),
                                     lr=self.learning_rate,
                                     lambd=0.0001,
                                     alpha=0.75,
                                     t0=1000000.0,
                                     weight_decay=0)
        return optimizer


if __name__ == "__main__":
    np_options_net = NPOptionsNet(13, 3)
    device = get_device_torch()
    net = np_options_net.to(device)
    xs = torch.rand((1, 3, 64, 295)).to(device)
    y = np_options_net(xs)
    print(y)
