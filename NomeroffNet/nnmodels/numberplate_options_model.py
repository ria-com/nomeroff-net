import torch
import torch.nn as nn
from torch.nn import functional
from .classification_model import ClassificationNet


class NPOptionsNet(ClassificationNet):
    def __init__(self,
                 region_output_size: int,
                 count_line_output_size: int,
                 img_h: int = 64,
                 img_w: int = 295,
                 batch_size: int = 1,
                 learning_rate: float = 0.005):
        super(NPOptionsNet, self).__init__()  # activation='relu'
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

        self.dropout_reg = nn.Dropout(0.2)
        self.fc1_reg = nn.Linear(128 * img_w * img_h, 512)
        self.fc2_reg = nn.Linear(512, 256)
        self.batch_norm_reg = nn.BatchNorm1d(512)
        self.fc3_reg = nn.Linear(256, region_output_size)
        
        self.dropout_line = nn.Dropout(0.2)
        self.fc1_line = nn.Linear(128 * img_w * img_h, 512)
        self.fc2_line = nn.Linear(512, 256)
        self.batch_norm_line = nn.BatchNorm1d(512)
        self.fc3_line = nn.Linear(256, count_line_output_size)

    def training_step(self, batch, batch_idx):
        loss, acc, acc_reg, acc_line = self.step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'test_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc_reg', acc_reg, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'acc_line', acc_line, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'test_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc_reg', acc_reg, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'acc_line', acc_line, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'test_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc_reg', acc_reg, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'acc_line', acc_line, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        x = self.pool(functional.relu(self.inp_conv(x)))
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = self.pool(functional.relu(self.conv3(x)))
        
        x1 = x.reshape(x.size(0), -1)
        x1 = self.dropout_reg(x1)
        x1 = functional.relu(self.fc1_reg(x1))
        if self.batch_size > 1:
            x1 = self.batch_norm_reg(x1)
        x1 = functional.relu(self.fc2_reg(x1))
        x1 = functional.softmax(self.fc3_reg(x1))
        
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
        loss = (loss_reg + loss_line) / 2

        acc_reg = (torch.max(outputs[0], 1)[1] == torch.max(label_reg, 1)[1]).float().sum() / self.batch_size
        acc_line = (torch.max(outputs[1], 1)[1] == torch.max(label_cnt, 1)[1]).float().sum() / self.batch_size
        acc = (acc_reg + acc_line) / 2

        return loss, acc, acc_reg, acc_line

    def configure_optimizers(self):
        # optimizer = torch.optim.Adamax(self.parameters(),
        #                                lr=self.learning_rate,
        #                                betas=(0.9, 0.999),
        #                                eps=1e-07)
        # optimizer = torch.optim.SGD(self.parameters(),
        #                                lr=self.learning_rate,
        #                                momentum=0.9)
        optimizer = torch.optim.ASGD(self.parameters(), lr=self.learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        return optimizer
