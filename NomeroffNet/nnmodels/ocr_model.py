import torch
import torch.nn as nn
from torch.nn import functional
import numpy as np
import pytorch_lightning as pl


def create_cnn():
    conv1 = nn.Conv2d(1,
                      16,
                      kernel_size=(3, 3),
                      stride=2,
                      padding=1)
    norm1 = nn.InstanceNorm2d(16)
    leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

    conv2 = nn.Conv2d(16,
                      16,
                      kernel_size=(3, 3),
                      stride=2,
                      padding=1)
    norm2 = nn.InstanceNorm2d(16)
    leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

    cnn = nn.Sequential()
    cnn.add_module('conv1', conv1)
    cnn.add_module('norm1', norm1)
    cnn.add_module('leaky_relu1', leaky_relu1)
    cnn.add_module('conv2', conv2)
    cnn.add_module('norm2', norm2)
    cnn.add_module('leaky_relu2', leaky_relu2)
    return cnn


class NPOcrNet(pl.LightningModule):
    def __init__(self,
                 letters_max: int,
                 img_h: int = 64,
                 img_w: int = 128):
        super().__init__()

        self.conv_to_rnn_dims = (img_w // (2 * 2),
                                 (img_h // (2 * 2)) * 16)

        self.cnn = create_cnn()
        self.gru_input_size = (img_h // (2 * 2)) * 16
        self.gru = nn.GRU(input_size=self.gru_input_size,
                          hidden_size=512,
                          batch_first=True,
                          num_layers=2,
                          bidirectional=True)

        self.fc = nn.Linear(512*2, letters_max)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.cnn(x)

        # x = x.reshape(self.conv_to_rnn_dims)
        x = x.reshape(batch_size, -1, self.gru_input_size)

        x, _ = self.gru(x)
        x = torch.stack([functional.log_softmax(self.fc(x[i]), dim=-1) for i in range(x.shape[0])])

        return x

    def step(self, batch):

        x, label = batch
        x, input_lengths, target_lengths = x

        output = self.forward(x)

        loss = functional.ctc_loss(output,
                                   label,
                                   input_lengths,
                                   target_lengths,
                                   zero_infinity=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(),
                                       lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(f'Batch {batch_idx} train_loss', loss)
        return {
            'loss': loss,
        }

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss)
        return {
            'val_loss': loss,
        }

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('test_loss', loss)
        return {
            'test_loss': loss,
        }

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0  # replace all nan/inf in gradients to zero
