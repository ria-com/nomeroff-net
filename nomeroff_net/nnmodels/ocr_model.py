"""
Numberplate OCR Model
python3 -m nomeroff_net.nnmodels.ocr_model -f nomeroff_net/nnmodels/ocr_model.py
"""
import torch
import torch.nn as nn
from typing import List, Any
import pytorch_lightning as pl
from torchvision.models import resnet18

from nomeroff_net.tools.ocr_tools import plot_loss, print_prediction
from nomeroff_net.tools.mcm import get_device_torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class BlockRNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, bidirectional):
        super(BlockRNN, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.bidirectional = bidirectional
        # layers
        self.gru = nn.LSTM(in_size, hidden_size, bidirectional=bidirectional, batch_first=True)

    def forward(self, batch, add_output=False):
        """
        in array:
            batch - [seq_len , batch_size, in_size]
        out array:
            out - [seq_len , batch_size, out_size]
        """
        outputs, hidden = self.gru(batch)
        out_size = int(outputs.size(2) / 2)
        if add_output:
            outputs = outputs[:, :, :out_size] + outputs[:, :, out_size:]
        return outputs


class NPOcrNet(pl.LightningModule):
    def __init__(self,
                 letters: List = None,
                 letters_max: int = 0,
                 max_plate_length: int = 8,
                 learning_rate: float = 0.02,
                 hidden_size: int = 32,
                 bidirectional: bool = True,
                 label_converter: Any = None,
                 val_dataset: Any = None,
                 weight_decay: float = 1e-5,
                 momentum: float = 0.9,
                 clip_norm: int = 5):
        super().__init__()
        self.save_hyperparameters()

        self.letters = letters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.momentum = momentum
        self.max_plate_length = max_plate_length

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.label_converter = label_converter
        
        # convolutions 
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)

        # RNN + Linear
        self.linear1 = nn.Linear(1024, 512)
        self.gru1 = BlockRNN(512, hidden_size, hidden_size,
                             bidirectional=bidirectional)
        self.gru2 = BlockRNN(hidden_size, hidden_size, letters_max,
                             bidirectional=bidirectional)
        self.linear2 = nn.Linear(hidden_size * 2, letters_max)

        self.automatic_optimization = True
        self.criterion = None
        self.val_dataset = val_dataset
        self.train_losses = []
        self.val_losses = []

    def forward(self, batch: torch.float64):
        """
        ------:size sequence:------
        torch.Size([batch_size, 3, 64, 128]) -- IN:
        torch.Size([batch_size, 16, 16, 32]) -- CNN blocks ended
        torch.Size([batch_size, 32, 256]) -- permuted
        torch.Size([batch_size, 32, 32]) -- Linear #1
        torch.Size([batch_size, 32, 512]) -- IN GRU
        torch.Size([batch_size, 512, 512]) -- OUT GRU
        torch.Size([batch_size, 32, vocab_size]) -- Linear #2
        torch.Size([32, batch_size, vocab_size]) -- :OUT
        """
        batch_size = batch.size(0)
        
        # convolutions
        batch = self.resnet(batch)

        # make sequences of image features
        batch = batch.permute(0, 3, 1, 2)
        n_channels = batch.size(1)
        batch = batch.reshape(batch_size, n_channels, -1)

        batch = self.linear1(batch)

        # rnn layers
        batch = self.gru1(batch, add_output=True)
        batch = self.gru2(batch)
        # output
        batch = self.linear2(batch)
        batch = batch.permute(1, 0, 2)
        return batch

    def init_loss(self):
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

    def calculate_loss(self, logits, texts):
        if self.criterion is None:
            self.init_loss()

        # get infomation from prediction
        device = logits.device
        input_len, batch_size, vocab_size = logits.size()
        # encode inputs
        logits = logits.log_softmax(2)
        encoded_texts, text_lens = self.label_converter.encode(texts)
        logits_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)
        # calculate ctc
        loss = self.criterion(
            logits,
            encoded_texts,
            logits_lens.to(device),
            text_lens)
        return loss

    def step(self, batch):

        x, texts = batch

        output = self.forward(x)

        loss = self.calculate_loss(output, texts)
        return loss

    def on_save_checkpoint(self, _):
        if self.current_epoch and self.val_dataset:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print_prediction(self, self.val_dataset, device, self.label_converter)
            plot_loss(self.current_epoch, self.train_losses, self.val_losses)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            nesterov=True,
            weight_decay=self.weight_decay,
            momentum=self.momentum)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'train_loss': loss,
        }
        return {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'val_loss': loss,
        }
        return {
            'val_loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tqdm_dict = {
            'test_loss': loss,
        }
        return {
            'test_loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }


if __name__ == "__main__":
    net = NPOcrNet(["4", "2"], letters_max=2)
    device = get_device_torch()
    net = net.to(device)
    xs = torch.rand((1, 3, 50, 200)).to(device)
    y = net(xs)
    print(y)


