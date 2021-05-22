import torch.nn as nn
from torch.nn import functional


class NPOptionsNet(nn.Module):
    def __init__(self, region_output_size: int, count_line_output_size: int, img_h: int = 64, img_w: int = 295):
        super(NPOptionsNet, self).__init__()  # activation='relu'
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

    def forward(self, x):
        x = self.pool(functional.relu(self.inp_conv(x)))
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = self.pool(functional.relu(self.conv3(x)))
        
        x1 = x.reshape(x.size(0), -1)
        x1 = self.dropout_reg(x1)
        x1 = functional.relu(self.fc1_reg(x1))
        x1 = self.batch_norm_reg(x1)
        x1 = functional.relu(self.fc2_reg(x1))
        x1 = functional.softmax(self.fc3_reg(x1))
        
        x2 = x.reshape(x.size(0), -1)
        x2 = self.dropout_line(x2)
        x2 = functional.relu(self.fc1_line(x2))
        x2 = self.batch_norm_line(x2)
        x2 = functional.relu(self.fc2_line(x2))
        x2 = functional.softmax(self.fc3_line(x2))

        return x1, x2
