import torch.nn as nn
import torch.nn.functional as F

class NPOptionsNet(nn.Module):
    def __init__(self):
        super(NPOptionsNet, self).__init__() #  activation='relu'
        self.inp_conv = nn.Conv2d(3, 32, (3, 3),
                                  stride=(1,1),
                                  padding=0
                                 )
        self.conv1 = nn.Conv2d(32, 64, (3, 3),
                               stride=(1,1),
                               padding=0
                              )
        self.conv2 = nn.Conv2d(64, 128, (3, 3),
                               stride=(1,1),
                               padding=0
                              )
        self.conv3 = nn.Conv2d(128, 128, (3, 3),
                               stride=(1,1),
                               padding=0
                              )
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout_reg = nn.Dropout(0.2)
        self.fc1_reg = nn.Linear(128 * 2 * 10, 512)
        self.fc2_reg = nn.Linear(512, 256)
        self.batch_norm_reg = nn.BatchNorm1d(512)
        self.fc3_reg = nn.Linear(256, 14)
        
        self.dropout_line = nn.Dropout(0.2)
        self.fc1_line = nn.Linear(128 * 2 * 10, 512)
        self.fc2_line = nn.Linear(512, 256)
        self.batch_norm_line = nn.BatchNorm1d(512)
        self.fc3_line = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.inp_conv(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x1 = x.reshape(x.size(0), -1)
        x1 = self.dropout_reg(x1)
        x1 = F.relu(self.fc1_reg(x1))
        x1 = self.batch_norm_reg(x1)
        x1 = F.relu(self.fc2_reg(x1))
        x1 = F.softmax(self.fc3_reg(x1))
        
        x2 = x.reshape(x.size(0), -1)
        x2 = self.dropout_line(x2)
        x2 = F.relu(self.fc1_line(x2))
        x2 = self.batch_norm_line(x2)
        x2 = F.relu(self.fc2_line(x2))
        x2 = F.softmax(self.fc3_line(x2))

        return x1, x2
