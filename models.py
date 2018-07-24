import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init

import pdb

class A3C_LSTM(torch.nn.Module):
    def __init__(self, target_num=None):
        super(A3C_LSTM, self).__init__()
        ## convolution network
        self.conv1 = nn.Conv2d(4, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        ## target attention network
        self.target_att_linear = nn.Linear(target_num, 64)

        ## a3c-lstm network
        self.linear = nn.Linear(64 * 7 * 10, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, 10)

    def forward(self, x, target, hx, cx):
        ## calculate images features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        img_feat = F.relu(self.conv4(x))
        ## img_feat size : (1, 64, 7, 10)

        ## calculate gated attention
        target_att = self.target_att_linear(target)
        target_att = target_att.expand(1, 7, 10, 64)
        target_att = target_att.permute(0, 3, 1, 2)
        target_att = F.sigmoid(target_att)

        ## apply gated attention
        x = img_feat * target_att

        ## flatten
        x = x.view(-1, 64 * 7 * 10)

        ## calculate action probability and value function
        x = F.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))

        return self.critic_linear(hx), self.actor_linear(hx), hx, cx