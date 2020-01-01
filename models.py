import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init

import pdb

class A3C_LSTM_GA(torch.nn.Module):
    def __init__(self):
        super(A3C_LSTM_GA, self).__init__()
        ## convolution network
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(64, track_running_stats=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(128, track_running_stats=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.batchnorm4 = nn.BatchNorm2d(128, track_running_stats=False)

        self.fc = nn.Linear(1024, 256)

        # Instruction Processing, MLP
        self.embedding = nn.Embedding(5, 25)
        # self.embedding = nn.Linear(5, 25)
        self.target_att_linear = nn.Linear(25, 256)

        ## a3c-lstm network
        self.lstm = nn.LSTMCell(512, 256)

        self.mlp = nn.Linear(512, 256)  #192

        self.mlp_policy = nn.Linear(256, 128)
        self.mlp_policy2 = nn.Linear(128, 64)
        self.actor_linear = nn.Linear(64, 10)

        self.mlp_value = nn.Linear(256, 64) #64
        self.mlp_value2 = nn.Linear(64, 32)
        self.critic_linear = nn.Linear(32, 1)

    def forward(self, state, instruction_idx, hx, cx, debugging=False):
        x = state

        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))

        if debugging is True:
            pdb.set_trace()

        x = x.view(x.size(0), -1)
        img_feat = F.relu(self.fc(x))

        # Get the instruction representation
        word_embedding = self.embedding(instruction_idx)
        word_embedding = word_embedding.view(word_embedding.size(0), -1)

        ## calculate gated attention
        word_embedding = self.target_att_linear(word_embedding)
        gated_att = torch.sigmoid(word_embedding)

        ## apply gated attention
        gated_fusion = torch.mul(img_feat, gated_att)
        lstm_input = torch.cat([gated_fusion, gated_att], 1)

        ## calculate action probability and value function
        _hx, _cx = self.lstm(lstm_input, (hx, cx))

        mlp_input = torch.cat([gated_fusion, _hx], 1)
        mlp_out = self.mlp(mlp_input)

        policy = self.mlp_policy(mlp_out)
        policy = self.mlp_policy2(policy)

        value = self.mlp_value(mlp_out)
        value = self.mlp_value2(value)

        return self.critic_linear(value), self.actor_linear(policy), _hx, _cx