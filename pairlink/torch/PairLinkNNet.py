# PairLinkNNet.py
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PairLinkNNet(nn.Module):
    def __init__(self, game, args):
        super(PairLinkNNet, self).__init__()
        # ゲームパラメータ
        # getFeatureSize()はペアごとの特徴量が変更されても常に最新の次元数を返す
        # 今回、各ペアあたり(2 + 2*C)次元に拡張され、achieved_any_flagが追加されたが、
        # ここでは特別な変更は不要。
        self.input_size = game.getFeatureSize()
        self.action_size = game.getActionSize()
        self.args = args

        hidden_size = args.num_channels
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc_policy = nn.Linear(hidden_size, self.action_size)
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, input_size)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        pi = self.fc_policy(x)
        v = self.fc_value(x)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
