import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        # シンプルなResidual Block: conv-bn-relu-conv-bn + skip
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ZeroMatrixNNet(nn.Module):
    def __init__(self, game, args):
        super(ZeroMatrixNNet, self).__init__()
        self.game = game
        self.args = args

        self.board_x, self.board_y = game.getBoardSize()  # (N+1, N)
        self.action_size = game.getActionSize()
        self.num_channels = args.num_channels  # 例えば args.num_channels = 128
        self.dropout = args.dropout

        # 入力を1チャネル -> num_channelsチャネルへ拡大
        self.conv_input = nn.Conv2d(1, self.num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(self.num_channels)

        # 複数のResidualBlockをスタック
        # これでネットワークを深くする
        self.res_blocks = nn.ModuleList([ResidualBlock(self.num_channels) for _ in range(5)])
        # 5個のResidualBlockで深みを増す

        # これまでの処理で特定の縮小は行っていないため、(N+1, N)のまま
        # 必要ならvalid paddingなどで空間サイズを縮小する
        # 今回は縮小しない簡易例として (N+1, N)を維持したままにする

        # Conv層を追加して空間縮小する場合は以下のような層を足す（例）
        # conv_reducer = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1, padding=0, bias=False)
        # bn_reducer = nn.BatchNorm2d(self.num_channels)
        # 上記で幅高さを (N+1-2, N-2)などに縮めても良い

        # 最終的な特徴マップのサイズを計算する必要がある
        # 簡易的には (N+1, N)のままと仮定し、そのままFlattenする
        final_x = self.board_x
        final_y = self.board_y
        fc_input_size = self.num_channels * final_x * final_y

        # 全結合層をより大きく/深く
        self.fc1 = nn.Linear(fc_input_size, 2048, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 1024, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 512, bias=False)
        self.bn_fc3 = nn.BatchNorm1d(512)

        self.fc_pi = nn.Linear(512, self.action_size)
        self.fc_v = nn.Linear(512, 1)

        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, s):
        # s: (batch_size, board_x, board_y)
        s = s.unsqueeze(1).float()  # (batch, 1, board_x, board_y)

        # 入力Conv
        s = F.relu(self.bn_input(self.conv_input(s)))

        # Residual blocks
        for block in self.res_blocks:
            s = block(s)

        # Flatten
        s = s.view(s.size(0), -1)

        # Fully connected layers
        s = self.fc1(s)
        s = self.bn_fc1(s)
        s = F.relu(s)
        s = self.dropout_layer(s)

        s = self.fc2(s)
        s = self.bn_fc2(s)
        s = F.relu(s)
        s = self.dropout_layer(s)

        s = self.fc3(s)
        s = self.bn_fc3(s)
        s = F.relu(s)
        s = self.dropout_layer(s)

        pi = self.fc_pi(s)  # policy head
        v = self.fc_v(s)    # value head

        # Return log-softmax for pi and tanh for v
        return F.log_softmax(pi, dim=1), torch.tanh(v)
