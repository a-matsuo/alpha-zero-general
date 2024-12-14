import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroMatrixNNet(nn.Module):
    def __init__(self, game, args):
        super(ZeroMatrixNNet, self).__init__()
        self.game = game
        self.args = args

        self.board_x, self.board_y = game.getBoardSize()  # (N+1, N)
        self.action_size = game.getActionSize()
        self.num_channels = args.num_channels
        self.dropout = args.dropout

        # Convolutional layers
        # First two with 'same' padding (we can emulate same padding by using padding=1 for kernel_size=3)
        self.conv1 = nn.Conv2d(1, self.num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)

        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.num_channels)

        # Next two with 'valid' padding (no padding, reduces dimension by 2 for each conv)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.num_channels)

        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(self.num_channels)

        # After conv4, the dimension is reduced by 4 in both directions:
        # Original: (N+1, N)
        # After conv3: ((N+1)-2, N-2)
        # After conv4: ((N+1)-4, (N)-4)
        # Let’s compute final conv output size:
        # final_x = (N+1)-4
        # final_y = N-4
        # We'll flatten this output.

        # Fully connected layers
        # Assuming the final spatial size is (final_x, final_y)
        # final_x = self.board_x - 4
        # final_y = self.board_y - 4
        # So the number of features after conv layers = num_channels * final_x * final_y
        # Make sure that board_x > 4 and board_y > 4 for this architecture to work as intended.
        final_x = self.board_x - 4
        final_y = self.board_y - 4
        fc_input_size = self.num_channels * final_x * final_y

        self.fc1 = nn.Linear(fc_input_size, 1024, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(512)

        self.fc_pi = nn.Linear(512, self.action_size)
        self.fc_v = nn.Linear(512, 1)

        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, s):
        # s: (batch_size, board_x, board_y)
        # Add channel dimension: (batch_size, 1, board_x, board_y)
        s = s.unsqueeze(1).float()

        # Convolutional layers
        s = F.relu(self.bn1(self.conv1(s)))  # shape: (batch_size, num_channels, board_x, board_y)
        s = F.relu(self.bn2(self.conv2(s)))  # same shape as above

        # valid conv reduces dimension by 2 (since kernel=3 and padding=0)
        s = F.relu(self.bn3(self.conv3(s)))  # (batch_size, num_channels, board_x-2, board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))  # (batch_size, num_channels, board_x-4, board_y-4)

        s = s.view(s.size(0), -1)  # flatten

        # Fully connected layers
        s = self.fc1(s)
        s = self.bn_fc1(s)
        s = F.relu(s)
        s = self.dropout_layer(s)

        s = self.fc2(s)
        s = self.bn_fc2(s)
        s = F.relu(s)
        s = self.dropout_layer(s)

        pi = self.fc_pi(s)  # policy head
        v = self.fc_v(s)    # value head

        # Return log-softmax for pi and tanh for v
        return F.log_softmax(pi, dim=1), torch.tanh(v)
