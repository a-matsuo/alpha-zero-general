import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ZeroMatrixNNet import ZeroMatrixNNet

class NNetWrapper:
    def __init__(self, game):
        self.game = game
        self.args = {
            'lr': 0.001,
            'dropout': 0.3,
            'num_channels': 64,
            'epochs': 10,
            'batch_size': 64,
            'cuda': torch.cuda.is_available()
        }

        self.nnet = ZeroMatrixNNet(game, self.args)
        self.device = torch.device("cuda" if self.args['cuda'] else "cpu")
        self.nnet.to(self.device)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.args['lr'])

    def train(self, examples):
        """
        examples: list of (board, pi, v)
        """
        for epoch in range(self.args['epochs']):
            print(f'Epoch {epoch+1}/{self.args["epochs"]}')
            self.nnet.train()

            np.random.shuffle(examples)
            batch_count = int(len(examples) / self.args['batch_size'])

            for i in range(batch_count):
                sample = examples[i*self.args['batch_size'] : (i+1)*self.args['batch_size']]
                boards, pis, vs = list(zip(*sample))

                boards = torch.FloatTensor(np.array(boards)).to(self.device)
                pis = torch.FloatTensor(np.array(pis)).to(self.device)
                vs = torch.FloatTensor(np.array(vs)).to(self.device)

                self.optimizer.zero_grad()
                out_pi, out_v = self.nnet(boards)
                loss_pi = -torch.sum(pis * out_pi) / pis.size(0)
                loss_v = torch.sum((vs - out_v.squeeze())**2)/vs.size(0)
                loss = loss_pi + loss_v

                loss.backward()
                self.optimizer.step()

            print(f"Loss_pi: {loss_pi.item():.4f}, Loss_v: {loss_v.item():.4f}")

    def predict(self, board):
        """
        board: numpy array of shape (N+1, N)
        """
        board = torch.FloatTensor(board).to(self.device)
        board = board.unsqueeze(0)  # add batch dimension
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        pi = pi.exp().cpu().numpy()[0]  # convert log_softmax to probabilities
        v = v.item()
        return pi, v

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print("No model in path {}".format(filepath))
            return
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])