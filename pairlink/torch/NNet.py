import os
import sys
import time
import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .PairLinkNNet import PairLinkNNet as plnnet

"""
PairLink用のNNetラッパークラス。
OthelloのNNet.py (NNetWrapper)を参考に、PairLinkの特徴量形式に合わせる。

examples: (board, pi, v)
boardはPairLinkGameのboard状態(np.array)
get_featuresで(L*C + P_max*(1+2*C))次元ベクトルへ変換してからNN入力へ。

train: 
  複数エポック、バッチごとにサンプルを取り出し、NN学習を行う。
predict:
  boardからfeaturesを生成してNNで予測(pi,v)を返す。
"""

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,  # MLPの隠れユニット数として利用
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.nnet = plnnet(game, args)
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of (board, pi, v)
          board: state array (from game.getInitBoard or nextState)
          pi: policy target
          v: value target
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, target_pis, target_vs = list(zip(*[examples[i] for i in sample_ids]))

                # boardsからfeatures生成
                features = []
                for b_state in boards:
                    f = self.game.get_features(b_state)  # (L*C + P_max*(1+2*C))次元ベクトル
                    features.append(f)
                features = np.array(features, dtype=np.float32)

                target_pis = np.array(target_pis, dtype=np.float32)
                target_vs = np.array(target_vs, dtype=np.float32)

                features = torch.FloatTensor(features)
                target_pis = torch.FloatTensor(target_pis)
                target_vs = torch.FloatTensor(target_vs)

                if args.cuda:
                    features = features.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                # 推論
                out_pi, out_v = self.nnet(features)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), features.size(0))
                v_losses.update(l_v.item(), features.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # 最適化
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np.array board state
        """
        # boardからfeaturesを生成
        features = self.game.get_features(board)
        features = torch.FloatTensor(features.astype(np.float32))
        if args.cuda:
            features = features.contiguous().cuda()

        features = features.view(1, -1)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(features)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        # 交差エントロピー（policyターゲットはone-hot）
        return -torch.sum(targets * outputs) / targets.size(0)

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size(0)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
