import numpy as np
from zero_matrix.ZeroMatrixGame import ZeroMatrixGame
from zero_matrix.torch.NNet import NNetWrapper as NNet
from Coach import Coach
from utils import dotdict
import torch

if __name__ == "__main__":
    # Game parameters
    N = 6
    coupling_map = [(i, i+1) for i in range(N-1)]

    game = ZeroMatrixGame(N, coupling_map, initial_mat=None)

    # Training parameters
    args = dotdict({
        'numIters': 10,            # number of training iterations
        'numEps': 10,              # number of self-play games per iteration
        'maxlenOfQueue': 200000,   # memory size
        'numMCTSSims': 25,         # number of MCTS simulations per move
        'cpuct': 1.0,
        'checkpoint': './checkpoint/',
        'load_model': False,
        'load_folder_file': ('./checkpoint','checkpoint.pth.tar'),
        'lr': 0.001,
        'dropout': 0.3,
        'num_channels': 64,
        'epochs': 10,
        'batch_size': 64,
        'cuda': True if torch.cuda.is_available() else False
    })

    nnet = NNet(game)
    c = Coach(game, nnet, args)
    c.learn()
