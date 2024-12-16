import logging

import coloredlogs
import torch
from Coach import Coach
from constants import *
from utils import dotdict
from zero_matrix.torch.NNet import NNetWrapper as NNet
from zero_matrix.ZeroMatrixGame import ZeroMatrixGame as Game

log = logging.getLogger(__name__)



coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.

if __name__ == "__main__":
    # Game parameters
    log.info("Loading %s...", Game.__name__)
    N = num_qubits
    coupling_map = [(i, i + 1) for i in range(N - 1)]
    game = Game(N, coupling_map, initial_mat_type=initial_mat_type)

    # Training parameters
    args = dotdict(
        {
            "numIters": 1000,  # number of training iterations
            "numEps": 100,  # number of self-play games per iteration
            "maxlenOfQueue": 200000,  # memory size
            "numMCTSSims": 400,  # number of MCTS simulations per move
            "cpuct": 1.0,
            "checkpoint": "./checkpoint/",
            "load_model": False,
            "load_folder_file": ("./checkpoint", "checkpoint.pth.tar"),
            "lr": 0.001,
            "dropout": 0.3,
            "num_channels": 64,
            "epochs": 10,
            "batch_size": 64,
            "cuda": True if torch.cuda.is_available() else False,
            "num_qubits": N,
            "tempThreshold": tempThreshold,
            "curriculum_learning": True,
        }
    )
    log.info("Loading %s...", NNet.__name__)
    nnet = NNet(game)
    log.info("Loading the Coach...")
    c = Coach(game, nnet, args)
    log.info("Starting the learning process 🎉")
    # save pi_losses and v_losses
    with open("./checkpoint/pi_losses.txt", "w") as f:
        f.write("")
    with open("./checkpoint/v_losses.txt", "a") as f:
        f.write("")
    c.learn()
