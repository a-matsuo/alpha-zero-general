import logging

import coloredlogs
import torch
from Coach import PairLinkCoach as Coach
from constants import *
from utils import dotdict
from pairlink.torch.NNet import NNetWrapper as NNet
from pairlink.PairLinkGame import PairLinkGame as Game

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.

if __name__ == "__main__":
    # Game parameters
    log.info("Loading %s...", Game.__name__)
    N = num_qubits
    game = Game(N)

    # Training parameters
    args = dotdict(
        {
            "numIters": 10,  # number of training iterations
            "numEps": 10,  # number of self-play games per iteration
            "maxlenOfQueue": 200000,  # memory size
            "numMCTSSims": 40,  # number of MCTS simulations per move
            "cpuct": 1.0,
            "checkpoint": "./checkpoint/",
            "load_model": False,
            "load_folder_file": ("./checkpoint/pairlink", "checkpoint.pth.tar"),
            "lr": 0.001,
            "dropout": 0.3,
            "num_channels": 64,
            "epochs": 10,
            "batch_size": 64,
            "cuda": True if torch.cuda.is_available() else False,
            "num_qubits": N,
            "tempThreshold": tempThreshold,
            "curriculum_learning": True,
            'arenaCompare': 100,
            'updateThreshold': 0.6,
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
