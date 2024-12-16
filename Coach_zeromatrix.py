import logging
import os
import pickle
import sys
import time

import numpy as np
from tqdm import tqdm

from zero_matrix.CurriculumEnvironment import CurriculumEnvironment
from MCTS import MCTS
from utils import *

log = logging.getLogger(__name__)


class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = None
        self.trainExamplesHistory = []  # history of examples from self-play
        self.env = CurriculumEnvironment(
            base_N=self.game.N,
            max_N=self.game.N,
            base_num_ones=2,
            max_num_ones=30,
            difficulty_steps=self.args.numIters,
            seed=None,
        )

        # Load model if provided
        if self.args.load_model:
            self.nnet.load_checkpoint(
                self.args.load_folder_file[0], self.args.load_folder_file[1]
            )

    def executeEpisode(self):
        """
        This function executes one episode of self-play.
        At each step, it performs MCTS simulations to get a policy π for the current board state, then it picks a move according to π, plays it, and moves on until the game ends.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, pi, v)
        """
        trainExamples = []
        if self.args.curriculum_learning:
            initial_mat = self.env.get_initial_state(len(self.trainExamplesHistory) + 1)
            board = self.game.getInitBoard(initial_mat)
        else:
            board = self.game.getInitBoard()
        player = 1
        episodeStep = 0

        self.mcts = MCTS(self.game, self.nnet, self.args)

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, player)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            action = np.random.choice(len(pi), p=pi)
            trainExamples.append([canonicalBoard, pi, None])

            board, player = self.game.getNextState(board, player, action)

            # visited stateを更新
            next_puzzle_board = board[:self.game.N, :]
            self.game.visited_states.add(self.game.stringRepresentation(next_puzzle_board))

            r = self.game.getGameEnded(board, player)

            if r != 0:
                # Game ended
                return [(x[0], x[1], r * ((-1) ** (1 - player))) for x in trainExamples]

    def learn(self):
        """
        This function executes the learning process.
        """
        for i in range(1, self.args.numIters + 1):
            log.info(f"Starting Iter #{i} ...")
            iterationTrainExamples = []
            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                iterationTrainExamples += self.executeEpisode()

            # Save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.maxlenOfQueue:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}"
                )
                self.trainExamplesHistory.pop(0)

            # Flatten examples
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)

            # Shuffle examples
            np.random.shuffle(trainExamples)

            self.nnet.train(trainExamples)

            # Save the model
            filename = f"checkpoint_qubits{self.args.num_qubits}_iter{i}_Eps{self.args.numEps}_MCTS{self.args.numMCTSSims}_lr{self.args.lr}_epochs{self.args.epochs}.pth.tar"
            dirname = os.path.join(
                self.args.checkpoint,
                f"qubits{self.args.num_qubits}_Eps{self.args.numEps}_MCTS{self.args.numMCTSSims}_lr{self.args.lr}_epochs{self.args.epochs}",
            )

            os.makedirs(dirname, exist_ok=True)

            self.nnet.save_checkpoint(folder=dirname, filename=filename)
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="last.pth.tar"
            )
