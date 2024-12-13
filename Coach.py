import sys
import os
import numpy as np
from utils import *
import pickle
import time
from MCTS import MCTS

class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = None
        self.trainExamplesHistory = []  # history of examples from self-play

        # Load model if provided
        if self.args.load_model:
            self.nnet.load_checkpoint(self.args.load_folder_file[0], self.args.load_folder_file[1])
    
    def executeEpisode(self):
        """
        This function executes one episode of self-play.
        At each step, it performs MCTS simulations to get a policy π for the current board state, then it picks a move according to π, plays it, and moves on until the game ends.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, pi, v)
        """
        trainExamples = []
        board = self.game.getInitBoard()
        player = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, player)
            valids = self.game.getValidMoves(canonicalBoard, 1)

            self.mcts = MCTS(self.game, self.nnet, self.args)
            pi = self.mcts.getActionProb(canonicalBoard, temp=1)
            action = np.random.choice(len(pi), p=pi)
            trainExamples.append([canonicalBoard, pi, None])

            board, player = self.game.getNextState(board, player, action)
            r = self.game.getGameEnded(board, player)

            if r != 0:
                # Game ended
                return [(x[0], x[1], r*((-1)**(1-player))) for x in trainExamples]

    def learn(self):
        """
        This function executes the learning process.
        """
        for i in range(1, self.args.numIters+1):
            print('------ITER ' + str(i) + '------')
            iterationTrainExamples = []
            for _ in range(self.args.numEps):
                iterationTrainExamples += self.executeEpisode()

            # Save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.maxlenOfQueue:
                self.trainExamplesHistory.pop(0)

            # Flatten examples
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            
            # Shuffle examples
            np.random.shuffle(trainExamples)

            self.nnet.train(trainExamples)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='checkpoint.pth.tar')
