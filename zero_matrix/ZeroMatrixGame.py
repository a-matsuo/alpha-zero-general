import numpy as np
import sys
sys.path.append('..')
from Game import Game
from .ZeroMatrixLogic import ZeroMatrixLogic

class ZeroMatrixGame(Game):
    def __init__(self, N, coupling_map, max_turns=None, initial_mat=None):
        if max_turns is None:
            max_turns = (N-2) * N/2
        coupling_map_mat = np.zeros((N, N))
        for (i, j) in coupling_map:
            coupling_map_mat[i, j] = 1
        coupling_map_mat = coupling_map_mat + coupling_map_mat.T
        self.logic = ZeroMatrixLogic(N, coupling_map, coupling_map_mat, max_turns, initial_mat)
        self.N = N

    def getInitBoard(self):
        return self.logic.get_initial_board()

    def getBoardSize(self):
        return self.logic.get_board_size()

    def getActionSize(self):
        return self.logic.get_action_size()

    def getNextState(self, board, player, action):
        next_board = self.logic.execute_action(board, action)
        return (next_board, player)

    def getValidMoves(self, board, player):
        return self.logic.get_valid_moves(board)

    def getGameEnded(self, board, player):
        return self.logic.get_game_ended(board)

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()