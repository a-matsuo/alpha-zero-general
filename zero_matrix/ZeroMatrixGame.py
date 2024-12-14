import numpy as np
import sys
sys.path.append('..')
from Game import Game
from .ZeroMatrixLogic import ZeroMatrixLogic

class ZeroMatrixGame(Game):
    def __init__(self, N, coupling_map, max_turns=None, initial_mat=None, seed=None):
        if max_turns is None:
            max_turns = (N-2) * N/2
        coupling_map_mat = np.zeros((N, N))
        for (i, j) in coupling_map:
            coupling_map_mat[i, j] = 1
        coupling_map_mat = coupling_map_mat + coupling_map_mat.T
        self.logic = ZeroMatrixLogic(N, coupling_map, coupling_map_mat, max_turns, initial_mat, seed)
        self.N = N
        self.coupling_map = coupling_map
        self.initVisitedStates()  # 訪問済み状態を記録するセット

    def getInitBoard(self):
        board = self.logic.get_initial_board()
        puzzle_board = board[:self.N, :]
        self.visited_states.add(self.stringRepresentation(puzzle_board))
        return board

    def getBoardSize(self):
        return self.logic.get_board_size()

    def getActionSize(self):
        return self.logic.get_action_size()

    def getNextState(self, board, player, action):
        next_board = self.logic.execute_action(board, action)
        return (next_board, player)

    def getValidMoves(self, board, player):
        valids = self.logic.get_valid_moves(board)
        # 現在のvisited_statesをもとに、各アクションをチェック
        for idx, (colA, colB) in enumerate(self.coupling_map):
            if valids[idx] == 1:
                # simulate action and check if the state has been visited
                next_board = self.logic.simulate_action(board, idx)
                next_puzzle_board = next_board[:self.N, :]
                next_state_str = self.stringRepresentation(next_puzzle_board)

                # もし既に訪問済みなら、無効化
                if next_state_str in self.visited_states:
                    valids[idx] = 0

        return valids

    def getGameEnded(self, board, player):
        return self.logic.get_game_ended(board)

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()

    def initVisitedStates(self):
        self.visited_states = set()
