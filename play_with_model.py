import numpy as np
from zero_matrix.ZeroMatrixGame import ZeroMatrixGame
from zero_matrix.torch.NNet import NNetWrapper as NNet
from MCTS import MCTS
from utils import dotdict

from qiskit import QuantumCircuit

if __name__ == "__main__":
    N = 6
    coupling_map = [(i, i+1) for i in range(N-1)]
    game = ZeroMatrixGame(N, coupling_map)

    # モデル読み込み
    nnet = NNet(game)
    nnet.load_checkpoint(folder='checkpoint', filename='checkpoint.pth.tar')

    mcts_args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts = MCTS(game, nnet, mcts_args)

    board = game.getInitBoard()
    player = 1
    print("Initial board:\n", board[:game.N, :])

    # 最大ターン数を設定し、ループを回す
    max_turns = 10000
    actions = []
    for t in range(max_turns):
        if game.getGameEnded(board, player) != 0:
            break

        # MCTSを用いて行動確率を求め、最も確率の高い行動を選択
        pi = mcts.getActionProb(board, temp=0)
        action = np.argmax(pi)
        actions.append(int(action))

        board, player = game.getNextState(board, player, action)
        print(f"Turn {t+1}, Chosen action: {action}")
        print("Current board:\n", board[:game.N, :])

    
    result = game.getGameEnded(board, player)
    if result > 0:
        print("Success: the puzzle was solved!")
        print("Actions taken:", actions)
        qc = QuantumCircuit(N)
        for action in actions:
            qc.swap(action, action+1)
        qc.draw(output='mpl', filename='result.png')
    elif result == -1:
        print("Failed to solve within constraints.")
    else:
        print("Game ended with no success or failure signal (0).")