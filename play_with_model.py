import numpy as np
from zero_matrix.ZeroMatrixGame import ZeroMatrixGame
from zero_matrix.torch.NNet import NNetWrapper as NNet
from MCTS import MCTS
from utils import dotdict
from constants import *
from tqdm import tqdm

from qiskit import QuantumCircuit
import sys
import random

from matplotlib import pyplot as plt

if __name__ == "__main__":
    N = num_qubits
    coupling_map = [(i, i+1) for i in range(N-1)]
    results = []
    
    run_random = ('random' in sys.argv)
    run_complete = ('complete' in sys.argv)
    with_fig = ('fig' in sys.argv)
    if with_fig:
        # clear results directory
        import os
        import shutil
        if os.path.exists('./results'):
            shutil.rmtree('./results')
        os.makedirs('./results')

    num_exps = 1  # デフォルトは1回実験
    # 引数の最後が整数であると仮定してパース
    if len(sys.argv) > 1:
        try:
            # 最後の引数をint変換して実験回数に使う
            num_exps = int(sys.argv[-1])
        except ValueError:
            pass

    experiments = tqdm(range(num_exps), desc="Experiments")
    for exp_i in experiments:
        # Check for 'random' argument
        if run_random:
            seed = random.randint(0, 10000)
            initial_mat_type = 'random'
        elif run_complete:
            initial_mat_type = 'complete'
            seed = 42
        else:
            seed = 42

        game = ZeroMatrixGame(N, coupling_map, initial_mat_type=initial_mat_type, seed=seed)

        # モデル読み込み
        nnet = NNet(game)
        nnet.load_checkpoint(folder='checkpoint', filename='last.pth.tar')
        # nnet.load_checkpoint(folder='checkpoint', filename='checkpoint_qubits6_iter10_Eps10_MCTS25_lr0.001_epochs10.pth.tar')

        mcts_args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts = MCTS(game, nnet, mcts_args)

        board = game.getInitBoard()
        player = 1
        if num_exps == 1:
            print("Initial board:\n", board[:game.N, :])

        # 最大ターン数を設定し、ループを回す
        max_turns = 10000
        actions = []
        
        for t in range(max_turns):
            if game.getGameEnded(board, player) != 0:
                break

            # MCTSを用いて行動確率を求め、最も確率の高い行動を選択
            pi = mcts.getActionProb(board, temp=0)
            if num_exps == 1:
                print("Action probabilities:", pi)
            action = np.argmax(pi)
            actions.append(int(action))

            board, player = game.getNextState(board, player, action)
            # 行動確定後にvisited_statesを更新
            puzzle_board = board[:game.N, :]
            game.visited_states.add(game.stringRepresentation(puzzle_board))
            
            if num_exps == 1:
                print(f"Turn {t+1}, Chosen action: {action}")
                print("Current board:\n", board[:game.N, :])

        
        result = game.getGameEnded(board, player)
        if num_exps == 1:
            print("Final board:\n", board[:game.N, :])
        if result > 0:
            results.append(1)
            if num_exps == 1:
                print("Success: the puzzle was solved!")
                print("Total turns:", t+1)
                print("Actions taken:", actions)
            if with_fig:
                qc = QuantumCircuit(N)
                for action in actions:
                    qc.swap(action, action+1)
                qc.draw(output='mpl', filename=f'./results/result_{exp_i+1}.png')
        elif result == -1:
            results.append(0)
            if num_exps == 1:
                print("Failed to solve within constraints.")
            if num_exps == 1 or with_fig:
                qc = QuantumCircuit(N)
                for action in actions:
                    qc.swap(action, action+1)
                qc.draw(output='mpl', filename=f'./results/result_failed_{exp_i+1}.png')
        else:
            print("Game ended with no success or failure signal (0).")

        experiments.set_postfix(success_rate=np.mean(np.array(results)))
        plt.close('all')
