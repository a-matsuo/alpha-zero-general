import sys
sys.path.append('..')
import numpy as np
from MCTS import MCTS
from utils import dotdict
from pairlink.PairLinkGame import PairLinkGame
from pairlink.torch.NNet import NNetWrapper  # あらかじめPairLinkNNetをWrapしたクラスとする
import os
import random
from tqdm import tqdm
from constants import *
from qiskit import QuantumCircuit
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # テスト用のパラメータ
    N = num_qubits  # カード枚数
    # target_pairs = [(1,5),(2,4)] # 固定したい場合はこちらを使用
    target_pairs = None  # ランダム問題を出したい場合Noneにする

    # Gameインスタンス作成
    game = PairLinkGame(N, target_pairs)

    # NNet読み込み設定
    args = dotdict({
        'numMCTSSims': 50,
        'cpuct': 1.0,
        'cuda': False,
    })

    with_fig = ('fig' in sys.argv)
    if with_fig:
    # clear results directory
        import os
        import shutil
        if os.path.exists('./results'):
            shutil.rmtree('./results')
        os.makedirs('./results')
    
    
    # 学習済みモデルの読み込み
    nnet = NNetWrapper(game)
    checkpoint_folder = 'checkpoint'
    checkpoint_file = 'best.pth.tar'  # 適宜変更
    nnet.load_checkpoint(folder=checkpoint_folder, filename=checkpoint_file)

    # MCTS生成
    mcts = MCTS(game, nnet, args)

    # テスト回数
    num_tests = 100

    successes = 0
    failures = 0
    t = tqdm(range(num_tests))
    for i in t:
        # 新しい問題を生成するため、target_pairs=Noneでgameを再インスタンス化しても良い
        # 今はゲームを固定としたい場合、この処理は不要
        # game = PairLinkGame(N, None) #毎回新問題を生成

        board = game.getInitBoard()

        # プレイ開始
        step = 0
        max_step_limit = 200  # 適宜設定、最大行動回数等
        result = 0
        actions = []
        while True:
            step += 1
            # 終了判定
            r = game.getGameEnded(board, 1)
            if r != 0:
                result = r
                break

            # モデルで行動選択
            pi = mcts.getActionProb(game.getCanonicalForm(board, 1), temp=0)
            action = np.argmax(pi)
            actions.append(int(action))

            # 行動適用
            board, _ = game.getNextState(board, 1, action)

            if step > max_step_limit:
                # 念のための保険
                result = -1
                break

        # print(f"Test {i+1}/{num_tests}: Result = {result}")
        if result == 1:
            successes += 1
            if with_fig:
                qc = QuantumCircuit(N)
                for action in actions:
                    qc.swap(action, action+1)
                qc.draw(output='mpl', filename=f'./results/result_{i+1}.png')
        else:
            failures += 1
            if with_fig:
                qc = QuantumCircuit(N)
                for action in actions:
                    qc.swap(action, action+1)
                qc.draw(output='mpl', filename=f'./results/result_failed_{i+1}.png')
 
        t.set_postfix(successes=successes, failures=failures)
        plt.close('all')

    print(f"Successes: {successes}, Failures: {failures}, Rate: {successes/num_tests*100:.2f}%")
