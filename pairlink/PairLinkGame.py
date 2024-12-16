from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import numpy as np
from .PairLinkLogic import Board

class PairLinkGame(Game):
    """
    PairLinkパズルのGameクラス
    OthelloGameと類似のインターフェースで実装。
    """

    def __init__(self, initial_cards, target_pairs, max_steps=20, max_card_id=None):
        self.initial_cards = initial_cards
        self.target_pairs = target_pairs
        self.max_steps = max_steps
        self.n = len(initial_cards)

        if max_card_id is None:
            max_card_id = max(initial_cards)
        self.max_card_id = max_card_id

    def getInitBoard(self):
        """
        初期状態のboardを返す
        shape: (n+1,)  board[0:n]=cards, board[n]=steps=0
        """
        b = Board(self.initial_cards, self.target_pairs, self.max_steps, self.max_card_id)
        return b.get_state_array()

    def getBoardSize(self):
        """
        ボードサイズ
        1次元だが、Gameクラスが2次元を想定していることが多いので、(n+1,1)などとする
        ここでは(n+1,)で返しておく
        """
        return (self.n+1,)

    def getActionSize(self):
        # アクション数：隣接入れ替え可能箇所 n-1
        return self.n - 1

    def getNextState(self, board, player, action):
        """
        board上でplayerがactionを行った次の状態
        playerは1人ゲームなので変化なし
        """
        b = Board(self.initial_cards, self.target_pairs, self.max_steps, self.max_card_id)
        b.set_state_from_array(board)
        b.execute_move(action)
        return (b.get_state_array(), player)

    def getValidMoves(self, board, player):
        """
        現在の状態で有効な手
        全ての隣接swapが有効(終局でなければ)
        """
        b = Board(self.initial_cards, self.target_pairs, self.max_steps, self.max_card_id)
        b.set_state_from_array(board)
        valids = [0]*self.getActionSize()
        if not b.is_terminal():
            for i in range(self.n - 1):
                valids[i] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        終了時の結果:
        1: 成功
        -1: 失敗
        継続:0
        """
        b = Board(self.initial_cards, self.target_pairs, self.max_steps, self.max_card_id)
        b.set_state_from_array(board)
        if b.is_terminal():
            return b.get_result()
        return 0

    def getCanonicalForm(self, board, player):
        """
        プレイヤー1視点に正規化
        1人ゲームではそのままでOK
        """
        return board

    def getSymmetries(self, board, pi):
        """
        対称性拡張: カード列の反転
        piも対応
        """
        assert len(pi) == self.getActionSize()
        cards = board[0:self.n]
        steps = board[self.n]

        l = [(board, pi)]

        rev_cards = np.flip(cards, axis=0)
        rev_pi = np.flip(pi, axis=0)
        rev_board = np.concatenate([rev_cards, [steps]]).astype(int)
        l.append((rev_board, rev_pi))

        return l

    def stringRepresentation(self, board):
        cards = board[0:self.n]
        steps = board[self.n]
        return " ".join(map(str, cards)) + f" | steps={steps}"

    @staticmethod
    def display(board):
        n = len(board)-1
        cards = board[0:n]
        steps = board[n]
        print("Cards:", cards)
        print("Steps:", steps)

    def getFeatureSize(self):
        """
        特徴ベクトルの次元数: L*C + P
        L = n (カード数)
        C = max_card_id
        P = len(target_pairs)
        """
        L = self.n
        C = self.max_card_id
        P = len(self.target_pairs)
        return L*C + P

    def get_features(self, board):
        """
        board(状態配列)から特徴ベクトルを生成
        """
        b = Board(self.initial_cards, self.target_pairs, self.max_steps, self.max_card_id)
        b.set_state_from_array(board)
        return b.get_features()
