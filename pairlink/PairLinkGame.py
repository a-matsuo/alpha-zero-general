from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import numpy as np
from .PairLinkBoard import PairLinkBoard as Board

class PairLinkGame(Game):
    """
    PairLinkパズルのGameクラス
    N と target_pairsを指定し、initial_cardsやmax_steps, max_card_idは内部で決定。
    target_pairsを省略すればランダムでペアを選ぶ。
    """

    def __init__(self, N):
        """
        N: カード枚数
        target_pairs: 指定なければランダム
        """
        self.N = N
        self.n = N
        self.max_card_id = N
        self.max_steps = (N - 2)*N // 2

    def getInitBoard(self, target_pairs=None):
        """
        初期状態のboardを返す
        shape: (n+1,)  board[0:n]=cards, board[n]=steps=0
        """
        b = Board(self.N)
        b.set_target_pairs(target_pairs)

        return b.get_state_array()

    def getBoardSize(self):
        return (self.n+1,)

    def getActionSize(self):
        # アクション数：隣接入れ替え可能箇所 n-1
        return self.n - 1

    def getNextState(self, board, player, action):
        """
        board上でplayerがactionを行った次の状態を返す
        常に新規Boardを生成
        """
        b = Board(self.N)
        b.set_state_from_array(board)
        b.execute_move(action)
        return (b.get_state_array(), player)

    def getValidMoves(self, board, player):
        """
        現在の状態で有効な手
        今回は全ての隣接swapが基本有効 (終局でなければチェック可能だが省略）
        """
        # 終局チェックをして厳密に制御したい場合は以下のようにする
        # b = Board(self.N, self.target_pairs)
        # b.set_state_from_array(board)
        # if b.is_terminal():
        #     return np.zeros(self.getActionSize(), dtype=int)

        valids = [1]*self.getActionSize()
        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        終了判定
        """
        b = Board(self.N)
        b.set_state_from_array(board)
        if b.is_terminal():
            return b.get_result()
        return 0

    def getCanonicalForm(self, board, player):
        # 1人ゲームなのでそのまま
        return board

    def getSymmetries(self, board, pi):
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
        # featureサイズを求めるため、一時的にBoardを生成
        b = Board(self.N)
        f = b.get_features()
        return f.shape[0]

    def get_features(self, board):
        # featuresを得るために新たにBoardを生成
        b = Board(self.N)
        b.set_state_from_array(board)
        return b.get_features()

    def reset_target_pairs(self):
        self.target_pairs = None
