import numpy as np
import random

class PairLinkBoard():
    def __init__(self, N, target_pairs=None):
        """
        N: カード枚数
        target_pairs: [(a,b), (c,d), ...] 必要なペア (a<b推奨)
                      None の場合はランダム生成

        initial_cards: [1..N]
        max_steps: (N - 2)*N/2
        max_card_id = N
        """
        self.n = N
        self.initial_cards = list(range(1, N+1))
        self.max_card_id = N
        self.max_steps = (N - 2)*N // 2

        # 全ペア集合を生成
        self.all_pairs_list = self._generate_all_pairs()
        self.P_max = len(self.all_pairs_list)

        if target_pairs is None:
            # ランダムにペアを選択
            num_pairs = random.randint(1, self.P_max)
            selected_pairs = random.sample(self.all_pairs_list, num_pairs)
            self.target_pairs = selected_pairs
        else:
            self.target_pairs = target_pairs[:]

        # 現在の状態
        self.cards = self.initial_cards[:]
        self.steps = 0

        # 一度でも隣接していたら達成済みとみなすためのセット
        self.achieved_pairs = set()

        # ペアカード特徴の事前計算
        self.pair_card_features = self._precompute_pair_card_features()

    def _generate_all_pairs(self):
        C = self.max_card_id
        all_pairs = []
        for x in range(1, C):
            for y in range(x+1, C+1):
                all_pairs.append((x,y))
        return all_pairs

    def set_state_from_array(self, board_array):
        # 前半: cards + steps
        self.cards = list(board_array[0:self.n])
        self.steps = int(board_array[self.n])

        # 後半: targetフラグ + achievedフラグ
        # targetフラグは board_array[n+1 : n+1+P_max]
        # achievedフラグは board_array[n+1+P_max : n+1+2*P_max]

        target_flags = board_array[self.n+1 : self.n+1+self.P_max]
        achieved_flags = board_array[self.n+1+self.P_max : self.n+1+2*self.P_max]

        # target_pairs復元
        # target_flags[i] == 1 のペアを target_pairs に含める
        # 0の場合は対象外
        new_target_pairs = []
        for i, (a,b) in enumerate(self.all_pairs_list):
            if target_flags[i] == 1:
                new_target_pairs.append((a,b))
        self.target_pairs = new_target_pairs

        # achieved_pairs復元
        new_achieved = set()
        for i, (a,b) in enumerate(self.all_pairs_list):
            if achieved_flags[i] == 1:
                pa, pb = (a,b) if a<b else (b,a)
                new_achieved.add((pa,pb))
        self.achieved_pairs = new_achieved

    def get_state_array(self):
        # 現在の状態を array 化
        # サイズ: n (cards) + 1 (steps) + P_max (target flags) + P_max (achieved flags)
        arr = np.array(self.cards + [self.steps], dtype=int)

        target_flags = np.zeros(self.P_max, dtype=int)
        # target_pairsに含まれるペアがあれば1にする
        target_set = set()
        for (a,b) in self.target_pairs:
            pa, pb = (a,b) if a<b else (b,a)
            target_set.add((pa,pb))
        for i,(a,b) in enumerate(self.all_pairs_list):
            pa, pb = (a,b) if a<b else (b,a)
            if (pa,pb) in target_set:
                target_flags[i] = 1

        achieved_flags = np.zeros(self.P_max, dtype=int)
        for i,(a,b) in enumerate(self.all_pairs_list):
            pa, pb = (a,b) if a<b else (b,a)
            if (pa,pb) in self.achieved_pairs:
                achieved_flags[i] = 1

        # 結合
        full_arr = np.concatenate([arr, target_flags, achieved_flags])
        return full_arr

    def execute_move(self, action):
        if action < 0 or action >= self.n - 1:
            raise ValueError("Invalid action.")
        self.cards[action], self.cards[action+1] = self.cards[action+1], self.cards[action]
        self.steps += 1

        self._update_achieved_pairs()

    def _update_achieved_pairs(self):
        # 現在隣接しているターゲットペアをachieved_pairsに追加
        target_set = set()
        for (a,b) in self.target_pairs:
            pa, pb = (a,b) if a<b else (b,a)
            target_set.add((pa,pb))

        for (a, b) in self.target_pairs:
            pa, pb = (a,b) if a<b else (b,a)
            if (pa, pb) not in self.achieved_pairs:
                if self._is_adjacent(a, b):
                    self.achieved_pairs.add((pa, pb))

    def is_goal_reached(self):
        target_set = set()
        for (a,b) in self.target_pairs:
            pa, pb = (a,b) if a<b else (b,a)
            target_set.add((pa,pb))
        # 全target_pairsがachievedに含まれているか
        return target_set.issubset(self.achieved_pairs)

    def _is_adjacent(self, a, b):
        for i in range(self.n - 1):
            if (self.cards[i] == a and self.cards[i+1] == b) or \
               (self.cards[i] == b and self.cards[i+1] == a):
                return True
        return False

    def is_terminal(self):
        if self.is_goal_reached():
            return True
        if self.steps >= self.max_steps:
            return True
        return False

    def get_result(self):
        if self.is_goal_reached():
            return 1
        else:
            return -1

    def get_one_hot_cards(self):
        L = self.n
        C = self.max_card_id
        one_hot = np.zeros((L, C), dtype=int)
        for i, card in enumerate(self.cards):
            idx = card - 1
            if idx < 0 or idx >= C:
                raise ValueError("カードIDが範囲外です")
            one_hot[i, idx] = 1
        return one_hot.flatten()

    def _precompute_pair_card_features(self):
        C = self.max_card_id
        pair_card_feats = []
        for (a, b) in self.all_pairs_list:
            vec_a = np.zeros(C, dtype=int)
            vec_b = np.zeros(C, dtype=int)
            vec_a[a-1] = 1
            vec_b[b-1] = 1
            pair_card_feat = np.concatenate([vec_a, vec_b])
            pair_card_feats.append(pair_card_feat)
        return pair_card_feats

    def get_features(self):
        # 特徴作成は従来通り
        C = self.max_card_id
        card_vec = self.get_one_hot_cards()

        target_set = set()
        for (a,b) in self.target_pairs:
            pa, pb = (a,b) if a<b else (b,a)
            target_set.add((pa,pb))

        pair_vecs = []
        for i, (a, b) in enumerate(self.all_pairs_list):
            pa, pb = (a,b) if a<b else (b,a)
            achieved_flag = 0
            if (pa, pb) in target_set:
                if (pa, pb) in self.achieved_pairs:
                    achieved_flag = 0  # 達成済みなので未達成フラグは0
                else:
                    achieved_flag = 1  # まだ達成していない

            pair_card_feat = self.pair_card_features[i]
            pair_feat = np.concatenate([[achieved_flag], pair_card_feat])
            pair_vecs.append(pair_feat)

        pair_vec = np.concatenate(pair_vecs)
        features = np.concatenate([card_vec, pair_vec])
        return features

# テスト例
if __name__ == "__main__":
    N = 5
    b = PairLinkBoard(N)
    print("Before any move:")
    arr = b.get_state_array()
    print("state_array shape:", arr.shape)
    print(arr)

    # 一手実行
    b.execute_move(0)
    arr2 = b.get_state_array()
    print("After one move:")
    print("state_array shape:", arr2.shape)
    print(arr2)

    # 新規Boardを作ってset_state_from_arrayで復元
    b2 = PairLinkBoard(N)  # 新規インスタンス
    b2.set_state_from_array(arr2)
    arr3 = b2.get_state_array()
    print("Copied board state_array:")
    print(arr3)
    print("Equal:", np.array_equal(arr2, arr3))
