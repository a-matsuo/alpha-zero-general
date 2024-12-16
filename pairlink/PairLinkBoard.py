import numpy as np

class PairLinkBoard():
    def __init__(self, initial_cards, target_pairs, max_steps, max_card_id):
        """
        initial_cards: [c1, c2, ..., cL] 1-based card ID
        target_pairs: [(a,b), (c,d), ...] 本ゲームで必要なペア (a<b推奨)
        max_steps: 最大手数
        max_card_id: C

        全ての可能なペア (a,b) (1<=a<b<=C) を定義したall_pairs_listを内部で生成。
        P_max = len(all_pairs_list)となる。

        特徴量:
        - カードone-hot: L*C次元
        - ペア特徴: P_maxペア分: 各ペア (1+2*C)次元
          1次元: achievedフラグ (0 or 1)
          2*C次元: (a,b)ペアを示すone-hot
        """

        self.initial_cards = initial_cards[:]
        self.n = len(initial_cards)
        self.target_pairs = target_pairs[:]
        self.max_steps = max_steps
        self.max_card_id = max_card_id

        self.cards = initial_cards[:]
        self.steps = 0

        # 全ペア集合を生成 (a<b)
        self.all_pairs_list = self._generate_all_pairs()
        self.P_max = len(self.all_pairs_list)

        # ペアカード特徴の事前計算
        self.pair_card_features = self._precompute_pair_card_features()

    def _generate_all_pairs(self):
        """
        max_card_idに基づいて全ての(a,b) (1<=a<b<=C)ペアを辞書順で生成
        """
        all_pairs = []
        C = self.max_card_id
        for x in range(1, C):
            for y in range(x+1, C+1):
                all_pairs.append((x,y))
        return all_pairs

    def set_state_from_array(self, board_array):
        """
        board_array: shape (n+1,)
        0～n-1: cards
        n: steps
        """
        self.cards = list(board_array[0:self.n])
        self.steps = int(board_array[self.n])

    def get_state_array(self):
        arr = np.array(self.cards + [self.steps], dtype=int)
        return arr

    def execute_move(self, action):
        if action < 0 or action >= self.n - 1:
            raise ValueError("Invalid action.")
        self.cards[action], self.cards[action+1] = self.cards[action+1], self.cards[action]
        self.steps += 1

    def is_goal_reached(self):
        for (a, b) in self.target_pairs:
            if not self._is_adjacent(a, b):
                return False
        return True

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
        """
        cardsをone-hotベクトル (L*C)へ
        """
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
        """
        all_pairs_listに基づき、ペアカード部分のone-hotを事前計算する。
        各ペアについて: (2*C)次元のベクトル(achievedを除く部分)
        """
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
        """
        特徴量生成:
        - カードone-hot: L*C
        - ペア特徴: P_max * (1+2*C)

        target_pairsに含まれるペア(i)について:
          achieved=1 if 未達成
          achieved=0 if 達成済
        target_pairsに含まれないペアはachieved=0 (対象外ペア)

        ペアはall_pairs_listの固定順序で対応
        """
        C = self.max_card_id
        card_vec = self.get_one_hot_cards()

        target_set = set()
        for p in self.target_pairs:
            a, b = p
            if a > b:
                a, b = b, a
            target_set.add((a, b))

        pair_vecs = []
        for i, (a, b) in enumerate(self.all_pairs_list):
            # achieved_flag判定
            if (a, b) in target_set:
                # 未達成=1, 達成=0
                achieved_flag = 1
                if self._is_adjacent(a, b):
                    achieved_flag = 0
            else:
                # ターゲットでないペアは0
                achieved_flag = 0

            pair_card_feat = self.pair_card_features[i]
            pair_feat = np.concatenate([[achieved_flag], pair_card_feat])
            pair_vecs.append(pair_feat)

        pair_vec = np.concatenate(pair_vecs)
        features = np.concatenate([card_vec, pair_vec])
        return features


# テスト例
if __name__ == "__main__":
    initial_cards = [1,2,3,4,5]
    target_pairs = [(1,5),(2,4)]
    max_steps = 20
    max_card_id = 5

    b = Board(initial_cards, target_pairs, max_steps, max_card_id)
    features = b.get_features()
    print("Features shape:", features.shape)
    print(features)
