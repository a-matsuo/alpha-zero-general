
import numpy as np

class InitialMatGenerator:
    def __init__(self, N, seed=None):
        """
        N: 行列サイズ N x N
        seed: 乱数シード (オプション)
        """
        self.N = N
        if seed is not None:
            np.random.seed(seed)

    def generate(self, type='random', num_ones=None):
        """
        N x N の対称な0/1行列を生成し返す。
        対角成分は0、
        Returns:
            initial_mat: N x N numpy array of 0/1
        """
        if type == 'random':
            return self._generate_random(num_ones)
        elif type == 'complete':
            return np.ones((self.N, self.N)) - np.eye(self.N)
        else:
            raise ValueError(f"Invalid type: {type}")

    def _generate_random(self, num_ones=None):
        """
        num_onesが指定されれば指定した個数の1を持つ対称行列を生成。
        num_onesがNoneなら以前と同様に制約なしでランダム生成する。
        """
        if num_ones is None:
            return self._generate_random_unconstrained()
        else:
            return self._generate_random_constrained(num_ones)

    def _generate_random_unconstrained(self):
        """
        上三角部分はランダムに0か1。
        下三角は上三角を対称化して作る。
        """
        # 上三角部分（対角の上）をランダムに0か1
        upper = np.triu(np.random.randint(0, 2, size=(self.N, self.N)), k=1)
        # 対称化
        sym_matrix = upper + upper.T
        # 対角は0になっているはず（upper生成時にk=1指定で対角上のみ）
        return sym_matrix

    def _generate_random_constrained(self, num_ones):
        """
        num_onesが指定された場合、指定した個数の1を持つ対称行列を生成する。
        num_onesは偶数、かつ2*M以下である必要がある（Mは上三角要素数）。
        """
        M = (self.N*(self.N-1))//2

        if num_ones % 2 != 0:
            raise ValueError("num_ones must be an even number.")
        if num_ones > 2*M:
            raise ValueError(f"num_ones is too large. Max is {2*M} for given N={self.N}.")

        half_ones = num_ones // 2

        # 上三角 (対角除く) の全インデックス
        upper_indices = [(r, c) for r in range(self.N) for c in range(r+1, self.N)]

        # half_ones個のインデックスをランダムに選択
        chosen = np.random.choice(len(upper_indices), size=half_ones, replace=False)

        upper = np.zeros((self.N, self.N), dtype=int)
        for idx in chosen:
            r, c = upper_indices[idx]
            upper[r, c] = 1

        # 対称化
        sym_matrix = upper + upper.T

        return sym_matrix

