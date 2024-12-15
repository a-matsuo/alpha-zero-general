import numpy as np
from .InitialMatGenerator import InitialMatGenerator

class CurriculumEnvironment:
    def __init__(self, base_N, max_N, base_num_ones, max_num_ones, difficulty_steps, seed=None):
        """
        base_N: 初期段階で使用するN
        max_N: 難易度最大時のN
        base_num_ones: 初期段階での1の個数
        max_num_ones: 難易度最大時の1の個数
        difficulty_steps: 難易度を上げるステップ数。学習イテレーションに応じて変化。

        例えば:
        base_N = 5, max_N = 10
        base_num_ones = 2, max_num_ones = 20
        difficulty_steps = args.numIters // 3 
        などと設定し、学習が進むたびにNやnum_onesを増やす。
        """
        self.base_N = base_N
        self.max_N = max_N
        self.base_num_ones = base_num_ones
        self.max_num_ones = max_num_ones
        self.difficulty_steps = difficulty_steps
        self.seed = seed

    def get_initial_state(self, iteration):
        """
        iteration: 現在の学習イテレーション番号 (1-based)
        iterationに応じて難易度を上げる。
        """
        # 難易度進行率を0~1で計算
        progress = min(1.0, iteration / self.difficulty_steps)

        # progressに応じてNやnum_onesを線形補間
        N = int(self.base_N + (self.max_N - self.base_N) * progress)
        # num_onesは偶数が望ましい例では偶数化
        desired_num_ones = int(self.base_num_ones + (self.max_num_ones - self.base_num_ones) * progress)
        # 偶数化(任意)
        if desired_num_ones % 2 != 0:
            desired_num_ones += 1

        # 一定の条件でnum_onesを設定せずに自由にランダム生成してもよい
        # ここでは難易度が低いときはnum_onesを指定、ある程度上がってきたら指定しないなども可能。
        # シンプルに常にnum_ones指定とする。
        
        gen = InitialMatGenerator(N, seed=self.seed)
        initial_mat = gen.generate(num_ones=desired_num_ones)
        return initial_mat


