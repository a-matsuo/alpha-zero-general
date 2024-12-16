import numpy as np
from .InitialMatGenerator import InitialMatGenerator


class ZeroMatrixLogic:
    def __init__(
        self,
        N,
        coupling_map,
        coupling_map_mat,
        max_turns=None,
        initial_mat=None,
        seed=None,
    ):
        self.N = N
        self.coupling_map = coupling_map
        self.coupling_map_mat = coupling_map_mat
        self.max_turns = max_turns
        self.initial_mat = initial_mat if initial_mat is not None else "random"
        self.initial_mat_generator = InitialMatGenerator(N, seed)
        np.random.seed(seed)

    def get_initial_board(self, initial_mat=None):
        if initial_mat is None:
            if self.initial_mat == "random":
                self.initial_mat = self.initial_mat_generator.generate("random")
            elif self.initial_mat == "complete":
                self.initial_mat = self.initial_mat_generator.generate("complete")
            elif isinstance(self.initial_mat, np.ndarray):
                if self.initial_mat.shape != (self.N, self.N):
                    raise ValueError(
                        f"Invalid shape for initial_mat: {self.initial_mat.shape}"
                    )
            else:
                raise ValueError(f"Invalid initial_mat_type: {self.initial_mat}")
        elif isinstance(initial_mat, np.ndarray):
            if initial_mat.shape != (self.N, self.N):
                raise ValueError(f"Invalid shape for initial_mat: {initial_mat.shape}")
            self.initial_mat = initial_mat
        
        board = np.zeros((self.N + 1, self.N), dtype=int)
        board[: self.N, :] = self.initial_mat.copy()

        # Apply mat’ = mat’ - (mat’ * coupling_map_mat)
        mat_section = board[: self.N, :]
        mat_section = mat_section - (mat_section * self.coupling_map_mat)
        next_board = np.zeros((self.N + 1, self.N), dtype=int)
        next_board[: self.N, :] = mat_section

        next_board[self.N, 0] = 0  # turn count
        if self.N > 1:
            next_board[self.N, 1] = 0
        return next_board

    def get_board_size(self):
        return (self.N + 1, self.N)

    def get_action_size(self):
        return len(self.coupling_map)

    def execute_action(self, board, action):
        next_board = board.copy()

        turn_count = next_board[self.N, 0]
        (i, j) = self.coupling_map[action]

        # Swap columns i, j
        next_board[: self.N, [i, j]] = next_board[: self.N, [j, i]]

        # Swap rows i, j
        next_board[[i, j], :] = next_board[[j, i], :]

        # Apply mat’ = mat’ - (mat’ * coupling_map_mat)
        mat_section = next_board[: self.N, :]
        mat_section = mat_section - (mat_section * self.coupling_map_mat)
        next_board[: self.N, :] = mat_section

        # Increment turn count
        turn_count += 1
        next_board[self.N, 0] = turn_count
        next_board[self.N, 1] = (1 << action)

        # レイヤーで使用されたアクションを記録。もしアクションが新たに使用された場合、ビットを立てる
        # そして、左右のビットをクリアする。
        # print(f"action: {action}")
        # if self.N > 1:
        #     used_actions = next_board[self.N, 1]  # ここにused_actionsビットマスクを格納する
        #     # 1. 現アクションを使用済みに設定
        #     used_actions |= (1 << action)

        #     # 2. 左隣のビットをクリア
        #     if action - 1 >= 0:
        #         used_actions &= ~(1 << (action - 1))

        #     # 3. 右隣のビットをクリア
        #     if action + 1 < self.get_action_size():
        #         used_actions &= ~(1 << (action + 1))

        #     next_board[self.N, 1] = used_actions

        return next_board

    def simulate_action(self, board, action):
        next_board = board.copy()

        (i, j) = self.coupling_map[action]

        # Swap columns i, j
        next_board[: self.N, [i, j]] = next_board[: self.N, [j, i]]

        # Swap rows i, j
        next_board[[i, j], :] = next_board[[j, i], :]

        # Apply mat’ = mat’ - (mat’ * coupling_map_mat)
        mat_section = next_board[: self.N, :]
        mat_section = mat_section - (mat_section * self.coupling_map_mat)
        next_board[: self.N, :] = mat_section

        return next_board

    def get_valid_moves(self, board):
        valids = [1] * self.get_action_size()

        # # used_actionsは使用済みアクションが立っているビットマスクとする
        # # 使用済みアクションはvalidsを0にする
        used_actions = board[self.N, 1]

        for i in range(self.get_action_size()):
            if (used_actions & (1 << i)) != 0:
                valids[i] = 0

        puzzle_board = board[: self.N, :]

        for idx, (colA, colB) in enumerate(self.coupling_map):
            # 2. Must involve rows/columns that have at least one '1'
            rowA_has_ones = puzzle_board[colA, :].any()
            rowB_has_ones = puzzle_board[colB, :].any()
            colA_has_ones = puzzle_board[:, colA].any()
            colB_has_ones = puzzle_board[:, colB].any()

            if not (rowA_has_ones or rowB_has_ones or colA_has_ones or colB_has_ones):
                valids[idx] = 0
                continue

            # 3. After simulating the swap, at least one '1' must have a smaller |r'-c'|
            #    than its original |r-c|.
            improved = False
            ones_positions = np.argwhere(puzzle_board == 1)
            for r, c in ones_positions:
                # Compute the new positions after swapping rows/columns colA and colB
                if r == colA:
                    r_new = colB
                elif r == colB:
                    r_new = colA
                else:
                    r_new = r

                if c == colA:
                    c_new = colB
                elif c == colB:
                    c_new = colA
                else:
                    c_new = c

                old_dist = abs(r - c)
                new_dist = abs(r_new - c_new)
                if new_dist < old_dist:
                    improved = True
                    break

            if not improved:
                valids[idx] = 0

        return np.array(valids)

    def get_game_ended(self, board):
        if self.is_solved(board):
            turn_count = self.get_turn_count(board)
            return 1
            # return 1 - (turn_count / self.max_turns)
        if self.get_turn_count(board) >= self.max_turns:
            return -1
        return 0

    def is_solved(self, board):
        return np.all(board[: self.N, :] == 0)

    def get_turn_count(self, board):
        return board[self.N, 0]
