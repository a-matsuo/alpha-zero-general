import numpy as np

class ZeroMatrixLogic:
    def __init__(self, N, initial_mat, coupling_map, coupling_map_mat, max_turns=50):
        self.N = N
        self.initial_mat = initial_mat
        self.coupling_map = coupling_map
        self.coupling_map_mat = coupling_map_mat
        self.max_turns = max_turns

    def get_initial_board(self):
        board = np.zeros((self.N+1, self.N), dtype=int)
        board[:self.N, :] = self.initial_mat.copy()
        board[self.N, 0] = 0   # turn count
        if self.N > 1:
            board[self.N, 1] = -1  # last action = -1 (no previous action)
        return board

    def get_board_size(self):
        return (self.N+1, self.N)

    def get_action_size(self):
        return len(self.coupling_map)

    def execute_action(self, board, action):
        next_board = board.copy()

        turn_count = next_board[self.N, 0]
        (i, j) = self.coupling_map[action]

        # Swap columns i, j
        next_board[:self.N, [i, j]] = next_board[:self.N, [j, i]]

        # Swap rows i, j
        next_board[[i, j], :] = next_board[[j, i], :]

        # Apply mat’ = mat’ - (mat’ * coupling_map_mat)
        mat_section = next_board[:self.N, :]
        mat_section = mat_section - (mat_section * self.coupling_map_mat)
        next_board[:self.N, :] = mat_section

        # Increment turn count
        turn_count += 1
        next_board[self.N, 0] = turn_count

        # Record this action as the last action
        if self.N > 1:
            next_board[self.N, 1] = action

        return next_board

    def get_valid_moves(self, board):
        valids = [1]*self.get_action_size()

        # Check the last action taken
        last_action = -1
        if self.N > 1:
            last_action = board[self.N, 1]

        puzzle_board = board[:self.N, :]

        for idx, (colA, colB) in enumerate(self.coupling_map):
            # 1. No repeating the last action
            if idx == last_action:
                valids[idx] = 0
                continue

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
            for (r, c) in ones_positions:
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
            return 1
        if self.get_turn_count(board) >= self.max_turns:
            return -1
        return 0

    def is_solved(self, board):
        return np.all(board[:self.N, :] == 0)

    def get_turn_count(self, board):
        return board[self.N, 0]