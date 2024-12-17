import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

class Arena():
    """
    An Arena class to compare two models/agents on a single-player puzzle.
    Both players solve the exact same set of puzzles, so we can fairly compare their performance.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        player1, player2: functions that take a board (canonical form) and return an action.
        game: Game object (single-player puzzle)
        display: a function to display the board (for debugging/verbose)
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, player, initial_board, verbose=False):
        """
        Run one game attempt using the given player (model) on a provided initial_board.
        Returns:
          1  if the player solved the puzzle (game ended with success)
          -1 if the player failed (game ended in failure)
          (or another value if game.getGameEnded returns something else)
        """
        board = initial_board.copy()  # Ensure we don't modify the original

        while self.game.getGameEnded(board, 1) == 0:
            if verbose and self.display is not None:
                self.display(board)
            action = player(self.game.getCanonicalForm(board, 1))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, 1), 1)
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            board, _ = self.game.getNextState(board, 1, action)

        # print(f'Game ended with result: {self.game.getGameEnded(board, 1)}')
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        """
        Tests each model on the puzzle `num` times.
        Both players solve the SAME `num` puzzles.

        Returns:
          oneWon: number of successful solves by player1
          twoWon: number of successful solves by player2
          draws:  number of attempts that ended in a state that is not success or fail
        """
        oneWon = 0
        twoWon = 0
        draws = 0

        # Generate all initial puzzles first
        # If getInitBoard is randomized, this ensures both players face the same puzzles.
        initial_boards = [self.game.getInitBoard() for _ in range(num)]

        t1 = tqdm(initial_boards, desc="Arena.playGames Player1")
        # Test player1 on all puzzles
        for ib in t1:
            result = self.playGame(self.player1, ib, verbose=verbose)
            if result == 1:
                oneWon += 1
            elif result == -1:
                pass
            else:
                print(result)
                draws += 1
            t1.set_postfix(win_rate=oneWon / len(initial_boards))
            

        # Test player2 on the same puzzles
        t2 = tqdm(initial_boards, desc="Arena.playGames Player2")
        for ib in t2:
            result = self.playGame(self.player2, ib, verbose=verbose)
            if result == 1:
                twoWon += 1
            elif result == -1:
                pass
            else:
                draws += 1
            t2.set_postfix(win_rate=twoWon / len(initial_boards))

        return oneWon, twoWon, draws
