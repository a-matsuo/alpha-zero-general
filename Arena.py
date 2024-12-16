import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

class Arena():
    """
    An Arena class where we test two different models/agents on a single-player game (puzzle).
    We measure how often each model solves the puzzle successfully.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        player1, player2: functions that take a board (canonical form) and return an action.
                          These represent two different models or agents trying to solve the puzzle.
        game: Game object (single-player puzzle)
        display: a function to display the board (for debugging/verbose)
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, player, verbose=False):
        """
        Run one game attempt using the given player (model).
        Returns:
          1  if the player solved the puzzle (game ended with success)
          -1 if the player failed to solve within constraints (game ended in failure)
        """
        board = self.game.getInitBoard()

        while self.game.getGameEnded(board, 1) == 0:  # player=1 is a dummy; single-player view
            if verbose and self.display is not None:
                self.display(board)

            # player gives the action
            action = player(self.game.getCanonicalForm(board, 1))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, 1), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            board, _ = self.game.getNextState(board, 1, action)

        # Game ended, return the result from the perspective of player=1
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        """
        Tests each model on the puzzle `num` times.

        Returns:
          oneWon: number of successful solves by player1
          twoWon: number of successful solves by player2
          draws:  number of attempts that ended in a state that is not success or fail
                  (If your game does not have a draw state, this will remain 0)
        """
        oneWon = 0
        twoWon = 0
        draws = 0

        # Test player1
        for _ in tqdm(range(num), desc="Arena.playGames Player1"):
            result = self.playGame(self.player1, verbose=verbose)
            if result == 1:
                oneWon += 1
            elif result == -1:
                # failed
                pass
            else:
                # handle draw or other outcomes if exist
                draws += 1

        # Test player2
        for _ in tqdm(range(num), desc="Arena.playGames Player2"):
            result = self.playGame(self.player2, verbose=verbose)
            if result == 1:
                twoWon += 1
            elif result == -1:
                # failed
                pass
            else:
                # handle draw or other outcomes if exist
                draws += 1

        return oneWon, twoWon, draws
