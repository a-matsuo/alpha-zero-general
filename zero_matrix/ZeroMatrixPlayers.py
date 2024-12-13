import numpy as np

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # Get all valid moves
        valids = self.game.getValidMoves(board, 1)  # player=1 is arbitrary, single-player game
        valid_actions = np.where(valids == 1)[0]
        # Choose a random valid action
        action = np.random.choice(valid_actions)
        return action

class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # Get all valid moves
        valids = self.game.getValidMoves(board, 1)
        valid_actions = np.where(valids == 1)[0]

        # If no valid actions, just return -1 (terminal)
        if len(valid_actions) == 0:
            print("No valid actions available.")
            return -1

        # List the valid actions for the human to see
        print("Valid actions: ", valid_actions.tolist())

        # Since each action corresponds to a pair (i, j), let's display them
        # The game must have a coupling_map. We'll assume we can access it from the game logic.
        # If not directly accessible, store it within the game or logic and provide a getter.
        if hasattr(self.game, 'logic'):
            # Retrieve the coupling map to show action details
            coupling_map = self.game.logic.coupling_map
            print("Action details (action_index: (colA, colB)):")
            for a_idx in valid_actions:
                print(f"{a_idx}: {coupling_map[a_idx]}")

        # Ask the user to choose an action
        action = None
        while action not in valid_actions:
            try:
                user_input = input("Enter your chosen action index: ")
                action = int(user_input)
                if action not in valid_actions:
                    print("Invalid action. Please choose from the listed valid actions.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        return action