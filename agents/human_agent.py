import random


class HumanAgent:
    """Agent controlled via keyboard input."""

    def __init__(self):
        self.name = 'HumanAgent'

    def observe(self, game, player):
        self.hand = player.hand
        self.briscola = game.briscola
        self.played_cards = game.played_cards

    def select_action(self, actions, render=False):
        """Parse user input from the keyboard.
        If it's not a valid action index, do something random.
        """
        if not render:
            print("Your turn!")
            print(f"The briscola is {self.briscola.name}.")
            print(f"Your hand is: {[card.name for card in self.hand]}.")

            try:
                action = int(input('Input: '))
            except ValueError:
                print("Error, not a number!!")
                action = random.choice(actions)

            if action not in actions:
                print("Error, out of bounds action selected!!")
                action = random.choice(actions)
        else:
            action = random.choice(actions)

        return action

    def update(self, reward):
        pass

    def make_greedy(self):
        pass

    def restore_epsilon(self):
        pass
