import random
import time


class HumanAgent:
    """Agent controlled via keyboard input."""

    def __init__(self):
        self.action = None
        self.played_cards = None
        self.hand = None
        self.briscola = None
        self.name = 'HumanAgent'

    def observe(self, game, player):
        self.hand = player.hand
        self.briscola = game.briscola
        self.played_cards = game.played_cards

    def played_card(self, action):
        """
        Method called from the GUI necessary for specifying the played card

        @param action: index [0, 2] that is the played card
        """
        self.action = action

    def select_action(self, actions, gui_obj=None, render=False):
        """Parse user input from the keyboard.
        If it's not a valid action index, do something random.

        @param actions: list of available actions
        @param gui_obj: BriscolaGui object for controlling the GUI
        @param render: True is you want logs, False otherwise

        @return the index of the chosen action, -1 if the gui called a reset
        """
        if not render:
            print("Your turn!")
            print(f"The briscola is {self.briscola.name}.")
            print(f"Your hand is: {[card.name for card in self.hand]}.")
            try:
                # GUI: waiting for the user to select a card from the GUI
                if gui_obj is not None:
                    with gui_obj.cond:
                        gui_obj.activate_restart(gui_obj)
                        gui_obj.cond.wait()
                    if not gui_obj.reset:
                        gui_obj.deactivate_restart()
                        action = self.action
                    else:
                        gui_obj.reset = False
                        return -1
                else:
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
    
    def reset(self):
        pass