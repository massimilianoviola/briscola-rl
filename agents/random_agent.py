import random


class RandomAgent:
    """Agent selecting random available actions."""

    def __init__(self):
        self.name = 'RandomAgent'

    def observe(self, *_):
        pass

    def select_action(self, actions):
        return random.choice(actions)

    def update(self, reward):
        pass

    def make_greedy(self):
        pass

    def restore_epsilon(self):
        pass