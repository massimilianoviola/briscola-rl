import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import environment as brisc


class AIAgent:
    """Agent playing using predefined heuristics."""

    def __init__(self):
        self.name = 'AIAgent'

    def observe(self, game, player):
        """Store information about the state of the game to be used in the decisional process."""
        self.hand = player.hand
        self.points = player.points
        self.played_cards = game.played_cards
        self.briscola_seed = game.briscola.seed

    def select_action(self, actions):
        """Default set of rules that the agent follows to play."""

        # count how many points are present on the table
        points_on_table =  0
        for played_card in self.played_cards:
            points_on_table += played_card.points

        if points_on_table:
            # there is at least 1 card on the table that it's worth some points
            # retrieve the winning cards in hand
            win_actions = []
            points = []
            for action_index, card in enumerate(self.hand):
                for played_card in self.played_cards:
                    winner = brisc.scoring(self.briscola_seed, played_card, card)
                    if winner:
                        win_actions.append(action_index)
                        points.append(card.points)
            
            # the agent can win this hand
            if win_actions:
                # cards that can win the hand, sorted according to the points that can be earned by playing them
                sorted_win_actions = [x for _, x in sorted(zip(points, win_actions), reverse=True)]
                best_action = sorted_win_actions[0]
                best_card = self.hand[best_action]

                if self.points + points_on_table + best_card.points > 60:
                    # winning this hand means winning the game, do it
                    return best_action

                # if it is possible to win without using a briscola, do it
                for win_action in sorted_win_actions:
                    if self.hand[win_action].seed != self.briscola_seed:
                        return win_action

                # if here, all the cards winning the hand are briscola
                if len(win_actions) == 1:
                    # if there is only one card to win the hand (which is a briscola)
                    if points_on_table >= 10:
                        # always play it if there are many points on the table
                        return best_action

                    # if there are not many points on the table and the other losing cards in hand are carichi, play the briscola
                    # if a low-point, non-briscola alternative can be played to lose the hand, play the alternative
                    lose_action = -1
                    lose_points = 10
                    for action_index, card in enumerate(self.hand):
                        if action_index not in win_actions and card.points < lose_points and card.seed != self.briscola_seed:
                            lose_action = action_index
                            lose_points = card.points
                    if lose_action == -1:
                        # only have carichi as losing cards and 1 winning briscola in hand, play the briscola
                        return best_action
                    elif best_card.points >= 3:
                        # if the only winning briscola is a knight or higher, lose the hand on purpose, 
                        # as there are not many points on the table
                        return lose_action
                    elif lose_points == 4:
                        # two possible hands: 2 kings + 1 briscola up to jack or 1 carico + 1 king + 1 briscola up to jack
                        if points_on_table >= 3:
                            # if there are at least 3 points on the table, play briscola up to jack
                            return best_action
                        else:
                            return lose_action
                    else:
                        # a less than 4 points, non-briscola, losing alternative can be played, do it
                        return lose_action
                else:
                    # there is more than one briscola available to win the hand, play the weakest
                    win_cards = [self.hand[i] for i in win_actions]
                    weakest_win_index, weakest_win_card = brisc.get_weakest_card(self.briscola_seed, win_cards)
                    return win_actions[weakest_win_index]

        # if here, it is not possible to win the hand or there are no points on the table (there may be no cards at all)
        # find the weakest card (it may be a carico if the other cards in hand are briscola) and play it
        weakest_index, weakest_card = brisc.get_weakest_card(self.briscola_seed, self.hand)
        if weakest_card.points > 4:
            # it is safer/better to play a small briscola than a carico
            low_points_sorted_cards = sorted(self.hand, key=lambda card: card.strength)
            return self.hand.index(low_points_sorted_cards[0])
        else:
            return weakest_index

    def update(self, reward):
        pass

    def make_greedy(self):
        pass

    def restore_epsilon(self):
        pass