import random
from utils import BriscolaLogger


class BriscolaCard:
    """Create a Briscola card with its attributes."""

    def __init__(self):
        self.id = -1  # index of the one-hot encoded card in the deck [0, len(deck) - 1]
        self.name = ""  # name to display
        self.seed = -1  # seed/suit of the card [0, 3]
        self.number = -1  # face value of the card [0, 9]
        self.strength = -1  # card rank during the game [0, 9]
        self.points = -1  # point value of the card [0, 11]

    def __str__(self):
        return self.name


class BriscolaDeck:
    """Initialize a deck of Briscola cards with its attributes."""

    def __init__(self):
        self.create_decklist()
        self.reset()

    def create_decklist(self):
        """Create all the BriscolaCards and add them to the deck."""
        points = [11, 0, 10, 0, 0, 0, 0, 2, 3, 4]
        strengths = [9, 0, 8, 1, 2, 3, 4, 5, 6, 7]
        seeds = ["Bastoni", "Coppe", "Denari", "Spade"]
        names = [
            "Asso",
            "Due",
            "Tre",
            "Quattro",
            "Cinque",
            "Sei",
            "Sette",
            "Fante",
            "Cavallo",
            "Re",
        ]

        self.deck = []
        id = 0
        for s, seed in enumerate(seeds):
            for n, name in enumerate(names):
                card = BriscolaCard()
                card.id = id
                card.name = f"{name} di {seed}"
                card.seed = s
                card.number = n
                card.strength = strengths[n]
                card.points = points[n]
                self.deck.append(card)
                id += 1

    def reset(self):
        """Prepare the deck for a new game."""
        self.briscola = None
        self.end_deck = False
        self.current_deck = self.deck.copy()
        self.shuffle()

    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.current_deck)

    def place_briscola(self, briscola):
        """Set a card as briscola and allow to draw it after the last card of the deck."""
        if self.briscola is not None:
            raise ValueError(
                "Trying BriscolaDeck.place_briscola, but BriscolaDeck.briscola is not None"
            )
        self.briscola = briscola

    def draw_card(self):
        """If the deck is not empty, draw a card, otherwise return the briscola or nothing."""
        if self.current_deck:
            drawn_card = self.current_deck.pop()
        else:
            drawn_card = self.briscola
            self.briscola = None
            self.end_deck = True
        return drawn_card

    def get_deck_size(self):
        """Size of the full deck."""
        return len(self.deck)

    def get_current_deck_size(self):
        """Size of the current deck."""
        current_deck_size = len(self.current_deck)
        current_deck_size += 1 if self.briscola else 0
        return current_deck_size

    def __len__(self):
        return len(self.current_deck)

    def __str__(self):
        str_ = ""
        for el in self.current_deck:
            str_ += el.__str__() + ", "
        return str_


class BriscolaPlayer:
    """Create basic player actions."""

    def __init__(self, _id):
        self.id = _id
        self.reset()

    def reset(self):
        """Reset hand and points when starting a new game."""
        self.hand = []
        self.points = 0

    def draw(self, deck):
        """Try to draw a card from the deck."""
        new_card = deck.draw_card()
        if new_card is not None:
            self.hand.append(new_card)
        if len(self.hand) > 3:
            raise ValueError(
                "Calling BriscolaPlayer.draw caused the player to have more than 3 cards in hand!"
            )

    def play_card(self, hand_index):
        """Try to play a card from the hand and return the chosen card or an exception if invalid index."""
        try:
            card = self.hand[hand_index]
            del self.hand[hand_index]
            return card
        except:
            raise ValueError("BriscolaPlayer.play_card called with invalid hand_index!")


class BriscolaGame:
    """Create the game environment with all the game stages."""

    def __init__(self, num_players=2, logger=BriscolaLogger()):
        self.num_players = num_players
        self.deck = BriscolaDeck()
        self.logger = logger
        self.won_the_match_points = False
        self.counter = 1
        self.screen = None

    def reset(self):
        """Start a new game."""
        # initialize the deck
        self.deck.reset()
        self.history = []
        self.played_cards = []

        # initialize the players
        self.players = [BriscolaPlayer(i) for i in range(self.num_players)]
        self.turn_player = random.randint(0, self.num_players - 1)
        self.players_order = self.get_players_order()

        # initialize the briscola
        self.briscola = self.deck.draw_card()
        self.deck.place_briscola(self.briscola)

        # initialize players' hands
        for _ in range(0, 3):
            for i in self.players_order:
                self.players[i].draw(self.deck)

        self.won_the_match_points = False
        self.counter = 1
        self.screen = None

    def reorder_hand(self, player_id):
        """Rearrange the cards in a player's hand from strongest to weakest,
        taking into account the Briscola seed.
        """
        player = self.players[player_id]
        # bubble sort algorithm using scoring() as a comparator
        for passnum in range(len(player.hand) - 1, 0, -1):
            for i in range(passnum):
                if scoring(
                    self.briscola.seed,
                    player.hand[i],
                    player.hand[i + 1],
                    keep_order=False,
                ):
                    temp = player.hand[i]
                    player.hand[i] = player.hand[i + 1]
                    player.hand[i + 1] = temp

    def get_player_actions(self, player_id):
        """Get the list of available actions for a player."""
        player = self.players[player_id]
        return list(range(len(player.hand)))

    def get_players_order(self):
        """Compute the clockwise players order starting from the current turn player."""
        players_order = [
            i % self.num_players
            for i in range(self.turn_player, self.turn_player + self.num_players)
        ]
        return players_order

    def draw_step(self):
        """Each player, in order, tries to draw a card from the deck."""
        self.logger.PVP(f"\n----------- NEW TURN -----------[{self.counter}]")
        # clear the table for the play_step
        self.played_cards = []
        # draw the cards in order
        for player_id in self.players_order:
            player = self.players[player_id]
            player.draw(self.deck)

    def play_step(self, action, player_id):
        """A player executes a chosen action."""
        player = self.players[player_id]

        self.logger.DEBUG(
            f"Player {player_id} hand: {[card.name for card in player.hand]}."
        )
        self.logger.DEBUG(f"Player {player_id} chose action {action}.")

        card = player.play_card(action)
        self.logger.PVP(f"Player {player_id} played {card.name}.")

        self.played_cards.append(card)
        self.history.append(card)

    def get_rewards_from_step(self):
        """Compute the reward for each player based on the cards just played.
        Note that this is a reward in RL terms, not actual game points.
        """
        winner_player_id, points = self.evaluate_step()

        rewards = []
        extra_points = 100  # Points for winning
        count = 0
        for player_id in self.get_players_order():
            reward = points if player_id is winner_player_id else -points
            # reward = points if player_id is winner_player_id else 0

            # Reward for winning the match

            player = self.players[player_id]
            if self.won_the_match_points == False:
                if player.points >= 60 and reward > 0:
                    # print(f"PLAYER POINTS {player.points}")
                    reward += extra_points
                    self.won_the_match_points = True
                    if count == 0:
                        return [reward, -reward]
                    else:
                        return [-reward, reward]

            count += 1
            rewards.append(reward)

        # print(rewards)
        return rewards

    def evaluate_step(self):
        """Look at the cards played and decide which player won the hand."""
        ordered_winner_id, strongest_card = get_strongest_card(
            self.briscola.seed, self.played_cards
        )
        winner_player_id = self.players_order[ordered_winner_id]

        points = sum([card.points for card in self.played_cards])
        winner_player = self.players[winner_player_id]

        self.update_game(winner_player, points)
        self.logger.PVP(
            f"Player {winner_player_id} wins {points} points with {strongest_card.name}."
        )

        return winner_player_id, points

    def check_end_game(self):
        """Check if the game is over."""
        end_deck = self.deck.end_deck
        player_has_cards = False
        for player in self.players:
            if player.hand:
                player_has_cards = True
                break

        return end_deck and not player_has_cards

    def get_winner(self):
        """Return the player with the most points and the winning amount."""
        winner_player_id = -1
        winner_points = -1

        for player in self.players:
            if player.points > winner_points:
                winner_player_id = player.id
                winner_points = player.points

        return winner_player_id, winner_points

    def end_game(self):
        """End the game and return the winner."""
        if not self.check_end_game():
            raise ValueError(
                "Calling BriscolaGame.end_game when the game has not ended!"
            )

        winner_player_id, winner_points = self.get_winner()
        self.logger.PVP(f"Player {winner_player_id} wins with {winner_points} points!!")

        return winner_player_id, winner_points

    def update_game(self, winner_player, points):
        """Update the scores and the order based on who won the previous hand."""
        winner_player_id = winner_player.id
        winner_player.points += points

        self.turn_player = winner_player_id
        self.players_order = self.get_players_order()
        self.counter += 1


def get_strongest_card(briscola_seed, cards):
    """Return the strongest card in the given set,
    taking into account the Briscola seed.
    """
    ordered_winner_id = 0
    strongest_card = cards[0]

    for ordered_id, card in enumerate(cards[1:]):
        ordered_id += 1  # adjustment since we are starting from the first element
        pair_winner = scoring(briscola_seed, strongest_card, card)
        if pair_winner == 1:
            ordered_winner_id = ordered_id
            strongest_card = card

    return ordered_winner_id, strongest_card


def get_weakest_card(briscola_seed, cards):
    """Return the weakest card in the given set,
    taking into account the Briscola seed.
    """
    ordered_loser_id = 0
    weakest_card = cards[0]

    for ordered_id, card in enumerate(cards[1:]):
        ordered_id += 1  # adjustment since we are starting from the first element
        pair_winner = scoring(briscola_seed, weakest_card, card, keep_order=False)
        if pair_winner == 0:
            ordered_loser_id = ordered_id
            weakest_card = card

    return ordered_loser_id, weakest_card


def scoring(briscola_seed, card_0, card_1, keep_order=True):
    """Compare a pair of cards and decide which one wins.
    The keep_order argument indicates whether the first card played has a priority.
    """
    # only one card is of the briscola seed
    if briscola_seed != card_0.seed and briscola_seed == card_1.seed:
        winner = 1
    elif briscola_seed == card_0.seed and briscola_seed != card_1.seed:
        winner = 0
    # same seed, briscola or not
    elif card_0.seed == card_1.seed:
        winner = 1 if card_1.strength > card_0.strength else 0
    # if of different seeds and none of them is briscola, the first one wins
    else:
        winner = 0 if keep_order or card_0.strength > card_1.strength else 1

    return winner


def play_episode(game, agents, train=True):
    """Play an entire game updating the environment at each step.
    rewards_log will contain as key the agent's name and as value a list
    containing all the rewards that the agent has received at each step.
    """
    game.reset()
    rewards_log = {agent.name: [] for agent in agents}
    rewards = []
    while not game.check_end_game():
        # action step
        players_order = game.get_players_order()

        # print("-"*140)

        for i, player_id in enumerate(players_order):
            player = game.players[player_id]
            agent = agents[player_id]

            # print(f"{agent.name} turn")

            # the agent observes the state before acting
            agent.observe(game, player)

            if train and rewards:
                agent.update(rewards[i])
                rewards_log[agent.name].append(rewards[i])

            available_actions = game.get_player_actions(player_id)
            action = agent.select_action(available_actions)

            # if agent.name == "QLearningAgent":
            # print(f"{agent.name} state {agent.state}")

            # print(f"{agent.name} plays {player.hand[action]} ({action})")

            game.play_step(action, player_id)

            # Update agents deck since it can only observe the environment when
            # it's its turn. So for example if it's the first one to play he
            # can only exclude the card he played and the one played by the
            # opponent. In the next turn played_cards will be empty.
            for card in game.played_cards:
                if agents[0].name == "QLearningAgent":
                    agents[0].deck[card.number][card.seed] = 1

        # print("PLAYED:")
        # for card in game.played_cards:
        # print(card)

        # update the environment
        rewards = game.get_rewards_from_step()

        # for i, player_id in enumerate(game.get_players_order()):
        # print(f"{agents[player_id].name} gets reward {rewards[i]}")

        game.draw_step()

    # update for the terminal state
    for i, player_id in enumerate(players_order):
        player = game.players[player_id]
        agent = agents[player_id]
        agent.observe(game, player)
        if train and rewards:
            agent.update(rewards[i])

    return *game.end_game(), rewards_log
