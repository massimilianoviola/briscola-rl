import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networks.dqn import ReplayMemory, DQN
from typing import List


class QAgent:
    """Q-learning agent"""

    def __init__(
            self,
            n_features: int,
            n_actions: int,
            replay_memory_capacity: int,
            minimum_training_samples: int,
            batch_size: int,
            discount: float,
            loss_fn,
            learning_rate: float,
            replace_every: int,
            epsilon: float,
            minimum_epsilon: float,
            epsilon_decay_rate: float,
            layers: List[int] = [256, 256],
            device=None,
    ) -> None:
        """"""
        self.name = "QLearningAgent"

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.n_features = n_features
        self.n_actions = n_actions

        self.last_state = None
        self.action = None
        self.reward = None
        self.state = None

        self.deck = np.zeros((10, 4))

        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.discount = discount

        self.replay_memory = ReplayMemory(replay_memory_capacity)
        self.minimum_training_samples = minimum_training_samples
        self.batch_size = batch_size

        self.layers = layers
        self.policy_net = DQN(n_features, n_actions, layers).to(self.device)
        self.target_net = DQN(n_features, n_actions, layers).to(self.device)

        self.loss_fn = loss_fn
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=learning_rate,
        )
        self.training_iterations = 0
        self.current_ep = 0
        self.replace_every = replace_every

        self.loss_log = []

    def observe(self, env, player):
        """Observes the environment and updates the state"""
        self.get_state_just_hand(env, player)

    def get_state_just_hand_one_hot_encoding(self, env, player):
        """The state vector encodes 5 cards: the 3 cards in the player's hand,
        the card on the desk and the briscola.
        Each card is encoded by 14 entries of the state, 10 for the value and
        4 for the seed. For example, "Asso di bastoni" is encoded as
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        """
        state = np.zeros(self.n_features)

        # Add player's hand to state
        for i, card in enumerate(player.hand):
            number_index = i * 14 + card.number
            seed_index = i * 14 + 10 + card.seed
            state[number_index] = 1
            state[seed_index] = 1

        # Add played cards to state
        for i, card in enumerate(env.played_cards):
            number_index = (i + 3) * 14 + card.number
            seed_index = (i + 3) * 14 + 10 + card.seed
            state[number_index] = 1
            state[seed_index] = 1

        # Add briscola to state
        number_index = 4 * 14 + env.briscola.number
        seed_index = 4 * 14 + 10 + env.briscola.seed
        state[number_index] = 1
        state[seed_index] = 1

        self.last_state = self.state
        self.state = state
        self.done = env.check_end_game()

    def get_state_just_hand(self, env, player):
        """The state vector encodes the player's score (points) and four cards.
        Each card is encoded as a vector of length 6:
        first entry: numerical value of the card.
        second entry: boolean value (1 if Briscola, 0 otherwise).
        last four entries: one hot encoding of the seeds.
        For example "Asso di bastoni" is encoded as:
        [0, 1, 1, 0, 0, 0] if the briscola is "bastoni".
        """
        state = np.zeros(self.n_features)
        value_offset = 2
        seed_offset = 2
        features_per_card = 6
        state[0] = player.points
        state[1] = env.counter

        for i, card in enumerate(player.hand):
            number_index = i * features_per_card + value_offset
            seed_index = i * features_per_card + seed_offset + card.seed + value_offset
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1

        for i, card in enumerate(env.played_cards):
            number_index = (i + 3) * features_per_card + value_offset
            seed_index = (i + 3) * features_per_card + seed_offset + card.seed + value_offset
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1

        self.last_state = self.state
        self.state = state
        self.done = env.check_end_game()

    def get_state(self, env, player):
        """To the state obtained from self.get_state_just_hand() is appended
        a vector of length 40. Each entry is associated with one card
        in the deck. At each step of the episode the agent observes the played
        cards and the cards in his hand and sets the corresponding entries to 1
        """
        state = np.zeros(26)
        value_offset = 2
        seed_offset = 2
        features_per_card = 6
        state[0] = player.points
        state[1] = env.counter

        for i, card in enumerate(player.hand):
            number_index = i * features_per_card + value_offset
            seed_index = i * features_per_card + seed_offset + card.seed + value_offset
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1

            self.deck[card.number][card.seed] = 1

        for i, card in enumerate(env.played_cards):
            number_index = (i + 3) * features_per_card + value_offset
            seed_index = (i + 3) * features_per_card + seed_offset + card.seed + value_offset
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1

            self.deck[card.number][card.seed] = 1

        deck = self.deck.flatten()
        state = np.concatenate((state, deck))

        self.last_state = self.state
        self.state = state
        self.done = env.check_end_game()

    def select_action(self, available_actions):
        """Selects action according to an epsilon-greedy policy"""
        state = torch.from_numpy(self.state).float().to(self.device)
        if np.random.uniform() < self.epsilon:
            # Select a random action with probability epsilon
            action = np.random.choice(available_actions)
        else:
            # Select a greedy action with probability 1 - epsilon
            self.policy_net.eval()
            with torch.no_grad():
                output = self.policy_net(state)
                sorted_actions = (-output).argsort()
            for predicted_action in sorted_actions:
                if predicted_action in available_actions:
                    action = predicted_action
                    break

        self.action = action
        return action

    def update(self, reward):
        """After receiving reward r, collects (s, a, r, s', done) and adds it
        to replay_memory and learns.
        """
        self.reward = reward
        if self.done:
            self.current_ep += 1
        if self.epsilon > self.minimum_epsilon:
            self.update_epsilon()
        else:
            self.epsilon = self.minimum_epsilon

        self.replay_memory.push(
            self.last_state, self.action, self.reward, self.state, self.done
        )

        self.learn()

    def learn(self):
        """Samples batch of experience from replay_memory,
        if it has enough samples, and trains the policy network on it
        """
        # If there are not enough samples in replay_memory: do nothing
        if len(self.replay_memory) < self.minimum_training_samples:
            return

        # Otherwise sample a batch
        self.training_iterations += 1
        batch = self.replay_memory.sample(self.batch_size)

        states = torch.tensor(
            np.array([x[0] for x in batch]),
            dtype=torch.float32,
        ).to(self.device)
        actions = torch.tensor(
            np.array([x[1] for x in batch]),
            dtype=torch.int64,
        ).to(self.device)
        rewards = torch.tensor(
            np.array([x[2] for x in batch]),
            dtype=torch.float32,
        ).to(self.device)
        next_states = torch.tensor(
            np.array([x[3] for x in batch]),
            dtype=torch.float32,
        ).to(self.device)
        done = torch.tensor(
            np.array([x[4] for x in batch]),
            dtype=torch.bool,
        ).to(self.device)

        self.policy_net.train()
        # computes Q(s, a1), Q(s, a_2), ... , Q(s, a_n)
        q_values = self.policy_net(states)
        # gets the right Q(s, a)
        q_state_action = q_values.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            self.target_net.eval()
            # computes Q'(s', a_1), Q'(s', a_2), ..., Q'(s', a_n)
            target_q_values = self.target_net(next_states)
        # gets max_a {Q(s', a)}
        next_state_max_q = target_q_values.max(dim=1)[0]

        # r + discount * Q_max(s)
        target = rewards + (self.discount * next_state_max_q * (1 - done.float()))
        target = target.unsqueeze(1)

        loss = self.loss_fn(q_state_action, target)
        self.optimizer.zero_grad()

        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        self.loss_log.append(loss.detach().cpu().numpy())

    def make_greedy(self):
        """Makes the agent greedy for evaluation"""
        self.epsilon_backup = self.epsilon
        self.epsilon = 0.0

    def restore_epsilon(self):
        """Restores epsilon to backup value"""
        self.epsilon = self.epsilon_backup

    def update_epsilon(self):
        """Updates epsilon"""
        self.epsilon *= self.epsilon_decay_rate

    def reset(self):
        self.deck = np.zeros((10, 4))

    def save(self, path):
        """Saves policy network's state dictionary to path"""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """Loads policy network's state dictionary from path"""
        self.policy_net.load_state_dict(torch.load(path))
