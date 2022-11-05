import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networks.dqn import DRQN, ReplayMemory


class RecurrentDeepQAgent:
    """Q-learning agent using a recurrent network"""

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
        num_recurrent_layers: int = 1,
        hidden_size: int = 256,
        fully_connected_layers: int = 256,
    ) -> None:
        """"""
        self.name = "RecurrentDeepQLearningAgent"

        # Network parameters
        self.n_features = n_features
        self.n_actions = n_actions

        self.policy_net = DRQN(
            n_features,
            hidden_size,
            n_actions,
            num_recurrent_layers,
            fully_connected_layers,
        )
        self.target_net = DRQN(
            n_features,
            hidden_size,
            n_actions,
            num_recurrent_layers,
            fully_connected_layers,
        )

        self.h, self.c = self.policy_net.init_hidden(batch_size=1)

        self.replay_memory = ReplayMemory(replay_memory_capacity)
        self.minimum_training_samples = minimum_training_samples
        self.replace_every = replace_every
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        # Reinforcement learning parameters
        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.discount = discount
        self.current_ep = 0
        self.training_iterations = 0

        self.last_state = None
        self.action = None
        self.reward = None
        self.state = None

    def observe(self, env, player) -> None:
        """"""
        self.get_state_just_hand(env, player)

    def get_state_just_hand(self, env, player):
        """The state vector encodes the player's score (points) and four cards.
        Each card is encoded as a vector of length 6:
        first entry: numerical value of the card.
        second entry: boolean value (1 if brisoscola 0 otherwise).
        last four entries: one hot encoding of the seeds.
        For example "Asso di bastoni" is encoded as:
        [0, 1, 1, 0, 0, 0] if the briscola is "bastoni".
        """
        state = np.zeros(self.n_features)
        value_offset = 1
        seed_offset = 3
        features_per_card = 6
        state[0] = player.points

        for i, card in enumerate(player.hand):
            number_index = i * features_per_card + value_offset
            seed_index = i * features_per_card + seed_offset + card.seed
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1

        for i, card in enumerate(env.played_cards):
            number_index = i + 3 * features_per_card + value_offset
            seed_index = i + 3 * features_per_card + seed_offset + card.seed
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1

        self.last_state = self.state
        self.state = state
        self.done = env.check_end_game()

    def select_action(self, available_actions):
        """Selects action according to an epsilon-greedy policy"""
        state = torch.from_numpy(self.state).float()
        state = state.reshape(1, -1, self.n_features)

        if np.random.uniform() < self.epsilon:
            # Select a random action with probability epsilon
            action = np.random.choice(available_actions)
        else:
            # Select a greedy action with probability 1 - epsilon
            self.policy_net.eval()
            with torch.no_grad():
                output, (self.h, self.c) = self.policy_net(state, self.h, self.c)
                output = output[0][0]
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
            self.h, self.c = self.policy_net.init_hidden(batch_size=1)
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
        ).reshape(self.batch_size, 1, self.n_features)
        actions = torch.tensor(
            np.array([x[1] for x in batch]),
            dtype=torch.int64,
        )
        rewards = torch.tensor(
            np.array([x[2] for x in batch]),
            dtype=torch.float32,
        )
        next_states = torch.tensor(
            np.array([x[3] for x in batch]),
            dtype=torch.float32,
        ).reshape(self.batch_size, 1, self.n_features)

        """
        print(f"STATES shape {states.shape}")
        print(f"ACTIONS shape {actions.shape}")
        print(f"REWARDS shape {rewards.shape}")
        print(f"NEXT_STATES shape {next_states.shape}")
        """
        self.policy_net.train()
        h, c = self.policy_net.init_hidden(self.batch_size)
        # computes Q(s, a1), Q(s, a_2), ... , Q(s, a_n)
        q_values, (h, c) = self.policy_net(states, h, c)
        # gets the right Q(s, a)
        # print("-"*140)
        # print(f"Q_VALUES shape {q_values.shape}")
        q_values = q_values.reshape(self.batch_size, self.n_actions)
        # print(f"Q_VALUES shape {q_values.shape}")
        q_state_action = q_values.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            self.target_net.eval()
            ht, ct = self.target_net.init_hidden(self.batch_size)
            # computes Q'(s', a_1), Q'(s', a_2), ..., Q'(s', a_n)
            target_q_values, (ht, ct) = self.target_net(next_states, ht, ct)
        # gets max_a {Q(s', a)}

        # print(f"TARGET_Q_VALUES shape {q_values.shape}")
        target_q_values = target_q_values.reshape(self.batch_size, self.n_actions)
        # print(f"TARGET_Q_VALUES shape {q_values.shape}")

        next_state_max_q = target_q_values.max(dim=1)[0]

        # r + discount * Q_max(s)
        target = rewards + self.discount * next_state_max_q
        target = target.unsqueeze(1)

        loss = self.loss_fn(q_state_action, target)
        self.optimizer.zero_grad()

        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # self.loss_log.append(loss.detach().numpy())

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

    def save(self, path):
        """Saves policy network's state dictionary to path"""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """Loads policy network's state dictionary from path"""
        self.policy_net.load_state_dict(torch.load(path))


