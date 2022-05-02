import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networks.dqn import *


class QAgent():
    """
    Q-learning agent
    """
    def __init__(self, n_features, n_actions, replay_memory_capacity, minimum_training_samples, batch_size, discount, loss_fn, learning_rate, replace_every, initial_epsilon, final_epsilon, decay_epsilon_until):
        """
        :param n_features: dimension of the state vector
        :param n_actions: number of possible actions
        :param replay_memory_capacity: capacity of the replay_memory
        :param minimum_training_samples: minimum amount of training samples in replay_memory required to start training
        :param batch_size: size of the batch extracted from replay_memory at each training step
        :param discount: discount factor for computing the return
        :param loss_fn: loss function used to optimize the policy network
        :param learning_rate: gradient descent learning rate
        :param replace_every: number of training iterations of the policy network before its weight are copied to the target network
        :param initial_epsilon: intial value of epsilon
        :param final_epsilon: final value of epsilon
        :param decay_epsilon_until: number of iteration after which epsilon=final_epsilon

        :param n_features: dimension of the state vector
        :param n_actions: number of possible actions
        :param policy_net: policy network
        :param target_net: target network
        :param optimizer: optimizer for the policy network
        :param training_iterations: training iteration counter
        :param current_ep: current episode number
        """
        self.name = "QLearningAgent"
        self.n_features = n_features
        self.n_actions = n_actions

        self.last_state = None
        self.action = None
        self.reward = None
        self.state = None

        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon = initial_epsilon
        self.discount = discount
        self.decay_epsilon_until = decay_epsilon_until

        self.replay_memory = ReplayMemory(replay_memory_capacity)
        self.minimum_training_samples = minimum_training_samples
        self.batch_size = batch_size

        self.policy_net = DQN(n_features, n_actions, nn.ReLU())
        self.target_net = DQN(n_features, n_actions, nn.ReLU())
        self.loss_fn = loss_fn
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.training_iterations = 0
        self.current_ep = 0
        self.replace_every = replace_every

    def observe(self, env, player):
        """
        Observes the environmet and updates the state
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

    def select_action(self, available_actions):
        """
        Selects one action from the available actions according to an epsilon-greedy policy
        """
        # state = torch.tensor(self.next_state, dtype=torch.float32)
        state = torch.from_numpy(self.state).float()
        #print(state)
        if np.random.uniform() > self.epsilon:
            action = np.random.choice(available_actions)
        else:
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
        """
        After receiving reward r, collects (s, a, r, s', done) and adds it to replay_memory and learns
        """
        self.reward = reward
        if self.done:
            self.current_ep += 1
        self.update_epsilon()
        self.replay_memory.push(self.last_state, self.action, self.reward, self.state, self.done)
        self.learn()

    def learn(self):
        """
        Samples batch of experience from replay_memory, if it has enough samples, and trains the policy network on it
        """
        
        # If there are not enough samples in replay_memory: do nothing
        if len(self.replay_memory) < self.minimum_training_samples:
            return
        # Otherwise sample a batch
        self.training_iterations += 1
        batch  = self.replay_memory.sample(self.batch_size)

        states = torch.tensor([x[0] for x in batch], dtype=torch.float32)
        actions = torch.tensor([x[1] for x in batch], dtype=torch.int64)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        next_states = torch.tensor([x[3] for x in batch], dtype=torch.float32)
        done = torch.tensor([x[4] for x in batch], dtype=torch.bool)

        self.policy_net.train()                                     
        q_values = self.policy_net(states)                          # computes Q(s, a_1), Q(s, a_2), ... , Q(s, a_n)
        q_state_action = q_values.gather(1, actions.unsqueeze(1))   # gets the right Q(s, a)

        with torch.no_grad():
            self.target_net.eval()                                  
            target_q_values = self.target_net(next_states)          # computes Q'(s', a_1), Q'(s', a_2), ..., Q'(s', a_n)
        next_state_max_q = target_q_values.max(dim=1)[0]            # gets max_a {Q(s', a)}

        target = rewards + self.discount * next_state_max_q         # r + discount * Q_max(s)
        target = target.unsqueeze(1)

        loss = self.loss_fn(q_state_action, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # After every replace_every iterations copy the weghts of policy_net to target_net
        if self.training_iterations % self.replace_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def make_greedy(self):
        """
        """
        self.epsilon_backup = self.epsilon
        self.epsilon = 1.0

    def restore_epsilon(self):
        """
        """
        self.epsilon = self.epsilon_backup

    def update_epsilon(self):
        """
        Updates epsilon 
        """
        temp =  (self.decay_epsilon_until - self.current_ep)/self.decay_epsilon_until
        r = max(temp, 0)
        self.epsilon = r * (self.initial_epsilon - self.final_epsilon) + self.final_epsilon

