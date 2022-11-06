import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch import Tensor
from typing import Tuple, List


class ReplayMemory(object):
    """Implements an experience replay memory"""

    def __init__(self, capacity: int) -> None:
        """
        :param capacity: capacity of the experience replay memory
        """
        self.memory = deque(maxlen=capacity)

    def __len__(self) -> int:
        """
        :returns: size of the experience replay memory
        """
        return len(self.memory)

    def push(self, state, action, reward, next_state, done) -> None:
        """Stores a tuple (s, a, r, s', done)
        :param state: current state s
        :param action: action taken a
        :param reward: reward r obtained taking action a being in state s
        :param next_state: state s' reached taking action a being in state s
        :param done: True if a final state is reached
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tensor:
        """
        :returns: a sample of size batch_size
        """
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)


class DQN(nn.Module):
    """Simple feed-forward neural network with two fully connected layers"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: List[int] = [256, 256],
        activation=nn.ReLU(),
    ) -> None:
        """
        :param input_dim: size of the state vector
        :param output_dim: number of possible actions
        :param activation: activation function for the hidden layers
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=layers[0])
        self.fc2 = nn.Linear(in_features=layers[0], out_features=layers[1])
        self.out = nn.Linear(in_features=layers[1], out_features=output_dim)
        self.act = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.out(x)
        return x


class DRQN(nn.Module):
    """Simple network with recurrent layers and one fully connected layer"""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        num_recurrent_layers: int,
        fc_size: int = 256,
        fc_activation=nn.ReLU(),
        dropout_rate: float = 0.5,
    ) -> None:
        """"""
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_dim, hidden_size, num_recurrent_layers, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.out = nn.Linear(fc_size, output_dim)
        self.act = fc_activation

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> Tensor:
        x, h = self.rnn(x, (h, c))
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.out(x)
        return x, h

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        h = torch.zeros(1, batch_size, self.hidden_size)
        c = torch.zeros(1, batch_size, self.hidden_size)
        return h, c
