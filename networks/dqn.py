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
        self.memory = deque(maxlen=capacity)

    def __len__(self) -> int:
        """
        :returns: size of the experience replay memory
        """
        return len(self.memory)

    def push(self, state, action, reward, next_state, done) -> None:
        """Stores a tuple (s, a, r, s', done)"""
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
