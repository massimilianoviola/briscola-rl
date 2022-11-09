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

    def push(self, transitions) -> None:
        """Stores a sequence of transitions (s, a, r, s', done)"""
        self.memory.append(transitions)

    def sample(self, batch_size: int) -> Tensor:
        """
        :returns: a sample of size batch_size
        """
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)


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
        # x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.out(x)
        return x, h

    def init_hidden(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> Tuple[Tensor, Tensor]:
        h = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return h, c
